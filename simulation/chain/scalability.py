"""Live PureChain scalability of the model-update gate: N gateways proposing updates concurrently.

  python -m chain.scalability --gateways 1 5 10 --rounds 5 --out results/chain_scalability

Each gateway runs `rounds` update cycles (propose -> 2 validator attestations, threshold 2) in its
own thread; the 3 validators are shared by all gateways (one validator committee). Measures per-tx
confirmation latency, end-to-end decision latency per update, throughput, gas, event-log bytes and
on-chain state growth. PureChain is gas-free (gas price 0), so gas is reported as work, not cost.
"""
import argparse
import hashlib
import json
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from chain.gate import CHAIN_ID, HERE, SECRETS, bps, connect

# storage written per accepted update (ModelUpdateGate.sol layout, 32-byte slots):
# Update struct = 4 slots (gateway | modelHash | parentHash | packed ids/counters/status),
# attested[id][validator] = 1 slot x 2 validators; activeModel and cumDropBps are overwritten.
NEW_SLOTS_PER_UPDATE = 4 + 2


class Chain:
    def __init__(self):
        from eth_account import Account
        self.w3 = connect()
        acc = json.loads(SECRETS.read_text())
        extra = acc.setdefault("gateways_extra", [])
        while len(extra) < 9:
            extra.append(Account.create().key.hex())
        SECRETS.write_text(json.dumps(acc, indent=2))
        self.owner = Account.from_key(acc["gateway"])
        self.gateways = [Account.from_key(k) for k in [acc["gateway"]] + extra]
        self.validators = [Account.from_key(k) for k in acc["validators"]]
        art = json.loads((HERE / "ModelUpdateGate.json").read_text())
        self.abi, self.bin = art["abi"], art["bin"]
        self.nonce, self.locks, self.glock = {}, {}, threading.Lock()
        self.records = []

    def _lock(self, addr):
        with self.glock:
            return self.locks.setdefault(addr, threading.Lock())

    def send(self, acct, fn_or_tx, label, meta=None):
        with self._lock(acct.address):  # one in-flight tx per account keeps nonces ordered
            if acct.address not in self.nonce:
                self.nonce[acct.address] = self.w3.eth.get_transaction_count(acct.address, "pending")
            tx = (fn_or_tx.build_transaction({"from": acct.address, "gas": 3_000_000, "gasPrice": 0})
                  if hasattr(fn_or_tx, "build_transaction") else fn_or_tx)
            for k in ("maxFeePerGas", "maxPriorityFeePerGas", "type"):
                tx.pop(k, None)
            tx.update({"from": acct.address, "nonce": self.nonce[acct.address], "chainId": CHAIN_ID,
                       "gasPrice": 0})
            signed = acct.sign_transaction(tx)
            t0 = time.perf_counter()
            h = self.w3.eth.send_raw_transaction(signed.raw_transaction)
            r = self.w3.eth.wait_for_transaction_receipt(h, timeout=300, poll_latency=0.1)
            dt = time.perf_counter() - t0
            self.nonce[acct.address] += 1
        assert r.status == 1, f"{label} reverted {h.hex()}"
        log_bytes = sum(len(l["data"]) + 32 * len(l["topics"]) for l in r.logs)
        self.records.append({"label": label, "latency_s": dt, "gas": r.gasUsed, "log_bytes": log_bytes,
                             "block": r.blockNumber, **(meta or {})})
        return r

    def deploy(self):
        C = self.w3.eth.contract(abi=self.abi, bytecode=self.bin)
        tx = C.constructor(2, bps(0.30), bps(0.01), bps(0.30), bps(0.10)).build_transaction(
            {"from": self.owner.address, "gas": 3_000_000, "gasPrice": 0})
        r = self.send(self.owner, tx, "deploy")
        self.c = self.w3.eth.contract(address=r.contractAddress, abi=self.abi)
        for v in self.validators:
            self.send(self.owner, self.c.functions.addValidator(v.address), "addValidator")
        self.active = {}
        for g in self.gateways:
            genesis = hashlib.sha256(f"genesis-{g.address}".encode()).digest()
            self.send(self.owner, self.c.functions.registerGateway(g.address, genesis), "register")
            self.active[g.address] = genesis
        return r.contractAddress


def gateway_worker(ch, gi, rounds, n, out):
    g = ch.gateways[gi]
    for r in range(rounds):
        mh = hashlib.sha256(f"{g.address}-{n}-{r}-{time.time()}".encode()).digest()
        t0 = time.perf_counter()
        rc = ch.send(g, ch.c.functions.proposeUpdate(mh, ch.active[g.address], r + 1, bps(0.8)),
                     "propose", {"n_gateways": n})
        uid = ch.c.events.UpdateProposed().process_receipt(rc)[0]["args"]["id"]
        v1, v2 = [(0, 1), (1, 2), (0, 2)][(gi + r) % 3]  # spread load over the committee
        for v in (v1, v2):
            rc = ch.send(ch.validators[v], ch.c.functions.attest(uid, 0, bps(0.97), bps(0.98),
                                                                 bps(0.95), bps(0.96)),
                         "attest", {"n_gateways": n})
        accepted = bool(ch.c.events.UpdateAccepted().process_receipt(rc))
        if accepted:
            ch.active[g.address] = mh
        out.append({"n_gateways": n, "gateway": gi, "round": r, "accepted": accepted,
                    "decision_s": time.perf_counter() - t0})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateways", type=int, nargs="*", default=[1, 5, 10])
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", default="results/chain_scalability")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ch = Chain()
    addr = ch.deploy()
    print(f"contract {addr}; {len(ch.gateways)} gateways, {len(ch.validators)} validators registered")
    decisions, summary = [], []
    for n in args.gateways:
        rec0, dec0 = len(ch.records), len(decisions)
        t0 = time.perf_counter()
        ths = [threading.Thread(target=gateway_worker, args=(ch, gi, args.rounds, n, decisions))
               for gi in range(n)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        wall = time.perf_counter() - t0
        recs = pd.DataFrame(ch.records[rec0:])
        dec = pd.DataFrame(decisions[dec0:])
        lat = recs["latency_s"].to_numpy()
        summary.append({
            "n_gateways": n, "updates": len(dec), "accepted": int(dec.accepted.sum()),
            "txs": len(recs), "wall_s": wall, "throughput_tx_s": len(recs) / wall,
            "updates_per_min": len(dec) / wall * 60,
            "tx_lat_mean_s": lat.mean(), "tx_lat_p95_s": np.percentile(lat, 95),
            "tx_lat_p99_s": np.percentile(lat, 99),
            "decision_mean_s": dec.decision_s.mean(), "decision_p95_s": np.percentile(dec.decision_s, 95),
            "blocks_used": recs.block.nunique(),
            "gas_propose": recs[recs.label == "propose"].gas.mean(),
            "gas_attest": recs[recs.label == "attest"].gas.mean(),
            "log_bytes_per_update": recs.log_bytes.sum() / len(dec),
            "state_bytes_per_update": NEW_SLOTS_PER_UPDATE * 32})
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary[-1].items()}))
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(ch.records).to_csv(out / "transactions.csv", index=False)
    pd.DataFrame(decisions).to_csv(out / "decisions.csv", index=False)
    (out / "contract.json").write_text(json.dumps({"address": addr}, indent=2))


if __name__ == "__main__":
    main()
