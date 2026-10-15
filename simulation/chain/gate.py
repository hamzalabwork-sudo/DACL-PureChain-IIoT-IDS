"""Model-update gate backends.

  LocalGate      same acceptance policy as ModelUpdateGate.sol, in Python (fast offline runs)
  PureChainGate  live PureChain (Geth-based, PoA^2, chain id 900520900520) via web3

Both expose: register(genesis_hash), propose(model_hash, drift_id, severity) -> update id,
attest(validator_idx, id, class_id, cand=(class_detectability, BA), parent=(...)) -> status,
and record per-call latency. Policy (same as ModelUpdateGate.sol): per-update class and
detection drops within tolerance, and a per-class cumulative (CUSUM) drop budget.

Accounts: generated locally on first use and stored in .secrets/purechain_accounts.json
(gitignore/keep private). PURECHAIN_PRIVATE_KEY, if set, is used as the owner/deployer.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SECRETS = HERE.parent / ".secrets" / "purechain_accounts.json"
DEPLOYMENT = HERE / "deployment.json"
RPC = "https://purechainnode.com"
CHAIN_ID = 900520900520

PENDING, ACCEPTED, REJECTED = 1, 2, 3


def model_hash(model):
    """SHA-256 over the model's parameters (deterministic order)."""
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.digest()


def bps(x):
    return int(round(float(np.clip(x, 0, 1)) * 10_000))


class LocalGate:
    backend = "local"

    def __init__(self, threshold=2, max_recall_drop=0.05, max_detect_drop=0.05, max_cum_drop=0.05,
                 soft_drop=1.0):
        self.threshold = threshold
        self.max_recall_drop, self.max_detect_drop = bps(max_recall_drop), bps(max_detect_drop)
        self.max_cum_drop, self.cum = bps(max_cum_drop), {}
        self.soft_drop, self.escalated = bps(soft_drop), set()
        self.updates, self.latency = [], []

    def register(self, genesis_hash):
        self.active = genesis_hash

    def propose(self, mhash, drift_id, severity):
        t = time.perf_counter()
        self.updates.append({"hash": mhash, "parent": self.active, "passes": 0, "fails": 0,
                             "status": PENDING})
        self.latency.append(("propose", time.perf_counter() - t))
        return len(self.updates) - 1

    def attest(self, v, uid, cls, cand, parent):
        """cand/parent = (worst known-class detectability, macro detectability over all classes)."""
        t = time.perf_counter()
        u = self.updates[uid]
        (cr, cd), (pr, pd) = [(bps(a), bps(b)) for a, b in (cand, parent)]
        d_cls, d_det = pr - cr, pd - cd
        cum = max(0, self.cum.get(cls, 0) + d_cls)
        ok = d_cls <= self.max_recall_drop and d_det <= self.max_detect_drop and cum <= self.max_cum_drop
        if u["status"] == PENDING:
            if ok:
                u["passes"] += 1
                if u["passes"] >= self.threshold:
                    u["status"], self.active, self.cum[cls] = ACCEPTED, u["hash"], cum
                    if d_cls > self.soft_drop:
                        self.escalated.add(uid)
            else:
                u["fails"] += 1
                if u["fails"] >= self.threshold:
                    u["status"] = REJECTED
        self.latency.append(("attest", time.perf_counter() - t))
        return u["status"]


def _accounts(n_validators=3):
    from eth_account import Account
    if SECRETS.exists():
        acc = json.loads(SECRETS.read_text())
    else:
        SECRETS.parent.mkdir(parents=True, exist_ok=True)
        acc = {"gateway": Account.create().key.hex(),
               "validators": [Account.create().key.hex() for _ in range(n_validators)]}
        SECRETS.write_text(json.dumps(acc, indent=2))
    acc["owner"] = os.environ.get("PURECHAIN_PRIVATE_KEY") or acc["gateway"]
    return acc


def connect(rpc=RPC):
    import truststore
    truststore.inject_into_ssl()
    from web3 import Web3
    from web3.middleware import ExtraDataToPOAMiddleware
    w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 60}))
    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    assert w3.eth.chain_id == CHAIN_ID, f"unexpected chain id {w3.eth.chain_id}"
    return w3


class PureChainGate:
    backend = "purechain"

    def __init__(self, threshold=2, max_recall_drop=0.05, max_detect_drop=0.05, max_cum_drop=0.05,
                 soft_drop=1.0, address=None, deploy_new=True):
        from eth_account import Account
        self.w3 = connect()
        acc = _accounts()
        self.owner = Account.from_key(acc["owner"])
        self.gateway = Account.from_key(acc["gateway"])
        self.validators = [Account.from_key(k) for k in acc["validators"]]
        art = json.loads((HERE / "ModelUpdateGate.json").read_text())
        self.abi, self.bin = art["abi"], art["bin"]
        self.latency, self.gas, self.nonce, self.last_block = [], [], {}, 0
        self.escalated = set()
        if address is None and not deploy_new and DEPLOYMENT.exists():
            address = json.loads(DEPLOYMENT.read_text())["address"]
        if address is None:
            address = self._deploy(threshold, bps(max_recall_drop), bps(max_detect_drop), bps(max_cum_drop),
                                   bps(soft_drop))
            for v in self.validators:
                self._tx(self.owner, self.c.functions.addValidator(v.address), "addValidator")
        self.c = self.w3.eth.contract(address=address, abi=self.abi)
        self.address = address

    def _send(self, acct, tx, label):
        # legacy (type-0) transaction: PureChain's gas price and base fee are 0
        for k in ("maxFeePerGas", "maxPriorityFeePerGas", "type"):
            tx.pop(k, None)
        if acct.address not in self.nonce:
            self.nonce[acct.address] = self.w3.eth.get_transaction_count(acct.address, "pending")
        tx.update({"from": acct.address, "nonce": self.nonce[acct.address],
                   "chainId": CHAIN_ID, "gasPrice": 0})
        tx.setdefault("gas", 3_000_000)
        signed = acct.sign_transaction(tx)
        t = time.perf_counter()
        h = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        rcpt = self.w3.eth.wait_for_transaction_receipt(h, timeout=180, poll_latency=0.1)
        dt = time.perf_counter() - t
        self.nonce[acct.address] += 1
        assert rcpt.status == 1, f"{label} reverted: {h.hex()}"
        self.last_block = rcpt.blockNumber
        self._sync()
        self.latency.append((label, dt))
        self.gas.append((label, rcpt.gasUsed))
        return rcpt

    def _sync(self, timeout=30):
        """Read-after-write: wait until the (load-balanced) RPC node has the last mined block."""
        t = time.time()
        while self.w3.eth.block_number < self.last_block and time.time() - t < timeout:
            time.sleep(0.05)

    def _tx(self, acct, fn, label, gas=500_000):
        # explicit gas: skips eth_estimateGas, which may run on a lagging node
        return self._send(acct, fn.build_transaction({"from": acct.address, "gas": gas,
                                                      "gasPrice": 0}), label)

    def _deploy(self, threshold, rdrop, ddrop, cdrop, sdrop):
        C = self.w3.eth.contract(abi=self.abi, bytecode=self.bin)
        tx = C.constructor(threshold, rdrop, ddrop, cdrop, sdrop).build_transaction(
            {"from": self.owner.address, "gas": 3_000_000, "gasPrice": 0})
        rcpt = self._send(self.owner, tx, "deploy")
        self.c = self.w3.eth.contract(address=rcpt.contractAddress, abi=self.abi)
        DEPLOYMENT.write_text(json.dumps({"address": rcpt.contractAddress, "tx": rcpt.transactionHash.hex(),
                                          "block": rcpt.blockNumber, "owner": self.owner.address,
                                          "gateway": self.gateway.address,
                                          "validators": [v.address for v in self.validators],
                                          "deployed_at": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
        return rcpt.contractAddress

    def register(self, genesis_hash):
        self._tx(self.owner, self.c.functions.registerGateway(self.gateway.address, genesis_hash),
                 "register")
        self.active = genesis_hash  # tracked locally: reads may hit a lagging RPC node

    def propose(self, mhash, drift_id, severity):
        rcpt = self._tx(self.gateway, self.c.functions.proposeUpdate(mhash, self.active, int(drift_id),
                                                                     bps(severity)), "propose")
        ev = self.c.events.UpdateProposed().process_receipt(rcpt)[0]
        self.pending_hash = mhash
        return ev["args"]["id"]

    def attest(self, v, uid, cls, cand, parent):
        fn = self.c.functions.attest(uid, int(cls), bps(cand[0]), bps(parent[0]), bps(cand[1]), bps(parent[1]))
        rcpt = self._tx(self.validators[v], fn, "attest")
        # status from the receipt's events (authoritative), not from a possibly stale read
        if self.c.events.UpdateAccepted().process_receipt(rcpt):
            self.active = self.pending_hash
            if self.c.events.UpdateEscalated().process_receipt(rcpt):
                self.escalated.add(uid)
            return ACCEPTED
        if self.c.events.UpdateRejected().process_receipt(rcpt):
            return REJECTED
        return PENDING


def make_gate(backend, **kw):
    return PureChainGate(**kw) if backend == "purechain" else LocalGate(**kw)
