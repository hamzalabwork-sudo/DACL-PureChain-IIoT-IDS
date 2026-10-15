"""Analysis of the net-security-benefit gate (grid `gate2`).

  python -m dacl.gate2_report --root results/gate2 --out results/gate2_report

Per dataset x setting x hard floor h x method: macro-F1, binary detection rate / FNR / FPR,
missed attacks during exposure (MAE), known- and new-attack detectability (any-attack) and exact
recall, gate rejection / escalation rates and poisoned-update acceptance (P4 attack). Wilcoxon tests
pool datasets x seeds (N = 25) per setting.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DS = {"wustl_iiot": "WUSTL", "xiiotid": "X-IIoTID", "edge_iiotset": "Edge-IIoTset",
      "forge_iiot": "ForgeIIOT", "mu_iot": "MU-IoT"}
SETTING = {"gate": "Clean", "poisonP2": "Poison during drift (P2)", "poisonP4": "Unlearning attack (P4)"}
METHODS = ["DACL", "DACL+Gate", "DACL+Gate-dual"]


def load(root, prefix="gate2"):
    rows = []
    for f in glob.glob(f"{root}/{prefix}_*/mlp/s*/summary_per_seed.csv"):
        p = Path(f)
        body = p.parts[-4][len(prefix) + 1:]
        rest, h = body.rsplit("_h", 1) if "_h" in body else (body, 30)
        ds, setting = rest.rsplit("_", 1)
        s = pd.read_csv(f).assign(dataset=ds, setting=setting, h=int(h))
        meta = json.loads((p.parent / "meta.json").read_text())
        p4 = meta["phase_start"][3]
        extra = {}
        for l in open(p.parent / "phase_eval.jsonl"):
            j = json.loads(l)
            fin = j["phase_eval"][-1]
            g = j.get("gate_log", [])
            pois = [x for x in g if setting == "poisonP4" and x["t"] >= p4]
            extra[j["method"]] = {
                "A_detect_final": fin.get("A_detect", np.nan), "B_detect_final": fin.get("B_detect", np.nan),
                "n_checks": len(g), "rejected": sum(not x["accepted"] for x in g),
                "escalated": sum(bool(x.get("escalated")) for x in g),
                "poisoned_checks": len(pois), "poisoned_accepted": sum(x["accepted"] for x in pois)}
        for k in ("A_detect_final", "B_detect_final", "n_checks", "rejected", "escalated",
                  "poisoned_checks", "poisoned_accepted"):
            s[k] = [extra.get(m, {}).get(k, np.nan) for m in s.method]
        rows.append(s)
    return pd.concat(rows, ignore_index=True)


def fmt(x, c, scale=100, d=1):
    return f"{x[c].mean() * scale:.{d}f} ± {x[c].std() * scale:.{d}f}"


def table(df, h):
    out = []
    for ds in DS:
        for st in SETTING:
            for m in METHODS:
                x = df[(df.dataset == ds) & (df.setting == st) & (df.h == h) & (df.method == m)]
                if x.empty:
                    continue
                gated = m != "DACL"
                chk = x.n_checks.sum()
                out.append({
                    "Dataset": DS[ds], "Setting": SETTING[st], "Method": m,
                    "Macro-F1 (%)": fmt(x, "f1"), "Detection rate (%)": fmt(x, "detect"),
                    "FPR (%)": fmt(x, "fpr", 100, 2), "MAE (flows)": fmt(x, "exposure_missed_attacks", 1, 0),
                    "Known-attack detectability (%)": fmt(x, "A_detect_final"),
                    "Known-attack exact recall (%)": fmt(x, "final_A_recall"),
                    "New-attack detectability (%)": fmt(x, "B_detect_final"),
                    "New-attack exact recall (%)": fmt(x, "plasticity_B"),
                    "Rejected updates": f"{x.rejected.sum() / chk * 100:.1f}%" if gated and chk else "--",
                    "Escalated updates": f"{x.escalated.sum() / chk * 100:.1f}%" if gated and chk else "--",
                    "Poisoned updates accepted": (
                        "--" if st != "poisonP4" else
                        ("100%" if not gated else
                         f"{x.poisoned_accepted.sum() / max(1, x.poisoned_checks.sum()) * 100:.1f}% "
                         f"({int(x.poisoned_accepted.sum())}/{int(x.poisoned_checks.sum())})"))})
    return pd.DataFrame(out)


def tests(df, h):
    lines = []
    for st in SETTING:
        for m in ("DACL+Gate", "DACL+Gate-dual"):
            x = df[(df.setting == st) & (df.h == h)]
            piv = x.pivot_table(index=["dataset", "seed"], columns="method",
                                values=["f1", "exposure_missed_attacks", "A_detect_final", "detect"])
            parts = []
            for k, lab, sc in (("f1", "F1", 100), ("detect", "detection", 100),
                               ("exposure_missed_attacks", "MAE", 1), ("A_detect_final", "known-attack det.", 100)):
                d = (piv[(k, m)] - piv[(k, "DACL")]).dropna()
                p = stats.wilcoxon(d).pvalue if d.abs().sum() > 0 else 1.0
                parts.append(f"{lab} {d.median() * sc:+.2f} (p={p:.1g})")
            lines.append(f"- {SETTING[st]}, {m} vs DACL (N = {len(d)}): " + "; ".join(parts))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/gate2")
    ap.add_argument("--out", default="results/gate2_report")
    ap.add_argument("--prefix", default="gate2")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load(args.root, args.prefix)
    df.to_csv(out / "all_runs.csv", index=False)
    md = ["# Net-security-benefit PureChain gate (MLP, 5 datasets x 5 seeds)", ""]
    for h in sorted(df.h.unique()):
        t = table(df, h)
        t.to_csv(out / f"gate2_h{h}.csv", index=False)
        md += [f"## Hard floor h = {h}% (soft/escalation limit 10%, macro tolerance 1%)", "",
               *tests(df, h), "", t.to_markdown(index=False), ""]
    summ = df.groupby(["setting", "h", "method"])[["f1", "detect", "fpr", "exposure_missed_attacks",
                                                  "A_detect_final", "B_detect_final"]].mean()
    summ.to_csv(out / "summary_by_setting_h_method.csv")
    md += ["## Mean over all datasets", "", summ.round(4).to_markdown(), ""]
    (out / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
