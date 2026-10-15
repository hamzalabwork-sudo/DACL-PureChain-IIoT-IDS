"""Label-scarcity analysis: 1 / 5 / 10 % labelled flows (grid `labels`) + 100 % (stage1b, MLP).

  python -m dacl.labels_report --out results/labels_report
"""
import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

DS = {"wustl_iiot": "WUSTL", "xiiotid": "X-IIoTID", "edge_iiotset": "Edge-IIoTset",
      "forge_iiot": "ForgeIIOT", "mu_iot": "MU-IoT"}
METHODS = ["Static", "ER", "DER++", "FullRetrain", "DACL"]
COLORS = {"Static": "#7f7f7f", "ER": "#1f77b4", "DER++": "#9467bd", "FullRetrain": "#8c564b", "DACL": "#2ca02c"}


def load(labels_root, full_root):
    rows = []
    for f in glob.glob(f"{labels_root}/lab_*/mlp/s*/summary_per_seed.csv"):
        _, b, ds = Path(f).parts[-4].split("_", 2)
        rows.append(pd.read_csv(f).assign(dataset=ds, budget=int(b)))
    for f in glob.glob(f"{full_root}/*/mlp/s*/summary_per_seed.csv"):
        ds = Path(f).parts[-4]
        rows.append(pd.read_csv(f).assign(dataset=ds, budget=100))
    df = pd.concat(rows, ignore_index=True)
    return df[df.method.isin(METHODS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-root", default="results/labels")
    ap.add_argument("--full-root", default="results/stage1b")
    ap.add_argument("--out", default="results/labels_report")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load(args.labels_root, args.full_root)
    df.to_csv(out / "all_runs.csv", index=False)
    md = ["# Label scarcity (MLP, 5 seeds per cell)", "",
          "Only the given fraction of flows in each batch is labelled. Learning uses labelled flows only; "
          "DACL's unsupervised drift signals (entropy, feature shift) see the whole batch, its error "
          "signal only the labelled flows.", ""]
    budgets = sorted(df.budget.unique())
    # overall tests: DACL vs best baseline (ER / DER++) per budget, pooled datasets x seeds
    md += ["## DACL vs baselines per label budget (macro-F1, N = 25 dataset x seed pairs)", ""]
    for b in budgets:
        piv = df[df.budget == b].pivot_table(index=["dataset", "seed"], columns="method", values="f1")
        parts = []
        for m in ("ER", "DER++", "FullRetrain"):
            d = (piv["DACL"] - piv[m]).dropna()
            p = stats.wilcoxon(d).pvalue if d.abs().sum() > 0 else 1.0
            parts.append(f"vs {m} {d.median() * 100:+.2f} pts (p={p:.1g}, wins {int((d > 0).sum())}/{len(d)})")
        md.append(f"- **{b}% labels**: " + "; ".join(parts))
    md.append("")
    fig, axes = plt.subplots(1, len(DS), figsize=(7.2, 1.9), sharey=False)
    for ax, (ds, lab) in zip(axes, DS.items()):
        x = df[df.dataset == ds]
        rows = []
        for b in budgets:
            for m in METHODS:
                y = x[(x.budget == b) & (x.method == m)]
                if y.empty:
                    continue
                rows.append({"Labels (%)": b, "Method": m,
                             "Macro-F1 (%)": f"{y.f1.mean() * 100:.2f} ± {y.f1.std() * 100:.2f}",
                             "Detection (%)": f"{y.detect.mean() * 100:.1f}",
                             "FPR (%)": f"{y.fpr.mean() * 100:.2f}",
                             "MAE (flows)": f"{y.exposure_missed_attacks.mean():.0f}",
                             "Forgetting (%)": f"{y.forgetting.mean() * 100:.1f}",
                             "Det. delay": (f"{y.det_delay.mean():.1f}" if "det_delay" in y and y.det_delay.notna().any() else "--"),
                             "Missed drifts": (f"{y.missed_drifts.mean():.1f}" if "missed_drifts" in y and y.missed_drifts.notna().any() else "--")})
        t = pd.DataFrame(rows)
        t.to_csv(out / f"{ds}_labels.csv", index=False)
        md += [f"## {lab}", "", t.to_markdown(index=False), ""]
        for m in METHODS:
            g = x[x.method == m].groupby("budget")["f1"].agg(["mean", "std"])
            ax.errorbar(g.index, g["mean"] * 100, yerr=g["std"] * 100, marker="o", ms=3, capsize=2,
                        color=COLORS[m], label=m, lw=1.2 if m == "DACL" else 0.8)
        ax.set_xscale("log")
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(b) for b in budgets], fontsize=6)
        ax.set_title(lab, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("Labelled flows (%)", fontsize=6)
    axes[0].set_ylabel("Macro-F1 (%)", fontsize=6)
    axes[0].legend(fontsize=5, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "f1_vs_label_budget.pdf")
    fig.savefig(out / "f1_vs_label_budget.png", dpi=300)
    plt.close(fig)
    md += ["![F1 vs label budget](f1_vs_label_budget.png)", ""]
    (out / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
