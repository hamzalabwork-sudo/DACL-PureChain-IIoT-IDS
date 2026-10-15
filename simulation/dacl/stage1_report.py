"""Stage-1 report: backbone x dataset tables, Friedman test and Nemenyi critical-difference diagram."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .report import COLORS  # noqa: F401  (shared style)

plt.rcParams.update({"font.size": 8, "font.family": "serif", "savefig.dpi": 300})

ARCH_LABEL = {"mlp": "MLP", "cnn": "1D-CNN", "lstm": "LSTM", "gru": "GRU",
              "cnn_lstm": "CNN-LSTM", "transformer": "FT-Transformer"}
DS_LABEL = {"wustl_iiot": "WUSTL-IIoT", "xiiotid": "X-IIoTID", "edge_iiotset": "Edge-IIoTset",
            "forge_iiot": "Forge-IIoT", "mu_iot": "MU-IoT"}
# Nemenyi critical values q_alpha (alpha = 0.05) for k = 2..10 (Demsar, JMLR 2006)
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def load_all(root, datasets, archs):
    frames = []
    for d in datasets:
        for a in archs:
            for f in sorted((Path(root) / d / a).glob("**/summary_per_seed.csv")):
                frames.append(pd.read_csv(f).assign(dataset=d, arch=a))
    return pd.concat(frames, ignore_index=True)


def friedman_nemenyi(scores, higher_better=True):
    """scores: DataFrame blocks x treatments. Returns (chi2, p, avg_ranks, CD)."""
    s = scores.dropna()
    ranks = (-s if higher_better else s).rank(axis=1, method="average")
    chi2, p = stats.friedmanchisquare(*[s[c] for c in s.columns])
    k, n = s.shape[1], s.shape[0]
    cd = Q05[k] * np.sqrt(k * (k + 1) / (6.0 * n))
    return chi2, p, ranks.mean().sort_values(), cd


def cd_diagram(avg_ranks, cd, path, title):
    """Demsar-style CD diagram: rank 1 (best) on the left, bars join groups not significantly different."""
    k = len(avg_ranks)
    names, vals = list(avg_ranks.index), avg_ranks.values
    fig, ax = plt.subplots(figsize=(3.5, 0.9 + 0.13 * k))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.plot([1, k], [0.9, 0.9], color="k", lw=0.8)
    for r in range(1, k + 1):
        ax.plot([r, r], [0.9, 0.93], color="k", lw=0.8)
        ax.text(r, 0.95, str(r), ha="center", fontsize=7)
    half = (k + 1) // 2
    for i, (name, r) in enumerate(zip(names, vals)):
        left = i < half
        y = 0.75 - 0.11 * (i if left else k - 1 - i)
        xe = 0.6 if left else k + 0.4
        ax.plot([r, r, xe], [0.9, y, y], color="k", lw=0.6)
        ax.text(xe + (-0.05 if left else 0.05), y, f"{ARCH_LABEL.get(name, name)} ({r:.2f})",
                ha="right" if left else "left", va="center", fontsize=7)
    cliques = []
    for i in range(k):
        j = max(j for j in range(i, k) if vals[j] - vals[i] < cd)
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    for n, (i, j) in enumerate(cliques):
        y = 0.84 - 0.04 * n
        ax.plot([vals[i] - 0.03, vals[j] + 0.03], [y, y], color="k", lw=2)
    # CD scale bar just below the lowest label (no empty band)
    y_cd = 0.75 - 0.11 * (half - 1) - 0.14
    ax.plot([1, 1 + cd], [y_cd, y_cd], color="k", lw=1)
    ax.text(1 + cd / 2, y_cd + 0.03, f"CD = {cd:.2f}", ha="center", fontsize=7)
    ax.set_ylim(y_cd - 0.05, 1)
    ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def build_stage1_report(root, datasets, archs):
    root = Path(root)
    df = load_all(root, datasets, archs)
    out = root / "report"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "all_runs.csv", index=False)

    metrics = ["acc", "f1", "mcc", "h_score", "forgetting", "exposure_missed_attacks",
               "update_cpu_s", "params", "flops", "lat1_ms_p50", "lat1_ms_p99"]
    agg = df.groupby(["dataset", "arch", "method"])[metrics].agg(["mean", "std"])
    agg.to_csv(out / "mean_std_by_dataset_arch_method.csv")

    dacl = df[df.method == "DACL"]
    lines = []
    for metric, hb in [("f1", True), ("h_score", True), ("acc", True)]:
        # blocks = datasets (seed-averaged) — the standard Demsar setting
        tab = dacl.groupby(["dataset", "arch"])[metric].mean().unstack()[archs]
        tab.rename(index=DS_LABEL, columns=ARCH_LABEL).round(4).to_csv(out / f"dacl_{metric}_dataset_x_arch.csv")
        chi2, p, ranks, cd = friedman_nemenyi(tab, hb)
        lines.append(f"[{metric}] blocks=datasets (N={len(tab.dropna())}): Friedman chi2={chi2:.2f} "
                     f"p={p:.4f}  CD={cd:.2f}  avg ranks: " +
                     ", ".join(f"{ARCH_LABEL[a]}={r:.2f}" for a, r in ranks.items()))
        cd_diagram(ranks, cd, out / "figures" / f"cd_{metric}_datasets", f"DACL {metric}, N = {len(tab)} datasets")
        # blocks = dataset x seed (more power; seeds within a dataset are not fully independent)
        tab2 = dacl.pivot_table(index=["dataset", "seed"], columns="arch", values=metric)[archs]
        chi2, p, ranks, cd = friedman_nemenyi(tab2, hb)
        lines.append(f"[{metric}] blocks=dataset x seed (N={len(tab2.dropna())}): Friedman chi2={chi2:.2f} "
                     f"p={p:.2e}  CD={cd:.2f}  avg ranks: " +
                     ", ".join(f"{ARCH_LABEL[a]}={r:.2f}" for a, r in ranks.items()))
        cd_diagram(ranks, cd, out / "figures" / f"cd_{metric}_runs", f"DACL {metric}, N = {len(tab2)} runs")

    # DACL vs ER per backbone: paired Wilcoxon over dataset x seed
    for a in archs:
        x = df[(df.arch == a)].pivot_table(index=["dataset", "seed"], columns="method", values="f1")
        if {"DACL", "ER"} <= set(x.columns) and len(x.dropna()) > 5:
            w = stats.wilcoxon(x["DACL"], x["ER"])
            lines.append(f"[Wilcoxon DACL vs ER, f1, {ARCH_LABEL[a]}] median diff="
                         f"{(x['DACL'] - x['ER']).median():+.4f} p={w.pvalue:.2e} (N={len(x)})")

    # cost table (deployed DACL model)
    cost = dacl.groupby("arch")[["params", "flops", "lat1_ms_p50", "lat1_ms_p99", "update_cpu_s"]].mean()
    cost.rename(index=ARCH_LABEL).round(4).to_csv(out / "backbone_cost.csv")
    _pareto(dacl, out / "figures" / "pareto_f1_vs_update_cpu", archs)
    (out / "stats.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(cost.rename(index=ARCH_LABEL).round(4).to_string())


def _pareto(dacl, path, archs):
    m = dacl.groupby("arch")[["f1", "update_cpu_s"]].mean().reindex(archs)
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for a, r in m.iterrows():
        ax.scatter(r.update_cpu_s, r.f1 * 100, s=18)
        ax.annotate(ARCH_LABEL[a], (r.update_cpu_s, r.f1 * 100), textcoords="offset points",
                    xytext=(4, 2), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Update CPU time per stream (s, log)")
    ax.set_ylabel("Macro-F1 (%), mean of 5 datasets")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
