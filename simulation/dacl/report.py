"""Aggregate results over seeds: tables (CSV + LaTeX) and IEEE-sized figures."""
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6,
                     "legend.fontsize": 7, "lines.linewidth": 1.0, "savefig.dpi": 300})

COLORS = {"Static": "#7f7f7f", "FineTune": "#d62728", "ER": "#1f77b4", "EWC": "#9467bd",
          "FullRetrain": "#8c564b", "DriftFT": "#ff7f0e", "DACL": "#2ca02c",
          "DACL-noKD": "#17becf", "DACL-noReplay": "#bcbd22", "DACL-fixedK": "#e377c2"}

MAIN_COLS = [("acc", "Acc.", 100), ("f1", "Macro-F1", 100), ("det_delay", "Det. delay", 1),
             ("recovery_batches", "Recovery", 1), ("forgetting", "Forgetting", 100),
             ("return_recall", "Return rec.", 100), ("infer_us_per_sample", "Inf. (us/s)", 1),
             ("update_ms_per_batch", "Upd. (ms/b)", 1), ("update_cpu_s", "Upd. CPU (s)", 1),
             ("state_mb", "Mem. (MB)", 1), ("energy_j", "Energy (J)", 1)]


def _fmt(m, s, scale):
    if np.isnan(m):
        return "--"
    return f"{m * scale:.2f} $\\pm$ {s * scale:.2f}"


def make_tables(summary, out_dir, methods):
    g = summary.groupby("method")
    mean, std = g.mean(numeric_only=True), g.std(numeric_only=True).fillna(0)
    mean = mean.reindex([m for m in methods if m in mean.index])
    std = std.reindex(mean.index)
    agg = mean.add_suffix("_mean").join(std.add_suffix("_std"))
    agg.to_csv(out_dir / "summary_mean_std.csv")
    cols = [c for c in MAIN_COLS if c[0] in mean.columns]
    lines = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
             "Method & " + " & ".join(c[1] for c in cols) + " \\\\", "\\midrule"]
    for m in mean.index:
        cells = [_fmt(mean.loc[m, c], std.loc[m, c], s) for c, _, s in cols]
        lines.append(f"{m} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out_dir / "table_main.tex").write_text("\n".join(lines))
    return mean, std


def _shade(ax, phase_start, phase_names, n):
    edges = list(phase_start) + [n]
    for i in range(len(phase_start)):
        if i % 2:
            ax.axvspan(edges[i], edges[i + 1], color="0.93", lw=0)
        ax.text((edges[i] + edges[i + 1]) / 2, 1.0, phase_names[i].split()[0],
                transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7)
    for s in phase_start[1:]:
        ax.axvline(s, color="0.4", ls="--", lw=0.6)


def timeline_figure(timelines, meta, methods, col, ylabel, path, smooth=5):
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    n = len(next(iter(timelines.values()))[0])
    _shade(ax, meta["phase_start"], meta["phase_names"], n)
    for m in methods:
        if m not in timelines:
            continue
        arr = np.stack([d[col].to_numpy() for d in timelines[m]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mu = pd.Series(np.nanmean(arr, 0)).rolling(smooth, min_periods=1).mean()
        ax.plot(mu.index, mu * 100, color=COLORS.get(m), label=m,
                lw=1.4 if m == "DACL" else 0.9, zorder=3 if m == "DACL" else 2)
    ax.set_xlabel("Stream batch")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, n)
    fig.legend(*ax.get_legend_handles_labels(), ncol=4, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), handlelength=1.5, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def tradeoff_figure(mean, std, path):
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for m in [m for m in mean.index if not m.startswith("DACL-")]:
        x, y = mean.loc[m, "update_cpu_s"], mean.loc[m, "acc"] * 100
        ax.errorbar(x, y, xerr=std.loc[m, "update_cpu_s"], yerr=std.loc[m, "acc"] * 100,
                    fmt="o", ms=4, color=COLORS.get(m), capsize=2, label=m)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("Total update CPU time (s)")
    ax.set_ylabel("Prequential accuracy (%)")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def build_report(out_dir, methods):
    out_dir = Path(out_dir)
    summary = pd.read_csv(out_dir / "summary_per_seed.csv")
    meta = json.loads((out_dir / "meta.json").read_text())
    methods = [m for m in methods if m in set(summary.method)]
    mean, std = make_tables(summary, out_dir, methods)
    timelines = {}
    for f in sorted((out_dir / "timelines").glob("*.csv")):
        m = f.stem.rsplit("_s", 1)[0]
        timelines.setdefault(m, []).append(pd.read_csv(f))
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    main = [m for m in methods if not m.startswith("DACL-")]
    timeline_figure(timelines, meta, main, "acc", "Accuracy (%)", fig_dir / "fig_accuracy_over_time")
    timeline_figure(timelines, meta, main, "A_recall", "Attack-A recall (%)", fig_dir / "fig_attackA_recall")
    abl = [m for m in methods if m.startswith("DACL") or m == "DriftFT"]
    if len(abl) > 1:
        timeline_figure(timelines, meta, abl, "acc", "Accuracy (%)", fig_dir / "fig_ablation_accuracy")
    tradeoff_figure(mean, std, fig_dir / "fig_accuracy_vs_update_cost")
    return mean, std
