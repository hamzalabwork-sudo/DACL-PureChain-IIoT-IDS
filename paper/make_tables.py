"""Generate every LaTeX table and paper figure directly from the simulation results.

  cd paper && python make_tables.py

No number in the paper's tables is typed by hand. Reads ../simulation/results/*.
"""
import glob
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
SIM = HERE.parent / "simulation"
R = SIM / "results"
sys.path.insert(0, str(SIM))
TAB, FIG = HERE / "tables", HERE / "figures"
TAB.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

DS = ["wustl_iiot", "xiiotid", "edge_iiotset", "forge_iiot", "mu_iot"]
DSL = {"wustl_iiot": "WUSTL", "xiiotid": "X-IIoTID", "edge_iiotset": "Edge-IIoTset",
       "forge_iiot": "ForgeIIOT", "mu_iot": "MU-IoT"}
NAT = ["natural_wustl_iiot", "natural_forge_iiot", "natural_xiiotid"]
ARCH = ["mlp", "cnn", "lstm", "gru", "cnn_lstm", "transformer"]
ARCHL = {"mlp": "MLP", "cnn": "1D-CNN", "lstm": "LSTM", "gru": "GRU", "cnn_lstm": "CNN-LSTM",
         "transformer": "FT-Transformer"}
METH = ["Static", "FineTune", "EWC", "LwF", "ER", "DER++", "PeriodicER", "WindowRetrain", "FullRetrain", "DACL"]


def ms(x, scale=100, d=1):
    x = pd.Series(x).dropna()
    return f"{x.mean() * scale:.{d}f}$\\pm${x.std(ddof=1) * scale:.{d}f}" if len(x) > 1 else f"{x.mean() * scale:.{d}f}"


def mn(x, scale=100, d=1):
    """Mean only: for rows pooled over datasets, a std across datasets would mislead."""
    return f"{pd.Series(x).dropna().mean() * scale:.{d}f}"


def write(name, body):
    (TAB / f"{name}.tex").write_text(body, encoding="utf-8")
    print("wrote", name)


def bold_best(cells, vals, higher=True, exclude=()):
    v = pd.Series(vals)
    v = v[[k for k in v.index if k not in exclude]].dropna()
    if v.empty:
        return cells
    best = v.idxmax() if higher else v.idxmin()
    cells[best] = f"\\textbf{{{cells[best]}}}"
    return cells


inj = pd.read_csv(R / "per_dataset" / "all_runs_derived.csv")
nat = pd.read_csv(R / "per_dataset" / "all_runs_natural_derived.csv")


# ---------------------------------------------------------------- datasets
def t_datasets():
    rows = []
    for d in DS:
        meta = json.loads(next((R / "stage1b" / d).glob("*/s0/meta.json")).read_text())
        cfg = json.loads((SIM / "configs" / f"{d}.json").read_text())
        info = meta["dataset_info"]
        g = cfg["scenario"]["groups"]
        rows.append(f"{DSL[d]} & {info['rows_used']:,} & {meta['n_features']} & "
                    f"{', '.join(g['A']).replace('_', '\\_')} & {', '.join(g['B']).replace('_', '\\_')} & "
                    f"{'yes' if f'natural_{d}' in NAT else 'no'} \\\\")
    write("datasets", "\n".join(rows))


# ---------------------------------------------------------------- detection quality per dataset (DACL, MLP)
def t_detection():
    rows = []
    for d in DS:
        x = inj[(inj.dataset == d) & (inj.arch == "mlp") & (inj.method == "DACL")]
        rows.append(f"{DSL[d]} & {ms(x.acc)} & {ms(x.precision)} & {ms(x.detect)} & {ms(x.f1)} & "
                    f"{x.fpr.mean() * 100:.2f} & {x.fnr.mean() * 100:.2f} & {x.mcc_test.mean():.3f} & "
                    f"{x.final_prauc_binary.mean():.3f} & {x.final_prauc_macro.mean():.3f} \\\\")
    write("detection", "\n".join(rows))


# ---------------------------------------------------------------- main F1 table (MLP)
def t_main_f1():
    lines = []
    cols = DS + NAT
    for m in METH:
        cells, vals = {}, {}
        for c in cols:
            src = nat if c.startswith("natural_") else inj
            x = src[(src.dataset == c) & (src.arch == "mlp") & (src.method == m)].f1
            cells[c] = ms(x) if len(x) else "--"
            vals[c] = x.mean() if len(x) else np.nan
        lines.append((m, cells, vals))
    # bold best per column excluding FullRetrain (upper bound)
    for c in cols:
        v = {m: vals[c] for m, cells, vals in lines if m != "FullRetrain"}
        best = max((k for k in v if not np.isnan(v[k])), key=lambda k: v[k])
        for m, cells, vals in lines:
            if m == best:
                cells[c] = f"\\textbf{{{cells[c]}}}"
    body = "\n".join(f"{('\\textit{FullRetrain}$^\\dagger$' if m == 'FullRetrain' else m)} & "
                     + " & ".join(cells[c] for c in cols) + " \\\\" + ("\\midrule" if m == "FullRetrain" else "")
                     for m, cells, vals in lines)
    write("main_f1", body)


# ---------------------------------------------------------------- lifecycle security (MLP)
def t_lifecycle():
    rows = []
    for d in DS:
        first = True
        for m in ["ER", "DER++", "WindowRetrain", "FullRetrain", "DACL"]:
            x = inj[(inj.dataset == d) & (inj.arch == "mlp") & (inj.method == m)]
            name = "\\textbf{DACL}" if m == "DACL" else m
            rows.append(f"{('\\multirow{5}{*}{' + DSL[d] + '}') if first else ''} & {name} & "
                        f"{ms(x.forgetting)} & {x.h_score.mean():.3f} & {x.recovery_batches.mean():.1f} & "
                        f"{x.exposure_missed_attacks.mean():,.0f} & {x.edge_update_s.mean():.2f} & "
                        f"{x.n_model_versions.mean():.1f} \\\\")
            first = False
        rows.append("\\midrule")
    write("lifecycle", "\n".join(rows[:-1]))


# ---------------------------------------------------------------- natural drift detail (MLP)
def t_natural():
    rows = []
    for d in NAT:
        first = True
        for m in ["Static", "ER", "DER++", "WindowRetrain", "FullRetrain", "DACL"]:
            x = nat[(nat.dataset == d) & (nat.arch == "mlp") & (nat.method == m)]
            if x.empty:
                continue
            name = "\\textbf{DACL}" if m == "DACL" else m
            lab = DSL[d.replace("natural_", "")]
            rows.append(f"{('\\multirow{6}{*}{' + lab + '}') if first else ''} & {name} & {ms(x.f1)} & "
                        f"{x.mcc_test.mean():.3f} & {ms(x.forgetting)} & {x.exposure_missed_attacks.mean():,.0f} & "
                        f"{x.n_model_versions.mean():.1f} & {x.edge_update_s.mean():.1f} \\\\")
            first = False
        rows.append("\\midrule")
    write("natural", "\n".join(rows[:-1]))


# ---------------------------------------------------------------- backbones (DACL)
def t_backbones():
    d = inj[inj.method == "DACL"]
    piv = d.pivot_table(index=["dataset", "seed"], columns="arch", values="f1")[ARCH]
    ranks = (-piv).rank(axis=1).mean()
    chi2, p = stats.friedmanchisquare(*[piv[a] for a in ARCH])
    from dacl.stage1_report import Q05
    cd = Q05[6] * np.sqrt(6 * 7 / (6.0 * len(piv)))
    prof = d.groupby("arch")[["params", "flops", "lat1_ms_mean", "lat1_ms_p99", "deadline_miss",
                              "energy_per_update_j"]].mean()
    rows = []
    for a in ARCH:
        f1 = " & ".join(ms(d[(d.dataset == x) & (d.arch == a)].f1) for x in DS)
        rows.append(f"{ARCHL[a]} & {f1} & {ranks[a]:.2f} & {prof.loc[a, 'params'] / 1e3:.1f}k & "
                    f"{prof.loc[a, 'flops'] / 1e6:.2f}M & {prof.loc[a, 'lat1_ms_mean']:.3f} / "
                    f"{prof.loc[a, 'lat1_ms_p99']:.3f} & {prof.loc[a, 'deadline_miss'] * 100:.1f} & "
                    f"{prof.loc[a, 'energy_per_update_j']:.2f} \\\\")
    write("backbones", "\n".join(rows))
    (TAB / "macros.tex").write_text(
        r"\newcommand{\BackboneStats}{" + f"Friedman $\\chi^2={chi2:.1f}$, $p={p:.1e}$; "
        f"Nemenyi CD $={cd:.2f}$ ($N={len(piv)}$)" + "}\n", encoding="utf-8")
    from dacl.stage1_report import cd_diagram
    cd_diagram(ranks.sort_values().rename(index=lambda a: a), cd, FIG / "cd_backbones",
               f"DACL macro-F1 across 5 datasets x 5 seeds (N = {len(piv)})")


# ---------------------------------------------------------------- statistics DACL vs baselines
def t_stats():
    rows = []
    for scen, src, dss in (("Injected", inj, DS), ("Natural", nat, NAT)):
        for arch in ("mlp", "transformer"):
            s = src[(src.arch == arch) & src.dataset.isin(dss)]
            piv = s.pivot_table(index=["dataset", "seed"], columns="method", values="f1")
            cells = []
            for m in ["ER", "DER++", "WindowRetrain", "FullRetrain"]:
                if m not in piv:
                    cells.append("--")
                    continue
                dd = (piv["DACL"] - piv[m]).dropna()
                p = stats.wilcoxon(dd).pvalue if dd.abs().sum() > 0 else 1.0
                cells.append(f"{dd.median() * 100:+.2f} ({int((dd > 0).sum())}/{len(dd)}, {p:.0e})")
            rows.append(f"{scen} & {ARCHL[arch]} & " + " & ".join(cells) + " \\\\")
    write("stats", "\n".join(rows))


# ---------------------------------------------------------------- ablation (stage2)
def t_ablation():
    rows = []
    s2 = []
    for f in glob.glob(str(R / "stage2" / "*" / "*" / "s*" / "summary_per_seed.csv")):
        p = Path(f)
        s2.append(pd.read_csv(f).assign(dataset=p.parts[-4], arch=p.parts[-3]))
    s2 = pd.concat(s2)
    for m, lab in (("DACL", "DACL (full)"), ("DACL-noKD", "w/o distillation"), ("DACL-noReplay", "w/o replay"),
                   ("DACL-fixedK", "fixed extent ($k=k_{\\max}$)"), ("DriftFT", "w/o replay and distillation"),
                   ("PeriodicER", "periodic trigger (no detector)")):
        cells = []
        for a in ("mlp", "transformer"):
            x = s2[(s2.method == m) & (s2.arch == a)]
            cells += [mn(x.f1), mn(x.forgetting), f"{x.grad_steps.mean():.0f}"]
        rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    write("ablation", "\n".join(rows))


# ---------------------------------------------------------------- gate (final)
def t_gate():
    g = pd.read_csv(R / "gate3_report" / "all_runs.csv")
    rows = []
    for st, lab in (("gate", "Clean"), ("poisonP2", "Poisoning during drift (P2)"),
                    ("poisonP4", "Unlearning attack (P4)")):
        first = True
        for m in ("DACL", "DACL+Gate", "DACL+Gate-dual"):
            x = g[(g.setting == st) & (g.method == m)]
            gated = m != "DACL"
            pois = ("--" if st != "poisonP4" else
                    ("100\\%" if not gated else f"{x.poisoned_accepted.sum() / x.poisoned_checks.sum() * 100:.1f}\\%"))
            rows.append(f"{('\\multirow{3}{*}{' + lab + '}') if first else ''} & {m} & {mn(x.f1)} & {mn(x.detect)} & "
                        f"{mn(x.fpr, 100, 2)} & {mn(x.A_detect_final)} & {x.exposure_missed_attacks.mean():,.0f} & "
                        f"{(f'{x.rejected.sum() / x.n_checks.sum() * 100:.1f}\\%' if gated else '--')} & {pois} \\\\")
            first = False
        rows.append("\\midrule")
    write("gate", "\n".join(rows[:-1]))
    lines = []
    for st in ("gate", "poisonP2", "poisonP4"):
        piv = g[g.setting == st].pivot_table(index=["dataset", "seed"], columns="method",
                                             values=["f1", "detect", "A_detect_final"])
        parts = []
        for k, lab in (("f1", "F1"), ("detect", "detection"), ("A_detect_final", "known-attack det.")):
            dd = (piv[(k, "DACL+Gate-dual")] - piv[(k, "DACL")]).dropna()
            p = stats.wilcoxon(dd).pvalue if dd.abs().sum() > 0 else 1.0
            parts.append(f"{lab} {dd.mean() * 100:+.1f} ($p={p:.0e}$)")
        lines.append(f"{st}: " + ", ".join(parts))
    (TAB / "gate_stats.txt").write_text("\n".join(lines), encoding="utf-8")
    per = g[g.setting == "poisonP4"].groupby(["dataset", "method"]).A_detect_final.mean().unstack() * 100
    write("gate_per_dataset", "\n".join(
        f"{DSL[d]} & {per.loc[d, 'DACL']:.1f} & {per.loc[d, 'DACL+Gate-dual']:.1f} & "
        f"{g[(g.dataset == d) & (g.setting == 'gate') & (g.method == 'DACL')].fpr.mean() * 100:.2f} & "
        f"{g[(g.dataset == d) & (g.setting == 'gate') & (g.method == 'DACL+Gate-dual')].fpr.mean() * 100:.2f} \\\\"
        for d in DS))


# ---------------------------------------------------------------- labels
def t_labels():
    d = pd.read_csv(R / "labels_report" / "all_runs.csv")
    rows = []
    for b in sorted(d.budget.unique()):
        s = d[d.budget == b]
        piv = s.pivot_table(index=["dataset", "seed"], columns="method", values="f1")
        means = " & ".join(mn(s[s.method == m].f1) for m in ["Static", "ER", "DER++", "FullRetrain", "DACL"])
        tests = []
        for m in ("ER", "DER++", "FullRetrain"):
            dd = (piv["DACL"] - piv[m]).dropna()
            p = stats.wilcoxon(dd).pvalue if dd.abs().sum() > 0 else 1.0
            tests.append(f"{dd.median() * 100:+.1f}{'$^*$' if p < 0.05 else ''}")
        dl = s[s.method == "DACL"]
        rows.append(f"{b}\\% & {means} & {' / '.join(tests)} & {dl.det_delay.mean():.1f} \\\\")
    write("labels", "\n".join(rows))


# ---------------------------------------------------------------- sensitivity
def t_sens():
    rows = []
    x = []
    for f in glob.glob(str(R / "sensitivity" / "sens_*" / "mlp" / "s*" / "summary_per_seed.csv")):
        n = Path(f).parts[-4][len("sens_"):]
        p, rest = n.split("_", 1)
        v, ds = rest.split("_", 1)
        x.append(pd.read_csv(f).assign(param=p, value=float(v)))
    x = pd.concat(x)
    lab = {"mem": "Replay memory $M$", "thr": "CUSUM threshold $h$", "kmax": "Max. steps $k_{\\max}$",
           "delay": "Label delay (batches)"}
    for p in ("mem", "thr", "kmax", "delay"):
        s = x[x.param == p].groupby(["value", "method"])[["f1", "grad_steps"]].mean().unstack()
        rows.append(f"\\multirow{{2}}{{*}}{{{lab[p]}}} & " + " & ".join(f"{int(v)}" for v in s.index) + " \\\\")
        rows.append(" & " + " & ".join(f"{s.loc[v, ('f1', 'DACL')] * 100:.1f} / {s.loc[v, ('f1', 'ER')] * 100:.1f}"
                                       for v in s.index) + " \\\\ \\addlinespace")
    write("sensitivity", "\n".join(rows))


# ---------------------------------------------------------------- chain
def t_chain():
    sc = pd.read_csv(R / "chain_scalability" / "summary.csv")
    rows = [f"{int(r.n_gateways)} & {int(r.txs)} & {r.throughput_tx_s:.2f} & {r.tx_lat_mean_s:.2f} / {r.tx_lat_p95_s:.2f} & "
            f"{r.decision_mean_s:.1f} / {r.decision_p95_s:.1f} & {int(r.blocks_used)} \\\\" for r in sc.itertuples()]
    write("chain_scalability", "\n".join(rows))
    tx = pd.read_csv(R / "chain_scalability" / "transactions.csv")
    info = {"gas_propose": int(tx[tx.label == "propose"].gas.mean()), "gas_attest": int(tx[tx.label == "attest"].gas.mean()),
            "log_bytes_update": int(sc.log_bytes_per_update.iloc[0]), "state_bytes_update": int(sc.state_bytes_per_update.iloc[0])}
    (TAB / "chain_info.json").write_text(json.dumps(info, indent=2))


# ---------------------------------------------------------------- lifetime figures (real data)
def fig_lifetime(ds, roots, methods, out_name, title, natural, smooth=5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    col = {"Static": "#7f7f7f", "ER": "#1f77b4", "DER++": "#9467bd", "WindowRetrain": "#ff7f0e",
           "FullRetrain": "#8c564b", "DACL": "#2ca02c"}
    meta = json.loads(next((roots[0] / ds / "mlp").glob("s*/meta.json")).read_text())
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    n = None
    for m in methods:
        curves = []
        for r in roots:
            curves += [pd.read_csv(f, usecols=["f1"]).f1.to_numpy()
                       for f in (r / ds / "mlp").glob(f"s*/timelines/{m}_s*.csv")]
        if not curves:
            continue
        mu = pd.Series(np.mean(curves, 0)).rolling(smooth, min_periods=1, center=natural).mean() * 100
        n = len(mu)
        ax.plot(mu, color=col[m], lw=1.5 if m == "DACL" else 0.9, label=m, zorder=3 if m == "DACL" else 2)
    if natural:
        for dp in meta["drift_points"]:
            ax.axvline(dp, color="0.6", ls=":", lw=0.7)
        ax.text(0.01, 0.02, "dotted: first appearance of a new attack", transform=ax.transAxes, fontsize=6,
                color="0.4")
    else:
        edges = meta["phase_start"] + [n]
        for i, nm in enumerate(meta["phase_names"]):
            if i % 2:
                ax.axvspan(edges[i], edges[i + 1], color="0.93", lw=0)
            ax.text((edges[i] + edges[i + 1]) / 2, 1.01, nm.split()[0], transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=7)
    ax.set_xlim(0, n)
    ax.set_xlabel("Stream batch (256 flows)")
    ax.set_ylabel("Macro-F1 (%)")
    if title:
        ax.set_title(title, fontsize=8)
    fig.legend(*ax.get_legend_handles_labels(), ncol=len(methods), fontsize=6, frameon=False,
               loc="lower center", handlelength=1.4, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / f"{out_name}.pdf")
    fig.savefig(FIG / f"{out_name}.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- figures
def figures():
    fig_lifetime("natural_xiiotid", [R / "natural", R / "window"], ["Static", "ER", "WindowRetrain", "DACL"],
                 "lifetime_xiiotid_natural", "", True, smooth=15)
    fig_lifetime("mu_iot", [R / "stage1b", R / "window"], ["Static", "ER", "DER++", "WindowRetrain", "DACL"],
                 "lifetime_mu_iot_injected", "", False)
    for src, dst in ((R / "per_dataset" / "figures" / "natural_xiiotid_f1_over_time.png", "natural_xiiotid_f1.png"),
                     (R / "per_dataset" / "figures" / "natural_xiiotid_confusion_per_phase.png", "natural_xiiotid_confusion.png"),
                     (R / "per_dataset" / "figures" / "xiiotid_sensitivity.png", "sensitivity_xiiotid.png"),
                     (R / "labels_report" / "f1_vs_label_budget.png", "labels.png"),
                     (R / "per_dataset" / "figures" / "mu_iot_confusion.png", "mu_iot_confusion.png")):
        for ext in (".png", ".pdf"):
            s = src.with_suffix(ext)
            if s.exists():
                shutil.copy(s, FIG / Path(dst).with_suffix(ext).name)


if __name__ == "__main__":
    for f in (t_datasets, t_detection, t_main_f1, t_lifecycle, t_natural, t_backbones, t_stats, t_ablation, t_gate,
              t_labels, t_sens, t_chain, figures):
        f()
