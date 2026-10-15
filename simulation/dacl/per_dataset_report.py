"""Per-dataset, per-backbone report over the five metric layers (+ PureChain layer).

  python -m dacl.per_dataset_report --root results/stage1b --out results/per_dataset

Edge costs come from profile_edge.py (single x86 CPU thread): update time = grad steps x measured
step cost; energy = time x `--cpu-power` W (an explicit power model, not a measurement).
"""
import argparse
import glob
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "savefig.dpi": 300})

DATASETS = ["wustl_iiot", "xiiotid", "edge_iiotset", "forge_iiot", "mu_iot"]
DS_LABEL = {"wustl_iiot": "WUSTL-IIoT-2021", "xiiotid": "X-IIoTID", "edge_iiotset": "Edge-IIoTset",
            "forge_iiot": "ForgeIIOT (IoTForge Pro)", "mu_iot": "MU-IoT"}
DS_LABEL.update({f"natural_{k}": f"{v} (natural drift)" for k, v in list(DS_LABEL.items())})
ARCHS = ["mlp", "cnn", "lstm", "gru", "cnn_lstm", "transformer"]
ARCH_LABEL = {"mlp": "MLP", "cnn": "1D-CNN", "lstm": "LSTM", "gru": "GRU", "cnn_lstm": "CNN-LSTM",
              "transformer": "FT-Transformer"}
METHODS = ["Static", "FineTune", "EWC", "LwF", "ER", "DER++", "PeriodicER", "WindowRetrain", "FullRetrain",
           "DACL"]
STEP = {"WindowRetrain": "step_ce_ms", "FineTune": "step_ce_ms", "EWC": "step_ce_ms", "LwF": "step_ce_ms", "FullRetrain": "step_ce_ms",
        "ER": "step_er_ms", "PeriodicER": "step_er_ms", "DER++": "step_dacl_ms", "DACL": "step_dacl_ms"}

# (column, label, scale, higher-is-better) per layer
LAYERS = {
    "L1_detection": [("acc", "Accuracy", 100, True), ("precision", "Precision", 100, True),
                     ("detect", "Recall (DR)", 100, True), ("f1", "Macro-F1", 100, True),
                     ("fpr", "FPR", 100, False), ("fnr", "FNR", 100, False), ("mcc", "MCC (stream)", 1, True),
                     ("mcc_test", "MCC (test)", 1, True),
                     ("final_prauc_binary", "PR-AUC (bin.)", 1, True),
                     ("final_prauc_macro", "PR-AUC (macro)", 1, True)],
    "L2_continual": [("acc", "Avg. acc.", 100, True), ("forgetting", "Forgetting", 100, False),
                     ("bwt_A", "BWT (A)", 100, True), ("intransigence", "Intransigence", 100, False),
                     ("h_score", "H-score", 1, True)],
    "L3_lifecycle": [("recovery_batches", "EW (batches)", 1, False),
                     ("exposure_missed_attacks", "MAE (flows)", 1, False),
                     ("lifecycle_f1", "Lifecycle F1", 100, True), ("det_delay", "Det. delay", 1, False),
                     ("false_alarms", "False alarms", 1, False), ("fa_per_100", "FA / 100 batches", 1, False),
                     ("missed_drifts", "Missed drifts", 1, False),
                     ("eff_per_cpu_s", "dF1 / CPU-s", 1, True), ("eff_per_j", "dF1 / J", 1, True)],
    "L4_edge": [("lat1_ms_mean", "Lat. mean (ms)", 1, False), ("lat1_ms_p95", "Lat. p95 (ms)", 1, False),
                ("lat1_ms_p99", "Lat. p99 (ms)", 1, False), ("throughput", "Throughput (flows/s)", 1, True),
                ("deadline_miss", "Deadline miss (%)", 100, False),
                ("upd_lat_mean_ms", "Upd. lat. mean (ms)", 1, False),
                ("upd_lat_max_ms", "Upd. lat. max (ms)", 1, False),
                ("params", "Params", 1, False), ("model_mb", "Model (MB)", 1, False),
                ("flops", "FLOPs/flow", 1, False), ("state_mb", "Peak mem. (MB)", 1, False),
                ("energy_per_update_j", "Energy/update (J)", 1, False)],
}


def load_runs(root):
    rows, phase, tl = [], {}, {}
    for f in glob.glob(f"{root}/*/*/s*/summary_per_seed.csv"):
        p = Path(f)
        ds, arch = p.parts[-4], p.parts[-3]
        df = pd.read_csv(f).assign(dataset=ds, arch=arch)
        rows.append(df)
        for l in open(p.parent / "phase_eval.jsonl"):
            j = json.loads(l)
            phase[(ds, arch, j["method"], j["seed"])] = j["phase_eval"]
        for t in (p.parent / "timelines").glob("*.csv"):
            m, s = t.stem.rsplit("_s", 1)
            tl[(ds, arch, m, int(s))] = t
    return pd.concat(rows, ignore_index=True), phase, tl


def mcc_from_confusion(cm):
    """Multiclass MCC (Gorodkin 2004) from a confusion matrix (rows = true, cols = predicted)."""
    cm = np.asarray(cm, float)
    t, p, c, s = cm.sum(1), cm.sum(0), np.trace(cm), cm.sum()
    den = np.sqrt((s ** 2 - (p ** 2).sum()) * (s ** 2 - (t ** 2).sum()))
    return float((c * s - (t * p).sum()) / den) if den > 0 else 0.0


def add_test_mcc(df, phase):
    """MCC on the held-out test sets at the final checkpoint (robust to single-class stream batches)."""
    df["mcc_test"] = [mcc_from_confusion(phase[(r.dataset, r.arch, r.method, r.seed)][-1]["confusion"])
                      if "confusion" in phase.get((r.dataset, r.arch, r.method, r.seed), [{}])[-1] else np.nan
                      for r in df.itertuples(index=False)]
    return df


def add_derived(df, tl, prof, power, deadline_ms):
    """Edge costs from the CPU profile, deadline misses from per-batch grad steps, efficiency."""
    prof = prof.set_index(["dataset", "arch"])
    out = []
    for r in df.itertuples(index=False):
        p = prof.loc[(r.dataset, r.arch)]
        d = r._asdict()
        step_ms = p[STEP[r.method]] if r.method in STEP else 0.0
        t = pd.read_csv(tl[(r.dataset, r.arch, r.method, r.seed)], usecols=["grad_steps"])
        upd_ms = t["grad_steps"].to_numpy() * step_ms
        batch_ms = p["infer256_ms"] + upd_ms
        d.update({k: p[k] for k in ("lat1_ms_mean", "lat1_ms_p95", "lat1_ms_p99", "flops", "params")})
        d["throughput"] = 256.0 / (p["infer256_ms"] / 1000.0)
        d["model_mb"] = p["params"] * 4 / 2**20
        d["deadline_miss"] = float((batch_ms > deadline_ms).mean())
        upd = upd_ms[upd_ms > 0]
        d["upd_lat_mean_ms"] = float(upd.mean()) if len(upd) else 0.0
        d["upd_lat_max_ms"] = float(upd.max()) if len(upd) else 0.0
        d["edge_update_s"] = float(upd_ms.sum() / 1000)
        n_upd = r.n_model_versions if r.n_model_versions else 0
        d["energy_per_update_j"] = d["edge_update_s"] * power / n_upd if n_upd else 0.0
        d["rollback_ms"] = p.get("rollback_ms", np.nan)
        fa = d.get("false_alarms", np.nan)
        d["fa_per_100"] = fa / len(t) * 100 if pd.notna(fa) else np.nan
        out.append(d)
    df = pd.DataFrame(out)
    static = df[df.method == "Static"].set_index(["dataset", "arch", "seed"])["lifecycle_f1"]
    full = df[df.method == "FullRetrain"].set_index(["dataset", "arch", "seed"])["plasticity_B"]
    key = list(zip(df.dataset, df.arch, df.seed))
    gain = df["lifecycle_f1"].to_numpy() - static.reindex(key).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        df["eff_per_cpu_s"] = np.where(df.edge_update_s > 0, gain / df.edge_update_s, np.nan)
        df["eff_per_j"] = np.where(df.edge_update_s > 0, gain / (df.edge_update_s * power), np.nan)
    # intransigence (Chaudhry et al. 2018): joint-training reference minus method, on the new attack
    df["intransigence"] = full.reindex(key).to_numpy() - df["plasticity_B"].to_numpy()
    return df


def fmt(m, s, scale):
    if pd.isna(m):
        return "--"
    v, e = m * scale, (0.0 if pd.isna(s) else s * scale)
    if e == 0:  # deterministic (edge profile) values: no spread to report
        return f"{v:,.0f}" if abs(v) >= 1000 else (f"{v:.1f}" if abs(v) >= 100 else f"{v:.3f}")
    if abs(v) >= 1000:
        return f"{v:,.0f} ± {e:,.0f}"
    if abs(v) >= 100:
        return f"{v:.1f} ± {e:.1f}"
    return f"{v:.3f} ± {e:.3f}" if scale == 1 and abs(v) < 10 else f"{v:.2f} ± {e:.2f}"


def layer_table(d, cols, method="DACL"):
    sub = d[d.method == method]
    g = sub.groupby("arch")
    rows = []
    for a in ARCHS:
        if a not in g.groups:
            continue
        x = g.get_group(a)
        rows.append({"Backbone": ARCH_LABEL[a],
                     **{lab: fmt(x[c].mean(), x[c].std(ddof=1), sc) for c, lab, sc, _ in cols if c in x}})
    t = pd.DataFrame(rows).set_index("Backbone")
    # bold the best backbone per column (mean)
    for c, lab, sc, hb in cols:
        if c not in sub or lab not in t:
            continue
        means = g[c].mean().reindex(ARCHS).dropna()
        if means.empty or means.nunique() <= 1:
            continue
        best = ARCH_LABEL[means.idxmax() if hb else means.idxmin()]
        t.loc[best, lab] = f"**{t.loc[best, lab]}**"
    return t


def method_matrix(d, metric="f1"):
    m = d.groupby(["method", "arch"])[metric].agg(["mean", "std"])
    t = pd.DataFrame(index=[m_ for m_ in METHODS if m_ in d.method.unique()],
                     columns=[ARCH_LABEL[a] for a in ARCHS if a in d.arch.unique()])
    for (meth, a), r in m.iterrows():
        t.loc[meth, ARCH_LABEL[a]] = f"{r['mean'] * 100:.2f} ± {r['std'] * 100:.2f}"
    for a in t.columns:
        arch = [k for k, v in ARCH_LABEL.items() if v == a][0]
        best = d[(d.arch == arch) & (d.method != "FullRetrain")].groupby("method")[metric].mean().idxmax()
        t.loc[best, a] = f"**{t.loc[best, a]}**"
    return t


def class_recall_table(phase, ds, method="DACL"):
    rows = {}
    for a in ARCHS:
        rec = [phase[k][-1]["class_recall"] for k in phase if k[0] == ds and k[1] == a and k[2] == method
               and "class_recall" in phase[k][-1]]
        if rec:
            rows[ARCH_LABEL[a]] = pd.DataFrame(rec).mean() * 100
    return pd.DataFrame(rows).T.round(2)


def stats_text(d):
    lines = []
    dacl = d[d.method == "DACL"].pivot_table(index="seed", columns="arch", values="f1")[
        [a for a in ARCHS if a in d.arch.unique()]].dropna()
    if dacl.shape[0] >= 3 and dacl.shape[1] >= 3:
        chi2, p = stats.friedmanchisquare(*[dacl[c] for c in dacl.columns])
        ranks = (-dacl).rank(axis=1).mean().sort_values()
        lines.append(f"- Friedman test across backbones (DACL macro-F1, blocks = {len(dacl)} seeds): "
                     f"chi2 = {chi2:.2f}, p = {p:.3g}. Average ranks: "
                     + ", ".join(f"{ARCH_LABEL[a]} {r:.2f}" for a, r in ranks.items()))
    piv = d.pivot_table(index=["arch", "seed"], columns="method", values="f1")
    for m in METHODS:
        if m in ("DACL",) or m not in piv:
            continue
        diff = (piv["DACL"] - piv[m]).dropna()
        if len(diff) >= 6 and diff.abs().sum() > 0:
            w = stats.wilcoxon(diff)
            lines.append(f"- DACL vs {m} (macro-F1, N = {len(diff)} backbone x seed pairs): "
                         f"median diff {diff.median() * 100:+.2f} pts, Wilcoxon p = {w.pvalue:.2g}")
    return "\n".join(lines)


def fig_f1_time(tl, ds, meta, path):
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    starts = meta["phase_start"]
    for s in starts[1:]:
        ax.axvline(s, color="0.5", ls="--", lw=0.6)
    for a in ARCHS:
        curves = [pd.read_csv(f, usecols=["f1"])["f1"].to_numpy() for k, f in tl.items()
                  if k[0] == ds and k[1] == a and k[2] == "DACL"]
        if curves:
            mu = pd.Series(np.mean(curves, 0)).rolling(5, min_periods=1).mean()
            ax.plot(mu * 100, label=ARCH_LABEL[a], lw=0.9)
    ax.set_xlabel("Stream batch")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title(f"{DS_LABEL[ds]}: DACL per backbone", fontsize=8)
    ax.legend(ncol=2, fontsize=6, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def fig_confusions(phase, ds, classes, path):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8))
    for ax, a in zip(axes.flat, ARCHS):
        cms = [np.array(phase[k][-1]["confusion"]) for k in phase
               if k[0] == ds and k[1] == a and k[2] == "DACL" and "confusion" in phase[k][-1]]
        if not cms:
            ax.axis("off")
            continue
        cm = np.mean(cms, 0)
        cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(ARCH_LABEL[a], fontsize=7)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=90, fontsize=5)
        ax.set_yticklabels(classes, fontsize=5)
        for i in range(len(classes)):
            for j in range(len(classes)):
                if cmn[i, j] >= 0.05:
                    ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=4,
                            color="white" if cmn[i, j] > 0.6 else "black")
    fig.suptitle(f"{DS_LABEL[ds]}: DACL confusion matrices at end of P4 (row-normalised, mean of seeds)",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def chain_section(gate_root, live_root):
    """Layer 5 (MLP): poisoned-update acceptance, tx latency, on-chain counts, rollback."""
    lines = []
    acc_p = tot_p = 0
    for f in glob.glob(f"{gate_root}/sweep_*_poisonP4_t10/mlp/s*/phase_eval.jsonl"):
        meta = json.loads((Path(f).parent / "meta.json").read_text())
        p4 = meta["phase_start"][3]
        for l in open(f):
            j = json.loads(l)
            for g in j.get("gate_log", []):
                if g["t"] >= p4:  # updates trained on poisoned (P4) labels
                    tot_p += 1
                    acc_p += g["accepted"]
    if tot_p:
        lines.append(f"| Poisoned-update acceptance rate (P4 attack, tau = 10%) | without gate: 100% | "
                     f"with PureChain gate: {acc_p / tot_p * 100:.1f}% ({acc_p}/{tot_p}) |")
    lat = {}
    for f in glob.glob(f"{live_root}/*/phase_eval.jsonl"):
        for l in open(f):
            for k, v in json.loads(l).get("chain_latency", []):
                lat.setdefault(k, []).append(v)
    for k in ("propose", "attest"):
        if k in lat:
            v = np.array(lat[k]) * 1000
            lines.append(f"| Confirmation latency, {k} (live PureChain) | mean {v.mean():.0f} ms | "
                         f"p95 {np.percentile(v, 95):.0f} ms (n = {len(v)}) |")
    return lines


def fig_phase_confusions(phase, ds, classes, arch, path):
    """Confusion matrix at the end of every phase/segment: Static (top) vs DACL (bottom)."""
    keys = {m: [k for k in phase if k[0] == ds and k[1] == arch and k[2] == m] for m in ("Static", "DACL")}
    n_ph = len(phase[keys["DACL"][0]]) - 1
    fig, axes = plt.subplots(2, n_ph, figsize=(1.9 * n_ph, 4.8))
    for row, m in enumerate(("Static", "DACL")):
        for p_i in range(n_ph):
            ax = axes[row, p_i]
            cm = np.mean([np.array(phase[k][p_i + 1]["confusion"]) for k in keys[m]], 0)
            cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
            ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
            ax.set_title(f"{m}: {phase[keys[m][0]][p_i + 1]['phase']}", fontsize=6)
            ax.set_xticks(range(len(classes)))
            ax.set_yticks(range(len(classes)))
            ax.set_xticklabels(classes if row else [], rotation=90, fontsize=4)
            ax.set_yticklabels(classes if p_i == 0 else [], fontsize=4)
    fig.suptitle(f"{DS_LABEL[ds]}: confusion per phase ({ARCH_LABEL[arch]}, row-normalised)", fontsize=7)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.45)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def phase_recall_table(phase, ds, arch, method):
    keys = [k for k in phase if k[0] == ds and k[1] == arch and k[2] == method]
    rows = {}
    for p_i, e in enumerate(phase[keys[0]]):
        rows[e["phase"]] = pd.DataFrame([phase[k][p_i]["class_recall"] for k in keys]).mean() * 100
    return pd.DataFrame(rows).T.round(2)


def cd_per_dataset(d, ds, path):
    from .stage1_report import cd_diagram, friedman_nemenyi
    tab = d[d.method == "DACL"].pivot_table(index="seed", columns="arch", values="f1")[
        [a for a in ARCHS if a in d.arch.unique()]].dropna()
    if tab.shape[0] < 3:
        return ""
    chi2, p, ranks, cd = friedman_nemenyi(tab, True)
    cd_diagram(ranks, cd, path, f"{DS_LABEL[ds]}: DACL macro-F1 (N = {len(tab)} seeds)")
    return f"![CD diagram](figures/{path.name}.png)"


def best_baseline_line(d):
    nb = d[~d.method.isin(["DACL", "FullRetrain", "Static"])].groupby("method")["f1"].mean()
    b = nb.idxmax()
    piv = d.pivot_table(index=["arch", "seed"], columns="method", values="f1")
    diff = (piv["DACL"] - piv[b]).dropna()
    w = stats.wilcoxon(diff)
    return (f"**Best baseline: {b}.** DACL - {b} macro-F1: median {diff.median() * 100:+.2f} pts, "
            f"mean {diff.mean() * 100:+.2f} pts; DACL better in {int((diff > 0).sum())}/{len(diff)} "
            f"backbone x seed pairs; Wilcoxon p = {w.pvalue:.2g}.")


def gate3_table(path, ds):
    """Final gate (balanced net security benefit, hard floor 30%, soft 10%) incl. dual serving."""
    if not Path(path).exists():
        return pd.DataFrame()
    d = pd.read_csv(path)
    d = d[d.dataset == ds]
    rows = []
    for st, lab in (("gate", "Clean (cost of security)"), ("poisonP2", "Poisoning during drift (P2)"),
                    ("poisonP4", "Unlearning attack on known attack (P4)")):
        for m in ("DACL", "DACL+Gate", "DACL+Gate-dual"):
            x = d[(d.setting == st) & (d.method == m)]
            if x.empty:
                continue
            gated = m != "DACL"
            chk = x.n_checks.sum()
            rows.append({
                "Setting": lab, "Method": m,
                "Macro-F1 (%)": fmt(x.f1.mean(), x.f1.std(), 100),
                "Detection (%)": fmt(x.detect.mean(), x.detect.std(), 100),
                "FPR (%)": fmt(x.fpr.mean(), x.fpr.std(), 100),
                "MAE (flows)": fmt(x.exposure_missed_attacks.mean(), x.exposure_missed_attacks.std(), 1),
                "Known-attack detectability (%)": fmt(x.A_detect_final.mean(), x.A_detect_final.std(), 100),
                "Rejected": f"{x.rejected.sum() / chk * 100:.1f}%" if gated and chk else "--",
                "Escalated": f"{x.escalated.sum() / chk * 100:.1f}%" if gated and chk else "--",
                "Poisoned updates accepted": ("--" if st != "poisonP4" else (
                    "100%" if not gated else
                    f"{x.poisoned_accepted.sum() / max(1, x.poisoned_checks.sum()) * 100:.1f}%"))})
    return pd.DataFrame(rows)


def gate_table(gate_root, ds, tau=10):
    rows = []
    for setting, lab in (("gate", "Clean (cost of security)"), ("poisonP2", "Poisoning during drift (P2)"),
                         ("poisonP4", "Unlearning attack on known attack (P4)")):
        fs = glob.glob(f"{gate_root}/sweep_{ds}_{setting}_t{tau:02d}/mlp/s*/summary_per_seed.csv")
        if not fs:
            continue
        x = pd.concat(pd.read_csv(f) for f in fs)
        acc = rej = pacc = ptot = 0
        for f in fs:
            meta = json.loads((Path(f).parent / "meta.json").read_text())
            p4 = meta["phase_start"][3]
            for l in open(Path(f).parent / "phase_eval.jsonl"):
                for g in json.loads(l).get("gate_log", []):
                    acc += g["accepted"]
                    rej += not g["accepted"]
                    if setting == "poisonP4" and g["t"] >= p4:
                        ptot += 1
                        pacc += g["accepted"]
        for m in ("DACL", "DACL+Gate"):
            y = x[x.method == m]
            rows.append({
                "Setting": lab, "Method": m,
                "Macro-F1 (%)": fmt(y.f1.mean(), y.f1.std(), 100),
                "Known-attack recall (%)": fmt(y.final_A_recall.mean(), y.final_A_recall.std(), 100),
                "New-attack recall (%)": fmt(y.plasticity_B.mean(), y.plasticity_B.std(), 100),
                "MAE (flows)": fmt(y.exposure_missed_attacks.mean(), y.exposure_missed_attacks.std(), 1),
                "Update rejection rate": (f"{rej / max(1, acc + rej) * 100:.1f}%" if m == "DACL+Gate"
                                          else "0% (no gate)"),
                "Poisoned-update acceptance": ("--" if setting != "poisonP4" else
                                               (f"{pacc / max(1, ptot) * 100:.1f}% ({pacc}/{ptot})"
                                                if m == "DACL+Gate" else "100%"))})
    return pd.DataFrame(rows)


def sensitivity_section(sens_root, ds, out):
    rows = []
    for f in glob.glob(f"{sens_root}/sens_*_{ds}/mlp/s*/summary_per_seed.csv"):
        name = Path(f).parts[-4]
        param, val = name[len("sens_"):-len(ds) - 1].split("_", 1)
        rows.append(pd.read_csv(f).assign(param=param, value=float(val)))
    if not rows:
        return [], None
    x = pd.concat(rows)
    labels = {"mem": "Replay memory size", "thr": "Detector threshold (CUSUM h)",
              "kmax": "Max adaptation steps (kmax)", "delay": "Label delay (batches)"}
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.9))
    tabs = []
    for ax, (pk, pl) in zip(axes, labels.items()):
        sub = x[x.param == pk]
        for m, c in (("DACL", "#2ca02c"), ("ER", "#1f77b4")):
            g = sub[sub.method == m].groupby("value")["f1"].agg(["mean", "std"])
            ax.errorbar(g.index, g["mean"] * 100, yerr=g["std"] * 100, marker="o", ms=3, capsize=2,
                        color=c, label=m, lw=1)
        ax.set_xlabel(pl, fontsize=6)
        ax.tick_params(labelsize=6)
        t = sub.groupby(["method", "value"])[["f1", "forgetting", "exposure_missed_attacks",
                                              "grad_steps"]].mean()
        t.insert(0, "parameter", pl)
        tabs.append(t)
    axes[0].set_ylabel("Macro-F1 (%)", fontsize=6)
    axes[0].legend(fontsize=6, frameon=False)
    fig.suptitle(f"{DS_LABEL[ds]}: sensitivity (MLP, mean ± std of 5 seeds)", fontsize=7)
    fig.tight_layout()
    path = out / "figures" / f"{ds}_sensitivity"
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    tab = pd.concat(tabs).reset_index()
    tab["f1"] = (tab["f1"] * 100).round(2)
    tab["forgetting"] = (tab["forgetting"] * 100).round(2)
    tab = tab.rename(columns={"f1": "Macro-F1 (%)", "forgetting": "Forgetting (%)",
                              "exposure_missed_attacks": "MAE", "grad_steps": "Grad steps"}).round(1)
    return [f"![Sensitivity](figures/{ds}_sensitivity.png)", ""], tab


def scenario_sections(d, phase, tl, root, ds, out, suffix=""):
    """Layers 1-4, method matrix, per-phase recall/confusion and statistics for one dataset/scenario."""
    meta = json.loads(next(Path(root, ds).glob("*/s*/meta.json")).read_text())
    md = [f"Classes: {', '.join(meta['classes'])}; {meta['n_features']} features.", ""]
    for name, cols in LAYERS.items():
        t = layer_table(d, cols)
        t.to_csv(out / f"{ds}_{name}.csv")
        md += [f"#### {name.replace('_', ' ').title()} (DACL, per backbone){suffix}", "", t.to_markdown(), ""]
    mm = method_matrix(d)
    mm.to_csv(out / f"{ds}_method_x_backbone_f1.csv")
    md += [f"#### Macro-F1 (%) of every method on every backbone{suffix}", "", mm.to_markdown(), "",
           best_baseline_line(d), ""]
    cr = class_recall_table(phase, ds)
    cr.to_csv(out / f"{ds}_class_recall_final.csv")
    md += ["#### Per-class recall (%) at the end of the stream (DACL)", "", cr.to_markdown(), ""]
    for m in ("Static", "DACL"):
        pr = phase_recall_table(phase, ds, "mlp", m)
        pr.to_csv(out / f"{ds}_class_recall_per_phase_{m}.csv")
        md += [f"#### Per-class recall (%) after each phase: {m} (MLP)", "", pr.to_markdown(), ""]
    md += ["#### Statistics", "", stats_text(d), "", cd_per_dataset(d, ds, out / "figures" / f"{ds}_cd"), ""]
    fig_f1_time(tl, ds, meta, out / "figures" / f"{ds}_f1_over_time")
    fig_confusions(phase, ds, meta["classes"], out / "figures" / f"{ds}_confusion")
    fig_phase_confusions(phase, ds, meta["classes"], "mlp", out / "figures" / f"{ds}_confusion_per_phase")
    md += [f"![F1 over time](figures/{ds}_f1_over_time.png)", "",
           f"![Confusion per phase](figures/{ds}_confusion_per_phase.png)", "",
           f"![Final confusion, all backbones](figures/{ds}_confusion.png)", ""]
    return md


def chain_global(scal_path, live_root, df, gate_root):
    md = ["## PureChain (global, live network)", "", "| Metric | Value | |", "|---|---|---|",
          *chain_section(gate_root, live_root),
          f"| On-chain model versions per run | DACL: {df[df.method == 'DACL'].n_model_versions.mean():.1f} | "
          f"continuous methods (FineTune, ER, DER++, ...): 280 |",
          f"| Rollback time (restore verified model) | MLP "
          f"{df[(df.method == 'DACL') & (df.arch == 'mlp')].rollback_ms.mean():.3f} ms | FT-Transformer "
          f"{df[(df.method == 'DACL') & (df.arch == 'transformer')].rollback_ms.mean():.3f} ms |", ""]
    if Path(scal_path).exists():
        sc = pd.read_csv(scal_path)
        md += ["### Scalability: concurrent gateways (3 shared validators; PoA² network with 4 signers)", "",
               sc.round(3).to_markdown(index=False), ""]
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/stage1b")
    ap.add_argument("--natural-root", default="results/natural")
    ap.add_argument("--sens-root", default="results/sensitivity")
    ap.add_argument("--profile", default="results/edge_profile_x86_1thread.csv")
    ap.add_argument("--gate-root", default="results/gatesweep")
    ap.add_argument("--window-root", default="results/window")
    ap.add_argument("--gate3", default="results/gate3_report/all_runs.csv")
    ap.add_argument("--labels-dir", default="results/labels_report")
    ap.add_argument("--live-root", default="results/gate_live")
    ap.add_argument("--scalability", default="results/chain_scalability/summary.csv")
    ap.add_argument("--out", default="results/per_dataset")
    ap.add_argument("--cpu-power", type=float, default=3.0)
    ap.add_argument("--deadline-ms", type=float, default=256.0,
                    help="per-batch budget: 256 flows at 1,000 flows/s")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    prof = pd.read_csv(args.profile)
    df, phase, tl = load_runs(args.root)
    win, wphase, wtl = (load_runs(args.window_root) if glob.glob(f"{args.window_root}/*/*/s*/summary_per_seed.csv")
                        else (pd.DataFrame(columns=["dataset"]), {}, {}))
    phase.update(wphase)
    tl.update(wtl)
    df = pd.concat([df, win[~win.dataset.str.startswith("natural_")]], ignore_index=True)
    df = add_test_mcc(add_derived(df, tl, prof, args.cpu_power, args.deadline_ms), phase)
    df.to_csv(out / "all_runs_derived.csv", index=False)
    nat = None
    if glob.glob(f"{args.natural_root}/*/*/s*/summary_per_seed.csv"):
        nat, nphase, ntl = load_runs(args.natural_root)
        nphase.update(wphase)
        ntl.update(wtl)
        nat = pd.concat([nat, win[win.dataset.str.startswith("natural_")]], ignore_index=True)
        nat = add_test_mcc(add_derived(nat, ntl, prof, args.cpu_power, args.deadline_ms), nphase)
        nat.to_csv(out / "all_runs_natural_derived.csv", index=False)

    md = ["# Per-dataset results", "",
          "Injected drift: 5 datasets x 6 backbones x 9 methods x 5 seeds. Natural drift: the 3 datasets with "
          "a usable recording order. Mean ± std over 5 seeds; bold = best backbone (per column) or best "
          "method excluding the FullRetrain upper bound.", "",
          f"Edge costs: single x86 CPU thread (profile_edge.py); energy = CPU time x {args.cpu_power} W "
          f"(power model, to be replaced by Raspberry Pi measurements); deadline = {args.deadline_ms:.0f} ms "
          "per 256-flow batch (1,000 flows/s). Within one dataset x backbone cell only 5 seeds exist (the "
          "smallest possible two-sided Wilcoxon p for 5 pairs is 0.0625), so tests pool backbones x seeds "
          "(N = 30).", ""]
    for ds in DATASETS:
        d = df[df.dataset == ds]
        if d.empty:
            continue
        md += [f"## {DS_LABEL[ds]}", "",
               "### A. Injected drift (P1 initial, P2 drift, P3 new attack, P4 known attack returns)", ""]
        md += scenario_sections(d, phase, tl, args.root, ds, out)
        gt = gate3_table(args.gate3, ds)
        if not gt.empty:
            gt.to_csv(out / f"{ds}_L5_purechain_gate.csv", index=False)
            md += ["#### Layer 5: PureChain update gate on this dataset (MLP; balanced net-security-benefit "
                   "policy, hard floor 30%, escalation at 10%; dual serving = alarm from the verified model, "
                   "attack type from an advisory model)", "", gt.to_markdown(index=False), ""]
        lab_csv = Path(args.labels_dir) / f"{ds}_labels.csv"
        if lab_csv.exists():
            lt = pd.read_csv(lab_csv)
            lt.to_csv(out / f"{ds}_label_budget.csv", index=False)
            md += ["#### Label scarcity (MLP): only the given share of flows is labelled", "",
                   lt.to_markdown(index=False), ""]
        lines, st = sensitivity_section(args.sens_root, ds, out)
        if st is not None:
            st.to_csv(out / f"{ds}_sensitivity.csv", index=False)
            md += ["### B. Sensitivity analysis (MLP; one parameter varied at a time)", "", *lines,
                   st.to_markdown(index=False), ""]
        nds = f"natural_{ds}"
        if nat is not None and nds in set(nat.dataset):
            md += ["### C. Natural drift (traffic replayed in recording order)", ""]
            md += scenario_sections(nat[nat.dataset == nds], nphase, ntl, args.natural_root, nds, out,
                                    " -- natural drift")
        elif ds in ("edge_iiotset", "mu_iot"):
            md += ["### C. Natural drift: not applicable", "",
                   ("Edge-IIoTset's frame.time has no date component, so the recording order cannot be "
                    "recovered." if ds == "edge_iiotset" else
                    "MU-IoT's normal traffic lies only in the first 1.6% of the timeline (a separate capture "
                    "session); replaying it in order would leave no benign background traffic."), ""]
    md += chain_global(args.scalability, args.live_root, df, args.gate_root)
    (out / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    with pd.ExcelWriter(out / "per_dataset_results.xlsx") as xw:
        for ds in DATASETS + [f"natural_{x}" for x in DATASETS]:
            files = [f for f in sorted(out.glob(f"{ds}_*.csv"))]
            if not files:
                continue
            r = 0
            for f in files:
                t = pd.read_csv(f, index_col=0)
                pd.DataFrame({f.stem: []}).to_excel(xw, sheet_name=ds[:31], startrow=r)
                t.to_excel(xw, sheet_name=ds[:31], startrow=r + 1)
                r += len(t) + 4
    print(f"wrote {out / 'REPORT.md'} and per_dataset_results.xlsx")


if __name__ == "__main__":
    main()
