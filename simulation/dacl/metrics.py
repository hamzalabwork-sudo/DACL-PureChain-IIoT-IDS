"""Over-time evaluation metrics (slide 8 / slide 10).

Learning behaviour: prequential accuracy / macro-F1, drift detection delay, missed and
false alarms, recovery time, adaptation time, forgetting, return dip.
Real-time cost: inference latency, update latency, CPU time, memory, energy estimate.
"""
import numpy as np
import pandas as pd


def summarize(rows, phase_eval, extra, scenario, cfg):
    mc = cfg["metrics"]
    df = pd.DataFrame(rows)
    acc = df["acc"].to_numpy()
    out = {"acc": df["acc"].mean(), "f1": df["f1"].mean(), "mcc": df["mcc"].mean(),
           "detect": df["detect"].mean(), "fpr": df["fpr"].mean()}
    if "precision" in df:
        out["precision"] = df["precision"].mean()
    out["fnr"] = 1.0 - out["detect"]
    # lifecycle F1 = normalised area under the prequential F1-over-time curve (== mean batch F1)
    out["lifecycle_f1"] = out["f1"]
    last = phase_eval[-1]
    for k in ("precision_macro", "prauc_macro", "prauc_binary", "f1_macro"):
        if k in last:
            out[f"final_{k}"] = last[k]
            out[f"mean_{k}"] = float(np.mean([e[k] for e in phase_eval if k in e]))
    for p, name in enumerate(scenario.phase_names):
        out[f"acc_P{p + 1}"] = df.loc[df.phase == p, "acc"].mean()

    # drift detection
    alarms = np.array(extra["alarms"], int)
    delays, missed, matched = [], 0, set()
    ends = scenario.phase_end[1:]
    # injected scenario: drift at phase starts; natural scenario: first appearance of new attacks
    windows = getattr(scenario, "drift_windows", None) or list(zip(scenario.drift_points, ends))
    for dp, end in windows:
        hit = alarms[(alarms >= dp) & (alarms < end)]
        if len(hit):
            delays.append(hit[0] - dp)
        else:
            missed += 1
        matched |= set(alarms[(alarms >= dp) & (alarms < dp + mc["false_alarm_window"])].tolist())
    if len(alarms) or extra.get("uses_detector"):
        out["det_delay"] = float(np.mean(delays)) if delays else np.nan
        out["missed_drifts"] = missed
        out["false_alarms"] = int(sum(a not in matched for a in alarms))
        out["n_alarms"] = len(alarms)

    # recovery and adaptation time per drift point
    rec, adapt_s, drop, mae = [], [], [], []
    for dp, end in windows:
        ref = acc[max(0, dp - mc["ref_window"]):dp].mean()
        post = acc[dp:end]
        drop.append(max(0.0, ref - post[:20].min()))
        r = len(post)
        for t in range(len(post)):
            if post[max(0, t - 4):t + 1].mean() >= ref - mc["recovery_tol"]:
                r = t
                break
        rec.append(r)
        adapt_s.append(df["update_wall"].iloc[dp:dp + r + 1].sum())
        mae.append(df["missed_attacks"].iloc[dp:dp + r + 1].sum())
    out["max_drop"] = float(np.mean(drop))
    out["recovery_batches"] = float(np.mean(rec))
    out["adapt_time_s"] = float(np.mean(adapt_s))
    # Missed Attacks during Exposure: attack samples passed as normal before recovery
    out["exposure_missed_attacks"] = float(np.sum(mae))

    # forgetting of attack group A after learning B (phase-end held-out evaluation)
    fg = mc["forgetting"]
    key = f"{fg['group']}_recall"
    hist = [e[key] for e in phase_eval[:fg["after_phase"] + 1]]  # P0 .. phase before
    after = phase_eval[fg["after_phase"] + 1][key]
    out["forgetting"] = float(max(hist) - after)
    rp = mc["return_phase"]
    s0 = scenario.phase_start[rp]
    out["return_recall"] = float(df[key].iloc[s0:s0 + mc["return_window"]].mean())
    out["final_A_recall"] = phase_eval[-1][key]
    # backward transfer: effect of learning the new attack (phase after_phase+1) on old groups
    # (negative = forgetting; GEM-style R_after - R_before, signed, no max over history)
    before, after_e = phase_eval[fg["after_phase"]], phase_eval[fg["after_phase"] + 1]
    out["bwt_A"] = float(after_e[key] - before[key])
    out["bwt_normal"] = float(after_e["Normal_recall"] - before["Normal_recall"])
    # stability-plasticity H-score after the new attack has been learned (end of P3)
    stab = max(0.0, 1.0 - out["forgetting"])
    plast = phase_eval[fg["after_phase"] + 1].get("B_recall", np.nan)
    out["plasticity_B"] = plast
    out["h_score"] = float(2 * stab * plast / (stab + plast)) if stab + plast > 0 else 0.0

    # real-time cost
    n = extra["n_samples"]
    out["infer_us_per_sample"] = df["infer_wall"].sum() / n * 1e6
    out["update_ms_per_batch"] = df["update_wall"].mean() * 1e3
    out["update_ms_max"] = df["update_wall"].max() * 1e3
    out["batch_ms_p99"] = float(np.percentile((df["infer_wall"] + df["update_wall"]) * 1e3, 99))
    out.update(extra["edge"])
    out["n_model_versions"] = (len(extra["alarms"]) if extra.get("uses_detector")
                               else int((df["grad_steps"] > 0).sum()))
    out["update_cpu_s"] = df["update_cpu"].sum()
    out["total_cpu_s"] = df["update_cpu"].sum() + df["infer_cpu"].sum()
    out["grad_steps"] = int(df["grad_steps"].sum())
    out["state_mb"] = extra["peak_state_bytes"] / 2**20
    out["energy_j"] = out["total_cpu_s"] * cfg["energy"]["cpu_power_w"]
    # deadline: fraction of batches whose inference+update exceeds the batch budget
    budget = cfg["metrics"].get("batch_deadline_ms")
    if budget:
        tot = (df["infer_wall"] + df["update_wall"]) * 1e3
        out["deadline_miss"] = float((tot > budget).mean())
    return out
