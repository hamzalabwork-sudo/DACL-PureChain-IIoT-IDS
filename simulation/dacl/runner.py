"""Prequential stream execution with real-time cost measurement."""
import copy
import time

import numpy as np
import psutil
import torch
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score)
from torch.utils.flop_counter import FlopCounterMode

from . import device as dev
from .learners import METHODS
from .models import build_model, n_params, train_epochs


def pretrain(scenario, cfg, seed):
    torch.manual_seed(seed)
    m = cfg["model"]
    arch = m.get("arch", "mlp")
    arch_cfg = m.get("arch_cfg", {}).get(arch, {"hidden": m.get("hidden", [128, 64])} if arch == "mlp" else {})
    model = build_model(arch, scenario.X_pre.shape[1], scenario.n_classes, arch_cfg).to(dev.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=m["lr"])
    train_epochs(model, opt, scenario.X_pre, scenario.y_pre, m["pretrain_epochs"],
                 m["train_batch"], np.random.default_rng(seed))
    return model


def count_flops(model, x):
    """FLOPs of one forward pass. FlopCounterMode misses fused RNN kernels and the
    Transformer inference fast path, so count in train mode (no fast path) and add
    LSTM cells analytically (2 FLOPs per MAC). GRU on CPU decomposes into counted ops
    (verified: counted value equals 2*T*3*H*(I+H) + head)."""
    rnn_flops = []

    def hook(m, inp, out):
        b, t, i = inp[0].shape
        h = m.hidden_size
        rnn_flops.append(2 * b * t * 4 * h * (i + h))

    hs = [m.register_forward_hook(hook) for m in model.modules()
          if isinstance(m, torch.nn.LSTM)]
    was_training = model.training
    model.train()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    model.train(was_training)
    for h in hs:
        h.remove()
    return int(fc.get_total_flops() + sum(rnn_flops))


def edge_profile(model, d, reps=300):
    """Single-sample (per-flow) inference latency and FLOPs of the deployed model (always on CPU)."""
    model = copy.deepcopy(model).cpu()
    model.eval()
    x = torch.randn(1, d)
    with torch.no_grad():
        for _ in range(20):
            model(x)
        ts = []
        for _ in range(reps):
            t = time.perf_counter()
            model(x)
            ts.append(time.perf_counter() - t)
    ts = np.array(ts) * 1e3
    return {"params": n_params(model), "flops": count_flops(model, x),
            "lat1_ms_p50": float(np.percentile(ts, 50)), "lat1_ms_p95": float(np.percentile(ts, 95)),
            "lat1_ms_p99": float(np.percentile(ts, 99))}


def evaluate_groups(learner, scenario, phase_idx):
    """Phase-end evaluation on the held-out test sets of ALL classes (incl. not-yet-seen ones),
    under the phase's operating condition: group recalls, per-class recall, confusion matrix,
    macro precision, and PR-AUC (macro one-vs-rest and binary attack-vs-normal)."""
    out, Xs, ys = {}, [], []
    for g in scenario.groups:
        X, y = scenario.test_set(g, phase_idx)
        Xs.append(X)
        ys.append(y)
        pred = learner.predict_proba(X).argmax(1)
        out[f"{g}_recall"] = float((pred == y).mean())
        if g == "Normal":
            out["Normal_fpr"] = float((pred != scenario.normal_idx).mean())
        else:
            out[f"{g}_detect"] = float((pred != scenario.normal_idx).mean())
    X, y = np.concatenate(Xs), np.concatenate(ys)
    prob = learner.predict_proba(X)
    pred = prob.argmax(1)
    labels = list(range(scenario.n_classes))
    out["confusion"] = confusion_matrix(y, pred, labels=labels).tolist()
    out["class_recall"] = {scenario.class_names[c]: float((pred[y == c] == c).mean()) for c in labels}
    out["precision_macro"] = float(precision_score(y, pred, labels=labels, average="macro", zero_division=0))
    out["f1_macro"] = float(f1_score(y, pred, labels=labels, average="macro", zero_division=0))
    out["prauc_macro"] = float(np.mean([average_precision_score(y == c, prob[:, c]) for c in labels]))
    atk = y != scenario.normal_idx
    out["prauc_binary"] = float(average_precision_score(atk, 1.0 - prob[:, scenario.normal_idx]))
    return out


def run_method(name, scenario, pretrained, cfg, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1)
    lcfg = {**cfg["learners"], "lr": cfg["model"]["lr"], "train_batch": cfg["model"]["train_batch"]}
    learner = METHODS[name](copy.deepcopy(pretrained), lcfg, rng, scenario)
    proc = psutil.Process()

    # label-flipping poisoning of the labels revealed to the learner (evaluation uses true labels)
    pz = cfg.get("poisoning", {})
    prng = np.random.default_rng(seed + 99)

    def poison(y, ph):
        if not pz.get("enabled") or ph not in pz["phases"]:
            return y
        y = y.copy()
        atk = np.flatnonzero(y != scenario.normal_idx)
        flip = atk[prng.random(len(atk)) < pz["rate"]]
        y[flip] = scenario.normal_idx
        return y

    phase_eval = [{"phase": "P0 Deployment", **evaluate_groups(learner, scenario, 0)}]
    rows, peak_state = [], learner.state_bytes()
    label_delay, pending = int(cfg.get("label_delay", 0)), []
    label_budget = float(cfg.get("label_budget", 1.0))
    lrng = np.random.default_rng(seed + 7)
    for b, (X, y, ph) in enumerate(scenario.stream):
        w0, c0 = time.perf_counter(), time.process_time()
        probs = learner.predict_proba(X)
        w1, c1 = time.perf_counter(), time.process_time()
        pred = probs.argmax(1)
        steps0, alarms0 = learner.grad_steps, len(learner.alarms)
        # labels (and thus learning) reach the learner `label_delay` batches after prediction
        pending.append((X, poison(y, ph), probs, ph))
        if len(pending) > label_delay:
            Xo, yo, po, pho = pending.pop(0)
            if label_budget < 1.0:
                # only a random fraction of flows is ever labelled; unsupervised drift signals
                # (entropy, feature shift) still see the whole batch via `full_batch`
                k = max(1, int(round(label_budget * len(yo))))
                sel = lrng.choice(len(yo), k, replace=False)
                learner.full_batch = (Xo, po)
                Xo, yo, po = Xo[sel], yo[sel], po[sel]
            learner.phase = pho
            learner.observe(Xo, yo, po)
            learner.full_batch = None
        w2, c2 = time.perf_counter(), time.process_time()
        peak_state = max(peak_state, learner.state_bytes())

        attack = y != scenario.normal_idx
        row = {"batch": b, "phase": ph,
               "acc": float((pred == y).mean()),
               "f1": float(f1_score(y, pred, labels=np.unique(y), average="macro", zero_division=0)),
               "mcc": float(matthews_corrcoef(y, pred)),
               "precision": float(precision_score(y, pred, labels=np.unique(y), average="macro",
                                                  zero_division=0)),
               "missed_attacks": int((attack & (pred == scenario.normal_idx)).sum()),
               "detect": float((pred[attack] != scenario.normal_idx).mean()) if attack.any() else np.nan,
               "fpr": float((pred[~attack] != scenario.normal_idx).mean()) if (~attack).any() else np.nan,
               "entropy": float(-(probs * np.log(probs + 1e-12)).sum(1).mean()),
               "infer_wall": w1 - w0, "infer_cpu": c1 - c0,
               "update_wall": w2 - w1, "update_cpu": c2 - c1,
               "grad_steps": learner.grad_steps - steps0,
               "alarm": int(len(learner.alarms) > alarms0)}
        for g, cs in scenario.groups.items():
            m = np.isin(y, [scenario.cidx[c] for c in cs])
            row[f"{g}_recall"] = float((pred[m] == y[m]).mean()) if m.any() else np.nan
        rows.append(row)
        if b + 1 == scenario.phase_end[ph]:
            phase_eval.append({"phase": scenario.phase_names[ph], **evaluate_groups(learner, scenario, ph)})

    extra = {"alarms": learner.alarms, "alarm_info": getattr(learner, "alarm_info", []),
             "extents": getattr(learner, "extents", []),
             "uses_detector": learner.uses_detector,
             "device": str(dev.DEVICE),
             "peak_state_bytes": peak_state, "rss_bytes": proc.memory_info().rss,
             "n_samples": sum(len(s[1]) for s in scenario.stream),
             "edge": edge_profile(learner.model, scenario.X_pre.shape[1]),
             "gate_log": getattr(learner, "gate_log", []),
             "chain_latency": getattr(getattr(learner, "gate", None), "latency", [])}
    return rows, phase_eval, extra
