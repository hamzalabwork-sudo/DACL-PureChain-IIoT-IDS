"""Edge cost profiling on a single CPU thread (run in isolation, nothing else busy).

For every (dataset, backbone): per-flow inference latency (p50/p99), batch-256 inference time,
and the CPU time of one gradient step of each update type:
  ce      plain cross-entropy on 256 samples (FineTune, EWC w/o penalty, FullRetrain step)
  er      256 new + 256 replay (Experience Replay)
  dacl    256 new + 256 window + 256 replay + KD teacher forward (DACL adaptation step)
Stage-1 grad-step counts x these step costs = device-independent edge update cost.

  python profile_edge.py --out results/edge_profile.csv
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dacl.models import build_model, n_params
from dacl.runner import count_flops

DATASETS = ["wustl_iiot", "xiiotid", "edge_iiotset", "forge_iiot", "mu_iot",
            "natural_wustl_iiot", "natural_forge_iiot", "natural_xiiotid"]
ARCHS = ["mlp", "cnn", "lstm", "gru", "cnn_lstm", "transformer"]


def dims(name):
    """(n_features, n_classes) from the stage-1 metadata if present, else from config + cache."""
    for meta in Path("results").glob(f"**/{name}/**/meta.json"):
        m = json.loads(meta.read_text())
        return m["n_features"], len(m["classes"])
    sys.argv = ["x"]
    from run_experiment import load_config
    from dacl.data import load_dataset
    cfg = load_config(f"configs/{name}.json")
    sc = cfg["scenario"]
    if sc.get("type") == "natural":
        X, y, _, _ = load_dataset(cfg["dataset"], keep_classes=None)
        _, cnt = np.unique(y, return_counts=True)
        return X.shape[1], int((cnt >= sc.get("min_class_rows", 200)).sum())
    keep = [sc["normal"]] + [c for g in sc["groups"].values() for c in g]
    X, _, _, _ = load_dataset(cfg["dataset"], keep_classes=keep)
    return X.shape[1], len(keep)


def timeit(fn, reps, warm=5):
    # high-resolution wall clock on an otherwise idle machine, single thread
    # (process_time on Windows ticks at ~15.6 ms, too coarse for sub-ms steps)
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return np.array(ts)


def wall(fn, reps, warm=20):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return np.array(ts)


def profile(arch, d, k, arch_cfg, reps):
    torch.manual_seed(0)
    m = build_model(arch, d, k, arch_cfg)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    teacher = copy.deepcopy(m).eval()
    x1 = torch.randn(1, d)
    xb, yb = torch.randn(256, d), torch.randint(0, k, (256,))
    x2, y2 = torch.randn(512, d), torch.randint(0, k, (512,))
    xr, yr = torch.randn(256, d), torch.randint(0, k, (256,))

    def infer1():
        with torch.no_grad():
            m(x1)

    def infer_b():
        with torch.no_grad():
            m(xb)

    def step(x, y, kd=False):
        def f():
            m.train()
            opt.zero_grad()
            loss = F.cross_entropy(m(x), y)
            if kd:
                out_r = m(xr)
                loss = loss + F.cross_entropy(out_r, yr)
                with torch.no_grad():
                    t = teacher(xr)
                loss = loss + F.kl_div(F.log_softmax(out_r / 2, 1), F.softmax(t / 2, 1),
                                       reduction="batchmean") * 4
            loss.backward()
            opt.step()
        return f

    m.eval()
    l1 = wall(infer1, reps * 5) * 1e3
    return {"params": n_params(m), "flops": count_flops(m, x1),
            "lat1_ms_mean": l1.mean(), "lat1_ms_p50": np.percentile(l1, 50),
            "lat1_ms_p95": np.percentile(l1, 95), "lat1_ms_p99": np.percentile(l1, 99),
            "rollback_ms": wall(lambda: m.load_state_dict(teacher.state_dict()), reps).mean() * 1e3,
            "infer256_ms": timeit(infer_b, reps).mean() * 1e3,
            "step_ce_ms": timeit(step(xb, yb), reps).mean() * 1e3,
            "step_er_ms": timeit(step(x2, y2), reps).mean() * 1e3,
            "step_dacl_ms": timeit(step(x2, y2, kd=True), reps).mean() * 1e3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/edge_profile.csv")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    arch_cfgs = json.loads(Path("configs/base.json").read_text())["model"]["arch_cfg"]
    rows = []
    for ds in DATASETS:
        d, k = dims(ds)
        for a in ARCHS:
            r = profile(a, d, k, arch_cfgs[a], args.reps)
            rows.append({"dataset": ds, "arch": a, "d": d, "k": k, **r})
            print(f"{ds:13s} {a:12s} " + " ".join(f"{key}={v:.3f}" if isinstance(v, float) else f"{key}={v}"
                                                   for key, v in r.items()), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
