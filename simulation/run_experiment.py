"""Run the drift-aware continual-learning IIoT IDS simulation.

  python run_experiment.py --config configs/synthetic.json
  python run_experiment.py --config configs/edge_iiotset.json --seeds 5
  python run_experiment.py --config configs/edge_iiotset.json --inspect   # list columns/classes
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dacl.data import Scenario, load_dataset
from dacl.natural import NaturalScenario
from dacl.metrics import summarize
from dacl.report import build_report
from dacl.runner import pretrain, run_method


def deep_merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path):
    path = Path(path)
    cfg = json.loads(path.read_text())
    if "extends" in cfg:
        cfg = deep_merge(load_config(path.parent / cfg.pop("extends")), cfg)
    return cfg


def inspect(cfg):
    d = cfg["dataset"]
    if d["type"] != "csv":
        print("inspect only applies to csv datasets")
        return
    head = pd.read_csv(d["path"], nrows=5, low_memory=False)
    print(f"{len(head.columns)} columns:\n{list(head.columns)}\n")
    counts = pd.read_csv(d["path"], usecols=[d["label_col"]])[d["label_col"]].value_counts()
    print(f"label column '{d['label_col']}':\n{counts.to_string()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--methods", nargs="*", default=None, help="subset of methods to run")
    ap.add_argument("--no-ablations", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--arch", default=None, help="backbone: mlp cnn lstm gru cnn_lstm transformer")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | cuda:N (learning only; edge costs on CPU)")
    ap.add_argument("--seed-list", type=int, nargs="*", default=None, help="explicit seeds (overrides --seeds)")
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.arch:
        cfg["model"]["arch"] = args.arch
    if args.inspect:
        inspect(cfg)
        return
    # Phase 5: resource-limited edge profile (CPU only, fixed thread count)
    torch.set_num_threads(cfg.get("threads", 1))
    from dacl.device import set_device
    set_device(args.device)
    cfg["device"] = args.device
    seeds = args.seeds or cfg.get("seeds", 5)
    methods = args.methods or (cfg["methods"] + ([] if args.no_ablations else cfg.get("ablations", [])))
    out = Path(args.out or f"results/{cfg['name']}/{cfg['model'].get('arch', 'mlp')}")
    (out / "timelines").mkdir(parents=True, exist_ok=True)
    (out / "config_used.json").write_text(json.dumps(cfg, indent=2))
    (out / "phase_eval.jsonl").unlink(missing_ok=True)
    for f in (out / "timelines").glob("*.csv"):
        f.unlink()

    sc = cfg["scenario"]
    keep = (None if sc.get("type") == "natural"
            else [sc["normal"]] + [c for g in sc["groups"].values() for c in g])
    t0 = time.time()
    X, y, feat_names, info = load_dataset(cfg["dataset"], keep_classes=keep)
    print(f"loaded {X.shape} in {time.time() - t0:.1f}s from {info['source']}")
    if "class_counts" in info:
        print(f"class counts: {info['class_counts']}")

    summaries = []
    seed_list = args.seed_list if args.seed_list is not None else list(range(seeds))
    for seed in seed_list:
        if sc.get("type") == "natural":
            scenario = NaturalScenario(X, y, info["times"], sc, seed)
        else:
            scenario = Scenario(X, y, sc, seed)
        if seed == seed_list[0]:
            print(scenario.describe())
            meta = {"phase_start": scenario.phase_start, "phase_names": scenario.phase_names,
                    "drift_points": scenario.drift_points, "classes": scenario.class_names,
                    "n_features": int(scenario.X_pre.shape[1]), "dataset_info": info}
            (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        model = pretrain(scenario, cfg, seed)
        for m in methods:
            t1 = time.time()
            rows, phase_eval, extra = run_method(m, scenario, model, cfg, seed)
            pd.DataFrame(rows).to_csv(out / "timelines" / f"{m}_s{seed}.csv", index=False)
            s = {"method": m, "seed": seed, **summarize(rows, phase_eval, extra, scenario, cfg)}
            summaries.append(s)
            with open(out / "phase_eval.jsonl", "a") as f:
                f.write(json.dumps({"method": m, "seed": seed, "phase_eval": phase_eval,
                                    "alarm_info": extra["alarm_info"],
                                    "extents": extra["extents"],
                                    "gate_log": extra.get("gate_log", []),
                                    "chain_latency": extra.get("chain_latency", [])}) + "\n")
            print(f"seed {seed} {m:14s} acc={s['acc']:.4f} forget={s['forgetting']:+.3f} "
                  f"delay={s.get('det_delay', np.nan)} upd_cpu={s['update_cpu_s']:.2f}s "
                  f"({time.time() - t1:.1f}s)", flush=True)
        pd.DataFrame(summaries).to_csv(out / "summary_per_seed.csv", index=False)

    mean, std = build_report(out, methods)
    cols = ["acc", "f1", "det_delay", "recovery_batches", "forgetting", "return_recall",
            "update_cpu_s", "infer_us_per_sample", "state_mb", "energy_j"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(mean[[c for c in cols if c in mean.columns]].round(4))
    print(f"\nresults in {out.resolve()}")


if __name__ == "__main__":
    sys.exit(main())
