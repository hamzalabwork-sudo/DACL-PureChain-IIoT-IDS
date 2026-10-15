"""Generic experiment grid: one job per (config, backbone, seed), spread over devices, resumable.

  python run_grid.py stage2 --archs mlp --devices cuda:0 cuda:1 cuda:2 --jobs 12
  python run_grid.py gate   --devices cuda:0 cuda:1 cuda:2 --jobs 12
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATASETS = ["wustl_iiot", "xiiotid", "edge_iiotset", "forge_iiot", "mu_iot"]
SENS = [f"sens_{p}_{v}_{d}" for p, vals in (("mem", (500, 1000, 2000, 5000)), ("thr", (6, 9, 12, 18)),
                                            ("kmax", (2, 3, 5, 8)), ("delay", (0, 5, 10, 20)))
        for v in vals for d in DATASETS]
GRIDS = {
    # natural (timestamp-ordered) drift: datasets with usable recording order
    "natural": {"configs": ["natural_wustl_iiot", "natural_forge_iiot", "natural_xiiotid"],
                "methods": ["Static", "FineTune", "EWC", "LwF", "ER", "DER++", "PeriodicER",
                            "FullRetrain", "DACL"]},
    # net-security-benefit gate: hard floor h in {20,30,50}%, soft (escalation) limit 10%
    "gate2": {"configs": [f"gate2_{d}_{v}_h{h}" for d in DATASETS
                          for v in ("gate", "poisonP2", "poisonP4") for h in (20, 30, 50)],
              "methods": ["DACL", "DACL+Gate", "DACL+Gate-dual"]},
    # final gate policy: balanced net benefit (half false alarms, half missed attacks), h = 30%
    "gate3": {"configs": [f"gate3_{d}_{v}" for d in DATASETS for v in ("gate", "poisonP2", "poisonP4")],
              "methods": ["DACL", "DACL+Gate", "DACL+Gate-dual"]},
    # label scarcity: only 1/5/10% of flows are labelled (MLP)
    "labels": {"configs": [f"lab_{b}_{d}" for b in (1, 5, 10) for d in DATASETS],
               "methods": ["Static", "ER", "DER++", "FullRetrain", "DACL"]},
    # IDS-specific drift-adaptive baseline (sliding-window retraining), injected + natural drift
    "window": {"configs": DATASETS + ["natural_wustl_iiot", "natural_forge_iiot", "natural_xiiotid"],
               "methods": ["WindowRetrain"]},
    # one-at-a-time sensitivity (MLP): memory, detector threshold, kmax, label delay
    "sensitivity": {"configs": SENS, "methods": ["ER", "DACL"]},
    # every backbone x every dataset with the full baseline set (per-dataset model comparison)
    "stage1b": {"configs": DATASETS,
                "methods": ["Static", "FineTune", "EWC", "LwF", "ER", "DER++", "PeriodicER",
                            "FullRetrain", "DACL"]},
    # full method comparison + ablations
    "stage2": {"configs": DATASETS,
               "methods": ["Static", "FineTune", "EWC", "LwF", "ER", "DER++", "PeriodicER",
                           "FullRetrain", "DriftFT", "DACL", "DACL-noKD", "DACL-noReplay", "DACL-fixedK"]},
    # blockchain-verified updates: clean (cost of security) and two poisoning attacks
    "gate": {"configs": [f"{d}_{v}" for d in DATASETS for v in ("gate", "poisonP2", "poisonP4")],
             "methods": ["DACL", "DACL+Gate"]},
    # gate tolerance sweep tau in {2,5,10,20}% (security-plasticity trade-off)
    "gatesweep": {"configs": [f"sweep_{d}_{v}_t{t:02d}" for d in DATASETS
                              for v in ("gate", "poisonP2", "poisonP4") for t in (2, 5, 10, 20)],
                  "methods": ["DACL", "DACL+Gate"]},
}


def run(job, devices, root, methods):
    i, (cfg, arch, seed) = job
    out = root / cfg / arch / f"s{seed}"
    if (out / "summary_per_seed.csv").exists():
        return 0
    log = root / "logs" / f"{cfg}_{arch}_s{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    with open(log, "w") as f:
        rc = subprocess.call([sys.executable, "run_experiment.py", "--config", f"configs/{cfg}.json",
                              "--arch", arch, "--seed-list", str(seed), "--methods", *methods,
                              "--device", devices[i % len(devices)], "--out", str(out)],
                             stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUTF8": "1"})
    print(f"[{time.strftime('%H:%M:%S')}] {cfg:22s} {arch:12s} s{seed} rc={rc} {time.time() - t:.0f}s",
          flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid", choices=GRIDS)
    ap.add_argument("--archs", nargs="*", default=["mlp"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--devices", nargs="*", default=["cpu"])
    args = ap.parse_args()
    g = GRIDS[args.grid]
    root = Path("results") / args.grid
    jobs = [(c, a, s) for c in g["configs"] for a in args.archs for s in range(args.seeds)]
    with ThreadPoolExecutor(args.jobs) as ex:
        rcs = list(ex.map(lambda j: run(j, args.devices, root, g["methods"]), enumerate(jobs)))
    print(f"failed jobs: {sum(r != 0 for r in rcs)}")


if __name__ == "__main__":
    main()
