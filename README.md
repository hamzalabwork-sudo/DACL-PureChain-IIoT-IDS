# DACL: Drift-Aware Continual Learning with PureChain-Verified Model Updates for IIoT Intrusion Detection

Code, smart contract, configurations and experiment results for the paper

> H. Ibrahim, L. A. C. Ahakonye, J.-M. Lee, D.-S. Kim, "Drift-Aware Continual Learning with PureChain-Verified Model Updates for Real-Time Industrial IoT Intrusion Detection," submitted to *IEEE Internet of Things Journal*.

DACL adapts an IIoT intrusion detection system online when traffic drifts and stops when the model has recovered. It also retains knowledge of known attacks. Each model update is verified by a smart contract on the PureChain network before the model may serve traffic.

DACL has four parts:
- **Detect:** a hybrid CUSUM drift detector (error, predictive entropy, feature shift).
- **Adapt:** severity-scaled adaptation, stopped by a recovery rule.
- **Retain:** a class-balanced replay memory plus selective distillation.
- **Verify:** `ModelUpdateGate.sol`, a 2-of-3 validator contract that accepts an update only if it yields a net security benefit.

## Repository layout

```
simulation/
  dacl/                 core package
    data.py, natural.py   dataset loading, injected and natural (timestamp-order) streams
    drift.py              hybrid CUSUM drift detector
    models.py             backbones: MLP, 1D-CNN, LSTM, GRU, CNN-LSTM, FT-Transformer
    learners.py           DACL and baselines (Static, FineTune, EWC, LwF, ER, DER++, PeriodicER,
                          WindowRetrain, FullRetrain, DriftFT, ablations) + verified-update logic
    runner.py, metrics.py prequential test-then-train loop, detection / lifecycle / edge metrics
    *_report.py           aggregation scripts that produce results/*_report and results/per_dataset
  chain/
    ModelUpdateGate.sol   update-verification contract (+ compiled ABI/bytecode in .json)
    gate.py               local (simulated) and live PureChain gate backends
    scalability.py        live multi-gateway scalability benchmark
    deployment.json       address of the contract deployed on PureChain (public data only)
  configs/              one JSON per experiment (inherit from base.json via "extends")
  run_experiment.py     run one configuration
  run_grid.py           run a full experiment grid (resumable, multi-GPU)
  profile_edge.py       single-core edge cost profile per backbone
  results/              results used in the paper (see below)
paper/
  make_tables.py        regenerates every table and result figure of the paper from simulation/results
```

## Setup

Python 3.10 or later.

```
pip install -r requirements.txt
```

A GPU is optional: learning metrics do not depend on the device, and edge costs are always measured on one CPU core.

## Datasets

The datasets are not redistributed here. Download them and place them in the **repository root**, at the paths used by `simulation/configs/*.json`:

| Dataset | Expected path | Source |
|---|---|---|
| WUSTL-IIoT-2021 | `wustl_iiot_2021.csv` | Zolanvari et al., IEEE DataPort, doi:10.21227/yftq-n229 |
| X-IIoTID | `X-IIoTID/X-IIoTID dataset.csv` | Al-Hawawreh et al., IEEE IoT J. 2022, doi:10.1109/JIOT.2021.3102056 (also on Kaggle: `munaalhawawreh/xiiotid-iiot-intrusion-dataset`) |
| Edge-IIoTset | `Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv` | Ferrag et al., IEEE Access 2022, doi:10.1109/ACCESS.2022.3165809 (Kaggle: `sibasispradhan/edge-iiotset-dataset`) |
| ForgeIIOT (IoTForge Pro) | `Forge_IIOT_Pro.csv` | IEEE IoT J., doi:10.1109/JIOT.2024.3501017 |
| MU-IoT | `MU-IoT.csv` | IEEE Access 2024, doi:10.1109/ACCESS.2024.3494052 |

On the first run, preprocessed arrays are cached in `simulation/cache/`.

## Reproducing the experiments

Run all commands from `simulation/`.

```
# a single configuration (5 seeds, all methods, MLP backbone)
python run_experiment.py --config configs/mu_iot.json --arch mlp --seeds 5

# full grids used in the paper (resumable; spread over GPUs if available)
python run_grid.py stage1b     --archs mlp cnn lstm gru cnn_lstm transformer --devices cuda:0 cuda:1 cuda:2
python run_grid.py stage2      # ablations
python run_grid.py natural     # natural (timestamp-ordered) drift
python run_grid.py window      # WindowRetrain baseline
python run_grid.py sensitivity # memory, CUSUM threshold, k_max, label delay
python run_grid.py labels      # label scarcity (1/5/10% labels)
python run_grid.py gate3       # final verified-update policy (clean, poisoning in P2, unlearning attack in P4)
python run_grid.py gate2       # gate-policy variants (supplementary)
python run_grid.py gatesweep   # strict per-class policy sweep (supplementary)

# edge cost profile (one CPU thread)
python profile_edge.py --threads 1 --out results/edge_profile_x86_1thread.csv

# aggregation
python -m dacl.gate2_report --root results/gate3 --prefix gate3 --out results/gate3_report
python -m dacl.labels_report
python -m dacl.per_dataset_report

# paper tables and figures (from the repository root)
python paper/make_tables.py
```

| Paper result | Grid / script | Result files |
|---|---|---|
| Detection quality, lifetime F1, significance, lifecycle security | `stage1b` | `results/per_dataset/*_L1..L4_*.csv`, `all_runs_derived.csv` |
| Natural drift | `natural`, `window` | `results/per_dataset/all_runs_natural_derived.csv` |
| Six backbones | `stage1b` (all archs) | `results/per_dataset/*_method_x_backbone_f1.csv` |
| PureChain-verified updates | `gate3` | `results/gate3_report/`, `results/per_dataset/*_L5_purechain_gate.csv` |
| Label scarcity, sensitivity | `labels`, `sensitivity` | `results/labels_report/`, `results/per_dataset/*_sensitivity.csv` |
| Ablation | `stage2` | `results/stage2/` |
| Chain scalability (1/5/10 gateways) | `chain.scalability` | `results/chain_scalability/` |
| Gate-policy variants (supplementary) | `gate2`, `gatesweep` | `results/gate_policy_ablation.csv`, `results/gate2_report/` |

### Included results

`simulation/results/` contains, for every run (`<grid>/<config>/<backbone>/s<seed>/`):
- `summary_per_seed.csv`: all metrics, one row per method;
- `phase_eval.jsonl`: held-out per-class recall per phase, drift alarms and gate decisions;
- `meta.json`: stream and phase layout.

It also contains the aggregated reports (`per_dataset/REPORT.md` and per-dataset CSVs for metric layers L1–L5). From these files `paper/make_tables.py` regenerates every table without rerunning the experiments.

Per-batch timelines (`timelines/*.csv`) are included only for the MLP runs of `stage1b`, `natural` and `window`, which the lifetime figures use. To keep the repository small, the other timelines, logs and per-run plots are left out; running the grids regenerates them.

## PureChain (live blockchain)

By default, verified-update experiments use a local simulation of the contract (`"backend": "local"`), which follows exactly the same acceptance logic. Two parts of the work run on the live PureChain network: `configs/wustl_iiot_live.json` and `chain/scalability.py`.

PureChain details:
- RPC: `https://purechainnode.com`
- Chain ID: 900520900520
- Consensus: PoA²
- Gas price: 0

Test accounts are generated on first use and stored locally in `simulation/.secrets/`. This folder is ignored by git, so never commit it. To deploy from your own account, set the `PURECHAIN_PRIVATE_KEY` environment variable.

The contract used in the paper is deployed at `0x3aa2875c2AB8316a759e9B0924C82de790d5428e` (see `chain/deployment.json` and `results/chain_scalability/contract.json`).

```
python run_experiment.py --config configs/wustl_iiot_live.json --methods DACL+Gate-dual --seeds 1
python -m chain.scalability --gateways 1 5 10 --rounds 5 --out results/chain_scalability
```

## License

MIT. See `LICENSE`. If you use this code, please cite the paper above.
