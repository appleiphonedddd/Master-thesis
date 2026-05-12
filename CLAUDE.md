# Master Thesis — FedFIP: Personalized Federated Learning

## Project Overview

This is a research codebase for **FedFIP** (Fisher-weighted Aggregation and Prototype Integration for Personalized Federated Learning), a master's thesis project built on top of the PFLlib framework.

The core idea: each client computes a **Fisher Information Matrix (FIM) trace** as a reliability score, and the server aggregates global models via softmax-weighted averaging. Simultaneously, clients share lightweight **class-wise feature prototypes** with the server, which forms global semantic centers to guide local training — enabling cross-client knowledge transfer without sharing raw data.

## Repository Structure

```
Master-thesis/
├── dataset/               # Dataset generators (one script per dataset)
│   ├── generate_*.py      # Downloads & partitions data for FL simulation
│   └── utils/             # dataset_utils.py, HAR_utils.py, language_utils.py
├── system/                # Main training code
│   ├── main.py            # Entry point — arg parsing, model & algorithm dispatch
│   ├── flcore/
│   │   ├── clients/       # clientbase.py + one file per algorithm (clientXXX.py)
│   │   ├── servers/       # serverbase.py + one file per algorithm (serverXXX.py)
│   │   ├── trainmodel/    # models.py, resnet.py, alexnet.py, bilstm.py, ...
│   │   ├── optimizers/    # fedoptimizer.py
│   │   └── loralib/       # LoRA layers
│   └── utils/             # data_utils, result_utils, mem_utils, ALA, FLAYER, INF
├── visualize/             # Plotting scripts for accuracy / loss curves
├── env.yaml               # Conda environment (name: PFL, Python 3.11)
└── README.md
```

## Environment Setup

```sh
conda env create -f env.yaml
conda activate PFL
```

Key dependencies: PyTorch 2.11.0, torchvision, numpy<2, scikit-learn, scipy, cvxpy, higher, transformers.

GPU: NVIDIA CUDA required (tested on RTX 3060, CUDA 12.x).

## How to Run an Experiment

**Step 1 — generate a partitioned dataset:**
```sh
cd dataset/
python generate_FashionMNIST.py iid balance -       # IID balanced
python generate_FashionMNIST.py noniid - pat        # Pathological non-IID
python generate_FashionMNIST.py noniid - dir        # Dirichlet non-IID
```

**Step 2 — run training:**
```sh
cd system/
python main.py -data FashionMNIST -m CNN -algo FedFIP -gr 300 -did 0
```

Common flags:
| Flag | Meaning | Default |
|------|---------|---------|
| `-data` | Dataset name | MNIST |
| `-m` | Model (CNN, ResNet18, DNN, …) | CNN |
| `-algo` | Algorithm key | FedAvg |
| `-gr` | Global communication rounds | 2000 |
| `-nc` | Number of clients | 20 |
| `-jr` | Join ratio per round | 1.0 |
| `-ls` | Local epochs per round | 1 |
| `-did` | CUDA device id(s) | 0 |
| `-ncl` | Number of classes | 10 |

**Step 3 — visualize results:**
```sh
cd visualize/
python plot_metric.py run1.csv run2.csv --metric test_acc --ema 0.15 --output out.png
python plot_metric.py run1.csv run2.csv --metric train_loss --ema 0.15 --output out.png
python plot_bar_chart.py
```

Per-round CSV files (`{algo}_run{i}_rs_test_acc.csv`, `{algo}_run{i}_rs_train_loss.csv`) are saved automatically in the working directory after each run.

## Implemented Algorithms

The algorithm passed to `-algo` maps to a `server{NAME}.py` / `client{NAME}.py` pair under `flcore/`. Key algorithms:

| Algorithm | Key |
|-----------|-----|
| **FedFIP** (this thesis) | `FedFIP` |
| FedAS (direct baseline) | `FedAS` |
| DCPFL | `DCPFL` |
| FedCPD | `FedCPD` |
| FedCALM | `FedCALM` |
| FedAvg | `FedAvg` |
| FedProto | `FedProto` |
| … many more | see `main.py` |

## Architecture Conventions

### Adding a New Algorithm

1. Create `system/flcore/clients/clientNAME.py` — extend `Client` from `clientbase.py`.
2. Create `system/flcore/servers/serverNAME.py` — extend `Server` from `serverbase.py`.
3. Register in `system/main.py`: add an `elif args.algorithm == "NAME":` branch.

Most algorithms that use a split head/backbone pattern call:
```python
args.head = copy.deepcopy(args.model.fc)
args.model.fc = nn.Identity()
args.model = BaseHeadSplit(args.model, args.head)
```
before instantiating the server.

### Adding a New Dataset

Create `dataset/generate_DATA.py` using `generate_MNIST.py` as a template:
```python
from utils.dataset_utils import separate_data, split_data, save_file
```

### Adding a New Model

Add it to `system/flcore/trainmodel/models.py` and handle it in the model-dispatch block in `main.py`.

## FedFIP-Specific Details

- **Fisher weighting** (`serverfip.py:aggregate_wrt_fisher`): collects `fim_trace_history[-1]` from all uploaded clients, normalises with softmax-style division, then does a weighted parameter average.
- **Prototype aggregation** (`serverfip.py:aggregate_prototypes`): averages same-class prototype vectors from clients that have local data for that class; zero vectors are excluded via an abs-sum mask.
- **Client training** (`clientfip.py:train`): runs standard CE loss + prototype alignment loss `α * ||z - proto_y||²` when global prototypes are available, then computes and stores the FIM trace. Non-selected clients still compute their FIM trace (no gradient update).
- **`-al` / `--alpha`** controls the prototype alignment loss weight (default 1.0).

## Output Files

- `{algo}_run{i}_rs_test_acc.csv` — per-round test accuracy
- `{algo}_run{i}_rs_train_loss.csv` — per-round train loss
- Global model checkpoints saved via `server.save_global_model()`
- Aggregated statistics via `average_data()` in `utils/result_utils.py`
