# Master Thesis — Personalized Federated Learning

## Project Overview

Research codebase for personalized federated learning experiments, built on top of the [PFLlib](https://github.com/TsingZ0/PFLlib) framework. The project implements and benchmarks multiple FL algorithms across standard vision and text datasets.

---

## Repository Structure

```
Master-thesis/
├── dataset/                        # One generator script per dataset
│   ├── generate_{NAME}.py
│   └── utils/
│       ├── dataset_utils.py        # separate_data, split_data, save_file
│       ├── HAR_utils.py
│       └── language_utils.py
├── system/
│   ├── main.py                     # Entry point: arg parsing, model dispatch, algorithm dispatch
│   └── flcore/
│       ├── clients/
│       │   ├── clientbase.py       # Base Client class — extend this for every algorithm
│       │   └── client{NAME}.py
│       ├── servers/
│       │   ├── serverbase.py       # Base Server class — extend this for every algorithm
│       │   └── server{NAME}.py
│       ├── trainmodel/
│       │   ├── models.py           # FedAvgCNN, DNN, MLR, BaseHeadSplit, etc.
│       │   ├── resnet.py
│       │   ├── alexnet.py
│       │   ├── bilstm.py
│       │   ├── mobilenet_v2.py
│       │   └── transformer.py
│       ├── optimizers/
│       │   └── fedoptimizer.py
│       └── loralib/
│           └── layers.py
├── system/utils/
│   ├── data_utils.py               # read_client_data
│   ├── result_utils.py             # average_data
│   ├── mem_utils.py                # MemReporter
│   └── dlg.py                      # Deep Leakage from Gradients
├── visualize/
│   ├── plot_metric.py              # Line curves from CSV files
│   ├── plot_bar_chart.py           # Bar chart (hardcoded accuracy table)
│   └── practical/
│       └── {Dataset}/
│           ├── Accuracy/           # {algo}_run{i}.csv — columns: round, test_acc
│           ├── Loss/               # {algo}_run{i}.csv — columns: round, train_loss
│           └── Result/             # {Dataset}_{algo}.json — best accuracy per run
├── results/                        # Legacy .h5 files
├── env.yaml
└── README.md
```

---

## Environment

```sh
conda env create -f env.yaml   # name: PFL, Python 3.11
conda activate PFL
```

Key deps: PyTorch 2.11.0, torchvision, numpy<2, scikit-learn, scipy, cvxpy, higher, transformers, h5py, pandas, matplotlib. GPU required (CUDA 12.x).

---

## Running Experiments

```sh
# Step 1 — generate partitioned data
cd dataset/
python generate_FashionMNIST.py iid balance -       # IID balanced
python generate_FashionMNIST.py noniid - pat        # pathological non-IID
python generate_FashionMNIST.py noniid - dir        # Dirichlet non-IID

# Step 2 — train
cd system/
python main.py -data FashionMNIST -m CNN -algo FedAvg -gr 300 -did 0

# Step 3 — visualize
cd visualize/
python plot_metric.py practical/FashionMNIST/Accuracy/FedAvg_run0.csv \
    --metric test_acc --ema 0.15 --output out.png
```

### Common flags

| Flag | Long form | Default | Notes |
|------|-----------|---------|-------|
| `-data` | `--dataset` | `MNIST` | Must match a generated dataset folder |
| `-m` | `--model` | `CNN` | See Model Keys below |
| `-algo` | `--algorithm` | `FedAvg` | See Algorithm Keys below |
| `-gr` | `--global_rounds` | `2000` | Communication rounds |
| `-nc` | `--num_clients` | `20` | Total clients |
| `-jr` | `--join_ratio` | `1.0` | Fraction selected per round |
| `-ls` | `--local_epochs` | `1` | Local update epochs |
| `-lr` | `--local_learning_rate` | `0.005` | SGD LR |
| `-lbs` | `--batch_size` | `10` | Mini-batch size |
| `-ncl` | `--num_classes` | `10` | Output classes |
| `-did` | `--device_id` | `"0"` | Sets `CUDA_VISIBLE_DEVICES` |
| `-eg` | `--eval_gap` | `1` | Evaluate every N rounds |
| `-t` | `--times` | `1` | Independent repetitions |
| `-ab` | `--auto_break` | `False` | Early stop when top-1 stalls |

---

## Design Patterns

### 1. Server / Client Pair

Every algorithm is a **matched pair** of one server file and one client file:

```
flcore/servers/server{NAME}.py   →   class {NAME}(Server)
flcore/clients/client{NAME}.py   →   class client{NAME}(Client)
```

The server owns the global model and the training loop. The client owns a local model copy and implements local training. The server instantiates all clients and drives all communication.

### 2. The Round Loop (server)

Every `server.train()` follows this skeleton — implement only what your algorithm changes:

```python
def train(self):
    for i in range(self.global_rounds + 1):
        self.selected_clients = self.select_clients()

        if i % self.eval_gap == 0:
            self.evaluate()                     # calls test_metrics + train_metrics on ALL clients

        self.send_models()                      # push global_model to selected clients
        for client in self.selected_clients:
            client.train()                      # local update
        self.receive_models()                   # collect uploaded models
        self.aggregate_parameters()             # FedAvg by default

    self.save_results()
    self.save_global_model()
```

Override `send_models` / `receive_models` / `aggregate_parameters` as needed. Add extra state (e.g. prototypes, control variates) as server attributes and pass them through `send_models`.

### 3. BaseHeadSplit — Backbone / Head Split

Many algorithms need separate access to the feature extractor (`base`) and the classifier (`head`). This is done in `main.py` before instantiating the server:

```python
args.head = copy.deepcopy(args.model.fc)
args.model.fc = nn.Identity()
args.model = BaseHeadSplit(args.model, args.head)
```

`BaseHeadSplit` (in `models.py`) wraps the two sub-modules and exposes:
- `model.base(x)` — feature extractor forward pass only
- `model.head(z)` — classifier forward pass only
- `model(x)` — full forward pass

**Use this pattern whenever** your algorithm needs to treat the backbone and classifier separately (prototype extraction, partial personalisation, head-only aggregation, etc.).

### 4. Client `set_parameters`

The base `set_parameters(model)` simply copies all parameters from the global model into the client's local model. Override it to implement personalised model receipt — e.g. only copy the backbone, align the backbone to local data before copying, or mix global and local weights.

### 5. Evaluation

`server.evaluate()` calls `client.test_metrics()` and `client.train_metrics()` on **all** clients (not just selected ones), then appends aggregate accuracy / loss to `self.rs_test_acc` / `self.rs_train_loss`. Override `test_metrics` on the client if the evaluation forward pass differs from training (e.g. uses a personalised head).

### 6. Algorithm-Specific Flags

Add new hyperparameters to the `argparse` block at the bottom of `main.py`. Group them with a comment. Access via `args.your_flag` in both server and client `__init__`.

---

## Adding a New Algorithm

1. **Client** — create `system/flcore/clients/client{NAME}.py`:
   ```python
   from flcore.clients.clientbase import Client

   class client{NAME}(Client):
       def __init__(self, args, id, train_samples, test_samples, **kwargs):
           super().__init__(args, id, train_samples, test_samples, **kwargs)
           # algorithm-specific state

       def train(self):
           # local update logic
   ```

2. **Server** — create `system/flcore/servers/server{NAME}.py`:
   ```python
   from flcore.servers.serverbase import Server
   from flcore.clients.client{NAME} import client{NAME}

   class {NAME}(Server):
       def __init__(self, args, times):
           super().__init__(args, times)
           self.set_slow_clients()
           self.set_clients(client{NAME})

       def train(self):
           # round loop
   ```

3. **Register** in `system/main.py` — add an import and an `elif` branch:
   ```python
   elif args.algorithm == "NAME":
       # apply BaseHeadSplit here if needed
       server = NAME(args, i)
   ```

4. **Flags** — add any new `parser.add_argument` entries at the bottom of `main.py`.

---

## Adding a New Dataset

Create `dataset/generate_{NAME}.py` following `generate_MNIST.py`:

```python
from utils.dataset_utils import separate_data, split_data, save_file
```

The script must accept three positional args: `{iid|noniid} {balance|-} {-|pat|dir}` and write per-client `.pkl` files under `dataset/{NAME}/`.

---

## Adding a New Model

1. Add the model class to `system/flcore/trainmodel/models.py` (or a new file in `trainmodel/`).
2. Add a dispatch branch in the model-selection block in `main.py`. Ensure the class has a `.fc` attribute if you plan to use the BaseHeadSplit pattern.

---

## Base Class API Reference

### `Client` (clientbase.py)

| Attribute / Method | Purpose |
|--------------------|---------|
| `self.model` | Deep copy of `args.model`; the client's local model |
| `self.optimizer` | SGD over `self.model.parameters()` |
| `self.loss` | `nn.CrossEntropyLoss()` |
| `self.local_epochs` | Local update epochs per round |
| `self.device` | CUDA/CPU device |
| `load_train_data(batch_size)` | Returns shuffled DataLoader (drop_last=True) |
| `load_test_data(batch_size)` | Returns DataLoader |
| `set_parameters(model)` | Copy all params from `model` → `self.model` |
| `test_metrics()` | Returns `(correct, total, auc)` |
| `train_metrics()` | Returns `(loss_sum, count)` |
| `save_item(item, name)` / `load_item(name)` | Persist tensors to `save_folder_name/` |

### `Server` (serverbase.py)

| Attribute / Method | Purpose |
|--------------------|---------|
| `self.global_model` | The shared global model |
| `self.clients` | List of all Client objects |
| `self.selected_clients` | Clients chosen this round |
| `self.uploaded_ids`, `self.uploaded_models`, `self.uploaded_weights` | Set by `receive_models()` |
| `self.rs_test_acc`, `self.rs_train_loss` | Per-round metrics list |
| `set_clients(clientObj)` | Instantiate all clients |
| `select_clients()` | Random sample of `num_join_clients` |
| `send_models()` | Push `global_model` to all clients |
| `receive_models()` | Pull models; applies drop_rate and time_threshold |
| `aggregate_parameters()` | Sample-count weighted FedAvg |
| `add_parameters(w, client_model)` | `global += w * client` |
| `evaluate()` | Aggregate metrics, print, append to `rs_*` |
| `save_results()` | Write `.h5` to `../results/` |
| `save_global_model()` | Save `models/{dataset}/{algo}_server.pt` |

---

## Output Files

| Path | Content |
|------|---------|
| `visualize/practical/{Dataset}/Accuracy/{algo}_run{i}.csv` | `round, test_acc` per round |
| `visualize/practical/{Dataset}/Loss/{algo}_run{i}.csv` | `round, train_loss` per round |
| `visualize/practical/{Dataset}/Result/{Dataset}_{algo}.json` | Best accuracy per run index |
| `results/{Dataset}_{algo}_{goal}_{times}.h5` | Legacy HDF5 (test_acc, test_auc, train_loss) |
| `system/models/{Dataset}/{algo}_server.pt` | Global model checkpoint |
