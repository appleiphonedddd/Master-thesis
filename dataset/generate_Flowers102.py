"""
Flowers102 federated split generator.

Purpose:
    - Download/prepare Oxford Flowers102 dataset.
    - Merge official train/val/test into a single pool.
    - Partition samples across federated clients (IID / Non-IID variants).
    - Persist per-client train/test splits and a distribution plot.

CLI (current):
    python generate_flowers102.py  <iid|noniid>  <balance|->  <pat|dir|->

    Examples:
      # IID & balanced across 20 clients
      python generate_flowers102.py iid balance -

      # Non-IID pathological across 20 clients
      python generate_flowers102.py noniid - pat

      # Non-IID Dirichlet (NOTE: the alpha is configured in utils/separate_data)
      python generate_flowers102.py noniid - dir

Outputs (side effects):
    Flowers102/
      ├── config.json            # metadata for reproducibility
      ├── train/                 # per-client train data files
      ├── test/                  # per-client test data files
      └── figures/client_data_distribution.png

Notes:
    - Images are resized & center-cropped to IMG_SIZE, normalized to [-1, 1].
    - A single-batch DataLoader loads the entire split into RAM (fast & simple).
      If you hit OOM on low-memory machines, change to iterative loading.
"""

import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file
import matplotlib.pyplot as plt

random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path = "Flowers102/"

IMG_SIZE = 64

def load_split(split_name, transform, root_dir):
    """
    Load a Flowers102 split and return images/labels as NumPy arrays.

    Args:
        split_name: One of {"train", "val", "test"} as defined by torchvision.
        transform: Transform applied by the dataset when fetching samples.
        root_dir: Root directory for raw cached data.

    Returns:
        images_np: float32 array of shape (N, 3, IMG_SIZE, IMG_SIZE), range [-1, 1].
        labels_np: int64 array of shape (N,), class ids in [0, num_classes-1].

    Notes:
        - We request a DataLoader with batch_size=len(dataset) to pull the
          entire split in one shot for simplicity. This assumes RAM is sufficient.
        - If you see memory pressure, refactor to iterate mini-batches and
          concatenate.
    """

    ds = torchvision.datasets.Flowers102(
        root=root_dir, split=split_name, download=True, transform=transform
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=False, num_workers=0)
    for imgs, targets in loader:
        images_np = imgs.cpu().detach().numpy()
        labels_np = targets.cpu().detach().numpy()
        return images_np, labels_np
    return np.empty((0, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32), np.empty((0,), dtype=np.int64)

def generate_dataset(dir_path, num_clients, niid, balance, partition):
    """
    Generate federated splits from Flowers102 and persist artifacts.

    Args:
        dir_path: Output directory (project root for generated artifacts).
        num_clients: Number of federated clients to partition into (>=1).
        niid: If True, produce a Non-IID partitioning; otherwise IID.
        balance: If True, balanced sample counts across clients (where applicable).
        partition: Non-IID strategy hint, e.g., "pat" (pathological) or "dir"
                   (Dirichlet). Use None or "-" to disable.

    Side Effects:
        - Creates folders/files under `dir_path` (see module docstring).
        - Saves a distribution bar plot per client.

    Raises:
        ValueError: If arguments are inconsistent (e.g., negative clients).
        RuntimeError: If persistence fails (I/O errors) or utils behave unexpectedly.

    Design Notes:
        - We merge train/val/test to simulate a single unlabeled pool before
          federated splitting. This is a choice to maximize sample variety per
          client; if you want to keep official test untouched, split before merge.
        - Images are normalized to mean=std=0.5 per channel => [-1, 1] range.
        - `class_per_client=2` is a strong inductive bias for label-skew; tune it
          per experiment design and record in config.json for traceability.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    config_path = os.path.join(dir_path, "config.json")
    train_path  = os.path.join(dir_path, "train/")
    test_path   = os.path.join(dir_path, "test/")

    # Fast-path: skip recomputation if config + outputs already match params
    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    root_raw = os.path.join(dir_path, "rawdata")
    
    # Load all official splits, then concatenate
    tr_imgs, tr_lbls = load_split("train", transform, root_raw)
    va_imgs, va_lbls = load_split("val",   transform, root_raw)
    te_imgs, te_lbls = load_split("test",  transform, root_raw)

    dataset_image = np.concatenate([tr_imgs, va_imgs, te_imgs], axis=0)
    dataset_label = np.concatenate([tr_lbls, va_lbls, te_lbls], axis=0)

    num_classes = len(set(dataset_label.tolist()))
    print(f'Number of classes: {num_classes}')

    # Partition across clients (utils encapsulates different schemes).
    X, y, statistic = separate_data(
        (dataset_image, dataset_label),
        num_clients, num_classes,
        niid, balance, partition,
        class_per_client=2
    )

    train_data, test_data = split_data(X, y)

    save_file(
        config_path, train_path, test_path,
        train_data, test_data,
        num_clients, num_classes, statistic,
        niid, balance, partition
    )

    # --- Visualization: per-client label histograms -------------------------
    rows = (num_clients + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(4 * 4, 3 * rows))
    axes = axes.flatten()
    width = 0.4

    for i in range(num_clients):
        y_train = train_data[i]['y']
        y_test  = test_data[i]['y']

        train_counts = [np.sum(y_train == c) for c in range(num_classes)]
        test_counts  = [np.sum(y_test  == c) for c in range(num_classes)]
        x = np.arange(num_classes)
        
        # Two bars per class: train vs test
        axes[i].bar(x - width/2, train_counts, width=width, label='Train', color='C0')
        axes[i].bar(x + width/2, test_counts,  width=width, label='Test',  color='C1')

        axes[i].set_title(f'Client {i}')
        axes[i].set_xlabel('Class')
        axes[i].set_ylabel('Samples')
        axes[i].legend(fontsize='small')
    
    # Delete empty subplots if num_clients is not multiple of 4.
    for j in range(num_clients, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    os.makedirs(os.path.join(dir_path, 'figures'), exist_ok=True)
    fig.savefig(os.path.join(dir_path, 'figures', 'client_data_distribution.png'))
    plt.close(fig)

if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_dataset(dir_path, num_clients, niid, balance, partition)
