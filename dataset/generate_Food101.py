"""
Food101 federated split generator.

Purpose:
    - Download/prepare Food-101 dataset (torchvision.datasets.Food101).
    - Merge official train/test into a single pool.
    - Partition samples across federated clients (IID / Non-IID variants).
    - Persist per-client train/test splits and a distribution plot.

CLI:
    python generate_food101.py  <iid|noniid>  <balance|->  <pat|dir|->

    Examples:
      # IID & balanced across 20 clients
      python generate_food101.py iid balance -

      # Non-IID pathological across 20 clients
      python generate_food101.py noniid - pat

      # Non-IID Dirichlet (NOTE: the alpha is configured in utils/separate_data)
      python generate_food101.py noniid - dir

Outputs (side effects):
    Food101/
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
dir_path = "Food101/"
IMG_SIZE = (64, 64)


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = os.path.join(dir_path, "config.json")
    train_path = os.path.join(dir_path, "train/")
    test_path = os.path.join(dir_path, "test/")

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    dataset_image = []
    dataset_label = []

    # Get Food-101 data (101 classes, 750 train + 250 test per class in the official split)
    # We merge train & test to form a single pool, then re-split per client via split_data().
    transform = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    def load_data(split="train"):
        # torchvision.datasets.Food101 supports split in {"train", "test"}
        ds = torchvision.datasets.Food101(
            root=os.path.join(dir_path, "rawdata"),
            split=split,
            download=True,
            transform=transform,
        )
        # Load entire split into one batch
        loader = torch.utils.data.DataLoader(
            ds, batch_size=len(ds), shuffle=False, num_workers=2
        )
        for _, batch in enumerate(loader, 0):
            # batch = (images, targets)
            images, targets = batch
            # Create attributes to keep symmetry with Flowers102 template
            ds.data, ds.targets = images, targets
        dataset_image.extend(ds.data.cpu().detach().numpy())
        dataset_label.extend(ds.targets.cpu().detach().numpy())

    load_data("train")
    load_data("test")

    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f"Number of classes: {num_classes}")

    # NOTE: class_per_client can be tuned. We keep 10 for parity with the Flowers102 generator.
    X, y, statistic = separate_data(
        (dataset_image, dataset_label),
        num_clients,
        num_classes,
        niid,
        balance,
        partition,
        class_per_client=10,
    )
    train_data, test_data = split_data(X, y)
    save_file(
        config_path,
        train_path,
        test_path,
        train_data,
        test_data,
        num_clients,
        num_classes,
        statistic,
        niid,
        balance,
        partition,
    )

    # --- Plot client data distribution ---
    rows = (num_clients + 3) // 4
    # Wider per-subplot because Food-101 has 101 classes
    fig_width = 5 * 4  # 5 inches each subplot column
    fig_height = 3.2 * rows
    fig, axes = plt.subplots(rows, 4, figsize=(fig_width, fig_height))
    axes = axes.flatten()
    width = 0.4

    for i in range(num_clients):
        y_train = train_data[i]["y"]
        y_test = test_data[i]["y"]

        train_counts = [np.sum(y_train == c) for c in range(num_classes)]
        test_counts = [np.sum(y_test == c) for c in range(num_classes)]
        x = np.arange(num_classes)

        axes[i].bar(x - width / 2, train_counts, width=width, label="Train")
        axes[i].bar(x + width / 2, test_counts, width=width, label="Test")

        axes[i].set_title(f"Client {i}")
        axes[i].set_xlabel("Class (0..100)")
        axes[i].set_ylabel("Samples")
        if i == 0:
            axes[i].legend(fontsize="small")

        # Make xticks sparser to avoid clutter
        if num_classes > 30:
            axes[i].set_xticks(np.arange(0, num_classes, 10))

    for j in range(num_clients, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    os.makedirs(os.path.join(dir_path, "figures"), exist_ok=True)
    fig.savefig(os.path.join(dir_path, "figures", "client_data_distribution.png"))
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python generate_food101.py <iid|noniid> <balance|-> <pat|dir|->")
        sys.exit(1)

    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_dataset(dir_path, num_clients, niid, balance, partition)
