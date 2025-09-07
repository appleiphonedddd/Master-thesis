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


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    dataset_image = []
    dataset_label = []
        
    # Get Flowers102 data
    transform = transforms.Compose(
        [transforms.Resize((64, 64)), 
        transforms.ToTensor(), 
        transforms.Normalize((0.5), (0.5))]
    )

    def load_data(split="train"):
        trainset = torchvision.datasets.Flowers102(
            root=dir_path+"rawdata", split=split, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=len(trainset), shuffle=False)
        for _, train_data in enumerate(trainloader, 0):
            trainset.data, trainset.targets = train_data
        dataset_image.extend(trainset.data.cpu().detach().numpy())
        dataset_label.extend(trainset.targets.cpu().detach().numpy())

    load_data("train")
    load_data("val")
    load_data("test")

    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition, class_per_client=10)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)
    
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

        axes[i].bar(x - width/2, train_counts, width=width, label='Train', color='C0')
        axes[i].bar(x + width/2, test_counts,  width=width, label='Test',  color='C1')

        axes[i].set_title(f'Client {i}')
        axes[i].set_xlabel('Class')
        axes[i].set_ylabel('Samples')
        axes[i].legend(fontsize='small')

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
