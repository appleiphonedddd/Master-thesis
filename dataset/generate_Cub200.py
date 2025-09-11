"""
CUB-200-2011 federated split generator.

Purpose
-------
- Download/prepare the CUB-200-2011 dataset (if not present).
- Load all images via a folder-structured dataset.
- Partition samples across federated clients (IID / Non-IID variants).
- Persist per-client train/test splits and a per-client distribution figure.

CLI (kept consistent with other generators)
-------------------------------------------
python generate_cub200.py <iid|noniid> <balance|-> <pat|dir|->

Examples:
  # IID & balanced across 20 clients
  python generate_cub200.py iid balance -

  # Non-IID pathological across 20 clients
  python generate_cub200.py noniid - pat

  # Non-IID Dirichlet
  python generate_cub200.py noniid - dir

Outputs (side effects)
----------------------
CUB_200_2011/
  ├── config.json
  ├── train/                   # per-client train tensors/labels
  ├── test/                    # per-client test tensors/labels
  └── figures/client_data_distribution.png

Notes
-----
- CUB images are variable-sized; we resize to 64x64 and normalize to [-1, 1]
  using mean/std = (0.5, 0.5, 0.5). This intentionally matches TinyImageNet
  memory footprint across generators for consistent benchmarking.
- We load the entire dataset in a single DataLoader batch for simplicity.
  If you hit OOM, refactor to iterative loading and concatenate.
- Design choice: class_per_client=20 to mirror TinyImageNet generator default
  (label-skew strength). Tune it and record in config.json for reproducibility.

Security/Compliance
-------------------
- Dataset is downloaded using shell `wget`/`tar` for brevity. Prefer Python
  stdlib (`urllib.request` + `tarfile`) and checksum verification in production.
"""

import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file
from torchvision.datasets import ImageFolder, DatasetFolder
import matplotlib.pyplot as plt

random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path = "CUB_200_2011/"

# https://github.com/QinbinLi/MOON/blob/6c7a4ed1b1a8c0724fa2976292a667a828e3ff5d/datasets.py#L148
class ImageFolder_custom(DatasetFolder):
    """
    A thin wrapper to optionally subset `ImageFolder` by indices.

    Args:
        root: Root directory containing class-subfolders (ImageNet-style).
        dataidxs: Optional numpy/int list of indices to subset the dataset.
        train: Kept for signature symmetry; not used in this implementation.
        transform: Transform pipeline applied to the loaded PIL image.
        target_transform: Optional transform for target label.

    Notes:
        - We delegate to an inner `ImageFolder` to reuse its loader/targets.
        - We expose `.samples` aligned with torchvision conventions.
        - We do NOT predefine `.data`/`.targets` attributes; those are filled
          later after we batch-load all samples once (consistent with other gens).
    """

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)

def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    # Get data
    if not os.path.exists(f"{dir_path}/rawdata/"):
        os.makedirs(f"{dir_path}/rawdata/", exist_ok=True)
        os.system(f"wget --directory-prefix {dir_path}/rawdata/ https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz")
        os.system(f"tar -xzf {dir_path}/rawdata/CUB_200_2011.tgz -C {dir_path}/rawdata/")
    else:
        print('rawdata already exists.\n')

    # Important: CUB images are variable-sized. Resize to 64x64
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Use the folder-structured images directory
    dataset_root = dir_path + 'rawdata/CUB_200_2011/images/'
    trainset = ImageFolder_custom(root=dataset_root, transform=transform)

    # Load all samples in one batch (follow the same design pattern)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset), shuffle=False
    )

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,
                                    niid, balance, partition, class_per_client=20)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, niid, balance, partition)

    rows = (num_clients + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(4 * 4, 3 * rows))
    axes = axes.flatten()
    width = 0.4

    for i in range(num_clients):
        y_train = train_data[i]['y']
        y_test = test_data[i]['y']

        train_counts = [np.sum(y_train == c) for c in range(num_classes)]
        test_counts = [np.sum(y_test == c) for c in range(num_classes)]
        x = np.arange(num_classes)

        axes[i].bar(x - width/2, train_counts, width=width, label='Train', color='C0')
        axes[i].bar(x + width/2, test_counts, width=width, label='Test', color='C1')

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
