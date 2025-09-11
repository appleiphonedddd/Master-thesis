"""
COIL-100 federated split generator.

Purpose
-------
- Download / prepare COIL-100 (if absent).
- Reshape raw flat files into an ImageFolder-compatible layout.
- Load all images and partition samples across federated clients (IID / Non-IID variants).
- Persist per-client train/test splits and a per-client distribution figure.

CLI (kept consistent with the Tiny-ImageNet generator template)
--------------------------------------------------------------
python generate_coil100.py <iid|noniid> <balance|-> <pat|dir|->

Examples:
  # IID & balanced across 20 clients
  python generate_coil100.py iid balance -

  # Non-IID pathological across 20 clients
  python generate_coil100.py noniid - pat

  # Non-IID Dirichlet
  python generate_coil100.py noniid - dir

Outputs (side effects)
----------------------
COIL100/
  ├── config.json
  ├── train/                        # per-client train tensors/labels
  ├── test/                         # per-client test tensors/labels
  └── figures/client_data_distribution.png

Notes
-----
- COIL-100 images are 128×128 color; original distribution uses PPM, many mirrors convert to PNG.
  We robustly handle .ppm/.png/.jpg by reorganizing files into an ImageFolder layout.
- Normalization: to [-1, 1] via mean/std=(0.5, 0.5, 0.5) for parity with other generators.
- We load the entire dataset in a single DataLoader batch (simple, fast). If you hit OOM,
  switch to chunked loading and concatenate.
- Design choice: class_per_client=10 to create moderate label-skew across 100 classes.
  Recorded in config.json for reproducibility.

Security/Compliance
-------------------
- Uses shell `wget`/`unzip` for brevity. Prefer Python stdlib (`urllib.request`, `zipfile`) with
  checksum verification (e.g., SHA256) in production.
"""

import os
import re
import sys
import shutil
import random
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder, DatasetFolder
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file

random.seed(1)
np.random.seed(1)

dir_path = "COIL100/"
num_clients = 20

# Source from Columbia CAVE; mirrors (e.g., TFDS/Kaggle) may differ in container/format.
COIL100_ZIP_URL = (
    "https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip"
)

# --- ImageFolder wrapper (kept from the Tiny-ImageNet template) ---------------------------
class ImageFolder_custom(DatasetFolder):
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
        target = int(self.samples[index][1])
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

# --- Helpers ------------------------------------------------------------------------------

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

# Reorganize raw COIL files (flat) into an ImageFolder-compatible tree.
# Expected filenames: obj<id>__<angle>.{ppm|png|jpg|jpeg}
# We place them into: <root>/coil-100-imagefolder/train/obj<id>/...

def build_imagefolder_tree(raw_root: str) -> str:
    # locate the directory that actually contains the images
    # after unzip we typically get raw_root/coil-100/*.png or *.ppm
    candidate_dirs = []
    for name in ["coil-100", "COIL-100", "coil100", "COIL100", "."]:
        p = os.path.join(raw_root, name)
        if os.path.isdir(p):
            candidate_dirs.append(p)
    # pick the deepest dir that has at least one matching file
    src_dir = None
    for d in candidate_dirs:
        has = any(
            f.lower().endswith((".ppm", ".png", ".jpg", ".jpeg"))
            for f in os.listdir(d)
            if os.path.isfile(os.path.join(d, f))
        )
        if has:
            src_dir = d
            break
    if src_dir is None:
        # maybe files are directly under raw_root
        src_dir = raw_root

    out_root = _ensure_dir(os.path.join(raw_root, "coil-100-imagefolder", "train"))

    pat = re.compile(r"^obj(\d+)__\d+\.(ppm|png|jpg|jpeg)$", re.IGNORECASE)
    count = 0
    for fname in os.listdir(src_dir):
        fpath = os.path.join(src_dir, fname)
        if not os.path.isfile(fpath):
            continue
        if not fname.lower().endswith((".ppm", ".png", ".jpg", ".jpeg")):
            continue
        m = pat.match(fname)
        if not m:
            # skip non-standard names silently
            continue
        obj_id = m.group(1)
        cls_dir = _ensure_dir(os.path.join(out_root, f"obj{obj_id}"))
        # copy rather than symlink for portability
        shutil.copy2(fpath, os.path.join(cls_dir, fname))
        count += 1

    if count == 0:
        raise RuntimeError(
            f"No COIL-100 images found in '{raw_root}'. Check that the archive extracted correctly."
        )
    return out_root

# --- Main generation ----------------------------------------------------------------------

def generate_dataset(dir_path: str, num_clients: int, niid: bool, balance: bool, partition: str):
    _ensure_dir(dir_path)

    # Standard output structure
    config_path = os.path.join(dir_path, "config.json")
    train_path = os.path.join(dir_path, "train/")
    test_path  = os.path.join(dir_path, "test/")

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    # Download raw data if missing
    raw_root = os.path.join(dir_path, "rawdata")
    if not os.path.exists(raw_root):
        _ensure_dir(raw_root)
        os.system(f"wget --no-verbose --directory-prefix {raw_root} {COIL100_ZIP_URL}")
        os.system(f"unzip -q {os.path.join(raw_root, 'coil-100.zip')} -d {raw_root}")
    else:
        print("rawdata already exists.\n")

    # Reorganize to ImageFolder format
    imgfolder_train_dir = build_imagefolder_tree(raw_root)

    # Transforms: force RGB -> tensor -> normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.Lambda(lambda im: im.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load all samples in one shot
    trainset = ImageFolder_custom(root=imgfolder_train_dir, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)

    for _, batch in enumerate(trainloader, 0):
        trainset.data, trainset.targets = batch

    dataset_image = np.array(trainset.data.cpu().detach().numpy())
    dataset_label = np.array(trainset.targets.cpu().detach().numpy())

    num_classes = len(set(dataset_label.tolist()))
    print(f"Number of classes: {num_classes}")

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

    # Visualize per-client distributions
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
        axes[i].bar(x - width/2, train_counts, width=width, label='Train')
        axes[i].bar(x + width/2, test_counts,  width=width, label='Test')
        axes[i].set_title(f'Client {i}')
        axes[i].set_xlabel('Class')
        axes[i].set_ylabel('Samples')
        axes[i].legend(fontsize='small')

    for j in range(num_clients, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    _ensure_dir(os.path.join(dir_path, 'figures'))
    fig.savefig(os.path.join(dir_path, 'figures', 'client_data_distribution.png'))
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python generate_coil100.py <iid|noniid> <balance|-> <pat|dir|->")
        sys.exit(1)
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None
    generate_dataset(dir_path, num_clients, niid, balance, partition)
