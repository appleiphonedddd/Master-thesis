"""
Office-Home federated split generator.

Purpose
-------
- (Optionally) Download/prepare Office-Home (if absent).
- Load all images (domain-aware) via an ImageFolder-compatible wrapper.
- Partition samples across federated clients (IID / Non-IID variants).
- Persist per-client train/test splits and a per-client distribution figure.

CLI (kept consistent with other generators)
-------------------------------------------
python generate_officehome.py <iid|noniid> <balance|-> <pat|dir|-> [domain|all] [img_size]

Examples:
  # IID & balanced across 20 clients, all 4 domains, images resized to 128
  python generate_officehome.py iid balance - all 128

  # Non-IID Dirichlet across 20 clients on the 'Art' domain only
  python generate_officehome.py noniid - dir art 128

Outputs (side effects)
----------------------
OfficeHome/
  ├── config.json
  ├── train/                        # per-client train tensors/labels
  ├── test/                         # per-client test tensors/labels
  └── figures/client_data_distribution.png

Notes
-----
- Office-Home has 4 domains: Art, Clipart, Product, Real World. Each domain
  contains the same 65 categories. We build a global label space (65 classes)
  and (by default) aggregate across domains to maximize domain shift.
- Images are resized to a square (default 128×128) then normalized to [-1, 1]
  via mean/std=(0.5, 0.5, 0.5) for parity with other generators.
- Unlike Tiny-ImageNet, loading every image at once may OOM. We therefore
  iterate the DataLoader in chunks and concatenate on CPU. Reduce img_size
  or increase chunk size conservatively if you face memory pressure.
- Design choice: class_per_client=10 to create moderate label-skew. Record
  this in config.json for reproducibility.

Security/Compliance
-------------------
- If dataset is missing, we *attempt* to fetch it using `gdown` (Google Drive).
  This convenience path requires an internet connection and `gdown` to succeed.
  For production, prefer mirroring with checksum verification (e.g., SHA256).
"""

import numpy as np
import os
import sys
import random
import json
import glob
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, DatasetFolder
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Project utilities (assumed available in the same repo as other generators)
from utils.dataset_utils import check, separate_data, split_data, save_file

random.seed(1)
np.random.seed(1)

# Defaults (align with other generators)
num_clients = 20
dir_path = "OfficeHome/"

# ---- Domain helpers ----

CANONICAL_DOMAINS = {
    "art": ["Art"],
    "clipart": ["Clipart", "Clip Art"],
    "product": ["Product"],
    "real": ["Real World", "Real_World", "Real-World", "RealWorld"],
}

def find_dataset_root(raw_root: str):
    """
    Returns the path that contains the domain subfolders (Art/Clipart/Product/Real*).
    Search common unzip patterns and allow manual env override.
    """
    # direct check
    candidates = [raw_root]

    # include common folder names (case variants)
    patterns = [
        "OfficeHome*", "officehome*", "Office_Home*", "office_home*",
        "OfficeHomeDataset*", "officehomedataset*"
    ]
    for pat in patterns:
        for p in glob.glob(os.path.join(raw_root, pat)):
            if os.path.isdir(p):
                candidates.append(p)

    # Also check one level deeper (some users unzip into another subdir)
    for base in list(candidates):
        for p in glob.glob(os.path.join(base, "*")):
            if os.path.isdir(p):
                candidates.append(p)

    # Pick the one that contains at least 2 domain dirs
    def has_domains(cand: str) -> bool:
        found = 0
        for variants in CANONICAL_DOMAINS.values():
            if any(os.path.isdir(os.path.join(cand, v)) for v in variants):
                found += 1
        return found >= 2

    # sort unique
    seen = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        if has_domains(cand):
            return cand
    return None


def _resolve_domain_dirs(dataset_root: str, domain_choice: str):
    """
    Given a dataset_root and a domain_choice in {'art','clipart','product','real','all'},
    return a list of actual existing domain directories.
    """
    domain_choice = (domain_choice or "all").lower()
    chosen = []
    if domain_choice == "all":
        keys = list(CANONICAL_DOMAINS.keys())
    else:
        if domain_choice not in CANONICAL_DOMAINS:
            raise ValueError(f"Unknown domain '{domain_choice}'. Use one of ['art','clipart','product','real','all'].")
        keys = [domain_choice]
    for k in keys:
        variants = CANONICAL_DOMAINS[k]
        for v in variants:
            p = os.path.join(dataset_root, v)
            if os.path.isdir(p):
                chosen.append(p)
                break  # prefer the first variant that exists
    if not chosen:
        raise FileNotFoundError(f"No domains found in {dataset_root} for choice '{domain_choice}'.")
    return chosen


# ---- Dataset wrappers ----

class ImageFolderWithGlobalMap(Dataset):
    """
    Wrap multiple domain roots with a shared global class_to_idx mapping.
    Returns (image_tensor, global_label). Domain id is not returned by default,
    since downstream partitioning is label-driven in dataset_utils.
    """
    def __init__(self, domain_roots, transform, global_class_to_idx):
        self.samples = []  # list of (path, global_label)
        self.transform = transform
        self.global_class_to_idx = global_class_to_idx

        for droot in domain_roots:
            # local mapping per domain
            local = ImageFolder(droot)  # only to read samples/targets/classes
            local_classes = local.classes
            # remap each local (path, local_idx) to global label via class name
            for path, local_idx in local.samples:
                cname = local_classes[local_idx]
                glabel = self.global_class_to_idx[cname]
                self.samples.append((path, glabel))

        self.loader = local.loader  # PIL loader from the last ImageFolder (same across)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, int(target)

    def __len__(self):
        return len(self.samples)


# ---- Download convenience ----

def try_download_officehome(raw_root: str):
    """
    Best-effort Google Drive download using gdown (fuzzy URL parsing).
    If a zip already exists under raw_root, try unzipping it.
    """
    os.makedirs(raw_root, exist_ok=True)

    # If user already placed a zip under raw_root, try that first.
    zips = sorted(glob.glob(os.path.join(raw_root, "*.zip")))
    for z in zips:
        try:
            print(f"[info] Found existing zip: {z}. Unzipping...")
            os.system(f'unzip -q "{z}" -d "{raw_root}"')
            return True
        except Exception as e:
            print("[warn] Unzip failed for", z, ":", e)

    # Provided by user (full share URL with resourcekey)
    urls = [
        "https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?usp=sharing&resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw",
        "https://drive.google.com/uc?id=0B81rNlvomiwed0V1YUxQdC1uOTg"
    ]
    out_zip = os.path.join(raw_root, "OfficeHomeDataset.zip")
    try:
        # Install/upgrade gdown, then try each URL with --fuzzy
        os.system('python -m pip -q install --upgrade gdown')
        for u in urls:
            print("[info] Trying gdown on:", u)
            exit_code = os.system(f'python -m gdown --fuzzy "{u}" -O "{out_zip}"')
            if exit_code == 0 and os.path.exists(out_zip) and os.path.getsize(out_zip) > 0:
                os.system(f'unzip -q "{out_zip}" -d "{raw_root}"')
                return True
        print("[warn] gdown could not download the file with given URLs.")
        return False
    except Exception as e:
        print("[warn] Download attempt failed:", e)
        return False
        # Unzip
        os.system(f'unzip -q "{out_zip}" -d "{raw_root}"')
        return True
    except Exception as e:
        print("[warn] Download attempt failed:", e)
        return False


# ---- Core generation ----

def build_global_class_map(domain_dirs):
    """Scan all domain dirs to construct a global class_to_idx mapping (sorted by name)."""
    class_names = set()
    for d in domain_dirs:
        # children of domain dir are class folders
        for cname in os.listdir(d):
            if os.path.isdir(os.path.join(d, cname)):
                class_names.add(cname)
    class_names = sorted(class_names)
    return {c: i for i, c in enumerate(class_names)}


def _load_all_images_as_numpy(dataset: Dataset, batch_size: int = 512, num_workers: int = 4):
    """
    Stream through the dataset and concatenate on CPU to avoid peak OOM.
    Returns (N,H,W,C)-like numpy array in CHW->HWC converted shape? We will keep
    the tensor in CHW normalized space (float32) for parity with other generators.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    xs = []
    ys = []
    for xb, yb in loader:
        xs.append(xb)  # (B, C, H, W)
        ys.append(yb)  # (B,)
    X = torch.cat(xs, dim=0).cpu().numpy()
    y = torch.cat(ys, dim=0).cpu().numpy()
    return X, y


def generate_dataset(dir_path, num_clients, niid, balance, partition, domain_choice="all", img_size=128):
    os.makedirs(dir_path, exist_ok=True)

    # Setup directory for train/test data
    config_path = os.path.join(dir_path, "config.json")
    train_path  = os.path.join(dir_path, "train/")
    test_path   = os.path.join(dir_path, "test/")

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    # Prepare raw data
    raw_root = os.path.join(dir_path, "rawdata/")
    os.makedirs(raw_root, exist_ok=True)

    env_root = os.environ.get("OFFICEHOME_DATASET_DIR")
    dataset_root = find_dataset_root(env_root) if env_root else None
    if dataset_root is None:
        dataset_root = find_dataset_root(raw_root)
    if dataset_root is None:
        print("[info] Office-Home not found under", raw_root)
        print("[info] Attempting to download via Google Drive (requires internet + gdown)...")
        try_download_officehome(raw_root)
        env_root = os.environ.get("OFFICEHOME_DATASET_DIR")
    dataset_root = find_dataset_root(env_root) if env_root else None
    if dataset_root is None:
        dataset_root = find_dataset_root(raw_root)
        if dataset_root is None:
            raise FileNotFoundError(
                "Office-Home dataset not found.\n"
                "Please manually download from the official link and unzip so that one of the following folders exists:\n"
                f"  - {raw_root}/OfficeHomeDataset_10072016\n"
                f"  - {raw_root}/OfficeHomeDataset\n"
                f"  - {raw_root} (containing 'Art', 'Clipart', 'Product', 'Real World')\n"
            )

    # Resolve domain directories
    domain_dirs = _resolve_domain_dirs(dataset_root, domain_choice)

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Global label mapping (65 classes)
    global_class_to_idx = build_global_class_map(domain_dirs)
    num_classes = len(global_class_to_idx)
    print(f'Number of classes: {num_classes} (expected ~65)  |  Domains used: {len(domain_dirs)}')

    # Build dataset and load into memory in chunks
    ds = ImageFolderWithGlobalMap(domain_dirs, transform, global_class_to_idx)
    dataset_image, dataset_label = _load_all_images_as_numpy(ds, batch_size=512, num_workers=4)

    # Partition
    X, y, statistic = separate_data(
        (dataset_image, dataset_label),
        num_clients, num_classes,
        niid, balance, partition,
        class_per_client=10
    )

    # Split & save
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path,
              train_data, test_data, num_clients, num_classes,
              statistic, niid, balance, partition)

    # Visualize per-client label distribution
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
    os.makedirs(os.path.join(dir_path, 'figures'), exist_ok=True)
    fig.savefig(os.path.join(dir_path, 'figures', 'client_data_distribution.png'))
    plt.close(fig)

if __name__ == "__main__":
    # Parse CLI (compatible with other generators)
    # argv: 1: iid|noniid, 2: balance|-, 3: pat|dir|-, 4: domain|all, 5: img_size
    if len(sys.argv) < 4:
        print("Usage: python generate_officehome.py <iid|noniid> <balance|-> <pat|dir|-> [domain|all] [img_size]")
        sys.exit(1)

    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None
    domain_choice = sys.argv[4] if len(sys.argv) >= 5 else "all"
    try:
        img_size = int(sys.argv[5]) if len(sys.argv) >= 6 else 64
    except ValueError:
        img_size = 64

    generate_dataset(dir_path, num_clients, niid, balance, partition, domain_choice, img_size)
