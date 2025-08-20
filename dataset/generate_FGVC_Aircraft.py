import numpy as np
import os
import sys
import random
import torch
import torchvision.transforms as transforms
from torchvision.datasets import FGVCAircraft
from utils.dataset_utils import check, separate_data, split_data, save_file
import matplotlib.pyplot as plt

random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path = "FGVC_Aircraft/"

def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = os.path.join(dir_path, "config.json")
    train_path = os.path.join(dir_path, "train/")
    test_path  = os.path.join(dir_path, "test/")

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    # Use torchvision to download official 2013b release under rawdata/
    raw_root = os.path.join(dir_path, "rawdata")
    os.makedirs(raw_root, exist_ok=True)

    # Keep pattern consistent; also ensure fixed size for batching all-at-once
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Merge official train+val+test, then we do our own split for FL
    trainval_set = FGVCAircraft(root=raw_root, split="trainval", annotation_level="variant",
                                download=True, transform=transform)
    test_set     = FGVCAircraft(root=raw_root, split="test", annotation_level="variant",
                                download=True, transform=transform)

    # Load all samples into memory (matches TinyImageNet pattern)
    tv_loader = torch.utils.data.DataLoader(trainval_set, batch_size=len(trainval_set), shuffle=False)
    te_loader = torch.utils.data.DataLoader(test_set,     batch_size=len(test_set),     shuffle=False)

    dataset_image = []
    dataset_label = []

    for imgs, labels in tv_loader:
        dataset_image.extend(imgs.cpu().numpy())
        dataset_label.extend(labels.cpu().numpy())

    for imgs, labels in te_loader:
        dataset_image.extend(imgs.cpu().numpy())
        dataset_label.extend(labels.cpu().numpy())

    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f"Number of classes: {num_classes}")

    # Partition to clients
    X, y, statistic = separate_data(
        (dataset_image, dataset_label),
        num_clients, num_classes,
        niid, balance, partition,
        class_per_client=20  # keep same knob as TinyImageNet script
    )

    # Client-wise train/test split
    train_data, test_data = split_data(X, y)

    # Persist
    save_file(config_path, train_path, test_path,
              train_data, test_data, num_clients, num_classes,
              statistic, niid, balance, partition)

    # Visualize distribution across clients (same style as TinyImageNet)
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
    # CLI pattern kept identical to generate_TinyImagenet.py
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_dataset(dir_path, num_clients, niid, balance, partition)
