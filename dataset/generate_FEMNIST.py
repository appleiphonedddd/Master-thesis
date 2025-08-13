import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import mnist as tv_mnist  # for EMNIST class monkeypatch
from utils.dataset_utils import check, separate_data, split_data, save_file
import matplotlib.pyplot as plt

random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path = "FEMNIST/"

# which now redirects to homepage and breaks MD5 check. New canonical mirror:
# https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip  (same MD5)
try:
    tv_mnist.EMNIST.url = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
    tv_mnist.EMNIST.md5 = "58c8d27c78d21e728a6bc7b3cc06412e"
except Exception:
    pass

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

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    raw_root = os.path.join(dir_path, "rawdata")

    trainset = torchvision.datasets.EMNIST(root=raw_root, split="byclass", train=True, download=True, transform=transform)
    testset  = torchvision.datasets.EMNIST(root=raw_root, split="byclass", train=False, download=True, transform=transform)

    train_imgs = trainset.data.numpy()    
    train_lbls = trainset.targets.numpy()
    test_imgs  = testset.data.numpy()
    test_lbls  = testset.targets.numpy()

    dataset_image = np.concatenate([train_imgs, test_imgs], axis=0)
    dataset_label = np.concatenate([train_lbls, test_lbls], axis=0)

    num_classes = len(set(dataset_label.tolist()))
    print(f"Number of classes: {num_classes}")

    class_per_client = min(20, num_classes)

    X, y, statistic = separate_data(
        (dataset_image, dataset_label),
        num_clients,
        num_classes,
        niid,
        balance,
        partition,
        class_per_client=class_per_client
    )

    train_data, test_data = split_data(X, y)

    save_file(
        config_path, train_path, test_path,
        train_data, test_data, num_clients, num_classes,
        statistic, niid, balance, partition
    )

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
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_dataset(dir_path, num_clients, niid, balance, partition)
