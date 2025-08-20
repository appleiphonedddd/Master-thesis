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

def _load_split(split_name, transform, root_dir):
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
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    config_path = os.path.join(dir_path, "config.json")
    train_path  = os.path.join(dir_path, "train/")
    test_path   = os.path.join(dir_path, "test/")

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    root_raw = os.path.join(dir_path, "rawdata")

    tr_imgs, tr_lbls = _load_split("train", transform, root_raw)
    va_imgs, va_lbls = _load_split("val",   transform, root_raw)
    te_imgs, te_lbls = _load_split("test",  transform, root_raw)

    dataset_image = np.concatenate([tr_imgs, va_imgs, te_imgs], axis=0)
    dataset_label = np.concatenate([tr_lbls, va_lbls, te_lbls], axis=0)

    num_classes = len(set(dataset_label.tolist()))
    print(f'Number of classes: {num_classes}')

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
