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
dir_path = "OxfordIIITPet/"

def generate_dataset(dir_path, num_clients, niid, balance, partition):
    os.makedirs(dir_path, exist_ok=True)
    config_path = os.path.join(dir_path, "config.json")
    train_path  = os.path.join(dir_path, "train/")
    test_path   = os.path.join(dir_path, "test/")

    # Skip if already generated
    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    # Define transforms: Resize → ToTensor → Normalize
    transform = transforms.Compose([
        transforms.Resize((128, 128)),      # Ensure fixed size for batching :contentReference[oaicite:10]{index=10}
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    # Load OxfordIIITPet datasets
    trainset = torchvision.datasets.OxfordIIITPet(
        root=os.path.join(dir_path, 'rawdata/'),
        split='trainval',
        target_types='category',
        transform=transform,
        download=True
    )
    testset = torchvision.datasets.OxfordIIITPet(
        root=os.path.join(dir_path, 'rawdata/'),
        split='test',
        target_types='category',
        transform=transform,
        download=True
    )

    # Aggregate all samples and labels into arrays
    images, labels = [], []
    for dataset in (trainset, testset):
        for img, lbl in dataset:
            images.append(img.numpy())
            labels.append(int(lbl))

    dataset_image = np.array(images)
    dataset_label = np.array(labels)
    num_classes = len(np.unique(dataset_label))
    print(f'Number of classes: {num_classes}')

    # Partition data across clients
    X_splits, y_splits, statistic = separate_data(
        (dataset_image, dataset_label),
        num_clients,
        num_classes,
        niid,
        balance,
        partition,
        class_per_client=20
    )
    train_data, test_data = split_data(X_splits, y_splits)

    # Save configuration and data files
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
        partition
    )

    # Visualize the train & test data distribution of each client and save the figure
    rows = (num_clients + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(4*4, 3*rows))
    axes = axes.flatten()
    width = 0.4

    for i in range(num_clients):
        y_tr = train_data[i]['y']
        y_te = test_data[i]['y']
        x = np.arange(num_classes)
        axes[i].bar(x - width/2,
                    [np.sum(y_tr == c) for c in x],
                    width=width, label='Train')
        axes[i].bar(x + width/2,
                    [np.sum(y_te == c) for c in x],
                    width=width, label='Test')
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
