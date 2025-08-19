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

"""
Generate a federated version of the Stanford Dogs Dataset.

Design pattern mirrors generate_TinyImagenet.py with one pragmatic tweak:
- We resize images to 64x64 to control VRAM/RAM usage because Stanford Dogs images
  are variable-sized and much larger than Tiny-ImageNet (64x64).

CLI (consistent with the template):
  python generate_StanfordDogs.py iid balance -
  python generate_StanfordDogs.py noniid - pat
  python generate_StanfordDogs.py noniid - dir
  python generate_StanfordDogs.py noniid - exdir   # extended-Dirichlet in dataset_utils

Outputs:
- StanfordDogs/config.json
- StanfordDogs/train/<client_id>.npz
- StanfordDogs/test/<client_id>.npz
- StanfordDogs/figures/client_data_distribution.png

Notes:
- We merge all images (official train+test) by reading from the "Images/" folder,
  then perform a client-wise split via dataset_utils (train_ratio applies there).
- Bounding boxes / annotations are not required for classification splits; only images are used.
"""

random.seed(1)
np.random.seed(1)
num_clients = 20
# Root output folder for this generator (parallel to TinyImagenet/)
dir_path = "StanfordDogs/"

# Custom ImageFolder wrapper following the same pattern as TinyImagenet template
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


def _ensure_rawdata(dir_path: str):
    """Download and extract Stanford Dogs (images.tar) into rawdata/ if missing.
    We only need images for classification; lists/annotations are optional.
    """
    raw_dir = os.path.join(dir_path, 'rawdata')
    images_dir = os.path.join(raw_dir, 'Images')
    if os.path.exists(images_dir) and len(os.listdir(images_dir)) > 0:
        print('rawdata already exists.\n')
        return

    os.makedirs(raw_dir, exist_ok=True)

    images_tar = os.path.join(raw_dir, 'images.tar')
    if not os.path.exists(images_tar):
        url = 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar'
        os.system(f'wget --no-verbose --directory-prefix {raw_dir} {url}')
    else:
        print('images.tar already downloaded.')

    # Extract
    os.system(f'tar -xf {images_tar} -C {raw_dir}')

    # (Optional) grab lists.tar for reference (not required by this generator)
    lists_tar = os.path.join(raw_dir, 'lists.tar')
    if not os.path.exists(lists_tar):
        url_lists = 'http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar'
        os.system(f'wget --no-verbose --directory-prefix {raw_dir} {url_lists}')
    else:
        print('lists.tar already downloaded.')


def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    # Download / prepare raw data if needed
    _ensure_rawdata(dir_path)

    # Transform: resize to 64x64 to make a full-tensor load feasible, then normalize like TinyImagenet
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = ImageFolder_custom(root=os.path.join(dir_path, 'rawdata', 'Images'), transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    # Load all data into memory once (similar to TinyImagenet generator)
    for _, batch in enumerate(dataloader, 0):
        dataset.data, dataset.targets = batch

    dataset_image = []
    dataset_label = []

    dataset_image.extend(dataset.data.cpu().detach().numpy())
    dataset_label.extend(dataset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    # Keep the same class_per_client setting (20) to align with the TinyImagenet template behavior
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,
                                    niid, balance, partition, class_per_client=20)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, niid, balance, partition)

    # Visualize the train & test data distribution of each client and save the figure
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
    # CLI exactly mirrors the TinyImagenet generator
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_dataset(dir_path, num_clients, niid, balance, partition)
