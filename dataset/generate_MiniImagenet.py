import numpy as np
import os
import sys
import random
import pickle
import torch
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file
import matplotlib.pyplot as plt

random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path = "MiniImagenet/"

def load_mini_imagenet(dir_path, split="train"):
    split_map = {
        "train": "miniImageNet_category_split_train_phase_train.pickle",
        "test":  "miniImageNet_category_split_test.pickle",
        "train_test": "miniImageNet_category_split_train_phase_test.pickle",
    }
    pkl = os.path.join(dir_path, "rawdata", split_map[split])
    with open(pkl, "rb") as f:
        obj = pickle.load(f, encoding="latin1")

    def _decode_keys(d):
        return { (k.decode("utf-8", "ignore") if isinstance(k, bytes) else k): v
                 for k, v in d.items() }
    if isinstance(obj, dict) and len(obj) and isinstance(next(iter(obj.keys())), (bytes,)):
        obj = _decode_keys(obj)

    imgs = obj.get("image_data", obj.get("data", obj.get("images", None)))
    if imgs is None:
        raise KeyError(f"{pkl} can not find image data key, found keys are: {list(obj.keys())[:10]}")

    if "class_dict" in obj:
        class_dict = obj["class_dict"]
        if isinstance(class_dict, dict) and len(class_dict) and isinstance(next(iter(class_dict.keys())), (bytes,)):
            class_dict = _decode_keys(class_dict)
        labels = np.empty(len(imgs), dtype=np.int64)
        for i, (_, idxs) in enumerate(sorted(class_dict.items())):
            labels[np.asarray(idxs, dtype=np.int64)] = i
    else:
        lbl = obj.get("labels", obj.get("targets", None))
        if lbl is None:
            raise KeyError(f"{pkl} can not find labels/targets/class_dict")
        labels = np.asarray(lbl, dtype=np.int64)
        _, labels = np.unique(labels, return_inverse=True)
        labels = labels.astype(np.int64)

    imgs = imgs.astype(np.float32)
    if imgs.max() > 1.5:
        imgs /= 255.0
    imgs = (imgs - 0.5) / 0.5

    if imgs.ndim == 4:
        if imgs.shape[-1] == 3:
            imgs = np.transpose(imgs, (0, 3, 1, 2))
        elif imgs.shape[1] == 3:
            pass
        else:
            raise ValueError(f"Unknown image shape: {imgs.shape}")

    return imgs, labels

def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    config_path = os.path.join(dir_path, "config.json")
    train_path  = os.path.join(dir_path, "train/")
    test_path   = os.path.join(dir_path, "test/")

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    rawdir = os.path.join(dir_path, "rawdata/")
    if not os.path.exists(rawdir):
        os.system(f'wget --directory-prefix {rawdir} https://cseweb.ucsd.edu/~weijian/static/datasets/mini-ImageNet/MiniImagenet.tar.gz')
        os.system(f'tar -xzvf {rawdir}/MiniImagenet.tar.gz -C {rawdir}/ --strip-components=1')
    else:
        print("rawdata already exists.\n")

    dataset_image, dataset_label = load_mini_imagenet(dir_path, split="train")
    num_classes = int(len(np.unique(dataset_label)))
    print(f"Number of classes: {num_classes}")

    class_per_client = min(20, num_classes)

    X, y, statistic = separate_data(
        (dataset_image, dataset_label),
        num_clients, num_classes,
        niid, balance, partition,
        class_per_client=class_per_client
    )
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path,
              train_data, test_data,
              num_clients, num_classes,
              statistic, niid, balance, partition)

    rows = (num_clients + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(16, 3 * rows))
    axes = axes.flatten()
    width = 0.4

    for i in range(num_clients):
        y_train = train_data[i]["y"]
        y_test  = test_data[i]["y"]
        train_counts = [np.sum(y_train == c) for c in range(num_classes)]
        test_counts  = [np.sum(y_test  == c) for c in range(num_classes)]
        x = np.arange(num_classes)
        axes[i].bar(x - width/2, train_counts, width=width, label="Train")
        axes[i].bar(x + width/2, test_counts,  width=width, label="Test")
        axes[i].set_title(f"Client {i}")
        axes[i].set_xlabel("Class")
        axes[i].set_ylabel("Samples")
        axes[i].legend(fontsize="small")

    for j in range(num_clients, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    os.makedirs(os.path.join(dir_path, "figures"), exist_ok=True)
    fig.savefig(os.path.join(dir_path, "figures", "client_data_distribution.png"))
    plt.close(fig)

if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None
    generate_dataset(dir_path, num_clients, niid, balance, partition)
