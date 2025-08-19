import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy(methods, datasets, accuracy):
    accuracy = np.array(accuracy)
    x = np.arange(len(datasets))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, method in enumerate(methods):
        bars = ax.bar(x + i*width, accuracy[i], width, label=method)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Comparison Across Methods and Datasets')
    ax.set_xticks(x + width * (len(methods)-1) / 2)
    ax.set_xticklabels(datasets, rotation=15, ha='right')
    ax.legend()

    ax.grid(False)

    plt.tight_layout()
    plt.savefig('Bar_chart.png')
    print("[Info] Saved accuracy comparison plot without grid to Bar_chart.png")

methods = ['FedAvg', 'FedDodm', 'FedAS', 'DCPFL']
datasets = ['MNIST', 'FMNIST', 'CIFAR10', 'CIFAR100', 'Tiny-ImageNet']
accuracy = [
    [98.18, 85.80, 56.68, 32.69, 19.50],  # FedAvg
    [99.65, 97.57, 89.90, 49.97, 35.50],  # FedDodm
    [99.68, 97.74, 91.55, 59.62, 40.06],  # FedAS
    [99.25, 97.36, 89.55, 46.82, 26.92]   # DCPFL
]

plot_accuracy(methods, datasets, accuracy)
