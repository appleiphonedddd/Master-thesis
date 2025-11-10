import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy(methods, datasets, accuracy):
    accuracy = np.array(accuracy)
    x = np.arange(len(datasets)) * 1.1
    width = 0.16
    fig, ax = plt.subplots(figsize=(15, 6))

    colors = ['#FF7F0E', '#2CA02C', '#1F77B4', '#D62728', '#9467BD',]

    for i, method in enumerate(methods):
        if method == "FedFIP":
            bars = ax.bar(x + i*width, accuracy[i], width,
                          label=method, color=colors[i],
                          edgecolor='black', linewidth=1.5, alpha=1.0)
        else:
            bars = ax.bar(x + i*width, accuracy[i], width,
                          label=method, color=colors[i],
                          alpha=0.6)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Comparison Across Methods and Datasets', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(methods)-1) / 2)
    ax.set_xticklabels(datasets, rotation=15, ha='right', fontsize=11, fontweight='bold')
    ax.tick_params(axis='y', labelsize=11)
    ax.legend(fontsize=11)

    ax.grid(False)

    plt.tight_layout()
    plt.savefig('Bar_chart.png', dpi=300)
    print("[Info] Saved accuracy comparison plot to Bar_chart.png")

methods = ['FedAvg', 'FedDodm', 'FedAS', 'DCPFL', 'FedFIP']
datasets = ['MNIST','FMNIST', 'CIFAR10', 'CIFAR100', 'Tiny-ImageNet']
accuracy = [
    [98.18, 85.80, 56.68, 32.48, 19.66],  # FedAvg
    [99.65, 97.57, 89.90, 56.52, 37.66],  # FedDodm
    [99.68, 97.74, 91.55, 59.78, 40.09],  # FedAS
    [99.25, 97.36, 89.55, 46.88, 27.00],  # DCPFL
    [99.68, 97.82, 91.56, 61.84, 42.84]   # FedFIP
]

plot_accuracy(methods, datasets, accuracy)
