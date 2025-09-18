import matplotlib.pyplot as plt
import numpy as np

clients = [f"Client{i}" for i in range(1, 21)]
x = np.arange(len(clients))
BAR_W = 0.22

def plot_dataset(
    title, fedavg, fedas, feddodm, dcpfl, 
    ylim=None, fname=None
):
    fig, ax = plt.subplots(figsize=(16, 8), constrained_layout=True)

    ax.bar(x - 1.5*BAR_W, fedavg,   width=BAR_W, label="FedAvg",  color="orange")
    ax.bar(x - 0.5*BAR_W, fedas,    width=BAR_W, label="FedAS",   color="blue")
    ax.bar(x + 0.5*BAR_W, feddodm,  width=BAR_W, label="FedDodm", color="green")
    ax.bar(x + 1.5*BAR_W, dcpfl,    width=BAR_W, label="DCPFL",   color="red")

    ax.set_xticks(x)
    ax.set_xticklabels(clients, rotation=35, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title(title, pad=12)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)

    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=4,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.6
    )

    if fname:
        fig.savefig(fname, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

# ---------- MNIST ----------
FedAS = [98.48,100.00,99.75,100.00,99.17,99.10,99.36,99.37,100.00,99.60,99.61,99.68,99.78,99.63,100.00,99.56,100.00,99.29,99.32,99.87]
FedAvg = [98.02,97.60,98.04,98.58,97.08,92.83,98.41,98.43,98.89,98.90,98.97,98.83,98.34,97.38,98.46,99.05,99.56,98.02,98.10,98.89]
FedDodm = [98.78,100.00,99.75,99.84,99.38,99.68,99.04,99.06,100.00,99.60,99.74,99.47,99.78,99.06,100.00,99.41,99.89,99.08,99.70,99.83]
DCPFL = [99.63,96.80,99.75,99.53,97.50,98.90,96.18,99.37,100.00,99.20,99.46,99.47,99.56,98.50,99.86,99.46,100.00,98.30,99.24,99.80]
plot_dataset("Client Accuracy on MNIST", FedAvg, FedAS, FedDodm, DCPFL, fname="MNIST.png")

# ---------- FMNIST ----------
FedAS_FMNIST   = [85.71,98.75,96.35,99.39,98.08,100.00,99.82,91.04,93.60,98.54,99.22,98.47,97.47,91.72,98.80,91.84,94.97,99.92,98.27,98.47]
FedAvg_FMNIST  = [90.48,87.50,93.61,86.94,91.76,90.28,94.53,73.66,57.84,93.03,94.49,60.73,90.55,74.52,96.39,65.31,88.71,97.76,92.54,93.11]
FedDodm_FMNIST = [85.71,96.25,98.17,98.92,97.53,100.00,99.73,90.50,93.38,94.48,99.48,98.53,97.52,89.17,98.49,87.76,94.69,99.92,98.35,98.64]
DCPFL_FMNIST   = [85.71,93.75,96.80,99.08,96.43,100.00,99.45,90.50,92.37,97.86,99.03,97.35,97.10,85.35,90.80,81.63,91.56,99.50,97.80,97.70]
plot_dataset("Client Accuracy on FMNIST", FedAvg_FMNIST, FedAS_FMNIST, FedDodm_FMNIST, DCPFL_FMNIST, fname="FMNIST.png")

# ---------- CIFAR10 ----------
FedAS_C10   = [60.10,92.29,89.11,88.08,99.14,88.37,75.28,89.48,93.68,88.13,85.22,81.67,91.79,78.27,96.00,86.23,91.06,100.00,96.92,99.85]
FedAvg_C10  = [62.07,59.69,74.26,64.04,30.27,70.73,50.28,64.02,48.92,63.29,66.32,70.47,55.10,38.22,45.68,62.29,75.35,77.20,39.77,32.18]
FedDodm_C10 = [57.88,91.85,87.13,85.96,99.38,93.34,70.28,88.11,92.74,86.40,81.10,73.52,89.79,74.08,96.00,84.10,90.69,100.00,96.70,99.85]
DCPFL_C10   = [58.13,34.58,86.47,84.04,99.45,92.03,65.28,87.04,91.26,87.27,79.04,69.04,89.66,78.53,94.97,82.62,87.96,100.00,96.63,97.70]
plot_dataset("Client Accuracy on CIFAR10", FedAvg_C10, FedAS_C10, FedDodm_C10, DCPFL_C10, fname="CIFAR10.png")

# ---------- CIFAR100 ----------
FedAS_C100   = [60.26,59.16,63.72,58.71,46.82,55.97,54.49,59.45,60.68,52.93,56.58,62.26,61.60,56.66,63.68,62.07,62.60,60.82,63.11,58.59]
FedAvg_C100  = [30.33,30.37,40.35,35.22,30.76,29.58,30.26,27.30,24.93,32.63,30.68,32.86,28.55,27.99,37.17,34.61,31.90,32.98,27.01,41.26]
FedDodm_C100 = [55.63,54.71,62.99,54.09,45.15,50.00,53.33,54.20,57.83,51.06,54.56,54.40,59.05,53.43,60.05,60.54,57.11,56.28,58.50,54.75]
DCPFL_C100   = [43.71,50.52,51.81,43.27,31.06,36.67,45.00,47.64,50.14,39.10,44.57,44.88,49.09,45.76,49.03,53.13,47.07,51.29,47.69,38.80]
plot_dataset("Client Accuracy on CIFAR100", FedAvg_C100, FedAS_C100, FedDodm_C100, DCPFL_C100, fname="CIFAR100.png")

# ---------- Tiny-ImageNet ----------
FedAS_Tiny   = [34.80,39.34,38.13,35.78,35.86,32.83,33.86,34.83,30.72,40.98,43.53,39.05,30.55,32.35,36.63,45.93,42.38,34.41,27.74,34.92]
FedAvg_Tiny  = [18.93,19.98,18.71,18.37,16.73,12.70,13.26,17.74,15.40,19.57,18.65,15.86,14.26,14.43,18.90,18.56,19.76,18.35,13.55,17.99]
FedDodm_Tiny = [34.96,40.56,42.61,36.58,39.19,32.38,31.18,35.41,32.81,38.91,43.89,39.83,28.84,33.90,37.88,48.13,40.38,33.15,30.50,36.52]
DCPFL_Tiny   = [23.02,31.13,27.74,22.76,26.49,21.04,20.52,23.71,20.84,26.32,31.02,26.67,18.86,18.39,23.69,33.24,32.20,21.10,17.42,22.74]
plot_dataset("Client Accuracy on Tiny-ImageNet", FedAvg_Tiny, FedAS_Tiny, FedDodm_Tiny, DCPFL_Tiny, fname="TINTIMAGENET.png")