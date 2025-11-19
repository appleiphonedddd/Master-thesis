import matplotlib.pyplot as plt
import csv
import math

def plot_all_clients(csv_path: str, rolling_window=None):

    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        round_cols = [c for c in reader.fieldnames if c.startswith("Round_")]
        rounds = [int(c.split("_")[1]) for c in round_cols]

        data_per_client = {}

        for row in reader:
            cid = row["Client_ID"]
            ys = []
            for col in round_cols:
                val = row[col]
                if val is None or val == "":
                    ys.append(math.nan)
                else:
                    ys.append(float(val))
            data_per_client[cid] = ys

    def apply_rolling_mean(values, window):
        if window is None:
            return values
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            segment = [v for v in values[start:i+1] if not math.isnan(v)]
            if len(segment) == 0:
                smoothed.append(math.nan)
            else:
                smoothed.append(sum(segment) / len(segment))
        return smoothed

    plt.figure(figsize=(12, 7))

    for cid, ys in data_per_client.items():
        if rolling_window is not None:
            ys = apply_rolling_mean(ys, rolling_window)

        plt.plot(rounds, ys, label=f"Client {cid}", linewidth=1)

    plt.xlabel("Communication Round", fontsize=14)
    plt.ylabel("FIM Trace", fontsize=14)
    plt.title("FIM Trace per Client on MNIST", fontsize=16)
    plt.grid(True, linewidth=0.3, alpha=0.6)
    plt.legend(fontsize=9, ncol=3)
    plt.tight_layout()
    plt.savefig("MNIST.png", dpi=300)
    print("Finished plotting FIM traces. Saved to fim_trace_per_client.png")

if __name__ == "__main__":
    plot_all_clients("MNIST.csv", rolling_window=None)
