import matplotlib.pyplot as plt
import csv
import math

def plot_fim(csv_path: str, target_clients=None):
    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        round_cols = [col for col in reader.fieldnames if col.startswith("Round_")]
        rounds = [int(col.split("_")[1]) for col in round_cols]

        data_per_client = {}

        for row in reader:
            cid = row["Client_ID"]
            if (target_clients is not None) and (cid not in target_clients):
                continue

            ys = []
            for col in round_cols:
                val = row[col]
                if val is None or val == "":
                    ys.append(math.nan)
                else:
                    ys.append(float(val))
            data_per_client[cid] = ys

    plt.figure(figsize=(10, 6))
    for cid, ys in data_per_client.items():
        plt.plot(rounds, ys, label=f"Client {cid}")

    plt.xlabel("Communication Round")
    plt.ylabel("FIM")
    plt.title("Client-wise FIM over Communication Rounds")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fim_plot.png")

if __name__ == "__main__":
    plot_fim("fim_trace_histories.csv")
