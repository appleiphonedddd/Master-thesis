#!/usr/bin/env python3
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Plot and optionally smooth curves (Accuracy/Loss) from multiple CSV files."
    )
    parser.add_argument(
        'csv_paths', nargs='+',
        help='Paths to one or more CSV files with columns "round" and the chosen metric'
    )
    parser.add_argument(
        '--metric', type=str, choices=['test_acc', 'train_loss'], required=True,
        help='Which metric to plot: "test_acc" or "train_loss"'
    )
    parser.add_argument(
        '--smooth', type=int, default=0,
        help='Optional smoothing window size (integer > 1) for a rolling average'
    )
    parser.add_argument(
        '--output', type=str, default="Result.png",
        help='Path to save the output PNG file'
    )
    args = parser.parse_args()

    plt.figure()
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for idx, csv_path in enumerate(args.csv_paths):
        if not os.path.isfile(csv_path):
            print(f"[Warning] File not found: {csv_path}")
            continue
        df = pd.read_csv(csv_path)

        if args.metric not in df.columns:
            print(f"[Warning] Metric '{args.metric}' not found in {csv_path}")
            continue

        label = os.path.splitext(os.path.basename(csv_path))[0]
        series = df[args.metric]

        if args.smooth and args.smooth > 1:
            series = series.rolling(
                window=args.smooth,
                center=True,
                min_periods=1
            ).mean()

        color = colors[idx % len(colors)]
        plt.plot(df['round'], series, label=label, color=color)

    plt.xlabel('Communication Round')
    ylabel = "Accuracy" if args.metric == "test_acc" else "Loss"
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(args.output)
    print(f"[Info] Saved comparison plot to {args.output}")

if __name__ == "__main__":
    main()
