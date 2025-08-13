#!/usr/bin/env python3
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Plot and optionally smooth accuracy curves from multiple CSV files."
    )
    parser.add_argument(
        'csv_paths', nargs='+',
        help='Paths to one or more CSV files with columns "round" and "test_acc"'
    )
    parser.add_argument(
        '--smooth', type=int, default=0,
        help='Optional smoothing window size (integer > 1) for a rolling average'
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
        label = os.path.splitext(os.path.basename(csv_path))[0]
        series = df['train_loss']

        if args.smooth and args.smooth > 1:
            # Apply rolling average smoothing
            series = series.rolling(
                window=args.smooth,
                center=True,
                min_periods=1
            ).mean()
            

        color = colors[idx % len(colors)]
        plt.plot(df['round'], series, label=label, color=color)

    plt.xlabel('Communication Round')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    png_path = 'Result.png'
    plt.savefig(png_path)
    print(f"[Info] Saved comparison plot to {png_path}")

if __name__ == "__main__":
    main()
