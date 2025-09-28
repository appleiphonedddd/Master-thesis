#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def ema(series: pd.Series, alpha: float) -> pd.Series:
    return series.ewm(alpha=alpha, adjust=False).mean()

def main():
    parser = argparse.ArgumentParser(
        description="Plot curves (Accuracy/Loss/etc.) from multiple CSV files with optional EMA smoothing. X-axis is fixed to 'round'."
    )
    parser.add_argument('csv_paths', nargs='+',
                        help='CSV file paths. Need columns: round and the chosen metric.')
    parser.add_argument('--metric', type=str, required=True,
                        help='Metric column to plot, e.g., train_loss, test_acc, etc.')
    parser.add_argument('--ema', type=float, default=0.0,
                        help='EMA alpha in [0,1]. 0 = no smoothing. (e.g., 0.15)')
    parser.add_argument('--title', type=str, default='',
                        help='Optional chart title.')
    parser.add_argument('--output', type=str, default='Result.png',
                        help='Output PNG path (default: Result.png).')
    parser.add_argument('--dpi', type=int, default=200,
                        help='Figure DPI (default: 200).')
    parser.add_argument('--sort-x', action='store_true',
                        help="Sort by 'round' before plotting (recommended if CSV is unsorted).")
    args = parser.parse_args()

    if args.ema < 0 or args.ema > 1:
        print("[Error] --ema must be in [0, 1]. 0 means no smoothing.")
        sys.exit(1)

    plt.figure()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    any_plotted = False

    for idx, csv_path in enumerate(args.csv_paths):
        if not os.path.isfile(csv_path):
            print(f"[Warning] File not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        if 'round' not in df.columns:
            print(f"[Warning] X column 'round' not found in {csv_path}")
            continue
        if args.metric not in df.columns:
            print(f"[Warning] Metric '{args.metric}' not found in {csv_path}")
            continue

        if args.sort_x:
            df = df.sort_values('round', kind='mergesort')

        x = df['round'].values
        y = pd.Series(df[args.metric].astype(float).values)

        y_plot = ema(y, args.ema) if args.ema > 0 else y
        label = os.path.splitext(os.path.basename(csv_path))[0]  # 只顯示方法（檔名）
        color = colors[idx % len(colors)]

        plt.plot(x, y_plot, label=label, linewidth=1.8, color=color)
        any_plotted = True

    if not any_plotted:
        print("[Error] No series plotted. Check file paths/columns.")
        sys.exit(1)

    plt.xlabel('Communication Round')
    ylabel_map = {"test_acc": "Accuracy", "train_loss": "Loss"}
    plt.ylabel(ylabel_map.get(args.metric, args.metric))
    if args.title:
        plt.title(args.title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=args.dpi)
    print(f"[Info] Saved plot to {args.output}")

if __name__ == "__main__":
    main()
