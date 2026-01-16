"""Generate plots for the loss modeling project."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--losses_csv", type=str, default="data/aggregate_losses.csv")
    ap.add_argument("--outdir", type=str, default="plots")
    args = ap.parse_args()

    losses = pd.read_csv(args.losses_csv)["aggregate_loss"].to_numpy()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Aggregate loss distribution (hist)
    plt.figure()
    plt.hist(losses, bins=60)
    plt.title("Simulated Aggregate Annual Loss")
    plt.xlabel("Aggregate Loss")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outdir/"aggregate_loss_hist.png", dpi=200)
    plt.close()

    # Tail zoom (top 5%)
    q95 = np.quantile(losses, 0.95, method="linear")
    tail = losses[losses >= q95]
    plt.figure()
    plt.hist(tail, bins=40)
    plt.title("Tail of Aggregate Loss Distribution (>= 95th pct)")
    plt.xlabel("Aggregate Loss (Tail)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outdir/"aggregate_loss_tail.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    main()
