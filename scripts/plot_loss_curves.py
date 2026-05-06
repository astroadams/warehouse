#!/usr/bin/env python3
"""Plot training vs validation loss curves from a YOLO results.csv.

Usage
-----
    python scripts/plot_loss_curves.py                          # default run
    python scripts/plot_loss_curves.py runs/my_run             # workspace dir
    python scripts/plot_loss_curves.py path/to/results.csv     # direct CSV path
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


def resolve_csv(path: str) -> Path:
    p = Path(path)
    if p.suffix == ".csv":
        return p
    # Treat as a workspace directory — find results.csv under training/runs/
    candidates = sorted(p.glob("training/runs/*/results.csv"))
    if not candidates:
        print(f"ERROR: no results.csv found under {p}/training/runs/")
        sys.exit(1)
    if len(candidates) > 1:
        print("Multiple results.csv found; using most recently modified:")
        candidates.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        for c in candidates:
            print(f"  {c}")
    return candidates[0]


def plot(csv: Path) -> Path:
    df = pd.read_csv(csv)
    df.columns = df.columns.str.strip()

    pairs = [
        ("train/box_loss", "val/box_loss", "Box loss"),
        ("train/seg_loss", "val/seg_loss", "Seg loss"),
        ("train/cls_loss", "val/cls_loss", "Cls loss"),
        ("train/dfl_loss", "val/dfl_loss", "DFL loss"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    title = f"Training vs Validation Losses — {csv.parent.name}"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, (train_col, val_col, subtitle) in zip(axes.flat, pairs):
        ax.plot(df["epoch"], df[train_col], label="train", linewidth=1.5)
        ax.plot(df["epoch"], df[val_col], label="val", linewidth=1.5, linestyle="--")
        ax.set_title(subtitle)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = csv.parent / "loss_curves.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Plot YOLO loss curves.")
    p.add_argument(
        "path",
        nargs="?",
        default="runs/reno_sparks_demo",
        help="Workspace dir or direct path to results.csv",
    )
    args = p.parse_args()

    csv = resolve_csv(args.path)
    print(f"Reading {csv}")
    out = plot(csv)
    print(f"Saved  → {out}")


if __name__ == "__main__":
    main()
