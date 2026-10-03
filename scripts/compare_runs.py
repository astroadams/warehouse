#!/usr/bin/env python3
"""Compare YOLO training runs in a workspace side by side.

Reads every <workspace>/training/runs/*/results.csv and reports, for each run,
the epoch with the best mask mAP50-95 along with its precision, recall, and
mAP50. Also plots validation mask mAP50-95 per epoch for all runs on one chart.

These are YOLO's built-in validation metrics, which undercount precision when
OSM labels are incomplete. Use evaluate_footprint.py --checkpoint on each run's
best.pt for footprint-anchored precision/recall/F1.

Usage
-----
    python scripts/compare_runs.py                       # default workspace
    python scripts/compare_runs.py runs/reno_sparks_demo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

KEY = "metrics/mAP50-95(M)"
COLUMNS = {
    "metrics/precision(M)": "P(M)",
    "metrics/recall(M)": "R(M)",
    "metrics/mAP50(M)": "mAP50(M)",
    KEY: "mAP50-95(M)",
    "metrics/mAP50(B)": "mAP50(B)",
}


def load_run(run_dir: Path) -> tuple[dict, pd.DataFrame] | None:
    csv = run_dir / "results.csv"
    df = pd.read_csv(csv)
    df.columns = df.columns.str.strip()
    if df.empty or KEY not in df.columns:
        return None

    args_yaml = run_dir / "args.yaml"
    args = yaml.safe_load(args_yaml.read_text()) if args_yaml.exists() else {}
    best = df.loc[df[KEY].idxmax()]
    row = {
        "run": run_dir.name,
        "model": Path(str(args.get("model", "?"))).name,
        "imgsz": args.get("imgsz", "?"),
        "batch": args.get("batch", "?"),
        "epochs": f"{len(df)}/{args.get('epochs', '?')}",
        "best_ep": int(best["epoch"]),
        **{short: round(float(best[col]), 3) for col, short in COLUMNS.items() if col in df},
    }
    if "time" in df.columns:
        row["min/ep"] = round(float(df["time"].diff().median()) / 60, 1)
    return row, df


def main() -> None:
    p = argparse.ArgumentParser(description="Compare YOLO training runs.")
    p.add_argument("workspace", nargs="?", default="runs/reno_sparks_demo")
    args = p.parse_args()

    runs_root = Path(args.workspace) / "training" / "runs"
    csvs = sorted(runs_root.glob("*/results.csv"))
    if not csvs:
        print(f"ERROR: no results.csv found under {runs_root}/")
        sys.exit(1)

    rows, curves = [], {}
    for csv in csvs:
        loaded = load_run(csv.parent)
        if loaded is None:
            print(f"  skipping {csv.parent.name}: no completed epochs")
            continue
        row, df = loaded
        rows.append(row)
        curves[row["run"]] = df

    table = pd.DataFrame(rows).sort_values("mAP50-95(M)", ascending=False)
    print(table.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, df in curves.items():
        ax.plot(df["epoch"], df[KEY], label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val mask mAP50-95")
    ax.set_title(f"Run comparison — {Path(args.workspace).name}")
    ax.grid(alpha=0.3)
    ax.legend()
    out = runs_root / "run_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out}")


if __name__ == "__main__":
    main()
