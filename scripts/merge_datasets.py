#!/usr/bin/env python3
"""Combine training patches from multiple metro workspaces into one dataset.yaml.

Each metro is prepared independently with prepare_training_data.py, which writes
patches to its own workspace. This script assembles those per-metro image
directories into a single dataset.yaml pointing to all of them. YOLO reads
multi-path dataset files natively, so no files are copied or merged.

Usage
-----
    # Combine two metros, write combined dataset.yaml to runs/combined/
    python scripts/merge_datasets.py configs/reno_sparks_demo.json configs/inland_empire.json

    # Specify a custom output workspace
    python scripts/merge_datasets.py configs/*.json --workspace runs/us_west

    # Train on the combined dataset
    python scripts/train_warehouse_detector.py runs/combined
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

from warehouse_growth import provenance
from warehouse_growth.config import load_config


def main(config_paths: list[Path], output_workspace: Path) -> None:
    train_dirs: list[str] = []
    val_dirs: list[str] = []
    skipped: list[str] = []

    for config_path in config_paths:
        config = load_config(config_path)
        train_dir = config.workspace / "training" / "images" / "train"
        val_dir = config.workspace / "training" / "images" / "val"

        if not train_dir.exists():
            skipped.append(
                f"{config.project_name} — training/images/train not found "
                f"(run prepare_training_data.py {config_path})"
            )
            continue

        train_dirs.append(str(train_dir.resolve()))
        val_dirs.append(str(val_dir.resolve()))

        n_train = sum(1 for _ in train_dir.glob("*.tif"))
        n_val = sum(1 for _ in val_dir.glob("*.tif")) if val_dir.exists() else 0
        print(f"  {config.project_name:<30s}  {n_train:5d} train  {n_val:4d} val")

    if skipped:
        print("\nSkipped (not yet prepared):")
        for msg in skipped:
            print(f"  {msg}")

    if not train_dirs:
        raise SystemExit("No prepared metros found — nothing to merge.")

    dataset_yaml = output_workspace / "training" / "dataset.yaml"
    dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
    dataset_yaml.write_text(
        yaml.dump(
            {
                "train": train_dirs,
                "val": val_dirs,
                "nc": 1,
                "names": ["warehouse"],
            },
            default_flow_style=False,
        )
    )
    provenance.write(
        dataset_yaml,
        metros=[str(p) for p in config_paths],
        n_merged=len(train_dirs),
    )

    print(f"\nMerged {len(train_dirs)} metro(s) → {dataset_yaml}")
    print("Next step:")
    print(f"  uv run python scripts/train_warehouse_detector.py {output_workspace}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)

    output_workspace = Path("runs/combined")
    config_args: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--workspace" and i + 1 < len(args):
            output_workspace = Path(args[i + 1])
            i += 2
        else:
            config_args.append(args[i])
            i += 1

    main([Path(p) for p in config_args], output_workspace)
