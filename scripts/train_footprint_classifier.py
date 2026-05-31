#!/usr/bin/env python3
"""Train a YOLO classify model on footprint crops.

Expects crops at <workspace>/training/crops/{train,val}/{warehouse,non_warehouse}/
produced by extract_footprint_crops.py.

Usage
-----
    python scripts/train_footprint_classifier.py configs/reno_sparks_demo.json
    python scripts/train_footprint_classifier.py configs/reno_sparks_demo.json \\
        --epochs 50 --model yolov8n-cls.pt --imgsz 224
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from warehouse_growth.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default="configs/reno_sparks_demo.json",
        help="Path to project config JSON/YAML",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--model",
        default="yolov8n-cls.pt",
        help="Pretrained YOLO classify base model (default: yolov8n-cls.pt)",
    )
    parser.add_argument("--imgsz", type=int, default=224, help="Input image size (default: 224)")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default=None, help="Device: cpu, 0, 0,1, … (default: auto)")
    args = parser.parse_args()

    config = load_config(args.config)
    crops_dir = config.workspace / "training" / "crops"

    if not crops_dir.exists():
        print(
            f"ERROR: crops directory not found at {crops_dir}\n"
            "Run extract_footprint_crops.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Project  : {config.project_name}")
    print(f"Crops dir: {crops_dir}")
    print("\nCrop counts:")
    total = 0
    for split in ("train", "val"):
        for label in ("warehouse", "non_warehouse"):
            d = crops_dir / split / label
            n = len(list(d.glob("*.jpg"))) if d.exists() else 0
            print(f"  {split}/{label}: {n:,}")
            total += n
    if total == 0:
        print("ERROR: no crops found — run extract_footprint_crops.py first.", file=sys.stderr)
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "ERROR: ultralytics not installed — run: uv sync --extra models",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = config.workspace / "training" / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBase model : {args.model}")
    print(f"Epochs     : {args.epochs}")
    print(f"Image size : {args.imgsz}")
    print(f"Batch      : {args.batch}")
    print(f"Output dir : {output_dir / 'warehouse_cls'}\n")

    model = YOLO(args.model)
    train_kwargs: dict = dict(
        data=str(crops_dir),  # YOLO classify auto-discovers {data}/{split}/{class}/
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(output_dir),
        name="warehouse_cls",
        exist_ok=True,
        # Augmentations suited for aerial imagery (no canonical orientation).
        flipud=0.5,
        fliplr=0.5,
        degrees=90.0,
        verbose=False,
    )
    if args.device is not None:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)

    best_pt = output_dir / "warehouse_cls" / "weights" / "best.pt"
    print("\n" + "─" * 50)
    if best_pt.exists():
        print(f"Best checkpoint → {best_pt}")
        print('\nAdd to your config JSON:')
        print('  "classifier": {')
        print('    "enabled": true,')
        print(f'    "checkpoint": "{best_pt}",')
        print('    "threshold": 0.5')
        print('  }')
    else:
        print(f"Best checkpoint not found at expected path: {best_pt}")
    print("─" * 50)


if __name__ == "__main__":
    main()
