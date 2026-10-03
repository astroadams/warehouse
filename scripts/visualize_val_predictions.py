#!/usr/bin/env python3
"""Visualise model predictions vs ground-truth labels on validation patches.

For each sampled patch the figure shows the RGB image with:
  - green outlines  → ground-truth warehouse polygons (from label .txt files)
  - red outlines    → model predictions above the confidence threshold

Usage
-----
    python scripts/visualize_val_predictions.py                       # defaults
    python scripts/visualize_val_predictions.py runs/my_run           # workspace
    python scripts/visualize_val_predictions.py runs/my_run --patches 24
    python scripts/visualize_val_predictions.py runs/my_run --all     # include negatives
    python scripts/visualize_val_predictions.py runs/my_run --conf 0.4
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualise val predictions vs ground truth.")
    p.add_argument("workspace", nargs="?", default="runs/reno_sparks_demo")
    p.add_argument("--patches", type=int, default=16,
                   help="Number of patches to show (default: 16)")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Model confidence threshold (default: 0.25)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--all", dest="include_negatives", action="store_true",
                   help="Include patches with no ground-truth warehouses")
    p.add_argument("--cols", type=int, default=4,
                   help="Number of columns in the output grid (default: 4)")
    return p.parse_args()


def read_image(img_path: Path):
    """Return (H, W, 3) uint8 numpy array from a GeoTIFF."""
    import rasterio

    with rasterio.open(img_path) as src:
        bands = src.read([1, 2, 3])  # shape (3, H, W)
    img = bands.transpose(1, 2, 0)  # → (H, W, 3)
    if img.dtype != "uint8":
        lo, hi = img.min(), img.max()
        if hi > lo:
            img = ((img - lo) / (hi - lo) * 255).astype("uint8")
        else:
            img = img.astype("uint8")
    return img


def parse_yolo_label(label_path: Path, width: int, height: int):
    """Return list of (N,2) float pixel-coord arrays from a YOLO segment label file."""
    import numpy as np

    polygons = []
    if not label_path.exists():
        return polygons
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 7:  # class_id + at least 3 xy pairs
            continue
        coords = list(map(float, parts[1:]))
        xs = [c * width  for c in coords[0::2]]
        ys = [c * height for c in coords[1::2]]
        polygons.append(np.array(list(zip(xs, ys))))
    return polygons


def run_model(model, img):
    """Return list of (N,2) float pixel-coord arrays from ultralytics result."""
    import numpy as np

    results = model(img, verbose=False)
    polygons = []
    for result in results:
        if result.masks is not None:
            for pts in result.masks.xy:
                if len(pts) >= 3:
                    polygons.append(np.array(pts))
        elif result.boxes is not None:
            for (x1, y1, x2, y2) in result.boxes.xyxy.cpu().numpy():
                polygons.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]))
    return polygons


def draw_polygon(ax, pts, color, linewidth=1.5):
    from matplotlib.patches import Polygon as MplPolygon
    patch = MplPolygon(pts, closed=True, edgecolor=color, facecolor="none",
                       linewidth=linewidth)
    ax.add_patch(patch)


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace)

    val_img_dir = workspace / "training" / "images" / "val"
    val_lbl_dir = workspace / "training" / "labels" / "val"
    weights = workspace / "training" / "runs" / "warehouse_seg" / "weights" / "best.pt"

    for path, label in [(val_img_dir, "val images"), (weights, "best.pt checkpoint")]:
        if not path.exists():
            print(f"ERROR: {label} not found at {path}")
            sys.exit(1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from ultralytics import YOLO

    all_imgs = sorted(val_img_dir.glob("*.tif"))
    if not all_imgs:
        print(f"ERROR: no .tif files found in {val_img_dir}")
        sys.exit(1)

    # Separate positive (has GT label content) from negative patches.
    positives, negatives = [], []
    for img_path in all_imgs:
        lbl = val_lbl_dir / (img_path.stem + ".txt")
        has_gt = lbl.exists() and lbl.stat().st_size > 0
        (positives if has_gt else negatives).append(img_path)

    rng = random.Random(args.seed)
    if args.include_negatives:
        pool = all_imgs[:]
    else:
        pool = positives
        if len(pool) < args.patches:
            print(f"Only {len(pool)} positive val patches; showing all of them.")

    rng.shuffle(pool)
    selected = pool[: args.patches]

    print(f"Loading model from {weights}")
    model = YOLO(str(weights))
    model.overrides["conf"] = args.conf

    n = len(selected)
    cols = min(args.cols, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes_flat = [axes] if n == 1 else list(np.array(axes).flat)

    print(f"Rendering {n} patches …")
    for ax, img_path in zip(axes_flat, selected):
        img = read_image(img_path)
        h, w = img.shape[:2]
        lbl_path = val_lbl_dir / (img_path.stem + ".txt")

        gt_polys = parse_yolo_label(lbl_path, w, h)
        pred_polys = run_model(model, img)

        ax.imshow(img)
        for poly in gt_polys:
            draw_polygon(ax, poly, color="lime")
        for poly in pred_polys:
            draw_polygon(ax, poly, color="red")

        ax.set_title(
            f"{img_path.stem[-24:]}\nGT={len(gt_polys)}  Pred={len(pred_polys)}",
            fontsize=7,
        )
        ax.axis("off")

    # Hide unused subplots.
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    # Legend
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], color="lime", linewidth=2, label="Ground truth"),
        Line2D([0], [0], color="red",  linewidth=2, label=f"Prediction (conf≥{args.conf})"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=2,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        f"Val predictions — {workspace.name}  "
        f"({len(positives)} positive / {len(negatives)} negative patches)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])

    out = workspace / "training" / "runs" / "warehouse_seg" / "val_predictions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
