#!/usr/bin/env python3
"""Extract padded square JPEG crops of labeled footprints for classifier training.

For each warehouse / non_warehouse footprint, finds the covering raw NAIP tile,
crops the footprint bounding box with padding, pads to square, and writes a JPEG
to training/crops/{split}/{label}/{tile_stem}_{footprint_idx}.jpg.

The train/val split assignment matches the existing YOLO patch split — footprints
are assigned to whichever split their covering tile was placed in during
prepare_training_data.py. Footprints from tiles not yet in the patch set are
skipped.

Usage
-----
    python scripts/extract_footprint_crops.py configs/reno_sparks_demo.json
    python scripts/extract_footprint_crops.py configs/reno_sparks_demo.json \\
        --padding 64 --epoch 2022
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from shapely.geometry import box
from shapely.strtree import STRtree
from tqdm import tqdm

from warehouse_growth.config import load_config


def _build_tile_index(raw_tile_dir: Path) -> tuple[list[Path], list, STRtree]:
    """Open every raw tile header, project bounds to EPSG:4326, build STRtree."""
    tile_paths: list[Path] = sorted(raw_tile_dir.glob("*.tif"))
    tile_boxes: list = []
    for p in tile_paths:
        with rasterio.open(p) as src:
            b4326 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
            tile_boxes.append(box(*b4326))
    return tile_paths, tile_boxes, STRtree(tile_boxes)


def _build_split_map(training_dir: Path) -> dict[str, str]:
    """Scan training/images/{train,val} patches to build tile_name → split map."""
    split_map: dict[str, str] = {}
    for split in ("train", "val"):
        for patch_path in (training_dir / "images" / split).glob("*.tif"):
            # patch stem format: {tile_name}_{x}_{y} — strip the last two components
            tile_name = patch_path.stem.rsplit("_", 2)[0]
            split_map[tile_name] = split
    return split_map


def _find_covering_tile(
    fp_geom,
    tile_paths: list[Path],
    tile_boxes: list,
    tile_tree: STRtree,
) -> Path | None:
    """Return the raw tile that best covers the footprint geometry."""
    hit_idxs = tile_tree.query(fp_geom, predicate="within")
    if len(hit_idxs) == 0:
        hit_idxs = tile_tree.query(fp_geom, predicate="intersects")
    if len(hit_idxs) == 0:
        return None
    if len(hit_idxs) == 1:
        return tile_paths[int(hit_idxs[0])]
    # Multiple tiles overlap — pick the one with greatest intersection area.
    best_idx = max(hit_idxs, key=lambda idx: fp_geom.intersection(tile_boxes[int(idx)]).area)
    return tile_paths[int(best_idx)]


def _extract_crop(
    fp_geom,
    fp_label: str,
    fp_idx: int,
    tile_path: Path,
    split: str,
    padding: int,
    crops_dir: Path,
) -> bool:
    """Crop, square-pad, and write one footprint as JPEG. Returns True on success."""
    out_path = crops_dir / split / fp_label / f"{tile_path.stem}_{fp_idx}.jpg"
    if out_path.exists():
        return True

    with rasterio.open(tile_path) as src:
        tile_crs_str = src.crs.to_string()
        tile_transform = src.transform
        bands = list(range(1, min(src.count, 3) + 1))

        xs, ys = zip(*list(fp_geom.exterior.coords))
        xt, yt = warp_transform("EPSG:4326", tile_crs_str, list(xs), list(ys))

        inv = ~tile_transform
        col_vals = []
        row_vals = []
        for gx, gy in zip(xt, yt):
            col, row = inv * (gx, gy)
            col_vals.append(col)
            row_vals.append(row)

        col_min = int(min(col_vals)) - padding
        col_max = int(max(col_vals)) + padding
        row_min = int(min(row_vals)) - padding
        row_max = int(max(row_vals)) + padding

        win_w = col_max - col_min
        win_h = row_max - row_min
        if win_w <= 0 or win_h <= 0:
            return False

        win = Window(col_off=col_min, row_off=row_min, width=win_w, height=win_h)
        # boundless=True zero-pads if window extends beyond tile edges.
        crop = src.read(bands, window=win, boundless=True, fill_value=0)  # (C, H, W)

    c, h, w = crop.shape
    side = max(h, w)
    if h != w:
        pad_h = side - h
        pad_w = side - w
        crop = np.pad(
            crop,
            ((0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
            mode="constant",
            constant_values=0,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path,
        "w",
        driver="JPEG",
        width=side,
        height=side,
        count=3,
        dtype="uint8",
        QUALITY=95,
    ) as dst:
        dst.write(crop.astype("uint8"))

    return True


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
    parser.add_argument(
        "--padding",
        type=int,
        default=32,
        metavar="PX",
        help="Pixel padding around footprint bbox (default: 32)",
    )
    parser.add_argument(
        "--epoch",
        default=None,
        metavar="NAME",
        help="Restrict to a single epoch name (default: all epochs)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    workspace = config.workspace
    training_dir = workspace / "training"
    raw_tile_dir = training_dir / "raw_tiles"
    crops_dir = training_dir / "crops"

    if not raw_tile_dir.exists() or not any(raw_tile_dir.glob("*.tif")):
        print(
            f"ERROR: no raw tiles found in {raw_tile_dir}\n"
            "Run prepare_training_data.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Building tile spatial index …")
    tile_paths, tile_boxes, tile_tree = _build_tile_index(raw_tile_dir)
    print(f"  {len(tile_paths)} tiles indexed")

    print("Building train/val split map …")
    split_map = _build_split_map(training_dir)
    if not split_map:
        print(
            "ERROR: no training patches found — run prepare_training_data.py first.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"  {len(split_map)} tiles mapped to splits")

    epochs = config.epochs
    if args.epoch:
        epochs = [e for e in epochs if e.name == args.epoch]
        if not epochs:
            print(f"ERROR: no epoch named {args.epoch!r} in config.", file=sys.stderr)
            sys.exit(1)

    total_written = total_skipped_no_tile = total_skipped_no_split = 0

    for epoch in epochs:
        labels_path = workspace / f"labeled_footprints_{epoch.name}.parquet"
        if not labels_path.exists():
            print(f"\nSkipping epoch {epoch.name!r}: {labels_path.name} not found")
            continue

        gdf = gpd.read_parquet(labels_path)
        gdf = gdf[gdf["label"].isin(["warehouse", "non_warehouse"])].reset_index(drop=True)
        print(f"\nEpoch {epoch.name!r}: {len(gdf):,} labeled footprints (warehouse + non_warehouse)")

        n_written = n_skipped_no_tile = n_skipped_no_split = 0

        for fp_idx, row in tqdm(gdf.iterrows(), total=len(gdf), unit=" fp"):
            fp_geom = row.geometry
            fp_label = row["label"]

            tile_path = _find_covering_tile(fp_geom, tile_paths, tile_boxes, tile_tree)
            if tile_path is None:
                n_skipped_no_tile += 1
                continue

            split = split_map.get(tile_path.stem)
            if split is None:
                n_skipped_no_split += 1
                continue

            try:
                ok = _extract_crop(fp_geom, fp_label, fp_idx, tile_path, split, args.padding, crops_dir)
                if ok:
                    n_written += 1
            except Exception as exc:
                warnings.warn(f"  SKIP footprint {fp_idx} ({tile_path.name}): {exc}")

        print(
            f"  Written: {n_written:,}  |  "
            f"No covering tile: {n_skipped_no_tile:,}  |  "
            f"Tile not in split map: {n_skipped_no_split:,}"
        )
        total_written += n_written
        total_skipped_no_tile += n_skipped_no_tile
        total_skipped_no_split += n_skipped_no_split

    print(f"\nTotal crops written: {total_written:,}")
    print("\nCrop counts by split/label:")
    for split in ("train", "val"):
        for label in ("warehouse", "non_warehouse"):
            d = crops_dir / split / label
            n = len(list(d.glob("*.jpg"))) if d.exists() else 0
            print(f"  {split}/{label}: {n:,}")


if __name__ == "__main__":
    main()
