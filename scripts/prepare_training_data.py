#!/usr/bin/env python3
"""Prepare a YOLO segmentation dataset from labeled footprints + NAIP tiles.

Steps
-----
1. Load labeled_footprints_2022.parquet (requires label_prototype_data.py first).
2. Search NAIP STAC for 2022 tiles covering the Reno-Sparks AOI.
3. For each tile:
   - Download the full GeoTIFF to raw_tiles/ (skipped on repeat runs).
   - Slice into PATCH_SIZE × PATCH_SIZE patches with STRIDE overlap.
   - For each patch: project warehouse polygons → normalised YOLO segment
     annotations.  Positive patches are always kept; negatives are sampled
     at NEG_SAMPLE_RATE.
4. Shuffle and split into train / val (80 / 20) at the TILE level so the
   same tile's patches don't appear in both splits.
5. Write dataset.yaml for `ultralytics YOLO.train()`.

Usage
-----
    python scripts/prepare_training_data.py [workspace_dir]

Default workspace: ./runs/reno_sparks_demo
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from urllib.parse import urlparse

import geopandas as gpd
import yaml
from shapely.geometry import box
from shapely.strtree import STRtree
from tqdm import tqdm

from warehouse_growth.adapters import NAIPImagerySource
from warehouse_growth.tiling import sliding_windows
from warehouse_growth.training import download_naip_tile, geo_polygon_to_yolo, write_label_file


def _fresh_uri(uri: str) -> str:
    """Return a freshly signed URL, stripping any existing SAS token first.

    MPC SAS tokens are valid for ~25 hours from issuance.  Re-signing before
    each download means a script paused overnight won't 403 on resume.
    """
    base = urlparse(uri)._replace(query="", fragment="").geturl()
    try:
        import planetary_computer as pc
        return pc.sign(base)
    except ImportError:
        return base

RENO_AOI = box(-120.05, 39.40, -119.45, 39.70)
EPOCH = "2022"
WAREHOUSE_CLASS_ID = 0
TRAIN_FRAC = 0.8
PATCH_SIZE = 1024
STRIDE = 768
NEG_SAMPLE_RATE = 0.05   # fraction of empty patches to keep as hard negatives
SEED = 42

random.seed(SEED)


def main(workspace: Path) -> None:
    labels_path = workspace / f"labeled_footprints_{EPOCH}.parquet"
    if not labels_path.exists():
        raise FileNotFoundError(f"Run label_prototype_data.py first: {labels_path}")

    print("Loading labeled footprints …")
    gdf = gpd.read_parquet(labels_path)
    warehouse_gdf = gdf[gdf["label"] == "warehouse"].reset_index(drop=True)
    print(f"  {len(warehouse_gdf):,} warehouse buildings (class 0)")

    warehouse_geoms = warehouse_gdf.geometry.tolist()
    wh_tree = STRtree(warehouse_geoms)

    # ── Fetch tile list from STAC (cached after first successful query) ────
    tile_cache = workspace / f"naip_tile_cache_{EPOCH}.json"
    if tile_cache.exists():
        print(f"\nLoading NAIP {EPOCH} tile list from cache …")
        raw_records = json.loads(tile_cache.read_text())
        from warehouse_growth.data_sources import ImageryAsset
        assets = [ImageryAsset(**r) for r in raw_records]
        print(f"  {len(assets)} tiles (cached)")
    else:
        print(f"\nSearching NAIP {EPOCH} tiles …")
        naip = NAIPImagerySource()
        assets = list(naip.assets_for_aoi(RENO_AOI, EPOCH))
        print(f"  {len(assets)} tiles")
        # Strip SAS tokens before caching — we re-sign fresh at download time.
        records = [
            {
                "uri": urlparse(a.uri)._replace(query="", fragment="").geturl(),
                "epoch": a.epoch,
                "capture_date": a.capture_date,
                "crs": a.crs,
                "resolution_meters": a.resolution_meters,
            }
            for a in assets
        ]
        tile_cache.write_text(json.dumps(records, indent=2))
        print(f"  tile list cached → {tile_cache.name}")

    # ── Shuffle and assign train / val splits at the TILE level ───────────
    shuffled = assets[:]
    random.shuffle(shuffled)
    split_at = int(len(shuffled) * TRAIN_FRAC)

    dataset_dir = workspace / "training"
    raw_tile_dir = dataset_dir / "raw_tiles"
    raw_tile_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── Import heavy dependencies lazily ──────────────────────────────────
    try:
        import numpy as np
        import rasterio
        from affine import Affine
        from rasterio.transform import array_bounds
        from rasterio.warp import transform_bounds
    except ImportError as e:
        raise ImportError(
            "rasterio, numpy, and affine are required — "
            "install via conda or pip install rasterio numpy affine"
        ) from e

    stats = {"train": {"pos": 0, "neg": 0}, "val": {"pos": 0, "neg": 0}}

    print(f"\nSlicing tiles into {PATCH_SIZE}×{PATCH_SIZE} patches (stride {STRIDE}) …")
    for rank, asset in enumerate(tqdm(shuffled, unit=" tile")):
        split = "train" if rank < split_at else "val"

        # urlparse extracts the clean path, ignoring the SAS token query string.
        tile_name = Path(urlparse(asset.uri).path).stem
        raw_path = raw_tile_dir / f"{tile_name}.tif"

        if not raw_path.exists():
            try:
                # Strip the old SAS token and re-sign fresh so a long-running
                # script or a resume-after-sleep doesn't hit 403 on expired tokens.
                download_uri = _fresh_uri(asset.uri)
                download_naip_tile(download_uri, raw_path)
            except Exception as exc:
                tqdm.write(f"  SKIP {tile_name}: {exc}")
                continue

        with rasterio.open(raw_path) as src:
            tile_crs = src.crs
            tile_transform = src.transform
            tile_w, tile_h = src.width, src.height
            # Read full tile into memory once; cheaper than many window reads.
            img_full = src.read([1, 2, 3])  # shape (3, H, W)

        for win in sliding_windows(tile_w, tile_h, PATCH_SIZE, STRIDE):
            # Affine transform for this patch (origin at patch top-left pixel).
            patch_transform = tile_transform * Affine.translation(win.x, win.y)

            # Geographic bounding box of the patch → EPSG:4326 for tree query.
            left, bottom, right, top = array_bounds(
                win.height, win.width, patch_transform
            )
            b4326 = transform_bounds(
                str(tile_crs), "EPSG:4326", left, bottom, right, top
            )
            patch_box_4326 = box(*b4326)

            hit_idxs = wh_tree.query(patch_box_4326, predicate="intersects")

            instances: list[tuple[int, list[float]]] = []
            for idx in hit_idxs.tolist():
                clipped = warehouse_geoms[idx].intersection(patch_box_4326)
                if clipped.is_empty:
                    continue
                if clipped.geom_type == "MultiPolygon":
                    clipped = max(clipped.geoms, key=lambda p: p.area)
                if clipped.geom_type != "Polygon":
                    continue
                pts = geo_polygon_to_yolo(
                    clipped,
                    patch_transform,
                    str(tile_crs),
                    win.width,
                    win.height,
                )
                if pts:
                    instances.append((WAREHOUSE_CLASS_ID, pts))

            is_positive = bool(instances)
            if not is_positive and random.random() > NEG_SAMPLE_RATE:
                continue

            patch_name = f"{tile_name}_{win.x}_{win.y}"
            img_path = dataset_dir / "images" / split / f"{patch_name}.tif"
            lbl_path = dataset_dir / "labels" / split / f"{patch_name}.txt"

            if not img_path.exists():
                patch_img = img_full[:, win.y:win.y + win.height, win.x:win.x + win.width]
                patch_profile = {
                    "driver": "GTiff",
                    "count": 3,
                    "dtype": patch_img.dtype,
                    "width": win.width,
                    "height": win.height,
                    "crs": tile_crs,
                    "transform": patch_transform,
                    "compress": "deflate",
                    "tiled": True,
                }
                with rasterio.open(img_path, "w", **patch_profile) as dst:
                    dst.write(patch_img)

            write_label_file(lbl_path, instances)
            stats[split]["pos" if is_positive else "neg"] += 1

    # ── Write dataset.yaml ─────────────────────────────────────────────────
    yaml_path = dataset_dir / "dataset.yaml"
    yaml_path.write_text(
        yaml.dump(
            {
                "path": str(dataset_dir.resolve()),
                "train": "images/train",
                "val": "images/val",
                "nc": 1,
                "names": ["warehouse"],
            },
            default_flow_style=False,
        )
    )

    print("\nDataset summary (patch level):")
    total = 0
    for split in ("train", "val"):
        p, n = stats[split]["pos"], stats[split]["neg"]
        total += p + n
        print(f"  {split:5s}  {p:4d} positive  {n:4d} negative")
    print(f"  total  {total} patches")
    print(f"\ndataset.yaml → {yaml_path}")
    print("Next step:")
    print("  python scripts/train_warehouse_detector.py")


if __name__ == "__main__":
    ws = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/reno_sparks_demo")
    main(ws)
