#!/usr/bin/env python3
"""Prepare a YOLO segmentation dataset from labeled footprints + NAIP tiles.

Steps
-----
1. Load labeled_footprints_2022.parquet (requires label_prototype_data.py first).
2. Search NAIP STAC for 2022 tiles covering the Reno-Sparks AOI.
3. For each tile (processed in parallel):
   - Download the full GeoTIFF to raw_tiles/ (skipped on repeat runs).
   - Slice into PATCH_SIZE × PATCH_SIZE patches with STRIDE overlap.
   - For each patch: project warehouse polygons → normalised YOLO segment
     annotations.  Positive patches are always kept; negatives are sampled
     at NEG_SAMPLE_RATE using a per-patch deterministic RNG.
4. Shuffle and split into train / val (80 / 20) at the TILE level so the
   same tile's patches don't appear in both splits.
5. Write dataset.yaml for `ultralytics YOLO.train()`.

Usage
-----
    python scripts/prepare_training_data.py [workspace_dir]

Parallelism is controlled by the TILING_WORKERS environment variable
(default: all available CPUs).

Default workspace: ./runs/reno_sparks_demo
"""
from __future__ import annotations

import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import geopandas as gpd
import planetary_computer as pc
import yaml
from tqdm import tqdm

from warehouse_growth.adapters import NAIPImagerySource
from warehouse_growth.training import init_patch_worker, process_tile_task

RENO_AOI_BBOX = (-120.05, 39.40, -119.45, 39.70)
EPOCH = "2022"
WAREHOUSE_CLASS_ID = 0
TRAIN_FRAC = 0.8
PATCH_SIZE = 1024
STRIDE = 768
NEG_SAMPLE_RATE = 0.20
SEED = 42


def main(workspace: Path) -> None:
    from shapely.geometry import box
    RENO_AOI = box(*RENO_AOI_BBOX)

    labels_path = workspace / f"labeled_footprints_{EPOCH}.parquet"
    if not labels_path.exists():
        raise FileNotFoundError(f"Run label_prototype_data.py first: {labels_path}")

    print("Loading labeled footprints …")
    gdf = gpd.read_parquet(labels_path)
    n_warehouse = (gdf["label"] == "warehouse").sum()
    print(f"  {n_warehouse:,} warehouse buildings (class 0)")

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
        # Strip SAS tokens before caching — re-signed fresh at download time.
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
    random.seed(SEED)
    shuffled = assets[:]
    random.shuffle(shuffled)
    split_at = int(len(shuffled) * TRAIN_FRAC)

    dataset_dir = workspace / "training"
    raw_tile_dir = dataset_dir / "raw_tiles"
    raw_tile_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── Build per-tile task descriptors ───────────────────────────────────
    tasks = [
        {
            # Always store the unsigned base URI; workers re-sign just before
            # downloading so a long-running job never hits an expired SAS token.
            "uri": urlparse(asset.uri)._replace(query="", fragment="").geturl(),
            "split": "train" if rank < split_at else "val",
            "dataset_dir": str(dataset_dir),
            "raw_tile_dir": str(raw_tile_dir),
            "patch_size": PATCH_SIZE,
            "stride": STRIDE,
            "neg_sample_rate": NEG_SAMPLE_RATE,
            "warehouse_class_id": WAREHOUSE_CLASS_ID,
        }
        for rank, asset in enumerate(shuffled)
    ]

    n_workers = min(
        int(os.environ.get("TILING_WORKERS", os.cpu_count() or 1)),
        len(tasks),
    )

    stats = {"train": {"pos": 0, "neg": 0}, "val": {"pos": 0, "neg": 0}}

    print(
        f"\nSlicing tiles into {PATCH_SIZE}×{PATCH_SIZE} patches "
        f"(stride {STRIDE}, {n_workers} workers) …"
    )
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_patch_worker,
        initargs=(str(labels_path),),
    ) as pool:
        futures = {pool.submit(process_tile_task, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), unit=" tile"):
            try:
                tile_result = future.result()
            except Exception as exc:
                task = futures[future]
                tile_name = Path(urlparse(task["uri"]).path).stem
                tqdm.write(f"  ERROR {tile_name}: {exc}")
                continue
            if tile_result["error"]:
                tqdm.write(f"  SKIP {tile_result['tile_name']}: {tile_result['error']}")
                continue
            split = tile_result["split"]
            stats[split]["pos"] += tile_result["pos"]
            stats[split]["neg"] += tile_result["neg"]

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
