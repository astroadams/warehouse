#!/usr/bin/env python3
"""Prepare a YOLO segmentation dataset from labeled footprints + NAIP tiles.

For each epoch defined in the project config:
  1. Load labeled_footprints_{epoch}.parquet (requires label_prototype_data.py first).
  2. Search NAIP STAC for tiles covering the AOI.
  3. For each tile (processed in parallel):
     - Download the full GeoTIFF to raw_tiles/ (skipped on repeat runs).
     - Slice into tile_size_px patches with stride_px overlap.
     - For each patch: project warehouse polygons → normalised YOLO segment
       annotations.  Positive patches are always kept; negatives are sampled
       at NEG_SAMPLE_RATE using a per-patch deterministic RNG.
  4. Shuffle and split into train / val (80 / 20) at the TILE level so the
     same tile's patches don't appear in both splits.
  5. Write dataset.yaml for use with train_warehouse_detector.py.

Epochs for which labeled_footprints_{epoch}.parquet is not yet present in the
workspace are skipped with a warning — run label_prototype_data.py first.

Usage
-----
    python scripts/prepare_training_data.py configs/reno_sparks_demo.json

Parallelism is controlled by the TILING_WORKERS environment variable
(default: all available CPUs).
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
from shapely.geometry import box
from tqdm import tqdm

from warehouse_growth.adapters import NAIPImagerySource
from warehouse_growth.config import EpochConfig, ProjectConfig, load_config
from warehouse_growth.training import init_patch_worker, process_tile_task

WAREHOUSE_CLASS_ID = 0
TRAIN_FRAC = 0.8
NEG_SAMPLE_RATE = 0.20
SEED = 42


def _process_epoch(
    epoch: EpochConfig,
    config: ProjectConfig,
    dataset_dir: Path,
    raw_tile_dir: Path,
    labels_path: Path,
    seed_offset: int,
) -> dict[str, dict[str, int]]:
    """Search, download, slice, and annotate all NAIP tiles for one epoch."""
    aoi = box(*config.aoi.bbox)

    tile_cache = config.workspace / f"naip_tile_cache_{epoch.name}.json"
    if tile_cache.exists():
        print(f"  Loading tile list from cache ({epoch.name}) …")
        from warehouse_growth.data_sources import ImageryAsset
        assets = [ImageryAsset(**r) for r in json.loads(tile_cache.read_text())]
        print(f"    {len(assets)} tiles (cached)")
    else:
        print(f"  Searching NAIP tiles ({epoch.name}) …")
        naip = NAIPImagerySource()
        assets = list(naip.assets_for_aoi(aoi, epoch.name))
        print(f"    {len(assets)} tiles")
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
        print(f"    tile list cached → {tile_cache.name}")

    if not assets:
        print(f"  No tiles found for epoch {epoch.name!r}, skipping.")
        return {"train": {"pos": 0, "neg": 0}, "val": {"pos": 0, "neg": 0}}

    # Use a per-epoch seed so each epoch gets an independent shuffle.
    random.seed(SEED + seed_offset)
    shuffled = assets[:]
    random.shuffle(shuffled)
    split_at = int(len(shuffled) * TRAIN_FRAC)

    tasks = [
        {
            # Store unsigned base URI; workers re-sign just before downloading
            # so long-running jobs never hit an expired SAS token.
            "uri": urlparse(asset.uri)._replace(query="", fragment="").geturl(),
            "split": "train" if rank < split_at else "val",
            "dataset_dir": str(dataset_dir),
            "raw_tile_dir": str(raw_tile_dir),
            "patch_size": config.tiling.tile_size_px,
            "stride": config.tiling.stride_px,
            "neg_sample_rate": NEG_SAMPLE_RATE,
            "warehouse_class_id": WAREHOUSE_CLASS_ID,
        }
        for rank, asset in enumerate(shuffled)
    ]

    n_workers = min(
        int(os.environ.get("TILING_WORKERS", os.cpu_count() or 1)),
        len(tasks),
    )
    stats: dict[str, dict[str, int]] = {"train": {"pos": 0, "neg": 0}, "val": {"pos": 0, "neg": 0}}

    print(
        f"  Slicing {len(tasks)} tiles into "
        f"{config.tiling.tile_size_px}×{config.tiling.tile_size_px} patches "
        f"(stride {config.tiling.stride_px}, {n_workers} workers) …"
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
                tqdm.write(f"    ERROR {tile_name}: {exc}")
                continue
            if tile_result["error"]:
                tqdm.write(f"    SKIP {tile_result['tile_name']}: {tile_result['error']}")
                continue
            split = tile_result["split"]
            stats[split]["pos"] += tile_result["pos"]
            stats[split]["neg"] += tile_result["neg"]

    return stats


def main(config_path: Path) -> None:
    config = load_config(config_path)
    print(f"Project : {config.project_name}")
    print(f"AOI     : {config.aoi.name}  {config.aoi.bbox}")
    print(f"Epochs  : {', '.join(e.name for e in config.epochs)}")

    dataset_dir = config.workspace / "training"
    raw_tile_dir = dataset_dir / "raw_tiles"
    raw_tile_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_stats: dict[str, dict[str, int]] = {"train": {"pos": 0, "neg": 0}, "val": {"pos": 0, "neg": 0}}

    for i, epoch in enumerate(config.epochs):
        labels_path = config.workspace / f"labeled_footprints_{epoch.name}.parquet"
        if not labels_path.exists():
            print(
                f"\nSkipping epoch {epoch.name!r} — {labels_path.name} not found "
                f"(run label_prototype_data.py first)"
            )
            continue

        print(f"\nEpoch {epoch.name}  ({epoch.start_date} → {epoch.end_date})")
        gdf = gpd.read_parquet(labels_path)
        n_warehouse = (gdf["label"] == "warehouse").sum()
        print(f"  {n_warehouse:,} warehouse buildings")

        epoch_stats = _process_epoch(epoch, config, dataset_dir, raw_tile_dir, labels_path, seed_offset=i)

        for split in ("train", "val"):
            total_stats[split]["pos"] += epoch_stats[split]["pos"]
            total_stats[split]["neg"] += epoch_stats[split]["neg"]

    # ── Write dataset.yaml ────────────────────────────────────────────────
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
        p, n = total_stats[split]["pos"], total_stats[split]["neg"]
        total += p + n
        print(f"  {split:5s}  {p:4d} positive  {n:4d} negative")
    print(f"  total  {total} patches")
    print(f"\ndataset.yaml → {yaml_path}")
    print("Next step:")
    print("  python scripts/train_warehouse_detector.py", config.workspace)


if __name__ == "__main__":
    cfg = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("configs/reno_sparks_demo.json")
    main(cfg)
