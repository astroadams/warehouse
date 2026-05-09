#!/usr/bin/env python3
"""Spatial-join Microsoft footprints with OSM tags and write labeled GeoParquet.

Reads the caches produced by download_prototype_data.py and runs
label_footprints() to assign WAREHOUSE / NON_WAREHOUSE / AMBIGUOUS
to each building footprint in the AOI.

Each epoch uses its own OSM tag snapshot (osm_tags_{epoch}.parquet) queried at
the epoch's end_date, so labels reflect building types as tagged at the time of
the imagery.  Epochs whose output file already exists are skipped.

Usage:
    python scripts/label_prototype_data.py configs/reno_sparks_demo.json
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from warehouse_growth import provenance
from warehouse_growth.config import load_config
from warehouse_growth.data_sources import VectorFeature
from warehouse_growth.labels import BuildingLabel, filter_trainable_labels, label_footprints


def load_features(path: Path) -> list[VectorFeature]:
    """Load a GeoParquet file as VectorFeature objects."""
    gdf = gpd.read_parquet(path)
    geom_col = gdf.geometry.name
    return [
        VectorFeature(
            geometry=row[geom_col],
            properties={k: v for k, v in row.items() if k != geom_col and pd.notna(v)},
        )
        for row in gdf.to_dict("records")
    ]


def main(config_path: Path) -> None:
    from collections import Counter

    config = load_config(config_path)
    workspace = config.workspace

    fp_path = workspace / "msft_buildings.parquet"
    if not fp_path.exists():
        raise FileNotFoundError(
            f"Missing cache file — run download_prototype_data.py first: {fp_path}"
        )

    print(f"Project : {config.project_name}")
    print(f"Epochs  : {', '.join(e.name for e in config.epochs)}\n")

    print(f"Loading footprints from {fp_path.name} …")
    footprints = load_features(fp_path)
    print(f"  {len(footprints):,} footprints\n")

    for epoch in config.epochs:
        output_path = workspace / f"labeled_footprints_{epoch.name}.parquet"
        if output_path.exists():
            provenance.check(output_path)
            print(f"[{epoch.name}] {output_path.name} already exists — skipping")
            continue

        osm_path = workspace / f"osm_tags_{epoch.name}.parquet"
        if not osm_path.exists():
            raise FileNotFoundError(
                f"Missing OSM cache — run download_prototype_data.py first: {osm_path}"
            )

        print(f"[{epoch.name}] Loading OSM tags from {osm_path.name} …")
        tags = load_features(osm_path)
        print(f"[{epoch.name}]   {len(tags):,} OSM buildings")

        instances = label_footprints(footprints, tags, epoch=epoch.name)

        counts = Counter(inst.label.value for inst in instances)
        print(f"[{epoch.name}] Label distribution:")
        for label, count in sorted(counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / len(instances)
            print(f"[{epoch.name}]   {label:<25s}  {count:>7,}  ({pct:.1f}%)")

        trainable = filter_trainable_labels(instances)
        warehouse = sum(1 for i in trainable if i.label is BuildingLabel.WAREHOUSE)
        print(f"[{epoch.name}] Trainable (excl. ambiguous):  {len(trainable):,}")
        print(f"[{epoch.name}]   Warehouse:      {warehouse:>7,}")
        print(f"[{epoch.name}]   Non-warehouse:  {len(trainable) - warehouse:>7,}")

        gdf = gpd.GeoDataFrame(
            [{"geometry": inst.geometry, "label": inst.label.value} for inst in instances],
            crs="EPSG:4326",
        )
        gdf.to_parquet(output_path)
        provenance.write(output_path, config_path, epoch=epoch.name)
        print(f"[{epoch.name}] Saved → {output_path.name}\n")

    print("Next step:")
    print(f"  uv run python scripts/prepare_training_data.py {config_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", nargs="?", default="configs/reno_sparks_demo.json",
                   type=Path, help="Path to project config JSON/YAML (default: configs/reno_sparks_demo.json)")
    main(p.parse_args().config)
