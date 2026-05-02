#!/usr/bin/env python3
"""Spatial-join Microsoft footprints with OSM tags and write labeled GeoParquet.

Reads the two caches produced by download_prototype_data.py and runs
label_footprints() to assign WAREHOUSE / NON_WAREHOUSE / AMBIGUOUS_INDUSTRIAL
to each of the 187k Reno-Sparks building footprints.

Usage:
    python scripts/label_prototype_data.py [workspace_dir]

Default workspace: ./runs/reno_sparks_demo
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from warehouse_growth.data_sources import VectorFeature
from warehouse_growth.labels import BuildingLabel, filter_trainable_labels, label_footprints

RENO_AOI = box(-120.05, 39.40, -119.45, 39.70)
EPOCH = "2022"


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


def main(workspace: Path) -> None:
    fp_path = workspace / "msft_buildings_reno.parquet"
    osm_path = workspace / "osm_tags_reno.parquet"

    for p in (fp_path, osm_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing cache file — run download_prototype_data.py first: {p}")

    print(f"Loading footprints from {fp_path.name} …")
    footprints = load_features(fp_path)
    print(f"  {len(footprints):,} footprints")

    print(f"Loading OSM tags from {osm_path.name} …")
    tags = load_features(osm_path)
    print(f"  {len(tags):,} OSM buildings\n")

    instances = label_footprints(footprints, tags, epoch=EPOCH)

    output_path = workspace / f"labeled_footprints_{EPOCH}.parquet"
    gdf = gpd.GeoDataFrame(
        [{"geometry": inst.geometry, "label": inst.label.value} for inst in instances],
        crs="EPSG:4326",
    )
    gdf.to_parquet(output_path)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\nLabel distribution:")
    for label, count in gdf["label"].value_counts().items():
        pct = 100 * count / len(gdf)
        print(f"  {label:<25s}  {count:>7,}  ({pct:.1f}%)")

    trainable = filter_trainable_labels(instances)
    warehouse = sum(1 for i in trainable if i.label is BuildingLabel.WAREHOUSE)
    print(f"\nTrainable (excl. ambiguous):  {len(trainable):,}")
    print(f"  Warehouse:      {warehouse:>7,}")
    print(f"  Non-warehouse:  {len(trainable) - warehouse:>7,}")
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    ws = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/reno_sparks_demo")
    main(ws)
