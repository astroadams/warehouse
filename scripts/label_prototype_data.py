#!/usr/bin/env python3
"""Spatial-join Microsoft footprints with OSM tags and write labeled GeoParquet.

Reads the two caches produced by download_prototype_data.py and runs
label_footprints() to assign WAREHOUSE / NON_WAREHOUSE / AMBIGUOUS
to each building footprint in the AOI.

The spatial join is epoch-agnostic (footprint geometry and OSM tags do not change
per imagery date), so the join runs once and the result is written to a separate
labeled_footprints_{epoch}.parquet for each epoch in the config. Epochs whose
output file already exists are skipped, enabling incremental re-runs.

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
    config = load_config(config_path)
    workspace = config.workspace

    fp_path = workspace / "msft_buildings.parquet"
    osm_path = workspace / "osm_tags.parquet"

    for p in (fp_path, osm_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing cache file — run download_prototype_data.py first: {p}"
            )

    print(f"Project : {config.project_name}")
    print(f"Epochs  : {', '.join(e.name for e in config.epochs)}\n")

    print(f"Loading footprints from {fp_path.name} …")
    footprints = load_features(fp_path)
    print(f"  {len(footprints):,} footprints")

    print(f"Loading OSM tags from {osm_path.name} …")
    tags = load_features(osm_path)
    print(f"  {len(tags):,} OSM buildings\n")

    # Run the spatial join once — labels are identical across epochs since
    # footprint geometry and OSM tags don't vary by imagery date.
    instances = label_footprints(footprints, tags, epoch=None)

    # ── Print label distribution (shown once, applies to all epochs) ───────
    label_values = [inst.label.value for inst in instances]
    from collections import Counter
    counts = Counter(label_values)
    print("\nLabel distribution:")
    for label, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(instances)
        print(f"  {label:<25s}  {count:>7,}  ({pct:.1f}%)")

    trainable = filter_trainable_labels(instances)
    warehouse = sum(1 for i in trainable if i.label is BuildingLabel.WAREHOUSE)
    print(f"\nTrainable (excl. ambiguous):  {len(trainable):,}")
    print(f"  Warehouse:      {warehouse:>7,}")
    print(f"  Non-warehouse:  {len(trainable) - warehouse:>7,}")

    # ── Write one output file per epoch ────────────────────────────────────
    base_gdf = gpd.GeoDataFrame(
        [{"geometry": inst.geometry, "label": inst.label.value} for inst in instances],
        crs="EPSG:4326",
    )

    print()
    for epoch in config.epochs:
        output_path = workspace / f"labeled_footprints_{epoch.name}.parquet"
        if output_path.exists():
            provenance.check(output_path)
            print(f"  {output_path.name} already exists — skipping")
            continue
        base_gdf.to_parquet(output_path)
        provenance.write(output_path, config_path, epoch=epoch.name)
        print(f"  Saved → {output_path.name}")

    print("\nNext step:")
    print(f"  python scripts/prepare_training_data.py {config_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", nargs="?", default="configs/reno_sparks_demo.json",
                   type=Path, help="Path to project config JSON/YAML (default: configs/reno_sparks_demo.json)")
    main(p.parse_args().config)
