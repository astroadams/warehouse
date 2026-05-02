#!/usr/bin/env python3
"""Download prototype data for the Reno-Sparks, NV demo.

Exercises all three data adapters against the I-80 warehouse corridor AOI and
writes cached outputs to a workspace directory.

Usage:
    python scripts/download_prototype_data.py [workspace_dir]

Default workspace: ./runs/reno_sparks_demo
"""
from __future__ import annotations

import sys
from pathlib import Path

from shapely.geometry import box

# Reno-Sparks metropolitan area: ~50 km × 30 km covering the I-80 warehouse
# corridor east of downtown, the South Meadows logistics park, and Sparks.
RENO_AOI = box(-120.05, 39.40, -119.45, 39.70)

NAIP_EPOCHS = ["2019", "2022"]  # two epochs for change-detection prototype


def download_msft_footprints(workspace: Path) -> None:
    from warehouse_growth.adapters import MicrosoftFootprintSource

    cache_path = workspace / "msft_buildings_reno.parquet"
    if cache_path.exists():
        print(f"[msft] Using cached footprints: {cache_path}")
        source = MicrosoftFootprintSource(cache_path)
    else:
        print("[msft] Querying Overture Maps for Nevada footprints …")
        source = MicrosoftFootprintSource.build_cache(RENO_AOI, cache_path)

    footprints = list(source.footprints_for_aoi(RENO_AOI))
    print(f"[msft] {len(footprints):,} building footprints in AOI\n")


def download_osm_tags(workspace: Path) -> None:
    import geopandas as gpd
    from warehouse_growth.adapters import OverpassTagSource

    cache_path = workspace / "osm_tags_reno.parquet"
    if cache_path.exists():
        print(f"[osm]  Using cached OSM tags: {cache_path}")
        gdf = gpd.read_parquet(cache_path)
        tag_series = gdf.get("building")
    else:
        print("[osm]  Querying Overpass API for buildings in AOI …")
        source = OverpassTagSource()
        tag_list = list(source.tags_for_aoi(RENO_AOI))

        gdf = gpd.GeoDataFrame(
            [{"geometry": t.geometry, **t.properties} for t in tag_list],
            crs="EPSG:4326",
        )
        gdf.to_parquet(cache_path)
        tag_series = gdf.get("building")

    print(f"[osm]  {len(gdf):,} building features")
    if tag_series is not None:
        print("[osm]  Top building= tags:")
        for tag, count in tag_series.value_counts().head(10).items():
            print(f"         {tag:<30s} {count:>5,}")
    print()


def list_naip_assets() -> None:
    from warehouse_growth.adapters import NAIPImagerySource

    source = NAIPImagerySource()
    for epoch in NAIP_EPOCHS:
        print(f"[naip] Searching STAC for epoch {epoch} …")
        assets = list(source.assets_for_aoi(RENO_AOI, epoch))
        print(f"[naip] {len(assets)} tile(s) found for {epoch}")
        for a in assets[:4]:
            uri_short = a.uri if len(a.uri) <= 80 else a.uri[:77] + "…"
            print(f"         {a.capture_date}  crs={a.crs}  {uri_short}")
        if len(assets) > 4:
            print(f"         … and {len(assets) - 4} more")
        print()


def main(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    print(f"Workspace: {workspace.resolve()}\n")

    download_msft_footprints(workspace)
    download_osm_tags(workspace)
    list_naip_assets()

    print("Done.  Run the labeling step next:")
    print("  python scripts/label_prototype_data.py")


if __name__ == "__main__":
    ws = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/reno_sparks_demo")
    main(ws)
