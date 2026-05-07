#!/usr/bin/env python3
"""Download source data for a warehouse-growth project AOI.

Queries all three data adapters (Microsoft building footprints, OSM tags, NAIP
imagery index) for the AOI and epochs defined in the project config, and writes
cached outputs to the config's workspace directory.

Usage:
    python scripts/download_prototype_data.py configs/reno_sparks_demo.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from shapely.geometry import box, mapping, shape

from warehouse_growth.config import load_config


def compute_road_mask_aoi(config, workspace: Path):
    """Fetch OSM roads, build road-mask AOI, cache to road_mask_aoi.geojson."""
    cache_path = workspace / "road_mask_aoi.geojson"
    if cache_path.exists():
        print(f"[road] Using cached road-mask AOI: {cache_path.name}")
        with open(cache_path) as f:
            return shape(json.load(f)["geometry"])

    print("[road] Fetching OSM roads to build road-mask AOI …")
    from warehouse_growth.adapters import OverpassRoadSource
    from warehouse_growth.road_mask import aoi_from_road_mask

    road_source = OverpassRoadSource()
    mask_geom = aoi_from_road_mask(config.aoi.bbox, config.road_mask, road_source)
    with open(cache_path, "w") as f:
        json.dump({"type": "Feature", "geometry": mapping(mask_geom), "properties": {}}, f)
    print(f"[road] Road-mask AOI cached → {cache_path.name}\n")
    return mask_geom


def download_msft_footprints(aoi, workspace: Path) -> None:
    from warehouse_growth.adapters import MicrosoftFootprintSource

    cache_path = workspace / "msft_buildings.parquet"
    if cache_path.exists():
        print(f"[msft] Using cached footprints: {cache_path.name}")
        source = MicrosoftFootprintSource(cache_path)
    else:
        print("[msft] Querying Overture Maps for building footprints …")
        source = MicrosoftFootprintSource.build_cache(aoi, cache_path)

    footprints = list(source.footprints_for_aoi(aoi))
    print(f"[msft] {len(footprints):,} building footprints in AOI\n")


def download_osm_tags(aoi, workspace: Path) -> None:
    import geopandas as gpd
    from warehouse_growth.adapters import OverpassTagSource

    cache_path = workspace / "osm_tags.parquet"
    if cache_path.exists():
        print(f"[osm]  Using cached OSM tags: {cache_path.name}")
        gdf = gpd.read_parquet(cache_path)
        tag_series = gdf.get("building")
    else:
        print("[osm]  Querying Overpass API for buildings in AOI …")
        source = OverpassTagSource()
        tag_list = list(source.tags_for_aoi(aoi))

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


def list_naip_assets(aoi, epoch_names: list[str]) -> None:
    from warehouse_growth.adapters import NAIPImagerySource

    source = NAIPImagerySource()
    for epoch in epoch_names:
        print(f"[naip] Searching STAC for epoch {epoch} …")
        assets = list(source.assets_for_aoi(aoi, epoch))
        print(f"[naip] {len(assets)} tile(s) found for {epoch}")
        for a in assets[:4]:
            uri_short = a.uri if len(a.uri) <= 80 else a.uri[:77] + "…"
            print(f"         {a.capture_date}  crs={a.crs}  {uri_short}")
        if len(assets) > 4:
            print(f"         … and {len(assets) - 4} more")
        print()


def main(config_path: Path) -> None:
    config = load_config(config_path)
    epoch_names = [e.name for e in config.epochs]

    config.workspace.mkdir(parents=True, exist_ok=True)
    print(f"Project  : {config.project_name}")
    print(f"AOI      : {config.aoi.name}  {config.aoi.bbox}")
    print(f"Epochs   : {', '.join(epoch_names)}")
    print(f"Workspace: {config.workspace.resolve()}\n")

    if config.road_mask is not None:
        aoi = compute_road_mask_aoi(config, config.workspace)
    else:
        aoi = box(*config.aoi.bbox)

    download_msft_footprints(aoi, config.workspace)
    download_osm_tags(aoi, config.workspace)
    list_naip_assets(aoi, epoch_names)

    print("Done.  Run the labeling step next:")
    print(f"  python scripts/label_prototype_data.py {config_path}")


if __name__ == "__main__":
    cfg = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("configs/reno_sparks_demo.json")
    main(cfg)
