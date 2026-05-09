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
from pathlib import Path

from shapely.geometry import box, mapping, shape

from warehouse_growth import provenance
from warehouse_growth.config import load_config


def compute_road_mask_aoi(config, workspace: Path, config_path: Path):
    """Fetch OSM roads, build road-mask AOI, cache to road_mask_aoi.geojson."""
    cache_path = workspace / "road_mask_aoi.geojson"
    if cache_path.exists():
        print(f"[road] Using cached road-mask AOI: {cache_path.name}")
        provenance.check(cache_path)
        with open(cache_path) as f:
            return shape(json.load(f)["geometry"])

    print("[road] Fetching OSM roads to build road-mask AOI …")
    from warehouse_growth.adapters import OverpassRoadSource
    from warehouse_growth.road_mask import aoi_from_road_mask

    road_source = OverpassRoadSource()
    mask_geom = aoi_from_road_mask(config.aoi.bbox, config.road_mask, road_source)
    with open(cache_path, "w") as f:
        json.dump({"type": "Feature", "geometry": mapping(mask_geom), "properties": {}}, f)
    provenance.write(cache_path, config_path)
    print(f"[road] Road-mask AOI cached → {cache_path.name}\n")
    return mask_geom


def download_msft_footprints(aoi, workspace: Path, config_path: Path) -> None:
    from warehouse_growth.adapters import MicrosoftFootprintSource

    cache_path = workspace / "msft_buildings.parquet"
    if cache_path.exists():
        print(f"[msft] Using cached footprints: {cache_path.name}")
        provenance.check(cache_path)
        source = MicrosoftFootprintSource(cache_path)
    else:
        print("[msft] Querying Overture Maps for building footprints …")
        source = MicrosoftFootprintSource.build_cache(aoi, cache_path)
        provenance.write(cache_path, config_path)

    footprints = list(source.footprints_for_aoi(aoi))
    print(f"[msft] {len(footprints):,} building footprints in AOI\n")


def download_osm_tags(
    aoi, workspace: Path, config_path: Path, epochs, osm_epoch_filter: bool = True
) -> None:
    """Download a per-epoch OSM tag snapshot for each epoch in the config.

    Each epoch gets its own ``osm_tags_{epoch.name}.parquet``.  When
    *osm_epoch_filter* is True the Overpass [date:...] setting is used so
    tags reflect OSM state at the epoch's end_date; when False, the current
    live OSM state is returned instead.  Epochs whose cache file already
    exists are skipped.
    """
    import geopandas as gpd
    from warehouse_growth.adapters import OverpassTagSource

    source = OverpassTagSource()

    for epoch in epochs:
        cache_path = workspace / f"osm_tags_{epoch.name}.parquet"
        if cache_path.exists():
            print(f"[osm]  Using cached OSM tags ({epoch.name}): {cache_path.name}")
            provenance.check(cache_path)
            gdf = gpd.read_parquet(cache_path)
        else:
            as_of = epoch.end_date if osm_epoch_filter else None
            date_label = f"at {epoch.end_date}" if as_of else "current state"
            print(f"[osm]  Querying Overpass API for buildings ({date_label}, epoch {epoch.name}) …")
            tag_list = list(source.tags_for_aoi(aoi, as_of_date=as_of))
            gdf = gpd.GeoDataFrame(
                [{"geometry": t.geometry, **t.properties} for t in tag_list],
                crs="EPSG:4326",
            )
            gdf.to_parquet(cache_path)
            provenance.write(cache_path, config_path, epoch=epoch.name)

        tag_series = gdf.get("building")
        print(f"[osm]  {len(gdf):,} building features (epoch {epoch.name})")
        if tag_series is not None:
            print(f"[osm]  Top building= tags (epoch {epoch.name}):")
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
        aoi = compute_road_mask_aoi(config, config.workspace, config_path)
    else:
        aoi = box(*config.aoi.bbox)

    download_msft_footprints(aoi, config.workspace, config_path)
    download_osm_tags(aoi, config.workspace, config_path, config.epochs, config.osm_epoch_filter)
    list_naip_assets(aoi, epoch_names)

    print("Done.  Run the labeling step next:")
    print(f"  python scripts/label_prototype_data.py {config_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", nargs="?", default="configs/reno_sparks_demo.json",
                   type=Path, help="Path to project config JSON/YAML (default: configs/reno_sparks_demo.json)")
    main(p.parse_args().config)
