from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree

import duckdb
import geopandas as gpd
from shapely import from_wkb

from warehouse_growth.adapters._progress import run_with_spinner
from warehouse_growth.data_sources import VectorFeature

# Microsoft Global ML Building Footprints are distributed through Overture Maps
# as GeoParquet shards on AWS S3 — publicly accessible, no credentials required.
#
#   s3://overturemaps-us-west-2/release/<version>/theme=buildings/type=building/

_OVERTURE_BUCKET = "overturemaps-us-west-2"
_OVERTURE_S3 = "s3://{bucket}/release/{release}/theme=buildings/type=building/*"
_S3_LIST_URL = "https://{bucket}.s3.us-west-2.amazonaws.com/"


def latest_overture_release() -> str:
    """Return the most recent Overture Maps release tag by listing the S3 bucket."""
    import requests

    resp = requests.get(
        _S3_LIST_URL.format(bucket=_OVERTURE_BUCKET),
        params={"list-type": "2", "prefix": "release/", "delimiter": "/"},
        timeout=15,
    )
    resp.raise_for_status()
    # S3 ListObjectsV2 returns XML; strip the namespace so ElementTree finds tags.
    xml = re.sub(r" xmlns=['\"][^'\"]*['\"]", "", resp.text)
    root = ElementTree.fromstring(xml)
    releases = sorted(
        p.text.removeprefix("release/").rstrip("/")
        for p in root.findall(".//CommonPrefixes/Prefix")
    )
    if not releases:
        raise RuntimeError("No releases found in Overture S3 bucket")
    return releases[-1]  # lexicographic sort works: YYYY-MM-DD.N


class MicrosoftFootprintSource:
    """Building footprints from a local Overture Maps buildings GeoParquet file.

    Use `build_cache()` to download and clip to an AOI once, then construct
    instances normally with the resulting path for subsequent reads.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def footprints_for_aoi(self, aoi: Any) -> Iterable[VectorFeature]:
        gdf = gpd.read_parquet(self.path)
        gdf = gdf[gdf.intersects(aoi)]

        for _, row in gdf.iterrows():
            props = {k: v for k, v in row.items() if k != gdf.geometry.name}
            yield VectorFeature(geometry=row.geometry, properties=dict(props))

    @classmethod
    def build_cache(
        cls,
        aoi: Any,
        output_path: Path,
        *,
        overture_release: str | None = None,
    ) -> "MicrosoftFootprintSource":
        """Query Overture Maps buildings for an AOI and write a GeoParquet cache.

        Pulls from the publicly accessible Overture S3 bucket (no auth needed).
        DuckDB pushes the bbox filter to the Parquet reader so only the relevant
        shards are scanned. A precise shapely intersects pass follows.

        When `overture_release` is None the latest available release is discovered
        automatically by listing the S3 bucket.
        """
        if overture_release is None:
            overture_release = run_with_spinner(
                "Discovering latest Overture release", latest_overture_release
            )

        minx, miny, maxx, maxy = aoi.bounds
        s3_glob = _OVERTURE_S3.format(bucket=_OVERTURE_BUCKET, release=overture_release)

        print(f"Querying Overture Maps release {overture_release} …")
        conn = duckdb.connect()

        run_with_spinner(
            "Loading DuckDB extensions",
            lambda: conn.execute("INSTALL httpfs; LOAD httpfs; SET s3_region='us-west-2';"),
        )

        sql = f"""
            SELECT id, geometry, height, class
            FROM read_parquet('{s3_glob}', hive_partitioning=1)
            WHERE bbox.xmin <= {maxx}
              AND bbox.xmax >= {minx}
              AND bbox.ymin <= {maxy}
              AND bbox.ymax >= {miny}
        """
        try:
            result = run_with_spinner(
                "Scanning Overture parquet shards (may take several minutes)",
                lambda: conn.execute(sql),
            )
            df = result.df()
        except Exception as e:
            raise RuntimeError(
                f"DuckDB query failed for release {overture_release!r} "
                f"at {s3_glob!r}."
            ) from e

        print(f"  {len(df):,} buildings in bbox, applying precise AOI filter …")
        df["geometry"] = from_wkb(df["geometry"].apply(bytes).to_numpy())
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
        gdf = gdf[gdf.intersects(aoi)].reset_index(drop=True)
        print(f"  {len(gdf):,} buildings within AOI")

        if gdf.empty:
            raise ValueError(f"No buildings found within bounds {aoi.bounds}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_parquet(output_path)
        print(f"Saved → {output_path}")
        return cls(output_path)
