from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from shapely.strtree import STRtree
from tqdm import tqdm

if TYPE_CHECKING:
    from warehouse_growth.data_sources import VectorFeature


class BuildingLabel(str, Enum):
    WAREHOUSE = "warehouse"
    NON_WAREHOUSE = "non_warehouse"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class BuildingInstance:
    geometry: Any
    label: BuildingLabel
    source_id: str | None = None
    epoch: str | None = None


# OSM `building=` values that map to each label class.
_WAREHOUSE_TAGS = frozenset({"warehouse", "logistics", "distribution_center", "storage"})
# Includes industrial types that could be warehouses, plus unspecified tags ("yes", "")
# that carry no type information — none are usable as confident negatives.
_AMBIGUOUS_TAGS = frozenset({
    "industrial", "manufacture", "factory", "works", "shed",
    "yes", "",
})


def label_from_osm_tags(tags: dict) -> BuildingLabel:
    """Map OSM building tags to a BuildingLabel.

    WAREHOUSE for known warehouse types, AMBIGUOUS for industrial/unspecified types,
    NON_WAREHOUSE for everything else (house, church, school, etc.).
    """
    building = tags.get("building", "").lower().strip()
    if building in _WAREHOUSE_TAGS:
        return BuildingLabel.WAREHOUSE
    if building in _AMBIGUOUS_TAGS:
        return BuildingLabel.AMBIGUOUS
    return BuildingLabel.NON_WAREHOUSE


def label_footprints(
    footprints: Iterable[VectorFeature],
    tags: Iterable[VectorFeature],
    epoch: str | None = None,
) -> list[BuildingInstance]:
    """Assign labels to Microsoft footprints by spatial join with OSM tag features.

    Each footprint is matched to the OSM feature with the greatest overlap area.
    Footprints with no OSM match are labelled AMBIGUOUS (building type unknown).

    Uses shapely 2.0's vectorised bulk STRtree query so the spatial index is
    traversed once for all footprints rather than once per footprint.
    """
    tag_list = list(tags)
    fp_list = list(footprints)

    if not tag_list:
        return [
            BuildingInstance(geometry=fp.geometry, label=BuildingLabel.AMBIGUOUS, epoch=epoch)
            for fp in fp_list
        ]

    tag_geoms = [t.geometry for t in tag_list]
    tree = STRtree(tag_geoms)

    # Single vectorised call returns (fp_indices, tag_indices) for every
    # intersecting pair — equivalent to a spatial join.
    fp_geoms = [fp.geometry for fp in fp_list]
    fp_idxs, tag_idxs = tree.query(fp_geoms, predicate="intersects")

    matches: dict[int, list[int]] = defaultdict(list)
    for fp_i, tag_i in zip(fp_idxs.tolist(), tag_idxs.tolist()):
        matches[fp_i].append(tag_i)

    instances: list[BuildingInstance] = []
    for i, fp in enumerate(tqdm(fp_list, desc="Labelling footprints", unit=" fp", leave=True)):
        candidates = matches.get(i)
        if not candidates:
            label = BuildingLabel.AMBIGUOUS
        elif len(candidates) == 1:
            label = label_from_osm_tags(tag_list[candidates[0]].properties)
        else:
            best = max(candidates, key=lambda j: fp.geometry.intersection(tag_geoms[j]).area)
            label = label_from_osm_tags(tag_list[best].properties)
        instances.append(BuildingInstance(geometry=fp.geometry, label=label, epoch=epoch))
    return instances


def filter_trainable_labels(instances: list[BuildingInstance]) -> list[BuildingInstance]:
    """Drop ambiguous instances from binary warehouse training sets."""
    return [item for item in instances if item.label is not BuildingLabel.AMBIGUOUS]


def label_footprints_duckdb(
    footprints_path: Path,
    osm_path: Path,
    epoch: str | None = None,
) -> list[BuildingInstance]:
    """Assign labels to footprints by spatial join with OSM features via DuckDB.

    Out-of-core alternative to ``label_footprints`` that reads GeoParquet files
    directly without loading all geometries into Python memory. Handles ~30M
    buildings with constant memory by pushing the spatial join into DuckDB.

    Each footprint is matched to the OSM feature with the greatest overlap area.
    Footprints with no OSM match are labelled AMBIGUOUS (building type unknown).
    """
    import duckdb
    from shapely import from_wkb

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")

    fp_str = str(footprints_path)
    osm_str = str(osm_path)

    osm_count = con.execute("SELECT COUNT(*) FROM read_parquet(?)", [osm_str]).fetchone()[0]

    if osm_count == 0:
        rows = con.execute(
            "SELECT ST_AsWKB(geometry) AS geom FROM read_parquet(?)", [fp_str]
        ).df()
        print(f"  No OSM features — all {len(rows):,} footprints labelled AMBIGUOUS")
        return [
            BuildingInstance(
                geometry=from_wkb(bytes(row["geom"])),
                label=BuildingLabel.AMBIGUOUS,
                epoch=epoch,
            )
            for _, row in rows.iterrows()
        ]

    fp_count = con.execute("SELECT COUNT(*) FROM read_parquet(?)", [fp_str]).fetchone()[0]
    print(f"  Spatial join: {fp_count:,} footprints × {osm_count:,} OSM features …")

    # LEFT JOIN so footprints with no OSM match produce a single NULL row.
    # ROW_NUMBER picks the best match (highest overlap area) per footprint;
    # NULLS LAST ensures the no-match NULL row sorts after any real matches,
    # and as the only row for unmatched footprints it still gets rn = 1.
    sql = """
        WITH fp AS (
            SELECT
                row_number() OVER () AS _fp_idx,
                geometry                AS _fp_geom
            FROM read_parquet(?)
        ),
        osm AS (
            SELECT
                geometry                       AS _osm_geom,
                COALESCE(building, '') AS building
            FROM read_parquet(?)
        ),
        intersections AS (
            SELECT
                fp._fp_idx,
                fp._fp_geom,
                osm.building,
                ST_Area(ST_Intersection(fp._fp_geom, osm._osm_geom)) AS overlap_area
            FROM fp
            LEFT JOIN osm ON ST_Intersects(fp._fp_geom, osm._osm_geom)
        ),
        ranked AS (
            SELECT
                _fp_idx,
                _fp_geom,
                building,
                ROW_NUMBER() OVER (
                    PARTITION BY _fp_idx
                    ORDER BY overlap_area DESC NULLS LAST
                ) AS rn
            FROM intersections
        )
        SELECT
            _fp_idx,
            ST_AsWKB(_fp_geom)         AS _fp_geom_wkb,
            COALESCE(building, '') AS building
        FROM ranked
        WHERE rn = 1
        ORDER BY _fp_idx
    """

    df = con.execute(sql, [fp_str, osm_str]).df()
    print(f"  Labelled {len(df):,} footprints")

    return [
        BuildingInstance(
            geometry=from_wkb(bytes(row["_fp_geom_wkb"])),
            label=label_from_osm_tags({"building": row["building"]}),
            epoch=epoch,
        )
        for _, row in df.iterrows()
    ]
