"""Tests for label_footprints_duckdb — DuckDB out-of-core spatial join."""
from __future__ import annotations

import pytest
import geopandas as gpd
from shapely.geometry import box

from warehouse_growth.labels import (
    BuildingLabel,
    label_footprints_duckdb,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_footprints_parquet(tmp_path, polygons):
    """Write a list of Shapely geometries as a GeoParquet footprint file."""
    path = tmp_path / "footprints.parquet"
    gdf = gpd.GeoDataFrame({"geometry": polygons}, crs="EPSG:4326")
    gdf.to_parquet(path)
    return path


def _make_osm_parquet(tmp_path, rows):
    """Write [{geometry, building, ...}] rows as a GeoParquet OSM tag file."""
    path = tmp_path / "osm_tags.parquet"
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    gdf.to_parquet(path)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_warehouse_match(tmp_path):
    fp = _make_footprints_parquet(tmp_path, [box(0, 0, 1, 1)])
    osm = _make_osm_parquet(tmp_path, [{"geometry": box(0, 0, 1, 1), "building": "warehouse"}])

    result = label_footprints_duckdb(fp, osm)

    assert len(result) == 1
    assert result[0].label is BuildingLabel.WAREHOUSE


def test_non_warehouse_match(tmp_path):
    fp = _make_footprints_parquet(tmp_path, [box(0, 0, 1, 1)])
    osm = _make_osm_parquet(tmp_path, [{"geometry": box(0, 0, 1, 1), "building": "house"}])

    result = label_footprints_duckdb(fp, osm)

    assert len(result) == 1
    assert result[0].label is BuildingLabel.NON_WAREHOUSE


def test_no_osm_match_is_ambiguous(tmp_path):
    # Footprint at (0,0)–(1,1), OSM feature far away at (10,10)–(11,11)
    fp = _make_footprints_parquet(tmp_path, [box(0, 0, 1, 1)])
    osm = _make_osm_parquet(tmp_path, [{"geometry": box(10, 10, 11, 11), "building": "warehouse"}])

    result = label_footprints_duckdb(fp, osm)

    assert len(result) == 1
    assert result[0].label is BuildingLabel.AMBIGUOUS


def test_empty_osm_all_ambiguous(tmp_path):
    fp = _make_footprints_parquet(tmp_path, [box(0, 0, 1, 1), box(5, 5, 6, 6)])
    osm = _make_osm_parquet(tmp_path, [{"geometry": box(100, 100, 101, 101), "building": "house"}])
    # Both footprints are out of range of the single OSM feature
    result = label_footprints_duckdb(fp, osm)

    assert len(result) == 2
    assert all(r.label is BuildingLabel.AMBIGUOUS for r in result)


def test_multiple_matches_picks_best_overlap(tmp_path):
    # Footprint covers (0,0)–(4,4).
    # OSM1: warehouse, covers (0,0)–(4,4) — full overlap (area 16)
    # OSM2: house, covers (3,3)–(4,4)    — small overlap (area 1)
    # Best match is OSM1 → WAREHOUSE
    fp = _make_footprints_parquet(tmp_path, [box(0, 0, 4, 4)])
    osm = _make_osm_parquet(tmp_path, [
        {"geometry": box(0, 0, 4, 4), "building": "warehouse"},
        {"geometry": box(3, 3, 4, 4), "building": "house"},
    ])

    result = label_footprints_duckdb(fp, osm)

    assert len(result) == 1
    assert result[0].label is BuildingLabel.WAREHOUSE


def test_multiple_matches_picks_best_overlap_non_warehouse(tmp_path):
    # Same layout but larger house footprint → NON_WAREHOUSE wins
    fp = _make_footprints_parquet(tmp_path, [box(0, 0, 4, 4)])
    osm = _make_osm_parquet(tmp_path, [
        {"geometry": box(0, 0, 1, 1), "building": "warehouse"},  # area 1
        {"geometry": box(0, 0, 4, 4), "building": "house"},       # area 16
    ])

    result = label_footprints_duckdb(fp, osm)

    assert len(result) == 1
    assert result[0].label is BuildingLabel.NON_WAREHOUSE


def test_epoch_is_propagated(tmp_path):
    fp = _make_footprints_parquet(tmp_path, [box(0, 0, 1, 1)])
    osm = _make_osm_parquet(tmp_path, [{"geometry": box(0, 0, 1, 1), "building": "warehouse"}])

    result = label_footprints_duckdb(fp, osm, epoch="2022")

    assert result[0].epoch == "2022"


def test_multiple_footprints_independent(tmp_path):
    # fp0 → warehouse, fp1 → house, fp2 → ambiguous (no match)
    fp = _make_footprints_parquet(tmp_path, [
        box(0, 0, 1, 1),
        box(5, 5, 6, 6),
        box(20, 20, 21, 21),
    ])
    osm = _make_osm_parquet(tmp_path, [
        {"geometry": box(0, 0, 1, 1), "building": "warehouse"},
        {"geometry": box(5, 5, 6, 6), "building": "church"},
    ])

    result = label_footprints_duckdb(fp, osm)

    assert len(result) == 3
    labels = [r.label for r in result]
    assert BuildingLabel.WAREHOUSE in labels
    assert BuildingLabel.NON_WAREHOUSE in labels
    assert BuildingLabel.AMBIGUOUS in labels


def test_ambiguous_building_tag(tmp_path):
    fp = _make_footprints_parquet(tmp_path, [box(0, 0, 1, 1)])
    osm = _make_osm_parquet(tmp_path, [{"geometry": box(0, 0, 1, 1), "building": "industrial"}])

    result = label_footprints_duckdb(fp, osm)

    assert result[0].label is BuildingLabel.AMBIGUOUS


def test_geometry_preserved(tmp_path):
    poly = box(1.5, 2.5, 3.5, 4.5)
    fp = _make_footprints_parquet(tmp_path, [poly])
    osm = _make_osm_parquet(tmp_path, [{"geometry": box(1.5, 2.5, 3.5, 4.5), "building": "warehouse"}])

    result = label_footprints_duckdb(fp, osm)

    # Geometry should round-trip through WKB with sufficient precision
    assert result[0].geometry.bounds == pytest.approx(poly.bounds, abs=1e-9)
