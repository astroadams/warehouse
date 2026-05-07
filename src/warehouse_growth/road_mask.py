from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from shapely.geometry import GeometryCollection, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from warehouse_growth.data_sources import VectorFeature

if TYPE_CHECKING:
    from warehouse_growth.config import RoadMaskConfig


def build_road_mask(
    roads: Iterable[VectorFeature],
    buffer_distance: float,
    clip_geometry: BaseGeometry | None = None,
) -> BaseGeometry:
    """Build a dissolved road-buffer mask.

    Inputs must already be projected into a meter-based CRS. The function is
    intentionally CRS-agnostic so callers can pick an equal-area/local projection
    suitable for their state or metro AOI.
    """
    buffered = [feature.geometry.buffer(buffer_distance) for feature in roads]
    if not buffered:
        return GeometryCollection()

    mask = unary_union(buffered)
    if clip_geometry is not None:
        mask = mask.intersection(clip_geometry)
    return mask


def aoi_from_road_mask(
    bbox: tuple[float, float, float, float],
    road_mask_config: "RoadMaskConfig",
    road_source: Any,
) -> BaseGeometry:
    """Return an AOI geometry built by buffering roads within *bbox*.

    Road features are fetched from *road_source*, projected to the UTM zone
    covering the bbox centroid, buffered, dissolved, and reprojected to
    EPSG:4326. The result is clipped to *bbox*.

    Falls back to the full bbox box if no roads are found or the mask is empty.
    """
    from pyproj import CRS, Transformer
    from shapely.ops import transform

    aoi_box = box(*bbox)

    classes = list(road_mask_config.road_classes)
    if road_mask_config.include_rail:
        classes.append("rail")
    if road_mask_config.include_ports:
        classes.append("port")

    roads = list(road_source.roads_for_aoi(aoi_box, classes))
    if not roads:
        return aoi_box

    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    zone = int((cx + 180) / 6) + 1
    epsg = 32600 + zone if cy >= 0 else 32700 + zone

    fwd = Transformer.from_crs(4326, epsg, always_xy=True).transform
    inv = Transformer.from_crs(epsg, 4326, always_xy=True).transform

    utm_roads = [
        VectorFeature(geometry=transform(fwd, f.geometry), properties=f.properties)
        for f in roads
    ]
    mask_utm = build_road_mask(
        utm_roads,
        road_mask_config.buffer_meters,
        clip_geometry=transform(fwd, aoi_box),
    )

    if mask_utm.is_empty:
        return aoi_box

    return transform(inv, mask_utm)

