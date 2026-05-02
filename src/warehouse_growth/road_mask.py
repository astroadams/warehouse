from __future__ import annotations

from collections.abc import Iterable

from shapely.geometry import GeometryCollection
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from warehouse_growth.data_sources import VectorFeature


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

