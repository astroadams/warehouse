from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol


@dataclass(frozen=True)
class ImageryAsset:
    uri: str
    epoch: str
    capture_date: str | None = None
    crs: str | None = None
    resolution_meters: float | None = None


@dataclass(frozen=True)
class VectorFeature:
    geometry: Any
    properties: dict


class ImagerySource(Protocol):
    def assets_for_aoi(self, aoi: Any, epoch: str) -> Iterable[ImageryAsset]:
        """Return imagery assets intersecting an AOI for an epoch."""


class RoadSource(Protocol):
    def roads_for_aoi(self, aoi: Any, classes: list[str]) -> Iterable[VectorFeature]:
        """Return road features intersecting an AOI."""


class FootprintSource(Protocol):
    def footprints_for_aoi(self, aoi: Any) -> Iterable[VectorFeature]:
        """Return existing building footprint features intersecting an AOI."""


class TagSource(Protocol):
    def tags_for_aoi(self, aoi: Any) -> Iterable[VectorFeature]:
        """Return building features with OSM-style tags intersecting an AOI."""


class LocalGeoPackageRoadSource:
    """Placeholder adapter for local road layers.

    This keeps the core pipeline independent from whether roads come from OSM,
    Overture Maps, Census TIGER, or a state DOT layer.
    """

    def __init__(self, path: Path, layer: str | None = None) -> None:
        self.path = path
        self.layer = layer

    def roads_for_aoi(self, aoi: Any, classes: list[str]) -> Iterable[VectorFeature]:
        raise NotImplementedError("Install geo extras and implement provider-specific loading.")
