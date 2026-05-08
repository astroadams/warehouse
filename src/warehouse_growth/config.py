from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass(frozen=True)
class AoiConfig:
    name: str
    bbox: tuple[float, float, float, float]
    crs: str = "EPSG:4326"


@dataclass(frozen=True)
class EpochConfig:
    name: str
    start_date: date
    end_date: date
    imagery_uri: str


@dataclass(frozen=True)
class RoadMaskConfig:
    road_classes: list[str] = field(default_factory=lambda: ["motorway", "trunk", "primary"])
    buffer_meters: float = 2000
    include_rail: bool = True
    include_ports: bool = False


@dataclass(frozen=True)
class TilingConfig:
    tile_size_px: int = 1024
    stride_px: int = 768
    resolution_meters: float = 0.5
    cache_tiles: bool = True


@dataclass(frozen=True)
class DetectorConfig:
    type: Literal["yolo", "clay", "olmoearth"] = "yolo"
    task: Literal["detect", "segment", "obb"] = "segment"
    checkpoint: str | None = None
    confidence_threshold: float = 0.25


@dataclass(frozen=True)
class ClassifierConfig:
    enabled: bool = True
    checkpoint: str | None = None
    threshold: float = 0.5


@dataclass(frozen=True)
class ProjectConfig:
    project_name: str
    workspace: Path
    aoi: AoiConfig
    epochs: list[EpochConfig]
    road_mask: RoadMaskConfig | None = None
    tiling: TilingConfig = field(default_factory=TilingConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            raw = yaml.safe_load(stream)
        else:
            raw = json.load(stream)
    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> ProjectConfig:
    aoi = raw["aoi"]
    epochs = raw["epochs"]
    return ProjectConfig(
        project_name=raw["project_name"],
        workspace=Path(raw["workspace"]),
        aoi=AoiConfig(name=aoi["name"], bbox=tuple(aoi["bbox"]), crs=aoi.get("crs", "EPSG:4326")),
        epochs=[
            EpochConfig(
                name=item["name"],
                start_date=date.fromisoformat(item["start_date"]),
                end_date=date.fromisoformat(item["end_date"]),
                imagery_uri=item["imagery_uri"],
            )
            for item in epochs
        ],
        road_mask=RoadMaskConfig(**raw["road_mask"]) if "road_mask" in raw else None,
        tiling=TilingConfig(**raw.get("tiling", {})),
        detector=DetectorConfig(**raw.get("detector", {})),
        classifier=ClassifierConfig(**raw.get("classifier", {})),
    )
