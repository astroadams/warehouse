from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Detection:
    geometry: Any
    score: float
    class_name: str
    tile_id: str | None = None


class BuildingDetector:
    def predict_tile(self, tile_path: Path) -> list[Detection]:
        raise NotImplementedError


class WarehouseClassifier:
    def predict(self, detections: list[Detection]) -> list[Detection]:
        raise NotImplementedError
