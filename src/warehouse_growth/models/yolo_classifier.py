from __future__ import annotations

import warnings
from collections import defaultdict
from pathlib import Path

from warehouse_growth.models.base import Detection, WarehouseClassifier


class YoloWarehouseClassifier(WarehouseClassifier):
    """Second-stage classifier: filters building detections to warehouses.

    Wraps a YOLO classify model. Crops each detection's bounding box from its
    source tile, zero-pads to a square, and batches all crops from the same
    tile through the model in one call.

    ``tile_dir`` must point to the directory containing the images that were
    passed to ``YoloBuildingDetector.predict_tile()`` — detections use
    ``tile_id = tile_path.stem`` as the lookup key.

    ultralytics is imported lazily — install with ``uv sync --extra models``.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        tile_dir: str | Path,
        padding_px: int = 32,
        threshold: float = 0.5,
    ) -> None:
        self.checkpoint = str(checkpoint)
        self.tile_dir = Path(tile_dir)
        self.padding_px = padding_px
        self.threshold = threshold
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.checkpoint)
        return self._model

    def predict(self, detections: list[Detection]) -> list[Detection]:
        if not detections:
            return []

        # Group by tile_id so each tile is opened at most once.
        by_tile: dict[str | None, list[tuple[int, Detection]]] = defaultdict(list)
        for i, det in enumerate(detections):
            by_tile[det.tile_id].append((i, det))

        results_map: dict[int, Detection] = {}
        warehouse_idx = self._warehouse_class_idx()

        for tile_id, tile_dets in by_tile.items():
            if tile_id is None:
                for i, det in tile_dets:
                    results_map[i] = det
                continue

            tile_path = self.tile_dir / f"{tile_id}.tif"
            if not tile_path.exists():
                warnings.warn(
                    f"YoloWarehouseClassifier: tile {tile_id!r} not found in {self.tile_dir}; "
                    "keeping all detections from this tile unfiltered.",
                    stacklevel=2,
                )
                for i, det in tile_dets:
                    results_map[i] = det
                continue

            crops = self._read_crops(tile_path, tile_dets)
            valid = [(i, det, crop) for i, det, crop in crops if crop is not None]
            for i, det, crop in crops:
                if crop is None:
                    results_map[i] = det

            if not valid:
                continue

            imgs = [crop for _, _, crop in valid]
            cls_results = self.model(imgs, verbose=False)

            for (i, det, _), cls_result in zip(valid, cls_results):
                probs = cls_result.probs.data.cpu().numpy()
                warehouse_prob = float(probs[warehouse_idx]) if warehouse_idx < len(probs) else 0.0
                if warehouse_prob >= self.threshold:
                    results_map[i] = Detection(
                        geometry=det.geometry,
                        score=warehouse_prob,
                        class_name="warehouse",
                        tile_id=det.tile_id,
                    )

        return [results_map[i] for i in range(len(detections)) if i in results_map]

    def _read_crops(
        self,
        tile_path: Path,
        tile_dets: list[tuple[int, Detection]],
    ) -> list[tuple]:
        import numpy as np
        import rasterio
        from rasterio.windows import Window

        results = []
        with rasterio.open(tile_path) as src:
            tile_transform = src.transform
            bands = list(range(1, min(src.count, 3) + 1))
            inv = ~tile_transform

            for i, det in tile_dets:
                geom = det.geometry
                try:
                    xs, ys = zip(*list(geom.exterior.coords))
                except AttributeError:
                    minx, miny, maxx, maxy = geom.bounds
                    xs, ys = [minx, maxx], [miny, maxy]

                col_vals = []
                row_vals = []
                for gx, gy in zip(xs, ys):
                    col, row = inv * (gx, gy)
                    col_vals.append(col)
                    row_vals.append(row)

                col_min = int(min(col_vals)) - self.padding_px
                col_max = int(max(col_vals)) + self.padding_px
                row_min = int(min(row_vals)) - self.padding_px
                row_max = int(max(row_vals)) + self.padding_px

                win_w = col_max - col_min
                win_h = row_max - row_min
                if win_w <= 0 or win_h <= 0:
                    results.append((i, det, None))
                    continue

                win = Window(col_off=col_min, row_off=row_min, width=win_w, height=win_h)
                crop = src.read(bands, window=win, boundless=True, fill_value=0)  # (C, H, W)

                c, h, w = crop.shape
                side = max(h, w)
                if h != w:
                    pad_h = side - h
                    pad_w = side - w
                    crop = np.pad(
                        crop,
                        ((0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
                        mode="constant",
                        constant_values=0,
                    )

                results.append((i, det, crop.transpose(1, 2, 0)))  # HWC for YOLO classify

        return results

    def _warehouse_class_idx(self) -> int:
        for idx, name in self.model.names.items():
            if name.lower() == "warehouse":
                return idx
        raise ValueError(
            f"'warehouse' class not found in model names: {self.model.names}. "
            "Ensure the classifier was trained with class name 'warehouse'."
        )
