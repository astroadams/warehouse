from __future__ import annotations

from pathlib import Path

from warehouse_growth.models.base import BuildingDetector, Detection


def _px_to_geo(geom, rasterio_transform):
    """Apply a rasterio Affine transform to convert pixel → geographic coords.

    Rasterio convention: geo_x = c + a*col + b*row
                         geo_y = f + d*col + e*row
    Shapely affine_transform takes [a, b, d, e, xoff, yoff] where
    (x, y) → (a*x + b*y + xoff, d*x + e*y + yoff), mapping col→x and row→y.
    """
    from shapely.affinity import affine_transform

    t = rasterio_transform
    return affine_transform(geom, [t.a, t.b, t.d, t.e, t.c, t.f])


class YoloBuildingDetector(BuildingDetector):
    """Warehouse detector backed by a fine-tuned Ultralytics YOLO model.

    Supports detect (bbox), segment (polygon), and obb (oriented bbox) tasks —
    inferred automatically from the loaded checkpoint.  Output geometries are in
    the tile's native CRS (read from the GeoTIFF via rasterio).

    The ultralytics import is delayed so the base package can be installed
    without ML dependencies. Use `pip install -e ".[models]"` to add it.
    """

    def __init__(self, checkpoint: str | Path, confidence_threshold: float = 0.25) -> None:
        self.checkpoint = str(checkpoint)
        self.confidence_threshold = confidence_threshold
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.checkpoint)
        return self._model

    def predict_tile(self, tile_path: Path) -> list[Detection]:
        try:
            import rasterio
            from shapely.geometry import Polygon, box
        except ImportError as e:
            raise ImportError(
                "rasterio and shapely are required — install via conda or pip install rasterio warehouse-growth[geo]"
            ) from e

        with rasterio.open(tile_path) as src:
            geo_transform = src.transform
            # NAIP is 4-band RGBIR; YOLO expects 3-channel uint8 (H, W, 3).
            bands = list(range(1, min(src.count, 3) + 1))
            img = src.read(bands).transpose(1, 2, 0)

        results = self.model(img, conf=self.confidence_threshold, verbose=False)

        tile_id = tile_path.stem
        detections: list[Detection] = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            if result.masks is not None:
                # Segmentation task — polygon vertices already in pixel coords.
                for poly_pts, conf, cls in zip(result.masks.xy, confs, classes):
                    if len(poly_pts) < 3:
                        continue
                    geom = _px_to_geo(Polygon(poly_pts), geo_transform)
                    detections.append(Detection(
                        geometry=geom, score=float(conf),
                        class_name=self.model.names[cls], tile_id=tile_id,
                    ))

            elif result.obb is not None:
                # OBB task — four corner points, shape (N, 4, 2).
                for pts, conf, cls in zip(result.obb.xyxyxyxy.cpu().numpy(), confs, classes):
                    geom = _px_to_geo(Polygon(pts), geo_transform)
                    detections.append(Detection(
                        geometry=geom, score=float(conf),
                        class_name=self.model.names[cls], tile_id=tile_id,
                    ))

            else:
                # Detect task — axis-aligned bounding boxes.
                for (x1, y1, x2, y2), conf, cls in zip(
                    result.boxes.xyxy.cpu().numpy(), confs, classes
                ):
                    geom = _px_to_geo(box(x1, y1, x2, y2), geo_transform)
                    detections.append(Detection(
                        geometry=geom, score=float(conf),
                        class_name=self.model.names[cls], tile_id=tile_id,
                    ))

        return detections
