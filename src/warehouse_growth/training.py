"""Helpers for preparing YOLO training data from labeled building footprints."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import rasterio
from rasterio.warp import transform as warp_transform
from shapely.geometry import Polygon


def geo_polygon_to_yolo(
    polygon: Any,
    tile_transform,
    tile_crs: str,
    tile_width: int,
    tile_height: int,
    *,
    src_crs: str = "EPSG:4326",
) -> list[float] | None:
    """Convert a geographic polygon to a flat YOLO segment annotation.

    Returns a flat list [x1, y1, x2, y2, …] with coordinates normalised to
    [0, 1] in tile pixel space, or None if the polygon is entirely outside
    the tile or has fewer than 3 valid interior points.

    The polygon is first reprojected from `src_crs` to the tile CRS, then the
    rasterio inverse Affine transform maps geographic coords → pixel coords.
    """
    if str(tile_crs) != src_crs:
        xs, ys = zip(*list(polygon.exterior.coords))
        xt, yt = warp_transform(src_crs, str(tile_crs), list(xs), list(ys))
        poly = Polygon(zip(xt, yt))
    else:
        poly = polygon

    inv = ~tile_transform
    coords = list(poly.exterior.coords)[:-1]  # drop the closing duplicate point
    pts: list[float] = []
    for gx, gy in coords:
        col, row = inv * (gx, gy)
        pts.extend([col / tile_width, row / tile_height])

    # Clamp all coordinates to [0, 1] (handles partially-outside polygons).
    pts = [max(0.0, min(1.0, v)) for v in pts]
    return pts if len(pts) >= 6 else None


def write_label_file(path: Path, instances: list[tuple[int, list[float]]]) -> None:
    """Write a YOLO annotations file (one line per instance).

    Line format: <class_id> <x1> <y1> <x2> <y2> … (space-separated).
    An empty file is a valid negative sample for YOLO training.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for class_id, coords in instances:
            f.write(f"{class_id} " + " ".join(f"{v:.6f}" for v in coords) + "\n")


def download_naip_tile(uri: str, output_path: Path) -> Path:
    """Read a NAIP COG and write a 3-band (RGB) GeoTIFF locally.

    Supports two URI schemes:
    - https://  Azure Blob Storage (Microsoft Planetary Computer) — opened
                via GDAL's /vsicurl/ virtual filesystem; SAS token already
                embedded in the URI by the NAIPImagerySource adapter.
    - s3://     AWS S3 with anonymous access (legacy; requester-pays buckets
                will fail with a 403 — use Planetary Computer instead).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if uri.startswith("s3://"):
        gdal_path = f"/vsis3/{uri.removeprefix('s3://')}"
        env = rasterio.Env(
            AWS_NO_SIGN_REQUEST="YES",
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
        )
    else:
        # HTTPS (Azure Blob with SAS token or any public COG).
        gdal_path = uri
        env = rasterio.Env(
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
        )

    with env:
        with rasterio.open(gdal_path) as src:
            profile = src.profile.copy()
            img = src.read([1, 2, 3])  # NAIP band order: R, G, B, NIR → take RGB

        profile.update(count=3, driver="GTiff", compress="deflate", tiled=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(img)

    return output_path
