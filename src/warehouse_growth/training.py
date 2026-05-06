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


# ---------------------------------------------------------------------------
# Parallel tile processing
# ---------------------------------------------------------------------------

_WORKER_STATE: dict = {}


def init_patch_worker(labels_parquet_path: str) -> None:
    """Load warehouse footprints and build a spatial index once per worker process."""
    import geopandas as gpd
    from shapely.strtree import STRtree

    gdf = gpd.read_parquet(labels_parquet_path)
    wh = gdf[gdf["label"] == "warehouse"].reset_index(drop=True)
    _WORKER_STATE["geoms"] = wh.geometry.tolist()
    _WORKER_STATE["tree"] = STRtree(_WORKER_STATE["geoms"])


def process_tile_task(task: dict) -> dict:
    """Download, slice, and annotate one NAIP tile. Returns per-split patch counts.

    Must run inside a worker process initialised with ``init_patch_worker``.
    Negative patches use a per-patch deterministic RNG keyed on the patch name
    so the selected set is identical regardless of worker execution order.
    """
    import random
    from urllib.parse import urlparse

    import planetary_computer as pc
    import rasterio
    from affine import Affine
    from rasterio.transform import array_bounds
    from rasterio.warp import transform_bounds
    from shapely.geometry import box

    from warehouse_growth.tiling import sliding_windows

    warehouse_geoms: list = _WORKER_STATE["geoms"]
    wh_tree = _WORKER_STATE["tree"]

    uri: str = task["uri"]
    split: str = task["split"]
    dataset_dir = Path(task["dataset_dir"])
    raw_tile_dir = Path(task["raw_tile_dir"])
    patch_size: int = task["patch_size"]
    stride: int = task["stride"]
    neg_sample_rate: float = task["neg_sample_rate"]
    warehouse_class_id: int = task["warehouse_class_id"]

    tile_name = Path(urlparse(uri).path).stem
    raw_path = raw_tile_dir / f"{tile_name}.tif"
    result: dict = {"tile_name": tile_name, "split": split, "pos": 0, "neg": 0, "error": None}

    if not raw_path.exists():
        try:
            download_naip_tile(pc.sign(uri), raw_path)
        except Exception as exc:
            result["error"] = str(exc)
            return result

    try:
        with rasterio.open(raw_path) as src:
            tile_crs = src.crs
            tile_transform = src.transform
            tile_w, tile_h = src.width, src.height
            img_full = src.read([1, 2, 3])
    except Exception as exc:
        result["error"] = str(exc)
        return result

    for win in sliding_windows(tile_w, tile_h, patch_size, stride):
        patch_transform = tile_transform * Affine.translation(win.x, win.y)
        left, bottom, right, top = array_bounds(win.height, win.width, patch_transform)
        b4326 = transform_bounds(str(tile_crs), "EPSG:4326", left, bottom, right, top)
        patch_box_4326 = box(*b4326)

        hit_idxs = wh_tree.query(patch_box_4326, predicate="intersects")

        instances: list[tuple[int, list[float]]] = []
        for idx in hit_idxs.tolist():
            geom = warehouse_geoms[idx]
            if geom.geom_type != "Polygon":
                continue
            if not patch_box_4326.contains(geom):
                continue
            pts = geo_polygon_to_yolo(geom, patch_transform, str(tile_crs), win.width, win.height)
            if pts:
                instances.append((warehouse_class_id, pts))

        is_positive = bool(instances)
        patch_name = f"{tile_name}_{win.x}_{win.y}"
        img_path = dataset_dir / "images" / split / f"{patch_name}.tif"
        lbl_path = dataset_dir / "labels" / split / f"{patch_name}.txt"

        if not img_path.exists():
            if not is_positive and random.Random(patch_name).random() > neg_sample_rate:
                continue

        if not img_path.exists():
            patch_img = img_full[:, win.y : win.y + win.height, win.x : win.x + win.width]
            patch_profile = {
                "driver": "GTiff",
                "count": 3,
                "dtype": patch_img.dtype,
                "width": win.width,
                "height": win.height,
                "crs": tile_crs,
                "transform": patch_transform,
                "compress": "deflate",
                "tiled": True,
            }
            with rasterio.open(img_path, "w", **patch_profile) as patch_dst:
                patch_dst.write(patch_img)

        write_label_file(lbl_path, instances)
        result["pos" if is_positive else "neg"] += 1

    return result
