#!/usr/bin/env python3
"""Plot validation patches with model detections and ground-truth footprints.

Reads the GeoParquet of detections saved by evaluate_footprint.py and overlays
them on each val patch image alongside green (warehouse) and red (non_warehouse)
ground-truth footprint polygons. Also generates a precision-recall curve.

Run evaluate_footprint.py at least once to produce the detections file before
using this script.

Usage
-----
    python scripts/plot_eval_results.py configs/reno_sparks_v2.json \\
        --plot-dir runs/reno_sparks_v2/eval_plots

    # Use a non-default detections file
    python scripts/plot_eval_results.py configs/reno_sparks_v2.json \\
        --detections path/to/eval_detections.parquet \\
        --plot-dir runs/reno_sparks_v2/eval_plots

    # Plot every val patch, not just those with detections or warehouse GT
    python scripts/plot_eval_results.py configs/reno_sparks_v2.json \\
        --plot-dir runs/reno_sparks_v2/eval_plots --plot-all

    # Generate only the PR curve (skip per-patch images)
    python scripts/plot_eval_results.py configs/reno_sparks_v2.json --no-patches
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import rowcol as _rasterio_rowcol
from rasterio.warp import transform_bounds, transform_geom as _warp_geom
from shapely.geometry import box, shape
from shapely.strtree import STRtree
from tqdm import tqdm

from warehouse_growth.config import load_config


def _load_footprints(path: Path) -> tuple[list, list, str]:
    gdf = gpd.read_parquet(path)
    epsg = gdf.crs.to_epsg()
    footprint_crs = f"EPSG:{epsg}" if epsg else gdf.crs.to_wkt()
    wh = gdf[gdf["label"] == "warehouse"].geometry.tolist()
    nwh = gdf[gdf["label"] == "non_warehouse"].geometry.tolist()
    print(f"  Footprints: {len(wh):,} warehouse, {len(nwh):,} non_warehouse — CRS: {footprint_crs}")
    return wh, nwh, footprint_crs


def _load_detections(path: Path) -> dict[str, list]:
    """Return detections grouped by tile_id as plain objects with .geometry."""
    from types import SimpleNamespace

    gdf = gpd.read_parquet(path)
    by_tile: dict[str, list] = {}
    for row in gdf.itertuples(index=False):
        det = SimpleNamespace(geometry=row.geometry, score=row.score,
                              class_name=row.class_name, tile_id=row.tile_id)
        by_tile.setdefault(row.tile_id, []).append(det)
    total = sum(len(v) for v in by_tile.values())
    print(f"  Detections: {total:,} across {len(by_tile)} tiles")
    return by_tile


def _geom_to_px(geom, tile_transform):
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    if geom.geom_type != "Polygon":
        return None
    xs = [c[0] for c in geom.exterior.coords]
    ys = [c[1] for c in geom.exterior.coords]
    rows, cols = _rasterio_rowcol(tile_transform, xs, ys)
    return list(zip(cols, rows))  # (x, y) for matplotlib


def plot_pr_curve(
    detections_path: Path,
    footprints_path: Path,
    iou_threshold: float,
    output_path: Path,
    all_buildings_path: Path | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    det_gdf = gpd.read_parquet(detections_path).sort_values("score", ascending=False)
    fp_gdf = gpd.read_parquet(footprints_path)

    warehouse_geoms = fp_gdf[fp_gdf["label"] == "warehouse"].geometry.tolist()
    non_warehouse_geoms = fp_gdf[fp_gdf["label"] == "non_warehouse"].geometry.tolist()
    n_positives = len(warehouse_geoms)

    # Load all buildings (warehouse + non_warehouse + ambiguous) to replicate the
    # evaluate_footprint.py default of counting detections over vacant land as FP.
    all_buildings_path = all_buildings_path or (footprints_path.parent / "eval_all_buildings.parquet")
    if all_buildings_path.exists():
        all_building_geoms = gpd.read_parquet(all_buildings_path).geometry.tolist()
        bldg_tree: STRtree | None = STRtree(all_building_geoms)
    else:
        bldg_tree = None

    print(f"  PR curve: {len(det_gdf):,} detections, {n_positives:,} warehouse footprints"
          + ("" if bldg_tree is None else f", {len(all_building_geoms):,} total buildings"))

    if n_positives == 0:
        print("  WARNING: no warehouse footprints — skipping PR curve")
        return

    wh_tree = STRtree(warehouse_geoms) if warehouse_geoms else None
    nwh_tree = STRtree(non_warehouse_geoms) if non_warehouse_geoms else None

    matched_wh: set[int] = set()
    # 1 = TP, 0 = FP, -1 = ignored
    is_tp: list[int] = []

    for row in det_gdf.itertuples(index=False):
        geom = row.geometry
        best_iou = 0.0
        best_idx: int | None = None

        if wh_tree is not None:
            for idx in wh_tree.query(geom, predicate="intersects").tolist():
                if idx in matched_wh:
                    continue
                try:
                    inter = geom.intersection(warehouse_geoms[idx]).area
                    union = geom.union(warehouse_geoms[idx]).area
                except Exception:
                    continue
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

        if best_iou >= iou_threshold and best_idx is not None:
            matched_wh.add(best_idx)
            is_tp.append(1)
        elif nwh_tree is not None and len(nwh_tree.query(geom, predicate="intersects")) > 0:
            is_tp.append(0)
        elif bldg_tree is not None and len(bldg_tree.query(geom, predicate="intersects")) == 0:
            is_tp.append(0)  # over confirmed-vacant land — FP, matching evaluate_footprint.py
        else:
            is_tp.append(-1)  # over ambiguous/unlabeled building — excluded from curve

    precisions = [1.0]
    recalls = [0.0]
    cum_tp = cum_fp = 0

    for label in is_tp:
        if label == 1:
            cum_tp += 1
        elif label == 0:
            cum_fp += 1
        else:
            continue
        denom = cum_tp + cum_fp
        precisions.append(cum_tp / denom if denom else 0.0)
        recalls.append(cum_tp / n_positives)

    p_arr = np.array(precisions)
    r_arr = np.array(recalls)
    ap = float(np.trapezoid(p_arr, r_arr))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(r_arr, p_arr, color="steelblue", linewidth=1.5)
    ax.fill_between(r_arr, p_arr, alpha=0.15, color="steelblue")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall  (IoU≥{iou_threshold:.2f}  AP={ap:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  PR curve saved → {output_path}  (AP={ap:.3f})")


def plot_val_patches(
    val_paths: list[Path],
    detections_by_tile: dict[str, list],
    warehouse_geoms: list,
    non_warehouse_geoms: list,
    footprint_crs: str,
    plot_dir: Path,
    plot_all: bool = False,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    plot_dir.mkdir(parents=True, exist_ok=True)

    wh_tree = STRtree(warehouse_geoms) if warehouse_geoms else None
    nwh_tree = STRtree(non_warehouse_geoms) if non_warehouse_geoms else None

    plotted = 0
    for patch_path in tqdm(val_paths, desc="Plotting", unit=" patch"):
        tile_id = patch_path.stem
        tile_dets = detections_by_tile.get(tile_id, [])

        with rasterio.open(patch_path) as src:
            tile_crs = src.crs
            tile_transform = src.transform
            tile_bounds = src.bounds
            bands = list(range(1, min(src.count, 3) + 1))
            img = src.read(bands).transpose(1, 2, 0)

        b_fp = transform_bounds(tile_crs, footprint_crs, *tile_bounds)
        tile_box_fp = box(*b_fp)

        wh_idxs = wh_tree.query(tile_box_fp, predicate="intersects").tolist() if wh_tree else []
        nwh_idxs = nwh_tree.query(tile_box_fp, predicate="intersects").tolist() if nwh_tree else []

        if not plot_all and not tile_dets and not wh_idxs:
            continue

        fig, ax = plt.subplots(figsize=(8, 8))
        lo, hi = img.min(), img.max()
        img_disp = ((img.astype(np.float32) - lo) / max(hi - lo, 1) * 255).astype(np.uint8)
        ax.imshow(img_disp)
        ax.set_title(tile_id, fontsize=8)
        ax.axis("off")

        def _draw_fp(fp_geom, facecolor, edgecolor):
            try:
                warped = _warp_geom(footprint_crs, tile_crs, fp_geom.__geo_interface__)
                tile_geom = shape(warped)
            except Exception:
                return
            pts = _geom_to_px(tile_geom, tile_transform)
            if pts:
                ax.add_patch(MplPolygon(
                    pts, closed=True, fill=True,
                    facecolor=facecolor, edgecolor=edgecolor, alpha=0.35, linewidth=1.5,
                ))

        for idx in nwh_idxs:
            _draw_fp(non_warehouse_geoms[idx], "red", "darkred")
        for idx in wh_idxs:
            _draw_fp(warehouse_geoms[idx], "lime", "green")

        for det in tile_dets:
            try:
                warped = _warp_geom(footprint_crs, tile_crs, det.geometry.__geo_interface__)
                det_geom = shape(warped)
            except Exception:
                continue
            pts = _geom_to_px(det_geom, tile_transform)
            if pts:
                ax.add_patch(MplPolygon(
                    pts, closed=True, fill=False, edgecolor="purple", linewidth=2.0,
                ))

        ax.legend(handles=[
            mpatches.Patch(color="lime", alpha=0.6, label="warehouse (GT)"),
            mpatches.Patch(color="red", alpha=0.6, label="non_warehouse (GT)"),
            mpatches.Patch(facecolor="none", edgecolor="purple", linewidth=2, label="detected"),
        ], loc="upper right", fontsize=7)

        fig.savefig(plot_dir / f"{tile_id}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        plotted += 1

    print(f"\nSaved {plotted} patch plots → {plot_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="Path to project config JSON/YAML")
    parser.add_argument("--detections", type=Path, default=None, metavar="FILE",
                        help="GeoParquet of detections from evaluate_footprint.py "
                             "(default: <workspace>/eval_detections.parquet).")
    parser.add_argument("--footprints", type=Path, default=None, metavar="FILE",
                        help="GeoParquet of val-coverage footprints from evaluate_footprint.py "
                             "(default: <workspace>/eval_footprints.parquet).")
    parser.add_argument("--plot-dir", type=Path, default=None, metavar="DIR",
                        help="Directory for output PNGs (default: <workspace>/eval_plots/).")
    parser.add_argument("--plot-all", action="store_true",
                        help="Plot every val patch; default skips patches with no detections or warehouse GT.")
    parser.add_argument("--no-patches", action="store_true",
                        help="Skip per-patch images; useful when you only want the PR curve.")
    parser.add_argument("--iou-threshold", type=float, default=0.5, metavar="T",
                        help="IoU threshold for TP matching used in PR curve (default: 0.5).")
    parser.add_argument("--pr-curve", type=Path, default=None, metavar="FILE",
                        help="Output path for the PR curve PNG "
                             "(default: <workspace>/eval_pr_curve.png).")
    parser.add_argument("--all-buildings", type=Path, default=None, metavar="FILE",
                        help="GeoParquet of all val buildings from evaluate_footprint.py "
                             "(default: <workspace>/eval_all_buildings.parquet). "
                             "Used to count detections over vacant land as FP, matching "
                             "evaluate_footprint.py default behavior.")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Project : {config.project_name}\n")

    det_path = args.detections or (config.workspace / "eval_detections.parquet")
    if not det_path.exists():
        print(f"ERROR: detections file not found: {det_path}", file=sys.stderr)
        print("Run evaluate_footprint.py first to generate it.", file=sys.stderr)
        sys.exit(1)

    fp_path = args.footprints or (config.workspace / "eval_footprints.parquet")
    if not fp_path.exists():
        print(f"ERROR: footprints file not found: {fp_path}", file=sys.stderr)
        print("Run evaluate_footprint.py first to generate it.", file=sys.stderr)
        sys.exit(1)

    print("Generating PR curve …")
    pr_curve_path = args.pr_curve or (config.workspace / "eval_pr_curve.png")
    plot_pr_curve(det_path, fp_path, args.iou_threshold, pr_curve_path,
                  all_buildings_path=args.all_buildings)

    if not args.no_patches:
        plot_dir = args.plot_dir or (config.workspace / "eval_plots")

        val_dir = config.workspace / "training" / "images" / "val"
        val_paths = sorted(val_dir.glob("*.tif"))
        if not val_paths:
            print(f"ERROR: no val patches found in {val_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"\nVal patches : {len(val_paths)}")

        print(f"Loading footprints from {fp_path} …")
        warehouse_geoms, non_warehouse_geoms, footprint_crs = _load_footprints(fp_path)

        print(f"Loading detections from {det_path} …")
        detections_by_tile = _load_detections(det_path)

        plot_val_patches(
            val_paths,
            detections_by_tile,
            warehouse_geoms,
            non_warehouse_geoms,
            footprint_crs,
            plot_dir,
            plot_all=args.plot_all,
        )


if __name__ == "__main__":
    main()
