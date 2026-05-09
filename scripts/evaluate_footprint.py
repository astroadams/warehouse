#!/usr/bin/env python3
"""Footprint-anchored evaluation of a trained warehouse detector.

Runs inference on the validation patch set and scores predictions against
labeled footprints (warehouse or non_warehouse). Detections in areas where
no building exists in the combined MSFT + OSM footprint set are counted as
false positives. Pass --ignore-vacant to revert to the legacy behavior of
ignoring such detections instead.

Usage
-----
    python scripts/evaluate_footprint.py configs/reno_sparks_demo.json
    python scripts/evaluate_footprint.py configs/reno_sparks_demo.json \\
        --checkpoint runs/reno_sparks_demo/training/runs/warehouse_seg/weights/best.pt \\
        --iou-threshold 0.5 \\
        --confidence 0.25
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.warp import transform_bounds, transform_geom as _warp_geom
from shapely.geometry import box, shape
from shapely.strtree import STRtree
from tqdm import tqdm

from warehouse_growth.config import load_config
from warehouse_growth.evaluation import binary_metrics, match_detections_to_footprints


def _load_footprints(workspace: Path) -> tuple[list, list, list, str]:
    """Load and combine labeled footprints from all epochs, split by label.

    Returns (warehouse_geoms, non_warehouse_geoms, all_building_geoms, crs_string).
    all_building_geoms is the union of all labels (warehouse + non_warehouse + ambiguous)
    and is used to penalize detections over areas confirmed to have no building.
    """
    parquets = sorted(workspace.glob("labeled_footprints_*.parquet"))
    if not parquets:
        raise FileNotFoundError(
            f"No labeled_footprints_*.parquet files found in {workspace}. "
            "Run label_prototype_data.py first."
        )

    frames = [gpd.read_parquet(p) for p in parquets]
    gdf = gpd.pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    epsg = gdf.crs.to_epsg()
    footprint_crs = f"EPSG:{epsg}" if epsg else gdf.crs.to_wkt()

    warehouse_geoms = gdf[gdf["label"] == "warehouse"].geometry.tolist()
    non_warehouse_geoms = gdf[gdf["label"] == "non_warehouse"].geometry.tolist()
    all_building_geoms = gdf.geometry.tolist()
    print(
        f"  Loaded footprints: {len(warehouse_geoms):,} warehouse, "
        f"{len(non_warehouse_geoms):,} non_warehouse, "
        f"{len(all_building_geoms) - len(warehouse_geoms) - len(non_warehouse_geoms):,} ambiguous "
        f"(from {len(parquets)} epoch file{'s' if len(parquets) != 1 else ''})"
        f" — CRS: {footprint_crs}"
    )
    return warehouse_geoms, non_warehouse_geoms, all_building_geoms, footprint_crs


def _filter_to_val_coverage(
    warehouse_geoms: list,
    non_warehouse_geoms: list,
    all_building_geoms: list,
    val_paths: list[Path],
) -> tuple[list, list, list, dict]:
    """Restrict footprint geometries to those intersecting the val patch area.

    Prevents train-area footprints from inflating false-negative counts.
    Reads only GeoTIFF headers (no pixel data).

    Returns (warehouse_geoms, non_warehouse_geoms, all_building_geoms, tile_crs_map) where
    tile_crs_map maps tile stem → rasterio CRS, used for detection reprojection.
    """
    bboxes: list = []
    tile_crs_map: dict = {}
    for p in val_paths:
        with rasterio.open(p) as src:
            b4326 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
            bboxes.append(box(*b4326))
            tile_crs_map[p.stem] = src.crs

    if not bboxes:
        return warehouse_geoms, non_warehouse_geoms, all_building_geoms, tile_crs_map

    coverage_tree = STRtree(bboxes)

    def _in_val(geoms: list) -> list:
        return [g for g in geoms if len(coverage_tree.query(g, predicate="intersects")) > 0]

    wh_in_val = _in_val(warehouse_geoms)
    nwh_in_val = _in_val(non_warehouse_geoms)
    all_in_val = _in_val(all_building_geoms)
    print(
        f"  Within val coverage: {len(wh_in_val):,} warehouse, "
        f"{len(nwh_in_val):,} non_warehouse, "
        f"{len(all_in_val):,} total buildings"
    )
    return wh_in_val, nwh_in_val, all_in_val, tile_crs_map


def _reproject_detections(detections: list, tile_crs_map: dict, dst_crs: str) -> list:
    """Reproject detection geometries from each tile's native CRS to dst_crs."""
    from warehouse_growth.models.base import Detection

    out = []
    for det in detections:
        src_crs = tile_crs_map.get(det.tile_id)
        if src_crs is None:
            continue
        try:
            warped = _warp_geom(src_crs, dst_crs, det.geometry.__geo_interface__)
            geom = shape(warped)
            if not geom.is_valid:
                geom = geom.buffer(0)
            out.append(Detection(geometry=geom, score=det.score,
                                 class_name=det.class_name, tile_id=det.tile_id))
        except Exception:
            pass
    return out


def _save_val_footprints(
    warehouse_geoms: list,
    non_warehouse_geoms: list,
    footprint_crs: str,
    path: Path,
) -> None:
    """Save the val-coverage footprint subset to GeoParquet for plot_eval_results.py."""
    labels = ["warehouse"] * len(warehouse_geoms) + ["non_warehouse"] * len(non_warehouse_geoms)
    gdf = gpd.GeoDataFrame(
        {"label": labels},
        geometry=warehouse_geoms + non_warehouse_geoms,
        crs=footprint_crs,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(path)
    print(
        f"  Val footprints saved → {path} "
        f"({len(warehouse_geoms):,} warehouse, {len(non_warehouse_geoms):,} non_warehouse)"
    )


def _save_val_all_buildings(all_building_geoms: list, footprint_crs: str, path: Path) -> None:
    """Save all val-coverage buildings (all labels) so plot_eval_results.py can apply
    the same vacant-area FP logic as evaluate_footprint.py."""
    gdf = gpd.GeoDataFrame(geometry=all_building_geoms, crs=footprint_crs)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(path)
    print(f"  All val buildings saved → {path} ({len(all_building_geoms):,} total)")


def _save_detections(detections: list, path: Path) -> None:
    """Persist inference results to GeoParquet for use with plot_eval_results.py."""
    if not detections:
        print(f"  No detections to save → {path}")
        return

    gdf = gpd.GeoDataFrame(
        {
            "score": [d.score for d in detections],
            "class_name": [d.class_name for d in detections],
            "tile_id": [d.tile_id for d in detections],
        },
        geometry=[d.geometry for d in detections],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(path)
    print(f"  Detections saved → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="Path to project config JSON/YAML")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Model checkpoint (.pt). Defaults to <workspace>/training/runs/warehouse_seg/weights/best.pt",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5, metavar="T",
                        help="IoU threshold for TP matching (default: 0.5)")
    parser.add_argument("--confidence", type=float, default=0.25, metavar="C",
                        help="Detector confidence threshold (default: 0.25)")
    parser.add_argument("--save-detections", type=Path, default=None, metavar="FILE",
                        help="Path for the GeoParquet of inference results "
                             "(default: <workspace>/eval_detections.parquet).")
    parser.add_argument(
        "--ignore-vacant",
        action="store_true",
        default=False,
        help="Detections over areas with no building footprint "
             "(MSFT + OSM) are ignored rather than counted as false positives.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Project : {config.project_name}")

    checkpoint = args.checkpoint or (
        config.workspace / "training" / "runs" / "warehouse_seg" / "weights" / "best.pt"
    )
    if not checkpoint.exists():
        print(f"ERROR: checkpoint not found at {checkpoint}", file=sys.stderr)
        print("Run train_warehouse_detector.py first, or pass --checkpoint.", file=sys.stderr)
        sys.exit(1)

    val_dir = config.workspace / "training" / "images" / "val"
    val_paths = sorted(val_dir.glob("*.tif"))
    if not val_paths:
        print(f"ERROR: no val patches found in {val_dir}", file=sys.stderr)
        print("Run prepare_training_data.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"\nVal patches: {len(val_paths)}")
    print(f"Checkpoint : {checkpoint}")
    print(f"IoU thresh : {args.iou_threshold}  confidence: {args.confidence}\n")

    print("Loading footprints …")
    warehouse_geoms, non_warehouse_geoms, all_building_geoms, footprint_crs = _load_footprints(config.workspace)

    print("Filtering to val coverage …")
    warehouse_geoms, non_warehouse_geoms, all_building_geoms, tile_crs_map = _filter_to_val_coverage(
        warehouse_geoms, non_warehouse_geoms, all_building_geoms, val_paths
    )
    _save_val_footprints(warehouse_geoms, non_warehouse_geoms, footprint_crs,
                         config.workspace / "eval_footprints.parquet")
    _save_val_all_buildings(all_building_geoms, footprint_crs,
                            config.workspace / "eval_all_buildings.parquet")

    print("\nLoading model …")
    from warehouse_growth.models.yolo import YoloBuildingDetector
    detector = YoloBuildingDetector(checkpoint, confidence_threshold=args.confidence)

    print("Running inference on val patches …")
    all_detections = []
    for patch_path in tqdm(val_paths, unit=" patch"):
        try:
            dets = detector.predict_tile(patch_path)
            all_detections.extend(dets)
        except Exception as exc:
            tqdm.write(f"  SKIP {patch_path.name}: {exc}")

    print(f"\nTotal detections: {len(all_detections)}")

    print("Reprojecting detections to footprint CRS …")
    all_detections = _reproject_detections(all_detections, tile_crs_map, footprint_crs)
    print(f"  {len(all_detections)} detections after reprojection")

    det_parquet = args.save_detections or (config.workspace / "eval_detections.parquet")
    _save_detections(all_detections, det_parquet)

    print("Matching detections to footprints …")
    tp, fp, fn, ignored = match_detections_to_footprints(
        all_detections,
        warehouse_geoms,
        non_warehouse_geoms,
        iou_threshold=args.iou_threshold,
        all_building_geoms=None if args.ignore_vacant else all_building_geoms,
    )

    metrics = binary_metrics(tp, fp, fn)

    print("\n" + "─" * 44)
    print("Footprint-anchored evaluation")
    print(f"  val patches          : {len(val_paths)}")
    print(f"  total detections     : {len(all_detections)}")
    print(f"  ignored (unlabeled)  : {ignored}")
    print(f"  TP={tp}  FP={fp}  FN={fn}")
    print("─" * 44)
    print(f"  Precision : {metrics.precision:.3f}")
    print(f"  Recall    : {metrics.recall:.3f}")
    print(f"  F1        : {metrics.f1:.3f}")
    print("─" * 44)

    _try_log_mlflow(config, metrics, tp, fp, fn, ignored, args)


def _try_log_mlflow(config, metrics, tp, fp, fn, ignored, args) -> None:
    try:
        import mlflow
    except ImportError:
        return

    mlflow_uri = config.workspace / "training" / "runs" / "mlruns"
    if not mlflow_uri.exists():
        mlflow_uri = config.workspace / "mlruns.db"
        if not mlflow_uri.exists():
            return

    mlflow.set_tracking_uri(f"sqlite:///{mlflow_uri}" if mlflow_uri.suffix == ".db" else str(mlflow_uri))
    mlflow.set_experiment(config.project_name)

    with mlflow.start_run(run_name="footprint_eval"):
        mlflow.log_params({
            "iou_threshold": args.iou_threshold,
            "confidence": args.confidence,
            "checkpoint": str(args.checkpoint or "best.pt"),
        })
        mlflow.log_metrics({
            "fp_precision": metrics.precision,
            "fp_recall": metrics.recall,
            "fp_f1": metrics.f1,
            "fp_tp": tp,
            "fp_fp": fp,
            "fp_fn": fn,
            "fp_ignored": ignored,
        })
    print(f"\nMetrics logged to MLflow ({mlflow_uri})")


if __name__ == "__main__":
    main()
