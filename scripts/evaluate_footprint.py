#!/usr/bin/env python3
"""Footprint-anchored evaluation of a trained warehouse detector.

Runs inference on the validation patch set and scores predictions only at
locations where labeled footprints (warehouse or non_warehouse) exist.
Detections in regions with no labeled footprint are ignored rather than
counted as false positives — appropriate for datasets with incomplete OSM
coverage.

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
from rasterio.warp import transform_bounds
from shapely.geometry import box
from shapely.strtree import STRtree
from tqdm import tqdm

from warehouse_growth.config import load_config
from warehouse_growth.evaluation import binary_metrics, match_detections_to_footprints


def _load_footprints(workspace: Path) -> tuple[list, list]:
    """Load and combine labeled footprints from all epochs, split by label."""
    parquets = sorted(workspace.glob("labeled_footprints_*.parquet"))
    if not parquets:
        raise FileNotFoundError(
            f"No labeled_footprints_*.parquet files found in {workspace}. "
            "Run label_prototype_data.py first."
        )

    frames = [gpd.read_parquet(p) for p in parquets]
    gdf = gpd.pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    warehouse_geoms = gdf[gdf["label"] == "warehouse"].geometry.tolist()
    non_warehouse_geoms = gdf[gdf["label"] == "non_warehouse"].geometry.tolist()
    print(
        f"  Loaded footprints: {len(warehouse_geoms):,} warehouse, "
        f"{len(non_warehouse_geoms):,} non_warehouse "
        f"(from {len(parquets)} epoch file{'s' if len(parquets) != 1 else ''})"
    )
    return warehouse_geoms, non_warehouse_geoms


def _filter_to_val_coverage(
    warehouse_geoms: list,
    non_warehouse_geoms: list,
    val_paths: list[Path],
) -> tuple[list, list]:
    """Restrict footprint geometries to those intersecting the val patch area.

    Prevents train-area footprints from inflating false-negative counts.
    Reads only GeoTIFF headers (no pixel data).
    """
    bboxes: list = []
    for p in val_paths:
        with rasterio.open(p) as src:
            b4326 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
            bboxes.append(box(*b4326))

    if not bboxes:
        return warehouse_geoms, non_warehouse_geoms

    coverage_tree = STRtree(bboxes)

    def _in_val(geoms: list) -> list:
        return [g for g in geoms if len(coverage_tree.query(g, predicate="intersects")) > 0]

    wh_in_val = _in_val(warehouse_geoms)
    nwh_in_val = _in_val(non_warehouse_geoms)
    print(
        f"  Within val coverage: {len(wh_in_val):,} warehouse, "
        f"{len(nwh_in_val):,} non_warehouse"
    )
    return wh_in_val, nwh_in_val


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
    warehouse_geoms, non_warehouse_geoms = _load_footprints(config.workspace)

    print("Filtering to val coverage …")
    warehouse_geoms, non_warehouse_geoms = _filter_to_val_coverage(
        warehouse_geoms, non_warehouse_geoms, val_paths
    )

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

    print("Matching detections to footprints …")
    tp, fp, fn, ignored = match_detections_to_footprints(
        all_detections,
        warehouse_geoms,
        non_warehouse_geoms,
        iou_threshold=args.iou_threshold,
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
