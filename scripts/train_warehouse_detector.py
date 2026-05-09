#!/usr/bin/env python3
"""Fine-tune a YOLO segmentation model on the prepared warehouse dataset.

Expects the dataset produced by prepare_training_data.py to exist at
<workspace>/training/dataset.yaml.

Usage
-----
    python scripts/train_warehouse_detector.py [workspace_dir] [--epochs N] [--model MODEL]

Arguments
---------
    workspace_dir   Path to the run workspace (default: ./runs/reno_sparks_demo)
    --epochs N      Number of training epochs (default: 100)
    --model MODEL   Pretrained YOLO checkpoint to fine-tune from
                    (default: yolov8n-seg.pt — downloads automatically on first run)
    --resume        Resume from the last saved checkpoint in the output directory

Outputs
-------
    <workspace>/training/runs/warehouse_seg/weights/best.pt   best checkpoint
    <workspace>/training/runs/warehouse_seg/results.csv       per-epoch metrics
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a YOLO warehouse segmentation model.")
    p.add_argument("workspace", nargs="?", default="runs/reno_sparks_demo")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--model", default="yolov8n-seg.pt",
                   help="Pretrained YOLO checkpoint (any ultralytics seg model)")
    p.add_argument("--imgsz", type=int, default=1024,
                   help="Training image size in pixels (default: 1024)")
    p.add_argument("--batch", type=int, default=8,
                   help="Batch size (default: 8). Use -1 to auto-detect via AutoBatch, "
                        "which requires ~2 GB of free system RAM for probe tensors.")
    p.add_argument("--device", default=None,
                   help="Training device: 0 (GPU), cpu, mps (Apple Silicon). "
                        "Auto-detected when omitted.")
    p.add_argument("--resume", action="store_true",
                   help="Resume training from the last checkpoint in the output directory.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace)
    dataset_yaml = workspace / "training" / "dataset.yaml"

    if not dataset_yaml.exists():
        print(f"ERROR: dataset.yaml not found at {dataset_yaml}")
        print("Run prepare_training_data.py first:")
        print("  python scripts/prepare_training_data.py")
        sys.exit(1)

    import os
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    # Store MLflow runs inside the workspace so each AOI keeps its own history.
    # MLFLOW_EXPERIMENT_NAME groups all runs for this workspace together.
    mlflow_uri = f"sqlite:///{workspace.resolve() / 'mlruns.db'}"
    os.environ.setdefault("MLFLOW_TRACKING_URI", mlflow_uri)
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", workspace.resolve().name)

    # End any runs left open by a previous interrupted session.  Ultralytics'
    # MLflow callback calls mlflow.log_metrics() with no error handling, so a
    # stale RUNNING run causes the training loop to crash after epoch 1.
    try:
        import mlflow
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(os.environ["MLFLOW_EXPERIMENT_NAME"])
        if exp:
            stale = client.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string="attributes.status = 'RUNNING'",
            )
            for run in stale:
                client.set_terminated(run.info.run_id, status="FAILED")
                print(f"Closed stale MLflow run {run.info.run_id[:8]}…")
    except Exception:
        pass  # MLflow unavailable or DB not yet created — fine

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics is not installed.")
        print("Install it with:  uv sync --extra models  or  uv pip install ultralytics")
        sys.exit(1)

    output_dir = workspace.resolve() / "training" / "runs"
    last_pt = output_dir / "warehouse_seg" / "weights" / "last.pt"

    if args.resume:
        if not last_pt.exists():
            print(f"ERROR: --resume requested but no checkpoint found at {last_pt}")
            sys.exit(1)
        print(f"Resuming from {last_pt}")
        model = YOLO(str(last_pt))
    else:
        print(f"Dataset  : {dataset_yaml}")
        print(f"Base model: {args.model}")
        model = YOLO(args.model)

    print(f"Epochs   : {args.epochs}")
    print(f"Img size : {args.imgsz}")
    print(f"Batch    : {args.batch}")
    print(f"Output   : {output_dir / 'warehouse_seg'}")
    print()

    train_kwargs: dict = dict(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(output_dir),
        name="warehouse_seg",
        exist_ok=True,
        resume=args.resume,
        # Augmentation — helps with the class-imbalance in aerial imagery.
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.2,
        flipud=0.5,        # aerial imagery has no canonical "up"
        fliplr=0.5,
        degrees=90.0,      # random 90° rotation steps
        mosaic=1.0,
        # Suppress per-batch console spam; results.csv still written.
        verbose=False,
    )
    if args.device is not None:
        train_kwargs["device"] = args.device

    try:
        results = model.train(**train_kwargs)
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        results = None
    finally:
        # Explicitly release GPU memory so the CUDA context doesn't linger in WSL2.
        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    best_pt = output_dir / "warehouse_seg" / "weights" / "best.pt"
    print("\nTraining complete.")
    if best_pt.exists():
        print(f"Best checkpoint → {best_pt}")
        print("\nNext step — run inference on a new tile:")
        print(f"  from warehouse_growth.models.yolo import YoloBuildingDetector")
        print(f"  detector = YoloBuildingDetector('{best_pt}')")
        print(f"  detections = detector.predict_tile(Path('path/to/tile.tif'))")
    else:
        print(f"(Best checkpoint not found at expected path {best_pt}; "
              "check {output_dir}/warehouse_seg/weights/)")


if __name__ == "__main__":
    main()
