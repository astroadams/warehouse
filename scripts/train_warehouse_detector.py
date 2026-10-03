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
    --name NAME     Run name; outputs go to <workspace>/training/runs/NAME
                    (default: warehouse_seg). Use a distinct name per model so
                    runs don't overwrite each other.
    --resume        Resume from the last saved checkpoint of run NAME

Outputs
-------
    <workspace>/training/runs/<name>/weights/best.pt   best checkpoint
    <workspace>/training/runs/<name>/results.csv       per-epoch metrics
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
    p.add_argument("--name", default="warehouse_seg",
                   help="Run name; outputs go to <workspace>/training/runs/NAME "
                        "(default: warehouse_seg).")
    p.add_argument("--workers", type=int, default=None,
                   help="Dataloader worker processes (Ultralytics default: 8). Each worker "
                        "holds its own copies of mosaic images; lower this if workers are "
                        "killed for running out of system RAM.")
    p.add_argument("--cache", choices=["ram", "disk"], default=None,
                   help="Cache decoded images in RAM or as .npy files on disk to skip TIFF "
                        "decompression each epoch. 'ram' needs ~3 MB per 1024px image.")
    p.add_argument("--resume", action="store_true",
                   help="Resume training from the last checkpoint of run NAME.")
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
    run_dir = output_dir / args.name
    last_pt = run_dir / "weights" / "last.pt"

    if args.resume:
        if not last_pt.exists():
            print(f"ERROR: --resume requested but no checkpoint found at {last_pt}")
            sys.exit(1)
        print(f"Resuming from {last_pt}")
        model = YOLO(str(last_pt))
    else:
        if (run_dir / "weights").exists():
            print(f"ERROR: run '{args.name}' already exists at {run_dir}")
            print("Pass --resume to continue it, or --name NEW_NAME to start a separate run.")
            sys.exit(1)
        print(f"Dataset  : {dataset_yaml}")
        print(f"Base model: {args.model}")
        model = YOLO(args.model)

    print(f"Epochs   : {args.epochs}")
    print(f"Img size : {args.imgsz}")
    print(f"Batch    : {args.batch}")
    print(f"Output   : {run_dir}")
    print()

    train_kwargs: dict = dict(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(output_dir),
        name=args.name,
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
    if args.workers is not None:
        train_kwargs["workers"] = args.workers
    if args.cache is not None:
        train_kwargs["cache"] = args.cache

    try:
        model.train(**train_kwargs)
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    finally:
        # Explicitly release GPU memory so the CUDA context doesn't linger in WSL2.
        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    best_pt = run_dir / "weights" / "best.pt"
    print("\nTraining complete.")
    if best_pt.exists():
        print(f"Best checkpoint → {best_pt}")
        print("\nNext step — run inference on a new tile:")
        print("  from warehouse_growth.models.yolo import YoloBuildingDetector")
        print(f"  detector = YoloBuildingDetector('{best_pt}')")
        print("  detections = detector.predict_tile(Path('path/to/tile.tif'))")
    else:
        print(f"(Best checkpoint not found at expected path {best_pt}; "
              f"check {run_dir}/weights/)")


if __name__ == "__main__":
    main()
