# Warehouse Growth Detector

Detects warehouse buildings in NAIP satellite imagery and measures warehouse
growth across time. The current prototype covers the Reno-Sparks, NV logistics
corridor using Microsoft building footprints, OSM tags, and a fine-tuned YOLOv8
segmentation model.

The intended pipeline is:

1. Build an area-of-interest mask around highways, primary roads, rail/ports, and
   known industrial zones.
2. Tile NAIP imagery for one or more epochs.
3. Run a YOLO segmentation detector to produce building footprints.
4. Classify detected buildings as warehouse / non-warehouse using imagery,
   footprint geometry, and context features.
5. Match detections across epochs to estimate new warehouse count and area.

## Quick Start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it, then:

```bash
uv sync --extra dev
uv run pytest
```

`uv sync` creates a virtual environment and installs all dependencies automatically.
Commit the generated `uv.lock` to the repo for reproducible installs.

## Training Pipeline

The full pipeline runs in six steps. Scripts 1–3 and 6 take a config file path;
the trainer (step 4) takes a workspace directory.

### 1. Download source data

Downloads Microsoft building footprints and OSM tags for the AOI.

```bash
uv run python scripts/download_prototype_data.py configs/reno_sparks_demo.json
```

### 2. Label footprints

Spatially joins footprints with OSM tags to assign `warehouse` / `non_warehouse`
/ `ambiguous_industrial` labels, then writes a GeoParquet file.

```bash
uv run python scripts/label_prototype_data.py configs/reno_sparks_demo.json
```

### 3. Prepare training dataset

Downloads NAIP tiles, slices them into 1024×1024 patches, and writes YOLO
segmentation annotations. Raw tiles are cached so re-runs only redo the slicing.

```bash
uv run python scripts/prepare_training_data.py configs/reno_sparks_demo.json
```

If you change annotation logic (boundary filter, sampling rate, class
definitions), delete `training/images/` and `training/labels/` before re-running
so stale label files are regenerated. Keep `training/raw_tiles/` — those are the
expensive downloads.

### 4. Train the detector

Fine-tunes a YOLOv8 segmentation model on the prepared dataset.

```bash
uv sync --extra models
uv run python scripts/train_warehouse_detector.py runs/reno_sparks_demo
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs N` | 150 | Training epochs |
| `--model MODEL` | `yolov8n-seg.pt` | Base checkpoint (downloaded automatically) |
| `--imgsz N` | 1024 | Input image size in pixels |
| `--batch N` | 8 | Batch size - reduce if GPU OOM |
| `--device DEVICE` | auto | `0` for GPU, `cpu`, `mps` for Apple Silicon |
| `--resume` | off | Resume from last saved checkpoint |

If the run is interrupted, resume it without losing progress:

```bash
uv run python scripts/train_warehouse_detector.py runs/reno_sparks_demo --resume
```

The best checkpoint is saved to `<workspace>/training/runs/warehouse_seg/weights/best.pt`.

### 5. Plot loss curves

After training starts, generate a training vs validation loss plot to check for overfitting:

```bash
uv run python scripts/plot_loss_curves.py                        # default workspace
uv run python scripts/plot_loss_curves.py runs/other_aoi         # different workspace
uv run python scripts/plot_loss_curves.py path/to/results.csv    # direct CSV path
```

The plot is saved as `loss_curves.png` alongside `results.csv` in the run directory.
Re-run it at any point during training to see the latest epochs.

### 6. Evaluate with footprint-anchored metrics

YOLO's built-in validation metrics count any detection without a matching label
file as a false positive. Because OSM coverage is incomplete, real warehouses that
lack OSM tags produce empty label files — making the built-in numbers misleading.

The footprint-anchored evaluation scores predictions against the combined MSFT +
OSM building footprint dataset:

- **TP** — detection whose IoU with a labeled warehouse footprint ≥ threshold
- **FP** — detection that overlaps a confirmed non-warehouse footprint, or a detection in an area where no building exists in the combined MSFT + OSM dataset
- **ignored** — detection that overlaps a building footprint whose type is uncertain (ambiguous label or warehouse overlap below the IoU threshold)
- **FN** — labeled warehouse footprint in the val coverage area not matched by any detection

```bash
uv run python scripts/evaluate_footprint.py configs/reno_sparks_demo.json
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint PATH` | `<workspace>/training/runs/warehouse_seg/weights/best.pt` | Model weights |
| `--iou-threshold T` | `0.5` | Minimum IoU to count a detection as a TP |
| `--confidence C` | `0.25` | Detector confidence threshold |
| `--ignore-vacant` | off | Revert to legacy behavior: detections over building-free areas are ignored rather than counted as FP |

Example output:

```
────────────────────────────────────────────
Footprint-anchored evaluation
  val patches          : 123
  total detections     : 87
  ignored (unlabeled)  : 34
  TP=44  FP=9  FN=14
────────────────────────────────────────────
  Precision : 0.831
  Recall    : 0.759
  F1        : 0.793
────────────────────────────────────────────
```

Both precision and recall are meaningful against the combined footprint dataset.
Compare the footprint-anchored F1 with the YOLO mAP from `results.csv` to get a
full picture. Pass `--ignore-vacant` if footprint coverage in the target area is
known to be sparse.

### Experiment tracking

MLflow is enabled automatically when the `models` extra is installed. After
training, launch the UI to compare runs:

```bash
uv run mlflow ui --backend-store-uri sqlite:///runs/reno_sparks_demo/mlruns.db
```

## Repository Layout

```text
configs/                        Experiment configuration files
  example.json                  Minimal config template
  reno_sparks_demo.json         Reno-Sparks NV prototype config
scripts/                        End-to-end pipeline scripts
  download_prototype_data.py    Fetch footprints and OSM tags
  label_prototype_data.py       Assign warehouse labels via OSM spatial join
  prepare_training_data.py      Slice NAIP tiles into YOLO-format patches
  train_warehouse_detector.py   Fine-tune YOLOv8 segmentation model
  plot_loss_curves.py           Plot training vs validation losses from results.csv
  evaluate_footprint.py         Footprint-anchored precision/recall over the val set
src/warehouse_growth/           Python package
  cli.py                        Command-line entry points
  config.py                     Config loading and validation
  data_sources.py               Imagery, roads, and footprint provider interfaces
  labels.py                     Label schema and building labeling logic
  training.py                   YOLO annotation helpers and tile downloader
  tiling.py                     Raster tile grid helpers
  road_mask.py                  Road-buffer AOI mask generation
  evaluation.py                 Detection/classification/change metrics
  change.py                     Cross-epoch matching and growth summaries
  adapters/                     Provider-specific data source implementations
    naip.py                     NAIP imagery via Microsoft Planetary Computer
    msft_footprints.py          Microsoft building footprints
    osm.py                      OpenStreetMap tag fetching
  models/                       Detector/classifier wrappers
    base.py                     Abstract detector interface
    yolo.py                     YOLOv8 segmentation wrapper
tests/                          Fast unit tests for core geometry/config logic
```

## Implementation Status

- [x] Labeled pilot dataset — Reno-Sparks NV, 2022 NAIP
- [x] YOLO segmentation training pipeline with MLflow experiment tracking
- [x] Checkpoint resume support
- [x] Footprint-anchored evaluation metrics (precision/recall over known footprint locations)
- [ ] Warehouse classifier over detected footprints and contextual features
- [ ] Multi-epoch change detection (cross-epoch footprint matching)
- [ ] Road-mask recall evaluation before applying mask at larger scale
- [ ] Cloud-native raster and GeoParquet readers for state-scale runs
