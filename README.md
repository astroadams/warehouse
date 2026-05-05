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

The full pipeline runs in four steps. Each script accepts an optional
`workspace_dir` argument (default: `./runs/reno_sparks_demo`).

### 1. Download source data

Downloads Microsoft building footprints and OSM tags for the AOI.

```bash
uv run python scripts/download_prototype_data.py
```

### 2. Label footprints

Spatially joins footprints with OSM tags to assign `warehouse` / `non_warehouse`
/ `ambiguous_industrial` labels, then writes a GeoParquet file.

```bash
uv run python scripts/label_prototype_data.py
```

### 3. Prepare training dataset

Downloads NAIP tiles, slices them into 1024×1024 patches, and writes YOLO
segmentation annotations. Raw tiles are cached so re-runs only redo the slicing.

```bash
uv run python scripts/prepare_training_data.py
```

If you change annotation logic (boundary filter, sampling rate, class
definitions), delete `training/images/` and `training/labels/` before re-running
so stale label files are regenerated. Keep `training/raw_tiles/` — those are the
expensive downloads.

### 4. Train the detector

Fine-tunes a YOLOv8 segmentation model on the prepared dataset.

```bash
uv sync --extra models
uv run python scripts/train_warehouse_detector.py
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
uv run python scripts/train_warehouse_detector.py --resume
```

The best checkpoint is saved to `<workspace>/training/runs/warehouse_seg/weights/best.pt`.

### Experiment tracking

MLflow is enabled automatically when the `models` extra is installed. After
training, launch the UI to compare runs:

```bash
mlflow ui --backend-store-uri sqlite:///runs/reno_sparks_demo/mlruns.db
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
- [ ] Warehouse classifier over detected footprints and contextual features
- [ ] Multi-epoch change detection (cross-epoch footprint matching)
- [ ] Road-mask recall evaluation before applying mask at larger scale
- [ ] Cloud-native raster and GeoParquet readers for state-scale runs
