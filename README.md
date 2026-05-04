# Warehouse Growth Detector

Project skeleton for detecting warehouse buildings in satellite imagery and
measuring warehouse growth across time in the United States.

The intended pipeline is:

1. Build an area-of-interest mask around highways, primary roads, rail/ports, and
   known industrial zones.
2. Tile imagery for one or more epochs.
3. Run a candidate building detector, initially YOLO segmentation or oriented
   bounding boxes.
4. Classify detected buildings as warehouse / non-warehouse using imagery,
   footprint geometry, and context features.
5. Match detections across epochs to estimate new warehouse count and area.

This repository starts with project structure and typed interfaces. Most modules
are intentionally small so the first real experiment can replace stubs with
provider-specific implementations.

## Quick Start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it, then:

```bash
uv sync --extra dev --extra geo
uv run pytest
```

`uv sync` creates a virtual environment and installs all dependencies automatically.
Commit the generated `uv.lock` to the repo for reproducible installs.

Copy `configs/example.json` to a new experiment config and edit the AOI,
imagery epochs, road-mask distance, and model checkpoint paths.

```bash
uv run warehouse-growth plan configs/example.json
```

## Repository Layout

```text
configs/                     Experiment configuration examples
src/warehouse_growth/         Python package
  cli.py                      Command-line entry points
  config.py                   Config loading and validation
  data_sources.py             Imagery, roads, and footprint provider interfaces
  road_mask.py                Road-buffer AOI mask generation
  tiling.py                   Raster tile grid helpers
  labels.py                   Label schema and conversion helpers
  models/                     Detector/classifier wrappers
  evaluation.py               Detection/classification/change metrics
  change.py                   Cross-epoch matching and growth summaries
tests/                        Fast unit tests for core geometry/config logic
```

## First Implementation Milestones

- Create a small labeled pilot set in 3-5 logistics-heavy metros.
- Train a YOLO segmentation/OBB candidate detector.
- Add a warehouse classifier over detected footprints and contextual features.
- Quantify road-mask recall loss before using the mask at larger scale.
- Add cloud-native raster and GeoParquet readers for state-scale runs.
