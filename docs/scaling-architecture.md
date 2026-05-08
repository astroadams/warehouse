# Scaling Architecture

This document describes the recommended changes to support training on ~1M patches
across multiple US states and running inference over entire state-scale AOIs.

The current prototype is a single-machine pipeline built around local files and
sequential processing. The changes below are ordered by impact and can be adopted
incrementally — each section is independently useful even if the others are not yet
in place.

---

## 1. Support both tile caching and COG streaming ✓ Implemented

**Current behavior:** `download_naip_tile` in `training.py` always downloads the
full NAIP GeoTIFF (~212 MB) before slicing. At state scale this produces ~2.4 TB
of intermediate files per state per epoch.

**The tradeoff:** Streaming COG windows avoids that storage cost, but it is not a
straightforward replacement — the right choice depends on the run:

| Scenario | Better mode |
|---|---|
| Re-running with updated annotation logic | **Download** — tiles are cached, only labels regenerate |
| First pass over new state tiles | **Stream** — avoids storing data you may discard |
| Running on Azure co-located with Planetary Computer | **Stream** — near-zero network cost |
| Running on AWS or a local machine | **Download** — avoids repeated cross-cloud egress |
| Sparse road mask (<20% of patches kept per tile) | **Stream** — reads far fewer bytes |
| Dense road mask (>60% of patches kept per tile) | **Download** — single request beats many range reads |

Streaming replaces one HTTP request (full tile) with many HTTP range requests —
roughly one per internal COG block (typically 256×256 px) per window read. For a
tile yielding 18 patches of 1024×1024 px, that is ~288 requests instead of 1.
With N parallel workers all streaming simultaneously, that pressure is multiplied.
Streaming only wins on total bytes when the road mask is sparse enough that you
skip a large fraction of each tile.

**Implemented:** `cache_tiles: bool = True` was added to `TilingConfig` in
`config.py`. `process_tile_task` in `training.py` branches on this flag:
the cache path (default) downloads the full GeoTIFF to `raw_tiles/` on first
use and re-reads it on subsequent runs; the streaming path opens the signed
remote COG URI directly in rasterio with no local file written. The flag is
threaded into each worker's task dict by `prepare_training_data.py`.

To enable streaming, add to your config:

```json
"tiling": { "cache_tiles": false }
```

---

## 2. Parallelize tile processing with a worker pool ✓ Implemented

Each `TileWindow` is an independent unit of work with no shared state.
`prepare_training_data.py` now wraps the inner tile loop with
`concurrent.futures.ProcessPoolExecutor`, where each worker calls
`YoloBuildingDetector.predict_tile` on its assigned window:

```python
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_window(args):
    uri, window, output_dir, config = args
    img = read_naip_window(uri, window)
    # label / infer / write patch
    return result

with ProcessPoolExecutor(max_workers=num_cpus) as pool:
    futures = [pool.submit(process_window, (uri, w, out, cfg)) for w in windows]
    for future in as_completed(futures):
        collect(future.result())
```

For multi-node scale (full state in parallel), replace `ProcessPoolExecutor` with
Ray or Dask distributed. The interface is the same; only the executor changes.
`YoloBuildingDetector` lazy-loads its model in each worker process to avoid CUDA
context issues across forks.

---

## 3. Replace in-memory spatial join with DuckDB spatial

**Current behavior:** `label_footprints` in `labels.py` loads all footprints and
OSM features into Python lists, builds a Shapely `STRtree`, and iterates over every
footprint. At state scale (~30M buildings) this OOMs before finishing.

**Change:** Push the spatial join into DuckDB using its `ST_Intersects` spatial
extension. The project already uses DuckDB for footprint downloads in
`adapters/msft_footprints.py`, so no new dependency is needed.

```python
import duckdb

con = duckdb.connect()
con.execute("INSTALL spatial; LOAD spatial;")

result = con.execute("""
    SELECT
        f.geometry,
        COALESCE(o.building, '') AS building_tag
    FROM read_parquet(?) f
    LEFT JOIN read_parquet(?) o
        ON ST_Intersects(ST_GeomFromWKB(f.geometry), ST_GeomFromWKB(o.geometry))
""", [footprints_path, osm_path]).fetchdf()
```

This runs entirely out-of-core against GeoParquet files, handles arbitrarily large
datasets with constant memory, and is ~10–50x faster than the Python STRtree loop
for large inputs. Partition the GeoParquet by state FIPS or H3 cell to make queries
on subregions fast.

---

## 4. Write training data as WebDataset shards

**Current behavior:** `prepare_training_data.py` writes one `.png` per patch to
`training/images/` and one `.txt` per patch to `training/labels/`. At 1M patches
this creates 2M files, which hits filesystem inode limits and makes distributed
training impractical.

**Change:** Write training data as [WebDataset](https://github.com/webdataset/webdataset)
tar shards (e.g., 1000 patches per shard). Shards can live on S3 or Azure Blob
and stream directly into a PyTorch `DataLoader` across multiple workers and nodes
without any local copy.

```python
import webdataset as wds

with wds.TarWriter(f"s3://bucket/training/shard-{shard_id:06d}.tar") as sink:
    for patch_id, img, label in patches:
        sink.write({
            "__key__": patch_id,
            "jpg": encode_jpeg(img),
            "txt": encode_yolo_label(label),
        })
```

Ultralytics supports custom dataset classes. Wrap a `wds.WebDataset` in a thin
`torch.utils.data.Dataset` adapter to keep the existing YOLO training script
largely unchanged.

---

## 5. Abstract file paths with fsspec

**Current behavior:** All I/O uses `pathlib.Path`, which is local-only. Output
GeoParquet files, model checkpoints, and result files all land on local disk.

**Change:** Replace `Path` with `fsspec.open` and `fsspec.implementations`
throughout the pipeline. `fsspec` supports `s3://`, `az://`, `gs://`, and `file://`
transparently and is already a transitive dependency via `pyarrow`.

```python
import fsspec

def write_parquet(df, uri: str) -> None:
    with fsspec.open(uri, "wb") as f:
        df.to_parquet(f)

def read_parquet(uri: str):
    with fsspec.open(uri, "rb") as f:
        return pd.read_parquet(f)
```

`ProjectConfig.workspace` should change from `Path` to `str` so it accepts
cloud URIs. No logic changes are needed beyond the I/O calls themselves.

---

## 6. Partition inference by geography

**Current behavior:** One `ProjectConfig` covers one AOI rectangle. Scaling to a
full state means one very large rectangle that cannot be processed on a single machine.

**Change:** Add a partition script that subdivides a state-level bounding box into
county- or grid-level AOI configs and fans them out as independent jobs. Each
partition is a self-contained `ProjectConfig` run — no inter-partition coordination
is needed until the change-detection step.

```
scripts/
  partition_aoi.py       # splits a state bbox into N configs by county FIPS or grid
  run_partition.py       # processes one partition config end-to-end
```

```bash
# Generate one config per county in California
uv run python scripts/partition_aoi.py configs/california.json --by county

# Submit each partition as an independent job
for cfg in runs/california/partitions/*.json; do
    uv run python scripts/run_partition.py "$cfg" &
done
```

Results from each partition are written to separate output GeoParquet files and
merged at the change-detection step using a DuckDB query over the full partition
set.

---

## Summary

| Change | Eliminates | Effort |
|---|---|---|
| Tile caching + COG streaming | 2.4 TB/state storage (streaming path) | Low — **implemented** |
| Worker pool | Sequential tile bottleneck | Low — **implemented** |
| DuckDB spatial join | OOM on large footprint sets | Medium |
| WebDataset shards | Inode limits, distributed training barrier | Medium |
| fsspec path abstraction | Local-disk-only outputs | Medium |
| Geographic partitioning | Single-machine AOI limit | Low |

The **worker pool** and **COG streaming** are both implemented. Together they
remove the sequential bottleneck and avoid intermediate storage costs, with no
changes to the core labeling or model code. The download cache remains the
default for iterative development workflows.
