# Cloud Training Guide

The training pipeline requires ~8–10 GB VRAM and 8–12 hours for 150 epochs. This document covers cloud GPU options and the workflow for running training remotely with spot instances to minimize cost.

## Provider Comparison

| Provider | GPU | VRAM | Spot $/hr | Persistent Storage | Notes |
|----------|-----|------|-----------|-------------------|-------|
| **RunPod Spot** | RTX 4090 | 24 GB | **~$0.24** | Network Volumes (instant re-attach) | Best value; cleanest spot workflow |
| RunPod Spot | A100 SXM | 40 GB | ~$0.79 | Network Volumes | More VRAM if batch 8 needed |
| AWS Spot | p3.2xlarge (V100) | 16 GB | ~$0.70 | EBS (survives termination) | VRAM limits batch to ~4–6 |
| GCP Spot | T4 | 16 GB | ~$0.11 | Persistent Disk (survives termination) | 24-hr max lifetime; manual disk re-attach |
| Azure Spot | NCas T4 v3 (T4) | 16 GB | ~$0.12 | Managed Disk (Deallocate policy) | NAIP data colocated in East US |
| Azure Spot | NVads A10 v5 (A10) | 24 GB | ~$0.20 | Managed Disk (Deallocate policy) | Good VRAM; same data-locality benefit |
| Azure Spot | NC24ads A100 v4 | 80 GB | ~$0.90 | Managed Disk (Deallocate policy) | Full batch 8+ headroom; ~70% off on-demand |

**Recommended:** RunPod Spot RTX 4090 at ~$0.24/hr. A 12-hour run costs ~$3. Use a RunPod **Network Volume** ($0.07/GB/month) so checkpoints survive interruption and data doesn't need re-uploading after a restart.

**Prefer Azure East US if** you plan to run `prepare_training_data.py` on the cloud instance — NAIP tiles are hosted on Azure Blob Storage (Planetary Computer), so downloads are within-datacenter (free egress). For multi-metro datasets with many tiles this saves meaningful time and cost.

## Persistent Storage on Spot Interruption

- **RunPod Network Volumes:** Completely independent of the pod. Terminate a spot pod, provision a new one, re-attach the same volume instantly.
- **AWS EBS:** Volume survives instance termination. Attach to a new spot instance with 1–2 CLI commands.
- **GCP Persistent Disk:** Same as EBS, but GCP Spot VMs also have a hard 24-hour maximum lifetime — plan for at least one disk rotation per day.
- **Azure Managed Disk:** Use **Deallocate** eviction policy (not Delete) so disks are retained on eviction. After eviction, Azure does not guarantee when capacity returns in the same region/SKU — you may need to switch SKU or region.

## Why the Training Script Is Spot-Friendly

`train_warehouse_detector.py` already supports `--resume`, which restarts from `last.pt` (saved every epoch). A spot interruption loses at most one epoch of progress. No code changes are needed to take advantage of this.

## Planned Helper Scripts

The following scripts are planned but not yet implemented:

### `scripts/train_spot.sh`
Auto-resume loop for spot instances — re-runs training with `--resume` after any interruption until it exits cleanly:

```bash
#!/usr/bin/env bash
while true; do
  uv run python scripts/train_warehouse_detector.py "$@" --resume
  EXIT=$?
  [[ $EXIT -eq 0 ]] && break
  echo "Interrupted (exit $EXIT), resuming in 30s..."
  sleep 30
done
```

### `scripts/cloud_setup.sh`
Bootstraps a fresh cloud instance: installs `uv`, runs `uv sync --extra models`.

### `scripts/sync_workspace.sh`
`rsync` wrapper for uploading the prepared training dataset and downloading results (`best.pt`, `results.csv`, `mlruns.db`).

## Workflow (Once Scripts Are Implemented)

```bash
# 1. Locally: prepare training data (CPU-only, no GPU needed)
uv run python scripts/download_prototype_data.py configs/reno_sparks_demo.json
uv run python scripts/label_prototype_data.py configs/reno_sparks_demo.json
uv run python scripts/prepare_training_data.py configs/reno_sparks_v2.json  # cache_tiles: false
uv run python scripts/extract_footprint_crops.py configs/reno_sparks_demo.json

# 2. Provision RunPod spot pod (RTX 4090) + attach Network Volume at /workspace

# 3. Upload training data to the volume (patches + footprint crops)
./scripts/sync_workspace.sh upload --host <pod-ip> --user root \
    runs/reno_sparks_demo/training /workspace/training

# 4. Bootstrap cloud environment (once per pod)
ssh root@<pod-ip> "bash -s" < scripts/cloud_setup.sh

# 5a. Train detector (auto-resumes after spot interruptions)
ssh root@<pod-ip> "cd /workspace && bash scripts/train_spot.sh runs/reno_sparks_demo --epochs 150"

# 5b. Train classifier (run after detector training completes)
ssh root@<pod-ip> "cd /workspace && uv run python scripts/train_footprint_classifier.py \
    configs/reno_sparks_demo.json --epochs 50"

# 6. After spot interruption: re-provision pod, re-attach same volume, repeat step 5
#    No data re-upload needed — last.pt is on the persistent volume

# 7. Pull results when done (detector + classifier checkpoints)
./scripts/sync_workspace.sh download --host <pod-ip> --user root \
    /workspace/training/runs/warehouse_seg/weights/best.pt \
    runs/reno_sparks_demo/training/runs/
./scripts/sync_workspace.sh download --host <pod-ip> --user root \
    /workspace/training/runs/warehouse_cls/weights/best.pt \
    runs/reno_sparks_demo/training/runs/
```

## Config Tips for Cloud

- Use `reno_sparks_v2.json` (or set `"cache_tiles": false`) to stream NAIP tiles rather than storing them — saves 10–50 GB of volume space per metro.
- Set `TILING_WORKERS` env var to match cloud vCPU count during data prep if running `prepare_training_data.py` on the cloud instance.
- MLflow is tracked locally in `<workspace>/mlruns.db`. To watch live loss curves from your laptop, forward port 5000 via SSH tunnel:
  ```bash
  ssh -L 5000:localhost:5000 root@<pod-ip>
  # On the pod:
  uv run mlflow ui --backend-store-uri sqlite:///runs/reno_sparks_demo/mlruns.db
  ```
