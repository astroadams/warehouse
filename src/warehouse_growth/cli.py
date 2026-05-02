from __future__ import annotations

import argparse
from pathlib import Path

from warehouse_growth.config import ProjectConfig, load_config


def _format_plan(config: ProjectConfig) -> str:
    epoch_names = ", ".join(epoch.name for epoch in config.epochs)
    road_classes = ", ".join(config.road_mask.road_classes)
    return "\n".join(
        [
            f"Project: {config.project_name}",
            f"Workspace: {config.workspace}",
            f"AOI: {config.aoi.name} {config.aoi.bbox} ({config.aoi.crs})",
            f"Epochs: {epoch_names}",
            f"Road mask: {config.road_mask.buffer_meters:g} m around {road_classes}",
            f"Tiling: {config.tiling.tile_size_px}px tiles, {config.tiling.stride_px}px stride",
            f"Detector: {config.detector.type} / {config.detector.task}",
            f"Warehouse classifier: {'enabled' if config.classifier.enabled else 'disabled'}",
        ]
    )


def plan(config_path: Path) -> None:
    """Print the execution plan for an experiment config."""
    config = load_config(config_path)
    print(_format_plan(config))


def make_workspace(config_path: Path) -> None:
    """Create the expected run directories for an experiment."""
    config = load_config(config_path)
    for dirname in ["tiles", "labels", "predictions", "metrics", "masks"]:
        (config.workspace / dirname).mkdir(parents=True, exist_ok=True)
    print(f"Created workspace at {config.workspace}")


def app() -> None:
    parser = argparse.ArgumentParser(
        prog="warehouse-growth",
        description="Warehouse detection and growth measurement workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help=plan.__doc__)
    plan_parser.add_argument("config_path", type=Path)

    workspace_parser = subparsers.add_parser("make-workspace", help=make_workspace.__doc__)
    workspace_parser.add_argument("config_path", type=Path)

    args = parser.parse_args()
    if args.command == "plan":
        plan(args.config_path)
    elif args.command == "make-workspace":
        make_workspace(args.config_path)


if __name__ == "__main__":
    app()
