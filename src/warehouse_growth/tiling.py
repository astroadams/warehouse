from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class TileWindow:
    x: int
    y: int
    width: int
    height: int


def sliding_windows(
    width: int,
    height: int,
    tile_size: int,
    stride: int,
) -> list[TileWindow]:
    """Create pixel windows that cover a raster, including right/bottom edges."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if tile_size <= 0 or stride <= 0:
        raise ValueError("tile_size and stride must be positive")

    x_steps = max(1, ceil((width - tile_size) / stride) + 1)
    y_steps = max(1, ceil((height - tile_size) / stride) + 1)

    windows: list[TileWindow] = []
    for y_index in range(y_steps):
        y = min(y_index * stride, max(0, height - tile_size))
        for x_index in range(x_steps):
            x = min(x_index * stride, max(0, width - tile_size))
            windows.append(TileWindow(x=x, y=y, width=min(tile_size, width), height=min(tile_size, height)))
    return windows

