from __future__ import annotations

from dataclasses import dataclass

from shapely.strtree import STRtree

from warehouse_growth.models.base import Detection


@dataclass(frozen=True)
class GrowthSummary:
    new_count: int
    new_area: float
    matched_count: int


def summarize_new_warehouses(
    baseline: list[Detection],
    comparison: list[Detection],
    min_iou: float = 0.2,
) -> GrowthSummary:
    """Summarize comparison detections that do not match baseline detections."""
    if not baseline:
        return GrowthSummary(
            new_count=len(comparison),
            new_area=sum(item.geometry.area for item in comparison),
            matched_count=0,
        )

    baseline_geoms = [item.geometry for item in baseline]
    tree = STRtree(baseline_geoms)
    new_area = 0.0
    new_count = 0
    matched_count = 0

    for detection in comparison:
        candidates = tree.query(detection.geometry)
        best_iou = 0.0
        for candidate_index in candidates:
            candidate = baseline_geoms[int(candidate_index)]
            union_area = detection.geometry.union(candidate).area
            if union_area:
                best_iou = max(best_iou, detection.geometry.intersection(candidate).area / union_area)
        if best_iou >= min_iou:
            matched_count += 1
        else:
            new_count += 1
            new_area += detection.geometry.area

    return GrowthSummary(new_count=new_count, new_area=new_area, matched_count=matched_count)

