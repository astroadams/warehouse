from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from warehouse_growth.models.base import Detection


@dataclass(frozen=True)
class BinaryMetrics:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


def binary_metrics(true_positives: int, false_positives: int, false_negatives: int) -> BinaryMetrics:
    precision = _safe_div(true_positives, true_positives + false_positives)
    recall = _safe_div(true_positives, true_positives + false_negatives)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return BinaryMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def match_detections_to_footprints(
    detections: list[Detection],
    warehouse_geoms: list,
    non_warehouse_geoms: list,
    iou_threshold: float = 0.5,
    all_building_geoms: list | None = None,
) -> tuple[int, int, int, int]:
    """Match predicted detections against labeled footprint geometries.

    Returns (tp, fp, fn, ignored) where:
    - TP: detection whose best IoU with a labeled warehouse ≥ iou_threshold
    - FP: detection that overlaps a labeled non-warehouse, OR (when all_building_geoms
          is provided) a detection with no intersection with any known building at all
    - FN: labeled warehouse footprints in the evaluation area not matched
    - ignored: detections that don't overlap any labeled footprint (or overlap only an
               ambiguous/low-confidence building when all_building_geoms is provided)

    Each warehouse footprint is matched at most once (greedy). Detections in
    regions with no labeled footprint are excluded from both precision and recall,
    which makes this suitable for datasets with incomplete OSM coverage.

    When all_building_geoms is provided (union of all MSFT + OSM footprints, all label
    types), detections over areas confirmed to have no building are counted as FP rather
    than ignored — this ensures the model is penalized for spurious detections.

    All geometries must share the same CRS (typically EPSG:4326).
    """
    from shapely.strtree import STRtree

    if not warehouse_geoms and not non_warehouse_geoms:
        return 0, 0, 0, len(detections)

    wh_tree = STRtree(warehouse_geoms) if warehouse_geoms else None
    nwh_tree = STRtree(non_warehouse_geoms) if non_warehouse_geoms else None
    bldg_tree = STRtree(all_building_geoms) if all_building_geoms else None

    matched_wh: set[int] = set()
    tp = fp = ignored = 0

    for det in detections:
        geom = det.geometry
        best_iou = 0.0
        best_idx: int | None = None

        if wh_tree is not None:
            hit_idxs = wh_tree.query(geom, predicate="intersects").tolist()
            for idx in hit_idxs:
                wh_geom = warehouse_geoms[idx]
                try:
                    inter = geom.intersection(wh_geom).area
                    union = geom.union(wh_geom).area
                except Exception:
                    continue
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

        if best_iou >= iou_threshold and best_idx is not None:
            tp += 1
            matched_wh.add(best_idx)
        elif nwh_tree is not None and len(nwh_tree.query(geom, predicate="intersects")) > 0:
            fp += 1
        elif bldg_tree is not None and len(bldg_tree.query(geom, predicate="intersects")) == 0:
            fp += 1
        else:
            ignored += 1

    fn = len(warehouse_geoms) - len(matched_wh)
    return tp, fp, fn, ignored

