from __future__ import annotations

from dataclasses import dataclass


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

