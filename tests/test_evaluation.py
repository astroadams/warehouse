import unittest

from shapely.geometry import box

from warehouse_growth.evaluation import binary_metrics, match_detections_to_footprints
from warehouse_growth.models.base import Detection


def _det(minx, miny, maxx, maxy):
    return Detection(geometry=box(minx, miny, maxx, maxy), score=0.9, class_name="warehouse")


class EvaluationTests(unittest.TestCase):
    def test_binary_metrics(self):
        metrics = binary_metrics(true_positives=8, false_positives=2, false_negatives=4)

        self.assertEqual(metrics.precision, 0.8)
        self.assertEqual(round(metrics.recall, 3), 0.667)
        self.assertEqual(round(metrics.f1, 3), 0.727)


class MatchDetectionsAllBuildingTests(unittest.TestCase):
    """Tests for the all_building_geoms false-positive penalization."""

    def test_no_building_detection_is_ignored_without_all_building_geoms(self):
        det = _det(10, 10, 11, 11)  # nowhere near any footprint
        warehouse = [box(0, 0, 1, 1)]
        tp, fp, fn, ignored = match_detections_to_footprints([det], warehouse, [])
        self.assertEqual(ignored, 1)
        self.assertEqual(fp, 0)

    def test_no_building_detection_is_fp_with_all_building_geoms(self):
        det = _det(10, 10, 11, 11)  # nowhere near any footprint
        warehouse = [box(0, 0, 1, 1)]
        all_buildings = [box(0, 0, 1, 1)]  # only the warehouse, nothing at (10,10)
        tp, fp, fn, ignored = match_detections_to_footprints(
            [det], warehouse, [], all_building_geoms=all_buildings
        )
        self.assertEqual(fp, 1)
        self.assertEqual(ignored, 0)

    def test_detection_over_ambiguous_building_is_still_ignored(self):
        # Ambiguous building at (5,5)-(6,6); det overlaps it but it's not in warehouse/nwh lists
        det = _det(5, 5, 6, 6)
        warehouse = [box(0, 0, 1, 1)]
        ambiguous = box(5, 5, 6, 6)
        all_buildings = [box(0, 0, 1, 1), ambiguous]
        tp, fp, fn, ignored = match_detections_to_footprints(
            [det], warehouse, [], all_building_geoms=all_buildings
        )
        self.assertEqual(ignored, 1)
        self.assertEqual(fp, 0)

    def test_detection_over_warehouse_below_iou_threshold_is_still_ignored(self):
        # Det covers only the right half of a warehouse → IoU < 0.5
        warehouse_geom = box(0, 0, 2, 2)
        det = _det(1, 0, 3, 2)  # overlaps right half: IoU = 1/3
        all_buildings = [warehouse_geom]
        tp, fp, fn, ignored = match_detections_to_footprints(
            [det], [warehouse_geom], [], iou_threshold=0.5, all_building_geoms=all_buildings
        )
        self.assertEqual(ignored, 1)
        self.assertEqual(fp, 0)
        self.assertEqual(tp, 0)
