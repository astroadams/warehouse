import importlib.util
import unittest

from warehouse_growth.data_sources import VectorFeature


@unittest.skipIf(importlib.util.find_spec("shapely") is None, "shapely is not installed")
class RoadMaskTests(unittest.TestCase):
    def test_build_road_mask_clips_to_aoi(self):
        from shapely.geometry import LineString, box

        from warehouse_growth.road_mask import build_road_mask

        road = VectorFeature(geometry=LineString([(0, 0), (10, 0)]), properties={})
        aoi = box(0, -1, 5, 1)

        mask = build_road_mask([road], buffer_distance=1, clip_geometry=aoi)

        self.assertFalse(mask.is_empty)
        self.assertEqual(mask.bounds, aoi.bounds)

    def test_build_road_mask_empty_input(self):
        from warehouse_growth.road_mask import build_road_mask

        mask = build_road_mask([], buffer_distance=1)

        self.assertTrue(mask.is_empty)
