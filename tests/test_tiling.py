import unittest

from warehouse_growth.tiling import sliding_windows


class TilingTests(unittest.TestCase):
    def test_sliding_windows_cover_edges(self):
        windows = sliding_windows(width=2500, height=1800, tile_size=1024, stride=768)

        self.assertEqual(windows[0].x, 0)
        self.assertEqual(windows[0].y, 0)
        self.assertEqual(max(window.x for window in windows), 1476)
        self.assertEqual(max(window.y for window in windows), 776)

    def test_sliding_windows_reject_invalid_size(self):
        with self.assertRaises(ValueError):
            sliding_windows(width=0, height=10, tile_size=256, stride=128)
