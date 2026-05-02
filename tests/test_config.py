import unittest

from warehouse_growth.config import load_config


class ConfigTests(unittest.TestCase):
    def test_load_example_config(self):
        config = load_config("configs/example.json")

        self.assertEqual(config.project_name, "inland_empire_pilot")
        self.assertEqual(config.detector.type, "yolo")
        self.assertEqual(len(config.epochs), 2)
