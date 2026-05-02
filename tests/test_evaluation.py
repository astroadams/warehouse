import unittest

from warehouse_growth.evaluation import binary_metrics


class EvaluationTests(unittest.TestCase):
    def test_binary_metrics(self):
        metrics = binary_metrics(true_positives=8, false_positives=2, false_negatives=4)

        self.assertEqual(metrics.precision, 0.8)
        self.assertEqual(round(metrics.recall, 3), 0.667)
        self.assertEqual(round(metrics.f1, 3), 0.727)
