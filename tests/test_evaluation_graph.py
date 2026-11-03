import unittest

import numpy as np

from evaluate_graph import boundary_counts, small_fire_counts, summarize


class EvaluationTests(unittest.TestCase):
    def test_perfect_small_fires_and_boundary(self):
        gt = np.zeros((16, 16), dtype=np.int64)
        gt[2:4, 2:4] = 2
        gt[10:12, 10:12] = 2
        valid = np.ones_like(gt, dtype=bool)
        np.testing.assert_array_equal(small_fire_counts(gt, gt, valid), [2, 2])
        counts = boundary_counts(gt, gt, valid)
        self.assertEqual(counts[0], counts[1])
        self.assertEqual(counts[2], counts[3])

    def test_one_prediction_cannot_match_two_fires(self):
        gt = np.zeros((8, 8), dtype=np.int64)
        gt[1, 1] = gt[1, 4] = 2
        pred = gt.copy()
        pred[1, 1:5] = 2
        np.testing.assert_array_equal(small_fire_counts(pred, gt, np.ones_like(gt, bool)), [1, 2])

    def test_large_and_ignored_fires_excluded(self):
        gt = np.zeros((16, 16), dtype=np.int64)
        gt[:10, :10] = 2
        gt[12, 12] = 2
        valid = np.ones_like(gt, bool)
        valid[12, 12] = False
        np.testing.assert_array_equal(small_fire_counts(gt, gt, valid), [0, 0])

    def test_empty_predictions_are_false_negatives(self):
        gt = np.zeros((8, 8), dtype=np.int64)
        gt[1, 1] = 2
        pred = np.zeros_like(gt)
        np.testing.assert_array_equal(small_fire_counts(pred, gt, np.ones_like(gt, bool)), [0, 1])

    def test_metric_summary_and_no_small_fire(self):
        summary = summarize(np.diag([5, 2, 3]), np.array([4, 4, 4, 4]),
                            np.array([0, 0]), ['background', 'smoke', 'fire'])
        self.assertEqual(summary['miou'], 1.)
        self.assertEqual(summary['boundary_f1'], 1.)
        self.assertIsNone(summary['small_fire_recall'])


if __name__ == '__main__':
    unittest.main()
