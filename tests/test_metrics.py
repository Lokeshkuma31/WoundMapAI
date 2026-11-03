"""Automated tests for segmentation metrics."""
import unittest
import numpy as np
from src.woundmap.metrics import compute_dice_score, compute_iou_score

class TestSegmentationMetrics(unittest.TestCase):
    def test_dice_perfect_overlap(self):
        y_true = np.array([[0, 1], [2, 3]])
        y_pred = np.array([[0, 1], [2, 3]])
        dice = compute_dice_score(y_true, y_pred, num_classes=4)
        for c in range(4):
            self.assertTrue(np.isclose(dice[c], 1.0), f"Class {c} failed perfect overlap Dice")

    def test_iou_disjoint(self):
        y_true = np.array([[0, 0], [0, 0]])
        y_pred = np.array([[1, 1], [1, 1]])
        iou = compute_iou_score(y_true, y_pred, num_classes=2)
        self.assertTrue(np.isclose(iou[0], 0.0, atol=1e-4))
        self.assertTrue(np.isclose(iou[1], 0.0, atol=1e-4))

    def test_metrics_absent_class_handling(self):
        y_true = np.array([[0, 1], [0, 1]])
        y_pred = np.array([[0, 1], [0, 1]])
        dice = compute_dice_score(y_true, y_pred, num_classes=3)
        self.assertTrue(np.isclose(dice[2], 1.0))

if __name__ == "__main__":
    unittest.main()
