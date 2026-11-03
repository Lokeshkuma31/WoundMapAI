"""Automated tests for segmentation metrics, confusion matrix, and latency."""
import unittest
import numpy as np
import torch
from src.woundmap.metrics import (
    compute_dice_score,
    compute_iou_score,
    compute_confusion_matrix,
    compute_classification_metrics,
    benchmark_inference_latency
)
from src.woundmap.models import build_model

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

    def test_confusion_matrix_shape_and_sums(self):
        # 3 classes, 4 pixels
        y_true = np.array([0, 1, 1, 2])
        y_pred = np.array([0, 1, 0, 2])
        cm = compute_confusion_matrix(y_true, y_pred, num_classes=3)
        self.assertEqual(cm.shape, (3, 3))
        self.assertEqual(int(cm.sum()), 4)
        # Class 1: 1 true pos at (1, 1), 1 misclassified as 0 at (1, 0)
        self.assertEqual(cm[1, 1], 1)
        self.assertEqual(cm[1, 0], 1)

    def test_classification_metrics_dual_macro(self):
        # 4 classes, but class 3 is absent from both true and pred
        y_true = np.array([[0, 1], [2, 2]])
        y_pred = np.array([[0, 1], [2, 2]])
        res = compute_classification_metrics(y_true, y_pred, num_classes=4)
        
        self.assertEqual(res["total_pixels_evaluated"], 4)
        # Classes 0, 1, 2 are present and perfect -> Dice = 1.0
        self.assertEqual(res["macro_dice_present"], 1.0)
        # Class 3 is absent -> Dice = 1.0 -> macro_dice_all = 1.0
        self.assertEqual(res["macro_dice_all"], 1.0)
        
        # Now test absent class with false positive
        y_pred_false = np.array([[0, 1], [2, 3]]) # class 3 falsely predicted
        res_false = compute_classification_metrics(y_true, y_pred_false, num_classes=4)
        # Class 3 support is 0, but predicted > 0 -> dice[3] should be 0.0
        self.assertEqual(res_false["per_class"][3]["support"], 0)
        self.assertEqual(res_false["per_class"][3]["pred_count"], 1)
        self.assertEqual(res_false["per_class"][3]["dice"], 0.0)

    def test_benchmark_inference_latency(self):
        model = build_model("unet", in_channels=3, num_classes=7, base_features=8)
        dummy_in = torch.randn(1, 3, 64, 64)
        bench = benchmark_inference_latency(model, dummy_in, torch.device("cpu"), warmups=2, repeats=5)
        self.assertIn("mean_ms", bench)
        self.assertIn("fps", bench)
        self.assertGreater(bench["mean_ms"], 0.0)
        self.assertGreater(bench["fps"], 0.0)

if __name__ == "__main__":
    unittest.main()
