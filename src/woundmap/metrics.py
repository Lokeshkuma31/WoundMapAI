"""Segmentation evaluation metrics: Dice, IoU, confusion analysis."""
from typing import Dict, List, Optional
import numpy as np

def compute_dice_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, smooth: float = 1e-6) -> Dict[int, float]:
    """Compute per-class Dice coefficient.
    
    Args:
        y_true: Ground truth mask (H, W) with discrete class IDs.
        y_pred: Predicted mask (H, W) with discrete class IDs.
        num_classes: Total number of classes.
        smooth: Smoothing epsilon for numerical stability.
    """
    dice_per_class = {}
    for cls in range(num_classes):
        true_cls = (y_true == cls)
        pred_cls = (y_pred == cls)
        
        intersection = np.logical_and(true_cls, pred_cls).sum()
        total = true_cls.sum() + pred_cls.sum()
        
        if total == 0:
            # Class absent from both ground truth and prediction
            dice = 1.0
        else:
            dice = (2.0 * intersection + smooth) / (total + smooth)
        dice_per_class[cls] = float(dice)
        
    return dice_per_class

def compute_iou_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, smooth: float = 1e-6) -> Dict[int, float]:
    """Compute per-class Intersection over Union (Jaccard Index)."""
    iou_per_class = {}
    for cls in range(num_classes):
        true_cls = (y_true == cls)
        pred_cls = (y_pred == cls)
        
        intersection = np.logical_and(true_cls, pred_cls).sum()
        union = np.logical_or(true_cls, pred_cls).sum()
        
        if union == 0:
            iou = 1.0
        else:
            iou = (intersection + smooth) / (union + smooth)
        iou_per_class[cls] = float(iou)
        
    return iou_per_class
