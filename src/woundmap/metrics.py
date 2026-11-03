"""Segmentation evaluation metrics: Dice, IoU, confusion matrix, precision, recall, and latency."""
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn

def compute_dice_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 7) -> Dict[int, float]:
    """Compute per-class Dice Similarity Coefficient (DSC)."""
    dice_per_class = {}
    for cls in range(num_classes):
        true_cls = (y_true == cls)
        pred_cls = (y_pred == cls)
        
        intersection = np.logical_and(true_cls, pred_cls).sum()
        total = true_cls.sum() + pred_cls.sum()
        
        if total == 0:
            dice = 1.0  # Absent from both ground truth and prediction
        elif intersection == 0:
            dice = 0.0
        else:
            dice = (2.0 * intersection) / total
        dice_per_class[cls] = float(dice)
        
    return dice_per_class

def compute_iou_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 7) -> Dict[int, float]:
    """Compute per-class Intersection over Union (Jaccard Index)."""
    iou_per_class = {}
    for cls in range(num_classes):
        true_cls = (y_true == cls)
        pred_cls = (y_pred == cls)
        
        intersection = np.logical_and(true_cls, pred_cls).sum()
        union = np.logical_or(true_cls, pred_cls).sum()
        
        if union == 0:
            iou = 1.0
        elif intersection == 0:
            iou = 0.0
        else:
            iou = intersection / union
        iou_per_class[cls] = float(iou)
        
    return iou_per_class

def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 7) -> np.ndarray:
    """Compute pixel-level confusion matrix of shape (num_classes, num_classes).
    
    Row i represents ground-truth class i; column j represents predicted class j.
    """
    y_true_flat = y_true.ravel().astype(np.int64)
    y_pred_flat = y_pred.ravel().astype(np.int64)
    
    # Filter valid range [0, num_classes - 1]
    valid_mask = (y_true_flat >= 0) & (y_true_flat < num_classes) & (y_pred_flat >= 0) & (y_pred_flat < num_classes)
    y_true_valid = y_true_flat[valid_mask]
    y_pred_valid = y_pred_flat[valid_mask]
    
    # Compute 2D bincount
    indices = y_true_valid * num_classes + y_pred_valid
    cm = np.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 7
) -> Dict[str, Any]:
    """Compute comprehensive segmentation evaluation metrics.
    
    Returns per-class Dice, IoU, Precision, Recall, Support, and Dual Macro Averages.
    """
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)
    
    tp = np.diag(cm).astype(np.float64)
    fn = (np.sum(cm, axis=1) - tp).astype(np.float64)
    fp = (np.sum(cm, axis=0) - tp).astype(np.float64)
    support = np.sum(cm, axis=1).astype(np.int64)
    pred_count = np.sum(cm, axis=0).astype(np.int64)
    
    per_class = []
    dices_all = []
    dices_present = []
    ious_all = []
    ious_present = []
    precisions = []
    recalls = []
    
    for c in range(num_classes):
        sup = int(support[c])
        pred_c = int(pred_count[c])
        
        # Dice calculation
        total_denom = 2.0 * tp[c] + fp[c] + fn[c]
        if total_denom == 0:
            dice = 1.0  # Absent in both
        elif tp[c] == 0:
            dice = 0.0
        else:
            dice = float((2.0 * tp[c]) / total_denom)
            
        # IoU calculation
        union_denom = tp[c] + fp[c] + fn[c]
        if union_denom == 0:
            iou = 1.0  # Absent in both
        elif tp[c] == 0:
            iou = 0.0
        else:
            iou = float(tp[c] / union_denom)
            
        # Precision & Recall
        if tp[c] + fp[c] == 0:
            prec = 1.0 if sup == 0 else 0.0
        elif tp[c] == 0:
            prec = 0.0
        else:
            prec = float(tp[c] / (tp[c] + fp[c]))
            
        if tp[c] + fn[c] == 0:
            rec = 1.0 if pred_c == 0 else 0.0
        elif tp[c] == 0:
            rec = 0.0
        else:
            rec = float(tp[c] / (tp[c] + fn[c]))
            
        per_class.append({
            "class_id": c,
            "support": sup,
            "pred_count": pred_c,
            "dice": dice,
            "iou": iou,
            "precision": prec,
            "recall": rec
        })
        
        dices_all.append(dice)
        ious_all.append(iou)
        precisions.append(prec)
        recalls.append(rec)
        
        if sup > 0:
            dices_present.append(dice)
            ious_present.append(iou)
            
    return {
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "macro_dice_present": float(np.mean(dices_present)) if dices_present else 0.0,
        "macro_dice_all": float(np.mean(dices_all)),
        "macro_iou_present": float(np.mean(ious_present)) if ious_present else 0.0,
        "macro_iou_all": float(np.mean(ious_all)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "total_pixels_evaluated": int(np.sum(cm))
    }

def benchmark_inference_latency(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    warmups: int = 5,
    repeats: int = 50
) -> Dict[str, float]:
    """Measure inference latency and throughput in milliseconds per image."""
    model.eval()
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        for _ in range(warmups):
            _ = model(input_tensor)
            
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = model(input_tensor)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
            
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "fps": float(1000.0 / np.mean(times))
    }
