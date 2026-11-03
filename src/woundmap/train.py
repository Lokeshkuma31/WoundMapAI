"""Training loops, loss functions, checkpointing, and evaluation routines."""
import os
import json
import time
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

import src.woundmap.compat  # Compatibility fix for Python 3.14 PyTorch optimizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.woundmap.metrics import compute_dice_score, compute_iou_score

class DiceLoss(nn.Module):
    """Soft multi-class Dice loss for semantic segmentation."""
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W)
        # targets: (B, H, W) long
        probs = F.softmax(logits, dim=1)
        num_classes = logits.shape[1]
        
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - torch.mean(dice)

class CompositeLoss(nn.Module):
    """Composite loss: weighted combination of Soft Dice Loss and Cross-Entropy Loss."""
    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dice = self.dice_loss(logits, targets)
        ce = self.ce_loss(logits, targets)
        total = self.dice_weight * dice + self.ce_weight * ce
        return total, dice, ce

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float, float]:
    """Execute one training epoch across all batches."""
    model.train()
    total_loss, total_dice_loss, total_ce_loss = 0.0, 0.0, 0.0
    num_batches = len(dataloader)

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss, d_loss, c_loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice_loss += d_loss.item()
        total_ce_loss += c_loss.item()

    return (
        total_loss / max(num_batches, 1),
        total_dice_loss / max(num_batches, 1),
        total_ce_loss / max(num_batches, 1)
    )

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 7
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model without gradients, computing validation loss and per-class Dice."""
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    all_dices = {c: [] for c in range(num_classes)}

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            loss, _, _ = criterion(logits, masks)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            targets_np = masks.cpu().numpy()

            for i in range(len(preds)):
                d_score = compute_dice_score(targets_np[i], preds[i], num_classes=num_classes)
                for c in range(num_classes):
                    all_dices[c].append(d_score[c])

    val_loss = total_loss / max(num_batches, 1)
    metrics_summary = {}
    macro_scores = []
    
    for c in range(num_classes):
        mean_c = float(np.mean(all_dices[c])) if len(all_dices[c]) > 0 else 0.0
        metrics_summary[f"dice_class_{c}"] = mean_c
        macro_scores.append(mean_c)
        
    metrics_summary["macro_dice"] = float(np.mean(macro_scores))
    return val_loss, metrics_summary

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 50,
    patience: int = 10,
    save_dir: str = "outputs/checkpoints",
    model_name: str = "best_baseline_unet.pt"
) -> Dict[str, Any]:
    """Full training procedure with model checkpointing and early stopping."""
    os.makedirs(save_dir, exist_ok=True)
    best_ckpt_path = os.path.join(save_dir, model_name)
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_macro_dice": [],
        "best_epoch": 0,
        "best_val_macro_dice": 0.0
    }
    
    best_dice = -1.0
    epochs_no_improve = 0

    print(f"Starting training for {epochs} epochs on device: {device}")
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_dice, tr_ce = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_metrics = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        
        v_dice = v_metrics["macro_dice"]
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(v_loss)
        history["val_macro_dice"].append(v_dice)

        print(f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - Train Loss: {tr_loss:.4f} | Val Loss: {v_loss:.4f} | Val Macro Dice: {v_dice:.4f}")

        # Checkpointing
        if v_dice > best_dice:
            best_dice = v_dice
            history["best_epoch"] = epoch
            history["best_val_macro_dice"] = best_dice
            epochs_no_improve = 0
            
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_dice": best_dice,
                "val_metrics": v_metrics
            }, best_ckpt_path)
            print(f"  --> Checkpoint saved to {best_ckpt_path} (Val Macro Dice: {best_dice:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs (no improvement for {patience} epochs).")
                break

    # Save training history JSON
    history_path = os.path.join(save_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return history

def overfit_one_batch_test(
    model: nn.Module,
    batch: Dict[str, Any],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    steps: int = 30
) -> List[float]:
    """Sanity test verifying model can memorize a single batch (loss -> 0)."""
    model.train()
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)
    loss_history = []

    for step in range(steps):
        optimizer.zero_grad()
        logits = model(images)
        loss, _, _ = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

    return loss_history
