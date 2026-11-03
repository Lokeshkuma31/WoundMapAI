"""Automated test suite for model output specifications and training sanity."""
import unittest
import src.woundmap.compat
import torch
import torch.optim as optim

from src.woundmap.models import UNet, build_model
from src.woundmap.train import CompositeLoss, overfit_one_batch_test

class TestModelOutputAndSanity(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.num_classes = 7

    def test_model_output_channels_and_shapes(self):
        """Mandatory check: Model channels, shapes, and class IDs match config."""
        model = build_model("unet", in_channels=3, num_classes=self.num_classes, base_features=16)
        dummy_input = torch.randn(2, 3, 128, 128, dtype=torch.float32)
        logits = model(dummy_input)
        
        # Verify 4D shape (B, C, H, W)
        self.assertEqual(logits.shape, (2, self.num_classes, 128, 128))
        self.assertEqual(logits.dtype, torch.float32)
        
        # Output should be raw unnormalized logits
        self.assertTrue(torch.is_tensor(logits))
        self.assertFalse(torch.isnan(logits).any(), "Model produced NaN in logits")

    def test_composite_loss_backward(self):
        """Verify CompositeLoss computes valid forward and backward gradients."""
        criterion = CompositeLoss(dice_weight=0.5, ce_weight=0.5)
        logits = torch.randn(2, self.num_classes, 32, 32, requires_grad=True)
        targets = torch.randint(0, self.num_classes, (2, 32, 32))
        
        total_loss, dice_loss, ce_loss = criterion(logits, targets)
        self.assertGreater(total_loss.item(), 0.0)
        self.assertGreater(dice_loss.item(), 0.0)
        self.assertGreater(ce_loss.item(), 0.0)
        
        total_loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertGreater(logits.grad.norm().item(), 0.0)

    def test_one_batch_overfit_sanity(self):
        """Sanity test: Verify network can overfit a single mini-batch (loss drops)."""
        model = build_model("unet", in_channels=3, num_classes=self.num_classes, base_features=16)
        criterion = CompositeLoss(dice_weight=0.5, ce_weight=0.5)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        synthetic_batch = {
            "image": torch.randn(2, 3, 64, 64),
            "mask": torch.randint(0, self.num_classes, (2, 64, 64)),
            "image_id": ["synthetic_1", "synthetic_2"]
        }
        
        loss_history = overfit_one_batch_test(
            model, synthetic_batch, criterion, optimizer, self.device, steps=25
        )
        
        self.assertEqual(len(loss_history), 25)
        # Verify significant loss reduction (>50%) over 25 steps
        initial_loss = loss_history[0]
        final_loss = loss_history[-1]
        self.assertLess(final_loss, initial_loss * 0.5, f"Overfit sanity failed: {initial_loss:.4f} -> {final_loss:.4f}")

if __name__ == "__main__":
    unittest.main()
