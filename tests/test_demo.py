"""Tests for the Streamlit research demo components and inference pipeline.

Validates image handling (PNG, JPEG, Grayscale, RGBA, non-square), model loading,
segmentation inference, alpha-blending, and quantitative composition metrics.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from app.streamlit_app import (
    build_overlay,
    compute_composition,
    load_model,
    run_segmentation,
)
from src.woundmap.masks import NUM_CLASSES
from src.woundmap.models import UNet


class TestStreamlitDemo(unittest.TestCase):
    """Test suite for WoundMap AI interactive Streamlit research demo."""

    @classmethod
    def setUpClass(cls):
        """Initialize mock model and mock images."""
        cls.model = UNet(in_channels=3, num_classes=NUM_CLASSES, base_features=16)
        cls.model.eval()

        # Create temporary directory for dummy images
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_path = Path(cls.temp_dir.name)

        # 1. Valid PNG (256x256 RGB)
        cls.png_path = cls.temp_path / "valid.png"
        Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)).save(cls.png_path)

        # 2. Valid JPEG (300x200 RGB non-square)
        cls.jpeg_path = cls.temp_path / "valid.jpg"
        Image.fromarray(np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)).save(cls.jpeg_path, "JPEG")

        # 3. Grayscale PNG (128x128 single channel)
        cls.gray_path = cls.temp_path / "grayscale.png"
        Image.fromarray(np.random.randint(0, 255, (128, 128), dtype=np.uint8), mode="L").save(cls.gray_path)

        # 4. RGBA PNG (256x256 with alpha)
        cls.rgba_path = cls.temp_path / "with_alpha.png"
        Image.fromarray(np.random.randint(0, 255, (256, 256, 4), dtype=np.uint8), mode="RGBA").save(cls.rgba_path)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_valid_png_inference(self):
        """Test complete inference pipeline with standard PNG image."""
        img = np.array(Image.open(self.png_path).convert("RGB"))
        resized_img, pred_indices, pred_rgb = run_segmentation(
            model=self.model,
            image_np=img,
            color_space="RGB",
            illumination_norm="none",
            target_size=(256, 256),
        )
        self.assertEqual(resized_img.shape, (256, 256, 3))
        self.assertEqual(pred_indices.shape, (256, 256))
        self.assertEqual(pred_rgb.shape, (256, 256, 3))
        self.assertEqual(resized_img.dtype, np.uint8)
        self.assertEqual(pred_indices.dtype, np.uint8)
        self.assertTrue(np.all(pred_indices >= 0) and np.all(pred_indices < NUM_CLASSES))

    def test_valid_jpeg_non_square_inference(self):
        """Test inference pipeline with non-square JPEG image."""
        img = np.array(Image.open(self.jpeg_path).convert("RGB"))
        resized_img, pred_indices, pred_rgb = run_segmentation(
            model=self.model,
            image_np=img,
            color_space="CIELAB",
            illumination_norm="clahe",
            target_size=(256, 256),
        )
        self.assertEqual(resized_img.shape, (256, 256, 3))
        self.assertEqual(pred_indices.shape, (256, 256))
        self.assertEqual(pred_rgb.shape, (256, 256, 3))

    def test_grayscale_input_converted(self):
        """Test grayscale input converted to 3-channel RGB is processed correctly."""
        pil_img = Image.open(self.gray_path).convert("RGB")
        img = np.array(pil_img)
        self.assertEqual(img.ndim, 3)
        self.assertEqual(img.shape[2], 3)
        resized_img, pred_indices, _ = run_segmentation(self.model, img)
        self.assertEqual(resized_img.shape, (256, 256, 3))
        self.assertEqual(pred_indices.shape, (256, 256))

    def test_rgba_input_converted(self):
        """Test RGBA input converted to 3-channel RGB is processed correctly."""
        pil_img = Image.open(self.rgba_path).convert("RGB")
        img = np.array(pil_img)
        self.assertEqual(img.ndim, 3)
        self.assertEqual(img.shape[2], 3)
        resized_img, pred_indices, _ = run_segmentation(self.model, img)
        self.assertEqual(resized_img.shape, (256, 256, 3))
        self.assertEqual(pred_indices.shape, (256, 256))

    def test_build_overlay_alphas(self):
        """Test alpha-blend at extremes (alpha=0.0 and alpha=1.0) and intermediate values."""
        base = np.full((10, 10, 3), 100, dtype=np.uint8)
        mask = np.full((10, 10, 3), 200, dtype=np.uint8)

        # Alpha 0.0 => exact original image
        overlay_0 = build_overlay(base, mask, alpha=0.0)
        np.testing.assert_array_equal(overlay_0, base)

        # Alpha 1.0 => exact mask
        overlay_1 = build_overlay(base, mask, alpha=1.0)
        np.testing.assert_array_equal(overlay_1, mask)

        # Alpha 0.5 => midpoint
        overlay_half = build_overlay(base, mask, alpha=0.5)
        self.assertEqual(overlay_half[0, 0, 0], 150)
        self.assertEqual(overlay_half.dtype, np.uint8)

    def test_compute_composition_math(self):
        """Test surface area composition table calculations and percentage totals."""
        # Create synthetic mask: 50% background (0), 30% granulation (1), 20% slough (2)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[:50, :] = 0  # 5,000 px
        mask[50:80, :] = 1  # 3,000 px
        mask[80:, :] = 2  # 2,000 px

        df = compute_composition(mask)

        # Verify columns exist
        expected_cols = [
            "Class ID", "Tissue Type", "Color (Hex)", "Pixel Count",
            "% of Total Image", "% of Wound Bed",
        ]
        self.assertListEqual(list(df.columns), expected_cols)

        # Total pixel sum must be 10,000
        self.assertEqual(df["Pixel Count"].sum(), 10000)

        # % of total image must sum to 100%
        self.assertAlmostEqual(df["% of Total Image"].sum(), 100.0, places=1)

        # Verify wound bed calculations: total wound pixels = 5,000
        # Granulation = 3000 / 5000 = 60.0%
        # Slough = 2000 / 5000 = 40.0%
        gran_row = df[df["Tissue Type"] == "Granulation"].iloc[0]
        slough_row = df[df["Tissue Type"] == "Slough"].iloc[0]
        bg_row = df[df["Tissue Type"] == "Background"].iloc[0]
        self.assertEqual(bg_row["% of Wound Bed"], 0.0)
        self.assertEqual(gran_row["% of Wound Bed"], 60.0)
        self.assertEqual(slough_row["% of Wound Bed"], 40.0)

    def test_load_model_missing_checkpoint(self):
        """Test load_model returns None gracefully when checkpoint path does not exist."""
        result = load_model("non_existent_model_checkpoint.pt")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
