"""Automated unit test suite for preprocessing, color conversions, and transforms."""
import unittest
import numpy as np
import torch
from PIL import Image

from src.woundmap.preprocessing import (
    apply_clahe_channel,
    convert_color_space,
    preprocess_image,
    paired_resize,
    paired_geometric_transform,
    image_only_color_jitter
)
from src.woundmap.data import WoundDataset

class TestPreprocessingAndTransforms(unittest.TestCase):
    def test_paired_resize_discrete_integrity(self):
        # Create a synthetic 100x100 mask with discrete IDs [0, 1, 2, 3]
        img_arr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        mask_arr = np.zeros((100, 100), dtype=np.uint8)
        mask_arr[0:50, 0:50] = 1
        mask_arr[0:50, 50:100] = 2
        mask_arr[50:100, :] = 3
        
        img = Image.fromarray(img_arr)
        mask = Image.fromarray(mask_arr)
        
        target_size = (256, 256)
        img_res, mask_res = paired_resize(img, mask, target_size=target_size)
        
        self.assertEqual(img_res.size, target_size)
        self.assertEqual(mask_res.size, target_size)
        
        # Verify categorical discrete integrity: only original discrete IDs exist
        mask_res_arr = np.array(mask_res)
        unique_labels = set(np.unique(mask_res_arr))
        self.assertTrue(unique_labels.issubset({0, 1, 2, 3}))

    def test_paired_geometric_alignment(self):
        # Place a unique marker at top-left (0, 0) in both image and mask
        img_arr = np.zeros((64, 64, 3), dtype=np.uint8)
        mask_arr = np.zeros((64, 64), dtype=np.int64)
        
        img_arr[0, 0] = [255, 255, 255]
        mask_arr[0, 0] = 1
        
        # Test horizontal flip: marker must move to (0, 63) in both
        img_h, mask_h = paired_geometric_transform(img_arr, mask_arr, hflip=True)
        self.assertEqual(tuple(img_h[0, 63]), (255, 255, 255))
        self.assertEqual(mask_h[0, 63], 1)
        self.assertEqual(tuple(img_h[0, 0]), (0, 0, 0))
        self.assertEqual(mask_h[0, 0], 0)
        
        # Test vertical flip: marker must move to (63, 0) in both
        img_v, mask_v = paired_geometric_transform(img_arr, mask_arr, vflip=True)
        self.assertEqual(tuple(img_v[63, 0]), (255, 255, 255))
        self.assertEqual(mask_v[63, 0], 1)

    def test_color_space_bounds(self):
        np.random.seed(42)
        rgb_img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        
        for space in ["RGB", "HSV", "CIELAB"]:
            converted = convert_color_space(rgb_img, space)
            self.assertEqual(converted.shape, (64, 64, 3))
            self.assertEqual(converted.dtype, np.float32)
            self.assertGreaterEqual(float(converted.min()), 0.0, f"{space} min < 0.0")
            self.assertLessEqual(float(converted.max()), 1.0, f"{space} max > 1.0")

    def test_clahe_channel_properties(self):
        # Test CLAHE on synthetic gradient
        gradient = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
        eq = apply_clahe_channel(gradient, clip_limit=2.0, grid_size=(8, 8))
        self.assertEqual(eq.shape, (64, 64))
        self.assertEqual(eq.dtype, np.uint8)
        self.assertGreaterEqual(eq.min(), 0)
        self.assertLessEqual(eq.max(), 255)

    def test_wound_dataset_loading_and_shapes(self):
        manifest_p = "data/manifests/train.csv"
        ds = WoundDataset(
            manifest_csv=manifest_p,
            target_size=(256, 256),
            color_space="RGB",
            illumination_norm="clahe",
            augment=True,
            seed=42
        )
        self.assertEqual(len(ds), 9)
        sample = ds[0]
        
        self.assertIn("image", sample)
        self.assertIn("mask", sample)
        self.assertIn("image_id", sample)
        
        img_t = sample["image"]
        mask_t = sample["mask"]
        
        self.assertEqual(img_t.shape, (3, 256, 256))
        self.assertEqual(mask_t.shape, (256, 256))
        self.assertEqual(img_t.dtype, torch.float32)
        self.assertEqual(mask_t.dtype, torch.int64)
        
        # Verify mask values are discrete valid IDs within [0, 6]
        self.assertGreaterEqual(int(mask_t.min()), 0)
        self.assertLessEqual(int(mask_t.max()), 6)

if __name__ == "__main__":
    unittest.main()
