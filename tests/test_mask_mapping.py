"""Automated test suite for mask decoding and categorical mapping."""
import unittest
import numpy as np
from src.woundmap.masks import rgb_to_mask, mask_to_rgb, WOUNDTISSUE_PALETTE

class TestMaskMapping(unittest.TestCase):
    def test_rgb_to_mask_roundtrip(self):
        # Create a synthetic mask using audited WoundTissue colors:
        # (0,0,255) background, (255,0,0) granulation, (255,255,0) slough, (0,0,0) necrotic
        rgb_input = np.zeros((2, 2, 3), dtype=np.uint8)
        rgb_input[0, 0] = [0, 0, 255]      # Class 0: background
        rgb_input[0, 1] = [255, 0, 0]      # Class 1: granulation
        rgb_input[1, 0] = [255, 255, 0]    # Class 2: slough
        rgb_input[1, 1] = [0, 0, 0]        # Class 3: necrotic
        
        mask = rgb_to_mask(rgb_input, palette=WOUNDTISSUE_PALETTE)
        expected = np.array([[0, 1], [2, 3]], dtype=np.int64)
        np.testing.assert_array_equal(mask, expected)
        
        # Test back to RGB
        reconstructed = mask_to_rgb(mask, palette=WOUNDTISSUE_PALETTE)
        np.testing.assert_array_equal(rgb_input, reconstructed)

    def test_rgba_mask_handling(self):
        # Masks from disk often come with an alpha channel: (H, W, 4)
        rgba_input = np.zeros((2, 2, 4), dtype=np.uint8)
        rgba_input[0, 0] = [0, 0, 255, 255]      # Class 0: background
        rgba_input[0, 1] = [255, 0, 0, 255]      # Class 1: granulation
        rgba_input[1, 0] = [255, 255, 255, 255]  # Class 4: maceration
        rgba_input[1, 1] = [255, 200, 124, 255]  # Class 5: bone
        
        mask = rgb_to_mask(rgba_input, palette=WOUNDTISSUE_PALETTE)
        expected = np.array([[0, 1], [4, 5]], dtype=np.int64)
        np.testing.assert_array_equal(mask, expected)

if __name__ == "__main__":
    unittest.main()
