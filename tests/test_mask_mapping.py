"""Automated test suite for mask decoding and categorical mapping."""
import unittest
import numpy as np
from src.woundmap.masks import rgb_to_mask, mask_to_rgb, DEFAULT_PALETTE

class TestMaskMapping(unittest.TestCase):
    def test_rgb_to_mask_roundtrip(self):
        # Create a 2x2 synthetic RGB mask: (0,0,0), (255,0,0), (255,255,0), (128,0,0)
        rgb_input = np.zeros((2, 2, 3), dtype=np.uint8)
        rgb_input[0, 0] = [0, 0, 0]        # Class 0: background
        rgb_input[0, 1] = [255, 0, 0]      # Class 1: granulation
        rgb_input[1, 0] = [255, 255, 0]    # Class 2: slough
        rgb_input[1, 1] = [128, 0, 0]      # Class 3: necrotic
        
        mask = rgb_to_mask(rgb_input, palette=DEFAULT_PALETTE)
        expected = np.array([[0, 1], [2, 3]], dtype=np.int64)
        np.testing.assert_array_equal(mask, expected)
        
        # Test back to RGB
        reconstructed = mask_to_rgb(mask, palette=DEFAULT_PALETTE)
        np.testing.assert_array_equal(rgb_input, reconstructed)

if __name__ == "__main__":
    unittest.main()

