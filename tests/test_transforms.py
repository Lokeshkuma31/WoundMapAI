"""Automated tests for image-mask alignment and transform invariance."""
import unittest
import numpy as np
from src.woundmap.preprocessing import convert_color_space

class TestTransforms(unittest.TestCase):
    def test_color_space_conversion_ranges(self):
        # Create random synthetic RGB image
        np.random.seed(42)
        rgb_img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        
        # Test RGB output normalized to [0, 1]
        rgb_norm = convert_color_space(rgb_img, "RGB")
        self.assertEqual(rgb_norm.shape, (64, 64, 3))
        self.assertTrue((rgb_norm >= 0.0).all() and (rgb_norm <= 1.0).all())
        
        # Test HSV output normalized
        hsv_norm = convert_color_space(rgb_img, "HSV")
        self.assertEqual(hsv_norm.shape, (64, 64, 3))
        self.assertTrue((hsv_norm >= 0.0).all() and (hsv_norm <= 1.0).all())

if __name__ == "__main__":
    unittest.main()

