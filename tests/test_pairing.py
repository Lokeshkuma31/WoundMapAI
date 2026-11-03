"""Automated tests for dataset pairing and manifest integrity."""
import unittest
import os
import pandas as pd

class TestPairing(unittest.TestCase):
    def test_manifest_pairing_contract(self):
        """Validates that any manifest generated adheres to the contract: every image and mask exists."""
        manifest_path = os.path.join("data", "manifests", "manifest.csv")
        if not os.path.exists(manifest_path):
            self.skipTest("Manifest file not generated yet (will be validated in Phase 1 audit)")
            
        df = pd.read_csv(manifest_path)
        required_cols = ["image_path", "mask_path", "image_id"]
        for col in required_cols:
            self.assertIn(col, df.columns)
            
        for _, row in df.iterrows():
            self.assertTrue(os.path.exists(row["image_path"]), f"Missing image: {row['image_path']}")
            self.assertTrue(os.path.exists(row["mask_path"]), f"Missing mask: {row['mask_path']}")

if __name__ == "__main__":
    unittest.main()

