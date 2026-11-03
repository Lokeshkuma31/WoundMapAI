"""Automated test suite for split integrity and data leakage prevention."""
import unittest
import os
import pandas as pd

class TestSplitIntegrity(unittest.TestCase):
    def setUp(self):
        self.train_path = os.path.join("data", "manifests", "train.csv")
        self.val_path = os.path.join("data", "manifests", "val.csv")
        self.test_path = os.path.join("data", "manifests", "test.csv")
        
    def test_split_files_exist(self):
        self.assertTrue(os.path.exists(self.train_path), "train.csv missing")
        self.assertTrue(os.path.exists(self.val_path), "val.csv missing")
        self.assertTrue(os.path.exists(self.test_path), "test.csv missing")
        
    def test_no_image_leakage_across_partitions(self):
        train_df = pd.read_csv(self.train_path)
        val_df = pd.read_csv(self.val_path)
        test_df = pd.read_csv(self.test_path)
        
        train_ids = set(train_df["image_id"])
        val_ids = set(val_df["image_id"])
        test_ids = set(test_df["image_id"])
        
        # Test mutual exclusivity (zero intersection)
        self.assertEqual(len(train_ids & val_ids), 0, f"Train-Val leakage: {train_ids & val_ids}")
        self.assertEqual(len(train_ids & test_ids), 0, f"Train-Test leakage: {train_ids & test_ids}")
        self.assertEqual(len(val_ids & test_ids), 0, f"Val-Test leakage: {val_ids & test_ids}")
        
        # Verify complete coverage
        total_unique = len(train_ids | val_ids | test_ids)
        self.assertEqual(total_unique, 13, f"Expected 13 total unique images, found {total_unique}")

if __name__ == "__main__":
    unittest.main()
