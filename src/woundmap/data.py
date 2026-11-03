"""PyTorch Dataset and DataLoader implementations for WoundMap AI."""
import os
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from src.woundmap.masks import rgb_to_mask, WOUNDTISSUE_PALETTE
from src.woundmap.preprocessing import (
    paired_resize,
    paired_geometric_transform,
    image_only_color_jitter,
    preprocess_image
)

class WoundDataset(Dataset):
    """PyTorch Dataset for multi-class chronic wound tissue segmentation."""
    
    def __init__(
        self,
        manifest_csv: str,
        target_size: Tuple[int, int] = (512, 512),
        color_space: str = "RGB",
        illumination_norm: str = "none",
        augment: bool = False,
        seed: Optional[int] = None
    ):
        super().__init__()
        if not os.path.exists(manifest_csv):
            raise FileNotFoundError(f"Manifest not found: {manifest_csv}")
            
        self.df = pd.read_csv(manifest_csv)
        self.target_size = target_size
        self.color_space = color_space
        self.illumination_norm = illumination_norm
        self.augment = augment
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        
    def __len__(self) -> int:
        return len(self.df)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        mask_path = row["mask_path"]
        img_id = row["image_id"]
        
        # Load raw images using PIL
        with Image.open(img_path) as raw_img, Image.open(mask_path) as raw_mask:
            img = raw_img.convert("RGB")
            mask = raw_mask.convert("RGB")
            
            # Deterministic paired resize (Bicubic image, Nearest-neighbor mask)
            img_res, mask_res = paired_resize(img, mask, self.target_size)
            
            img_arr = np.array(img_res, dtype=np.uint8)
            mask_rgb_arr = np.array(mask_res, dtype=np.uint8)
            
        # Map RGB mask to discrete integer class indices [0..6]
        mask_idx_arr = rgb_to_mask(mask_rgb_arr, palette=WOUNDTISSUE_PALETTE)
        
        # Apply paired geometric augmentation if requested
        if self.augment:
            hflip = bool(self.rng.random() > 0.5)
            vflip = bool(self.rng.random() > 0.5)
            rot_k = int(self.rng.integers(0, 4))
            
            img_arr, mask_idx_arr = paired_geometric_transform(
                img_arr, mask_idx_arr, hflip=hflip, vflip=vflip, rot90_k=rot_k
            )
            
            # Apply color jitter strictly to image array
            if self.rng.random() > 0.5:
                b_factor = float(self.rng.uniform(0.85, 1.15))
                c_factor = float(self.rng.uniform(0.85, 1.15))
                img_arr = image_only_color_jitter(img_arr, b_factor, c_factor)
                
        # Color conversion & optional illumination normalisation
        processed_img = preprocess_image(
            img_arr,
            color_space=self.color_space,
            illumination_norm=self.illumination_norm
        )
        
        # Convert to PyTorch tensors with contiguous memory guarantees
        img_tensor = torch.from_numpy(np.ascontiguousarray(processed_img)).permute(2, 0, 1).contiguous()
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_idx_arr)).long()
        
        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "image_id": img_id
        }

def build_dataloader(
    manifest_csv: str,
    batch_size: int = 4,
    shuffle: bool = False,
    target_size: Tuple[int, int] = (512, 512),
    color_space: str = "RGB",
    illumination_norm: str = "none",
    augment: bool = False,
    num_workers: int = 0,
    seed: Optional[int] = None
) -> DataLoader:
    """Factory creating configured DataLoader instances."""
    dataset = WoundDataset(
        manifest_csv=manifest_csv,
        target_size=target_size,
        color_space=color_space,
        illumination_norm=illumination_norm,
        augment=augment,
        seed=seed
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
