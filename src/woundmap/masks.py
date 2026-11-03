"""Mask decoding, categorical mapping, and verification utilities."""
from typing import Dict, Tuple
import numpy as np

# Standard WoundTissue label palette (to be validated in Phase 1 audit)
DEFAULT_PALETTE = {
    0: ("background", (0, 0, 0)),
    1: ("granulation", (255, 0, 0)),
    2: ("slough", (255, 255, 0)),
    3: ("necrotic", (128, 0, 0)),
}

def rgb_to_mask(rgb_arr: np.ndarray, palette: Dict[int, Tuple[str, Tuple[int, int, int]]] = DEFAULT_PALETTE) -> np.ndarray:
    """Map RGB segmentation mask image to discrete integer class indices [0..N-1]."""
    height, width, _ = rgb_arr.shape
    mask = np.zeros((height, width), dtype=np.int64)
    for class_id, (_, color) in palette.items():
        match = np.all(rgb_arr == color, axis=-1)
        mask[match] = class_id
    return mask

def mask_to_rgb(mask: np.ndarray, palette: Dict[int, Tuple[str, Tuple[int, int, int]]] = DEFAULT_PALETTE) -> np.ndarray:
    """Map discrete integer class indices back to RGB for visualization."""
    height, width = mask.shape
    rgb_arr = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id, (_, color) in palette.items():
        rgb_arr[mask == class_id] = color
    return rgb_arr
