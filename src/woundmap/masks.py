"""Mask decoding, categorical mapping, and verification utilities for WoundTissue."""
from typing import Dict, Tuple
import numpy as np

# Ground-truth WoundTissue palette audited from akabircs/WoundTissue
WOUNDTISSUE_PALETTE: Dict[int, Tuple[str, Tuple[int, int, int]]] = {
    0: ("background", (0, 0, 255)),       # Blue
    1: ("granulation", (255, 0, 0)),      # Red
    2: ("slough", (255, 255, 0)),         # Yellow
    3: ("necrotic", (0, 0, 0)),           # Black
    4: ("maceration", (255, 255, 255)),   # White
    5: ("bone", (255, 200, 124)),         # Cream / Peach
    6: ("tendon", (0, 191, 255)),         # Sky blue (per paper spec)
}

# Standard 4-class reduction for baseline comparison (background, granulation, slough, necrotic)
STANDARD_4CLASS_PALETTE: Dict[int, Tuple[str, Tuple[int, int, int]]] = {
    0: ("background", (0, 0, 255)),
    1: ("granulation", (255, 0, 0)),
    2: ("slough", (255, 255, 0)),
    3: ("necrotic", (0, 0, 0)),
}

DEFAULT_PALETTE = WOUNDTISSUE_PALETTE
NUM_CLASSES = len(WOUNDTISSUE_PALETTE)

CLASS_NAMES: Dict[int, str] = {cid: info[0].capitalize() for cid, info in WOUNDTISSUE_PALETTE.items()}
PALETTE_MAP: Dict[int, Tuple[int, int, int]] = {cid: info[1] for cid, info in WOUNDTISSUE_PALETTE.items()}

HEX_COLORS: Dict[int, str] = {
    0: "#0000FF",  # Background (Blue)
    1: "#FF0000",  # Granulation (Red)
    2: "#FFFF00",  # Slough (Yellow)
    3: "#000000",  # Necrotic (Black)
    4: "#FFFFFF",  # Maceration (White)
    5: "#FFC87C",  # Bone (Cream / Peach)
    6: "#00BFFF",  # Tendon (Sky Blue)
}

def mask_indices_to_rgb(mask: np.ndarray, palette: Dict[int, Tuple[str, Tuple[int, int, int]]] = DEFAULT_PALETTE) -> np.ndarray:
    """Alias for mask_to_rgb."""
    return mask_to_rgb(mask, palette=palette)

def rgb_to_mask(rgb_arr: np.ndarray, palette: Dict[int, Tuple[str, Tuple[int, int, int]]] = DEFAULT_PALETTE) -> np.ndarray:
    """Map RGB segmentation mask image to discrete integer class indices [0..N-1].
    
    If the mask has an alpha channel (RGBA), only the first 3 RGB channels are evaluated.
    """
    if rgb_arr.ndim == 3 and rgb_arr.shape[-1] >= 3:
        rgb_arr = rgb_arr[:, :, :3]
        
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
