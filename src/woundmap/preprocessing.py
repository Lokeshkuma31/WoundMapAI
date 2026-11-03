"""Preprocessing, color conversions, and illumination normalization routines.

Implements deterministic resizing, mathematical color conversions (sRGB, HSV, CIELAB),
Contrast-Limited Adaptive Histogram Equalization (CLAHE), and paired augmentations.
"""
from typing import Tuple, Optional
import numpy as np
from PIL import Image

def apply_clahe_channel(channel: np.ndarray, clip_limit: float = 2.0, grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply Contrast-Limited Adaptive Histogram Equalization to a 2D uint8 channel.
    
    Pure NumPy vectorized implementation ensuring zero external C++ dependencies.
    
    Args:
        channel: 2D uint8 array (H, W) in [0, 255].
        clip_limit: Contrast clipping threshold multiplier.
        grid_size: Number of grid tiles (rows, cols), e.g. (8, 8).
    
    Returns:
        Equalized 2D uint8 array (H, W) in [0, 255].
    """
    if channel.dtype != np.uint8:
        channel = np.clip(channel, 0, 255).astype(np.uint8)
        
    H, W = channel.shape
    gh, gw = grid_size
    th, tw = H // gh, W // gw
    
    # Precompute CDFs for each grid tile
    cdfs = np.zeros((gh, gw, 256), dtype=np.float32)
    nominal_clip = int(clip_limit * (th * tw) / 256.0)
    
    for r in range(gh):
        for c in range(gw):
            tile = channel[r * th:(r + 1) * th, c * tw:(c + 1) * tw]
            hist, _ = np.histogram(tile, bins=256, range=(0, 256))
            
            # Clip histogram and redistribute excess
            excess = np.maximum(hist - nominal_clip, 0).sum()
            hist = np.minimum(hist, nominal_clip) + (excess // 256)
            
            # Cumulative distribution function
            cdf = np.cumsum(hist).astype(np.float32)
            c_min, c_max = cdf.min(), cdf.max()
            denom = max(c_max - c_min, 1e-6)
            cdfs[r, c] = (cdf - c_min) / denom * 255.0
            
    # Bilinear interpolation across tile centers
    y_coords = np.arange(H, dtype=np.float32)
    x_coords = np.arange(W, dtype=np.float32)
    
    r_idx = np.clip((y_coords - th / 2.0) / th, 0, gh - 1.0001)
    c_idx = np.clip((x_coords - tw / 2.0) / tw, 0, gw - 1.0001)
    
    r0 = r_idx.astype(np.int32)
    r1 = np.minimum(r0 + 1, gh - 1)
    dr = (r_idx - r0)[:, None]
    
    c0 = c_idx.astype(np.int32)
    c1 = np.minimum(c0 + 1, gw - 1)
    dc = (c_idx - c0)[None, :]
    
    R0, C0 = np.meshgrid(r0, c0, indexing="ij")
    R1, C1 = np.meshgrid(r1, c1, indexing="ij")
    
    val00 = cdfs[R0, C0, channel]
    val01 = cdfs[R0, C1, channel]
    val10 = cdfs[R1, C0, channel]
    val11 = cdfs[R1, C1, channel]
    
    top = val00 * (1.0 - dc) + val01 * dc
    bottom = val10 * (1.0 - dc) + val11 * dc
    equalized = top * (1.0 - dr) + bottom * dr
    
    return np.clip(equalized, 0, 255).astype(np.uint8)

def convert_color_space(img_rgb_uint8: np.ndarray, target_space: str) -> np.ndarray:
    """Convert RGB image array (H, W, 3) in [0, 255] to specified color space.
    
    Supported spaces:
    - 'RGB': identity, scaled to [0.0, 1.0] float32.
    - 'HSV': Hue, Saturation, Value normalized to [0.0, 1.0] float32.
    - 'CIELAB': Lightness, a* opponent, b* opponent normalized to [0.0, 1.0] float32.
    """
    target = target_space.upper()
    pil_img = Image.fromarray(img_rgb_uint8)
    
    if target == "RGB":
        return img_rgb_uint8.astype(np.float32) / 255.0
    elif target == "HSV":
        hsv_img = pil_img.convert("HSV")
        return np.array(hsv_img, dtype=np.float32) / 255.0
    elif target in ("CIELAB", "LAB"):
        lab_img = pil_img.convert("LAB")
        return np.array(lab_img, dtype=np.float32) / 255.0
    else:
        raise ValueError(f"Unsupported color space: {target_space}. Must be RGB, HSV, or CIELAB.")

def preprocess_image(
    img_rgb_uint8: np.ndarray,
    color_space: str = "RGB",
    illumination_norm: str = "none",
    clip_limit: float = 2.0,
    grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Full preprocessing pipeline: optional illumination normalization + color conversion.
    
    Args:
        img_rgb_uint8: Input image (H, W, 3) uint8 in RGB.
        color_space: Target color space ('RGB', 'HSV', 'CIELAB').
        illumination_norm: 'none', 'clahe' (adaptive equalization).
        clip_limit: CLAHE clip limit.
        grid_size: CLAHE grid size (default: 8x8).
        
    Returns:
        Processed image array (H, W, 3) float32 in [0.0, 1.0].
    """
    target = color_space.upper()
    norm = illumination_norm.lower()
    
    if norm == "none":
        return convert_color_space(img_rgb_uint8, target)
        
    elif norm == "clahe":
        pil_img = Image.fromarray(img_rgb_uint8)
        
        if target == "RGB":
            # Apply CLAHE across each RGB channel
            eq_channels = [
                apply_clahe_channel(img_rgb_uint8[:, :, c], clip_limit, grid_size)
                for c in range(3)
            ]
            eq_rgb = np.stack(eq_channels, axis=-1)
            return eq_rgb.astype(np.float32) / 255.0
            
        elif target == "HSV":
            # Apply CLAHE strictly to V (Value / Luminance) channel
            hsv_arr = np.array(pil_img.convert("HSV"), dtype=np.uint8)
            hsv_arr[:, :, 2] = apply_clahe_channel(hsv_arr[:, :, 2], clip_limit, grid_size)
            return hsv_arr.astype(np.float32) / 255.0
            
        elif target in ("CIELAB", "LAB"):
            # Apply CLAHE strictly to L (Lightness / Luminance) channel
            lab_arr = np.array(pil_img.convert("LAB"), dtype=np.uint8)
            lab_arr[:, :, 0] = apply_clahe_channel(lab_arr[:, :, 0], clip_limit, grid_size)
            return lab_arr.astype(np.float32) / 255.0
        else:
            raise ValueError(f"Unknown color space: {color_space}")
    else:
        raise ValueError(f"Unsupported illumination normalization: {illumination_norm}")

def paired_resize(
    image: Image.Image,
    mask: Image.Image,
    target_size: Tuple[int, int] = (512, 512)
) -> Tuple[Image.Image, Image.Image]:
    """Deterministic paired resize preserving discrete categorical mask integrity.
    
    Image uses BICUBIC resampling for smooth anatomical rendering.
    Categorical mask uses strict NEAREST resampling to prevent synthetic label creation.
    """
    w, h = target_size
    img_res = image.resize((w, h), Image.Resampling.BICUBIC)
    mask_res = mask.resize((w, h), Image.Resampling.NEAREST)
    return img_res, mask_res

def paired_geometric_transform(
    image_arr: np.ndarray,
    mask_arr: np.ndarray,
    hflip: bool = False,
    vflip: bool = False,
    rot90_k: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply identical spatial transformations to image and categorical mask."""
    img_out = image_arr.copy()
    mask_out = mask_arr.copy()
    
    if hflip:
        img_out = np.fliplr(img_out)
        mask_out = np.fliplr(mask_out)
        
    if vflip:
        img_out = np.flipud(img_out)
        mask_out = np.flipud(mask_out)
        
    if rot90_k > 0:
        img_out = np.rot90(img_out, k=rot90_k, axes=(0, 1))
        mask_out = np.rot90(mask_out, k=rot90_k, axes=(0, 1))
        
    return img_out, mask_out

def image_only_color_jitter(
    img_rgb_uint8: np.ndarray,
    brightness_factor: float = 1.0,
    contrast_factor: float = 1.0
) -> np.ndarray:
    """Color/photometric perturbation applied strictly to image array (mask untouched)."""
    img_f = img_rgb_uint8.astype(np.float32) * brightness_factor
    mean = np.mean(img_f, axis=(0, 1), keepdims=True)
    img_f = (img_f - mean) * contrast_factor + mean
    return np.clip(img_f, 0, 255).astype(np.uint8)
