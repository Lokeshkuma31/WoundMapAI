"""Color space transformations and illumination normalization routines."""
import numpy as np
from PIL import Image

def convert_color_space(img_rgb: np.ndarray, target_space: str) -> np.ndarray:
    """Convert RGB image array (H, W, 3) in range [0, 255] to specified color space.
    
    Supported spaces:
    - 'RGB': identity
    - 'HSV': Hue (0-1), Saturation (0-1), Value (0-1)
    - 'CIELAB': L* (0-100), a* (-128 to 127), b* (-128 to 127)
    """
    target = target_space.upper()
    if target == "RGB":
        return img_rgb.astype(np.float32) / 255.0
    
    # Using PIL or OpenCV equivalents
    pil_img = Image.fromarray(img_rgb)
    if target == "HSV":
        hsv_img = pil_img.convert("HSV")
        return np.array(hsv_img, dtype=np.float32) / 255.0
    elif target == "CIELAB" or target == "LAB":
        lab_img = pil_img.convert("LAB")
        return np.array(lab_img, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported color space: {target_space}")

