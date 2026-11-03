"""WoundMap AI - Interactive Research Inspection Demonstration.

Streamlit web application for interactive wound tissue semantic segmentation,
color representation analysis, and surface area composition measurement.

RESEARCH PROTOTYPE ONLY - NOT FOR MEDICAL DIAGNOSIS OR CLINICAL USE.
"""
from __future__ import annotations

import io
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.woundmap.masks import (
    CLASS_NAMES,
    HEX_COLORS,
    NUM_CLASSES,
    PALETTE_MAP,
    mask_indices_to_rgb,
)
from src.woundmap.models import UNet
from src.woundmap.preprocessing import paired_resize, preprocess_image

# Page configuration
st.set_page_config(
    page_title="WoundMap AI Research Demo",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Biophysical tissue descriptions for clinical context
TISSUE_DESCRIPTIONS = {
    "Background": "Periwound skin, surrounding anatomy, surgical drape, or background (RGB: [0, 0, 255])",
    "Granulation": "Healthy vascularized collagen, high capillary density, proliferative stage (RGB: [255, 0, 0])",
    "Slough": "Devitalized extracellular fibrin, leukocytes, and proteinaceous debris (RGB: [255, 255, 0])",
    "Necrotic": "Avascular dead eschar, dehydrated collagenous crust (RGB: [0, 0, 0])",
    "Maceration": "Moisture-softened hyper-hydrated epidermal wound margin (RGB: [255, 255, 255])",
    "Bone": "Deep structural exposed osseous tissue (RGB: [255, 200, 124])",
    "Tendon": "Dense regular fibrous connective tissue / fascia (RGB: [0, 191, 255])",
}


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: str, in_channels: int = 3, num_classes: int = NUM_CLASSES) -> Optional[UNet]:
    """Load and cache trained PyTorch U-Net checkpoint, dynamically adapting base_features."""
    if not os.path.exists(checkpoint_path):
        return None
    try:
        state = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
        weights = state["model_state_dict"] if (isinstance(state, dict) and "model_state_dict" in state) else state
        # Automatically detect base_features from first convolutional layer
        base_features = weights["inc.block.0.weight"].shape[0] if "inc.block.0.weight" in weights else 16
        model = UNet(in_channels=in_channels, num_classes=num_classes, base_features=base_features)
        model.load_state_dict(weights)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading checkpoint {checkpoint_path}: {e}")
        return None


def run_segmentation(
    model: UNet,
    image_np: np.ndarray,
    color_space: str = "RGB",
    illumination_norm: str = "none",
    target_size: Tuple[int, int] = (256, 256),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess image, execute forward pass, and compute color mask and blend overlay.
    
    Returns:
        resized_img: (H, W, 3) uint8 image.
        pred_indices: (H, W) uint8 categorical class index map.
        pred_rgb: (H, W, 3) uint8 colored segmentation mask.
    """
    resized_img, _ = paired_resize(image_np, target_size=target_size)
    processed_np = preprocess_image(
        resized_img,
        color_space=color_space,
        illumination_norm=illumination_norm,
    )
    tensor_input = torch.from_numpy(np.ascontiguousarray(processed_np)).permute(2, 0, 1).unsqueeze(0).float()
    with torch.no_grad():
        logits = model(tensor_input)
        preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    pred_rgb = mask_indices_to_rgb(preds)
    return resized_img, preds, pred_rgb


def compute_composition(pred_indices: np.ndarray) -> pd.DataFrame:
    """Compute surface area breakdown by pixel count, % of total FOV, and % of wound bed."""
    total_pixels = int(pred_indices.size)
    background_pixels = int(np.sum(pred_indices == 0))
    wound_pixels = total_pixels - background_pixels

    rows = []
    for cls_id, name in CLASS_NAMES.items():
        px = int(np.sum(pred_indices == cls_id))
        pct_total = (px / total_pixels) * 100.0 if total_pixels > 0 else 0.0
        pct_wound = (px / wound_pixels) * 100.0 if (wound_pixels > 0 and cls_id > 0) else 0.0

        rows.append({
            "Class ID": cls_id,
            "Tissue Type": name,
            "Color (Hex)": HEX_COLORS[cls_id],
            "Pixel Count": px,
            "% of Total Image": round(pct_total, 2),
            "% of Wound Bed": round(pct_wound, 2),
        })
    return pd.DataFrame(rows)


def build_overlay(resized_img: np.ndarray, pred_rgb: np.ndarray, alpha: float) -> np.ndarray:
    """Alpha-blend the RGB segmentation mask onto the original image."""
    blend = (1.0 - alpha) * resized_img.astype(np.float32) + alpha * pred_rgb.astype(np.float32)
    return np.clip(blend, 0, 255).astype(np.uint8)


def main():
    st.title("🔬 WoundMap AI: Multi-Class Wound Tissue Segmentation")
    st.caption("Reproducible biophysical computer-vision research prototype for tissue composition analysis.")

    # Scientific Safeguard Banner
    st.warning(
        """
        ⚠️ **RESEARCH PROTOTYPE ONLY — NOT FOR MEDICAL DIAGNOSIS OR CLINICAL TREATMENT**
        
        This demonstration is an academic research prototype evaluating multi-class semantic segmentation
        across color representations (sRGB, HSV, CIELAB) and localized illumination normalisation (CLAHE).
        **Ordinary RGB photography cannot diagnose perfusion, tissue oxygenation, bacterial infection,
        or histological depth.** Surface appearances do not replace clinical wound care or histological biopsy.
        """
    )

    # Sidebar: Model and Configuration Controls
    st.sidebar.header("⚙️ Experiment Configuration")

    checkpoints_dir = REPO_ROOT / "outputs" / "checkpoints"
    available_models = {
        "Baseline U-Net (sRGB Raw)": {
            "path": checkpoints_dir / "baseline" / "best_baseline_unet_rgb.pt",
            "color_space": "RGB",
            "illumination_norm": "none",
            "notes": "Trained on raw sRGB. Best global macro dice (0.3006).",
        },
        "RUN-06: CIELAB + CLAHE-L (High Necrotic Sensitivity)": {
            "path": checkpoints_dir / "ablations" / "RUN-06" / "best_model.pt",
            "color_space": "CIELAB",
            "illumination_norm": "clahe",
            "notes": "Lightness CLAHE. Highest sensitivity for necrotic tissue in shadowed cavities.",
        },
        "RUN-01: sRGB Baseline Ablation": {
            "path": checkpoints_dir / "ablations" / "RUN-01" / "best_model.pt",
            "color_space": "RGB",
            "illumination_norm": "none",
            "notes": "Ablation run under locked split.",
        },
        "RUN-02: HSV Raw": {
            "path": checkpoints_dir / "ablations" / "RUN-02" / "best_model.pt",
            "color_space": "HSV",
            "illumination_norm": "none",
            "notes": "Ablation run in cylindrical HSV color space.",
        },
        "RUN-03: CIELAB Raw": {
            "path": checkpoints_dir / "ablations" / "RUN-03" / "best_model.pt",
            "color_space": "CIELAB",
            "illumination_norm": "none",
            "notes": "Ablation run in perceptual CIELAB color space.",
        },
        "RUN-04: sRGB + CLAHE": {
            "path": checkpoints_dir / "ablations" / "RUN-04" / "best_model.pt",
            "color_space": "RGB",
            "illumination_norm": "clahe",
            "notes": "sRGB with localized contrast normalisation.",
        },
        "RUN-05: HSV + CLAHE-V": {
            "path": checkpoints_dir / "ablations" / "RUN-05" / "best_model.pt",
            "color_space": "HSV",
            "illumination_norm": "clahe",
            "notes": "HSV with Value-channel localized CLAHE.",
        },
    }

    selected_model_name = st.sidebar.selectbox("Select Model Condition", list(available_models.keys()))
    model_info = available_models[selected_model_name]

    # Preprocessing parameter controls (auto-populated by model selection, customizable)
    color_space = st.sidebar.selectbox(
        "Color Space",
        ["RGB", "HSV", "CIELAB"],
        index=["RGB", "HSV", "CIELAB"].index(model_info["color_space"]),
    )
    illum_norm = st.sidebar.selectbox(
        "Illumination Normalisation",
        ["none", "clahe"],
        index=0 if model_info["illumination_norm"] == "none" else 1,
    )
    alpha = st.sidebar.slider("Overlay Blend Opacity (α)", min_value=0.0, max_value=1.0, value=0.45, step=0.05)

    st.sidebar.markdown(f"**Condition Details:** {model_info['notes']}")

    # Quick Sample Loader
    sample_images = {
        "Select sample image...": None,
        "Test Sample 1: img (20).png (Necrotic & Maceration)": REPO_ROOT / "data" / "raw" / "WoundTissue" / "image" / "img (20).png",
        "Test Sample 2: img (27).png (Granulation & Slough)": REPO_ROOT / "data" / "raw" / "WoundTissue" / "image" / "img (27).png",
        "Validation Sample: img (12).png (Large Wound Bed)": REPO_ROOT / "data" / "raw" / "WoundTissue" / "image" / "img (12).png",
        "Training Sample: img (1).png (Multi-Tissue)": REPO_ROOT / "data" / "raw" / "WoundTissue" / "image" / "img (1).png",
    }
    selected_sample = st.sidebar.selectbox("Or Load a Benchmark Sample", list(sample_images.keys()))

    uploaded_file = st.file_uploader(
        "Upload Wound Image (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
        help="Upload an ordinary RGB wound photograph for tissue semantic segmentation.",
    )

    image_to_process: Optional[np.ndarray] = None
    image_name = "sample_image.png"

    if uploaded_file is not None:
        try:
            pil_img = Image.open(uploaded_file).convert("RGB")
            image_to_process = np.array(pil_img)
            image_name = uploaded_file.name
        except Exception as e:
            st.error(f"Failed to load uploaded image: {e}")
    elif selected_sample != "Select sample image..." and sample_images[selected_sample] is not None:
        sample_path = sample_images[selected_sample]
        if sample_path.exists():
            pil_img = Image.open(sample_path).convert("RGB")
            image_to_process = np.array(pil_img)
            image_name = sample_path.name

    if image_to_process is None:
        st.info("👆 Please upload a wound photograph or select a benchmark sample image from the sidebar to begin.")
        return

    # Load selected model
    checkpoint_path = str(model_info["path"])
    model = load_model(checkpoint_path)

    if model is None:
        st.error(f"Model checkpoint not found at: `{checkpoint_path}`. Please verify training checkpoints.")
        return

    # Execute segmentation pipeline
    with st.spinner("Running tissue segmentation and biophysical analysis..."):
        resized_img, pred_indices, pred_rgb = run_segmentation(
            model=model,
            image_np=image_to_process,
            color_space=color_space,
            illumination_norm=illum_norm,
        )
        overlay_img = build_overlay(resized_img, pred_rgb, alpha=alpha)
        composition_df = compute_composition(pred_indices)

    # Visual Inspection Panels
    st.subheader("🖼️ Tissue Segmentation & Blend Visualisation")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**1. Original Input (256×256)**")
        st.image(resized_img, use_container_width=True)

    with col2:
        st.markdown("**2. Predicted Tissue Mask**")
        st.image(pred_rgb, use_container_width=True)

    with col3:
        st.markdown(f"**3. Alpha-Blend Overlay (α = {alpha:.2f})**")
        st.image(overlay_img, use_container_width=True)

    # Tissue Palette Legend
    with st.expander("🎨 Clinical Tissue Palette & Legend", expanded=True):
        legend_cols = st.columns(len(CLASS_NAMES))
        for idx, (cid, cname) in enumerate(CLASS_NAMES.items()):
            hex_color = HEX_COLORS[cid]
            with legend_cols[idx]:
                st.markdown(
                    f"<div style='background-color:{hex_color}; width:100%; height:20px; border-radius:3px; border:1px solid #999;'></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**{cname}**")
                st.caption(TISSUE_DESCRIPTIONS[cname])

    # Quantitative Surface Area Analysis
    st.subheader("📊 Quantitative Tissue Surface Area Analysis")
    
    total_px = int(pred_indices.size)
    bg_px = int(np.sum(pred_indices == 0))
    wound_px = total_px - bg_px

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Total Field of View (FOV)", f"{total_px:,} pixels")
    metric_col2.metric("Wound Bed Area (Excl. Background)", f"{wound_px:,} pixels", f"{(wound_px/total_px)*100:.1f}% of FOV")
    metric_col3.metric("Background / Non-Wound Area", f"{bg_px:,} pixels", f"{(bg_px/total_px)*100:.1f}% of FOV")

    # Display Breakdown Table & Chart
    tab1, tab2 = st.tabs(["📋 Composition Table", "📈 Composition Bar Chart"])

    with tab1:
        st.dataframe(composition_df, use_container_width=True, hide_index=True)

    with tab2:
        # Chart tissue distribution (excluding background to emphasize wound bed)
        wound_chart_data = composition_df[composition_df["Class ID"] > 0].copy()
        wound_chart_data["% of Wound Bed"] = pd.to_numeric(wound_chart_data["% of Wound Bed"])
        st.bar_chart(
            data=wound_chart_data,
            x="Tissue Type",
            y="% of Wound Bed",
            color="Tissue Type",
        )

    # Exportable Report CSV
    st.subheader("📥 Export Tissue Composition Report")
    csv_buffer = io.StringIO()
    # Write metadata header
    csv_buffer.write(f"# WoundMap AI Tissue Composition Report\n")
    csv_buffer.write(f"# Timestamp: {datetime.now().isoformat()}\n")
    csv_buffer.write(f"# Input Image: {image_name}\n")
    csv_buffer.write(f"# Model Condition: {selected_model_name}\n")
    csv_buffer.write(f"# Checkpoint: {checkpoint_path}\n")
    csv_buffer.write(f"# Color Space: {color_space}\n")
    csv_buffer.write(f"# Illumination Normalisation: {illum_norm}\n")
    csv_buffer.write(f"# Total FOV Pixels: {total_px}\n")
    csv_buffer.write(f"# Wound Bed Pixels: {wound_px}\n")
    csv_buffer.write(f"# Disclaimer: Research prototype only. Not for medical diagnosis.\n")
    composition_df.to_csv(csv_buffer, index=False)
    
    report_filename = f"woundmap_report_{Path(image_name).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    st.download_button(
        label="📄 Download Composition Report (CSV)",
        data=csv_buffer.getvalue(),
        file_name=report_filename,
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
