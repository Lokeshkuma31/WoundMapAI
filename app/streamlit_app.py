"""WoundMap AI - Interactive Research Inspection Demonstration.

RESEARCH PROTOTYPE ONLY - NOT FOR MEDICAL OR DIAGNOSTIC USE.
"""
import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="WoundMap AI Research Demo",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 WoundMap AI: Tissue Segmentation Prototype")
st.markdown(
    """
    > ⚠️ **Non-Clinical Research Notice**:
    > This application is an undergraduate computer-vision research demonstration evaluating multi-class tissue
    > segmentation across color spaces (RGB, HSV, CIELAB). It is **not** a clinical diagnostic tool and must not
    > be used to determine patient treatment or prognosis.
    """
)

st.sidebar.header("Model Settings")
model_ckpt = st.sidebar.selectbox("Checkpoint", ["baseline_unet_rgb (pending training)"])
color_space = st.sidebar.selectbox("Color Space", ["RGB", "HSV", "CIELAB"])

uploaded_file = st.file_uploader("Upload Wound Photograph", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input Image")
        st.image(image, use_column_width=True)
    with col2:
        st.subheader("Predicted Tissue Segmentation")
        st.info("Model checkpoint will be integrated upon completion of Phase 4 & Phase 7.")

