"""Generate visual evidence and mock dashboard snapshot for the Streamlit demo using pure Pillow."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.streamlit_app import (
    build_overlay,
    compute_composition,
    load_model,
    run_segmentation,
)
from src.woundmap.masks import CLASS_NAMES, HEX_COLORS, PALETTE_MAP


def main():
    os.makedirs(REPO_ROOT / "reports" / "figures", exist_ok=True)
    checkpoint_path = REPO_ROOT / "outputs" / "checkpoints" / "baseline" / "best_baseline_unet_rgb.pt"
    model = load_model(str(checkpoint_path))

    if model is None:
        print("Model checkpoint not found!")
        return

    sample_path = REPO_ROOT / "data" / "raw" / "WoundTissue" / "image" / "img (20).png"
    img_pil = Image.open(sample_path).convert("RGB")
    img_np = np.array(img_pil)

    resized_img, pred_indices, pred_rgb = run_segmentation(
        model=model,
        image_np=img_np,
        color_space="RGB",
        illumination_norm="none",
        target_size=(256, 256),
    )
    overlay_img = build_overlay(resized_img, pred_rgb, alpha=0.45)
    df = compute_composition(pred_indices)

    # Canvas dimensions
    canvas_w = 980
    canvas_h = 750
    canvas = Image.new("RGB", (canvas_w, canvas_h), (248, 249, 250))
    draw = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype("arialbd.ttf", 20)
        font_sub = ImageFont.truetype("arial.ttf", 13)
        font_header = ImageFont.truetype("arialbd.ttf", 14)
        font_body = ImageFont.truetype("arial.ttf", 12)
        font_body_bold = ImageFont.truetype("arialbd.ttf", 12)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = font_title
        font_header = font_title
        font_body = font_title
        font_body_bold = font_title

    # Header title
    draw.rectangle([(0, 0), (canvas_w, 65)], fill=(24, 43, 73))
    draw.text((25, 12), "WoundMap AI - Streamlit Research Demonstration Dashboard", fill=(255, 255, 255), font=font_title)
    draw.text((25, 40), "Multi-Class Tissue Segmentation, Color Space Inspection & Quantitative Composition Analysis", fill=(180, 205, 235), font=font_sub)

    # Scientific disclaimer callout box
    draw.rectangle([(25, 80), (canvas_w - 25, 125)], fill=(254, 243, 199), outline=(245, 158, 11), width=1)
    draw.text(
        (35, 87),
        "RESEARCH PROTOTYPE ONLY - NOT FOR MEDICAL DIAGNOSIS OR CLINICAL USE",
        fill=(146, 64, 14),
        font=font_body_bold,
    )
    draw.text(
        (35, 105),
        "Ordinary RGB photographs cannot diagnose perfusion, tissue oxygenation, bacterial presence, or depth of injury.",
        fill=(180, 83, 9),
        font=font_sub,
    )

    # Panels row: Input Image, Predicted Mask, Alpha Blend Overlay
    panel_y = 145
    panel_spacing = 35
    col_w = 270

    # 1. Input Image
    x1 = 30
    draw.text((x1, panel_y), "1. Resized Input (256x256)", fill=(30, 41, 59), font=font_header)
    canvas.paste(Image.fromarray(resized_img).resize((col_w, col_w)), (x1, panel_y + 25))
    draw.rectangle([(x1, panel_y + 25), (x1 + col_w, panel_y + 25 + col_w)], outline=(203, 213, 225), width=2)

    # 2. Predicted Mask
    x2 = x1 + col_w + panel_spacing
    draw.text((x2, panel_y), "2. Predicted Tissue Mask", fill=(30, 41, 59), font=font_header)
    canvas.paste(Image.fromarray(pred_rgb).resize((col_w, col_w)), (x2, panel_y + 25))
    draw.rectangle([(x2, panel_y + 25), (x2 + col_w, panel_y + 25 + col_w)], outline=(203, 213, 225), width=2)

    # 3. Alpha-Blend
    x3 = x2 + col_w + panel_spacing
    draw.text((x3, panel_y), "3. Overlay Blend (alpha=0.45)", fill=(30, 41, 59), font=font_header)
    canvas.paste(Image.fromarray(overlay_img).resize((col_w, col_w)), (x3, panel_y + 25))
    draw.rectangle([(x3, panel_y + 25), (x3 + col_w, panel_y + 25 + col_w)], outline=(203, 213, 225), width=2)

    # Lower section: Table of quantitative composition & Legend
    table_y = panel_y + col_w + 40
    draw.text((30, table_y), "Quantitative Tissue Surface Area Breakdown & Clinical Palette", fill=(30, 41, 59), font=font_header)

    # Table header
    th_y = table_y + 25
    draw.rectangle([(30, th_y), (canvas_w - 30, th_y + 26)], fill=(30, 41, 59))
    draw.text((45, th_y + 6), "Class / Tissue Type", fill=(255, 255, 255), font=font_body_bold)
    draw.text((280, th_y + 6), "Color Badge", fill=(255, 255, 255), font=font_body_bold)
    draw.text((430, th_y + 6), "Pixel Count", fill=(255, 255, 255), font=font_body_bold)
    draw.text((580, th_y + 6), "% Total Image (FOV)", fill=(255, 255, 255), font=font_body_bold)
    draw.text((760, th_y + 6), "% Wound Bed Area", fill=(255, 255, 255), font=font_body_bold)

    # Table rows
    row_y = th_y + 26
    row_h = 24
    for idx, row in df.iterrows():
        cid = int(row["Class ID"])
        bg_col = (255, 255, 255) if idx % 2 == 0 else (241, 245, 249)
        draw.rectangle([(30, row_y), (canvas_w - 30, row_y + row_h)], fill=bg_col, outline=(226, 232, 240), width=1)

        # Tissue name
        draw.text((45, row_y + 4), f"{cid}: {row['Tissue Type']}", fill=(15, 23, 42), font=font_body)

        # Color swatch
        color_rgb = PALETTE_MAP[cid]
        draw.rectangle([(280, row_y + 4), (350, row_y + row_h - 4)], fill=color_rgb, outline=(100, 100, 100), width=1)

        # Pixel count
        draw.text((430, row_y + 4), f"{int(row['Pixel Count']):,} px", fill=(15, 23, 42), font=font_body)

        # % of Total
        draw.text((580, row_y + 4), f"{row['% of Total Image']:.1f}%", fill=(15, 23, 42), font=font_body)

        # % of Wound Bed
        wound_pct_str = f"{row['% of Wound Bed']:.1f}%" if cid > 0 else "N/A (Background)"
        draw.text((760, row_y + 4), wound_pct_str, fill=(15, 23, 42), font=font_body_bold if cid > 0 else font_body)

        row_y += row_h

    # Stated denominators summary banner
    total_px = int(pred_indices.size)
    bg_px = int(np.sum(pred_indices == 0))
    wound_px = total_px - bg_px
    summary_y = row_y + 12
    draw.rectangle([(30, summary_y), (canvas_w - 30, summary_y + 30)], fill=(238, 242, 255), outline=(199, 210, 254), width=1)
    summary_text = (
        f"Stated Denominators: Total Field of View = {total_px:,} px (100.0%) | "
        f"Wound Bed (Excl. Background) = {wound_px:,} px ({(wound_px/total_px)*100:.1f}%) | "
        f"Periwound/Background = {bg_px:,} px ({(bg_px/total_px)*100:.1f}%)"
    )
    draw.text((40, summary_y + 8), summary_text, fill=(49, 46, 129), font=font_body_bold)

    out_path = REPO_ROOT / "reports" / "figures" / "streamlit_demo_preview.png"
    canvas.save(out_path)
    print(f"Generated dashboard preview at: {out_path}")


if __name__ == "__main__":
    main()
