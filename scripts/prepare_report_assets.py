"""Prepare individual figure assets for the LaTeX academic report."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
fig_dir = REPO_ROOT / "report" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# 1. Load preview
preview_path = fig_dir / "streamlit_demo_preview.png"
if preview_path.exists():
    img = Image.open(preview_path)
    w, h = img.size

    # Crop matched triplet (Panel 1, 2, 3)
    triplet = img.crop((15, 135, w - 15, 450))
    triplet.save(fig_dir / "fig_triplet.png")

    # Crop palette and quantitative summary table
    table_img = img.crop((15, 460, w - 15, h - 10))
    table_img.save(fig_dir / "fig_palette_table.png")

# 2. Generate clean IIT (BHU) logo emblem
logo = Image.new("RGBA", (400, 400), (255, 255, 255, 0))
draw = ImageDraw.Draw(logo)

# Concentric circles in maroon (#800000)
draw.ellipse([(20, 20), (380, 380)], outline=(128, 0, 0, 255), width=8)
draw.ellipse([(45, 45), (355, 355)], outline=(128, 0, 0, 255), width=4)
draw.ellipse([(110, 110), (290, 290)], fill=(128, 0, 0, 255))

try:
    font_bold = ImageFont.truetype("arialbd.ttf", 26)
    font_small = ImageFont.truetype("arialbd.ttf", 18)
except Exception:
    font_bold = ImageFont.load_default()
    font_small = font_bold

draw.text((200, 75), "INDIAN INSTITUTE OF TECHNOLOGY", fill=(128, 0, 0, 255), font=font_small, anchor="mm")
draw.text((200, 200), "IIT (BHU)", fill=(255, 255, 255, 255), font=font_bold, anchor="mm")
draw.text((200, 325), "VARANASI", fill=(128, 0, 0, 255), font=font_small, anchor="mm")

logo.save(fig_dir / "iit_bhu_logo.png")
print("Report assets generated successfully.")
