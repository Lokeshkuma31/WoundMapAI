# Notebooks Workflow & Colab Execution

This directory contains standalone, reproducible Jupyter notebooks designed to be opened directly in Google Colab or run locally in JupyterLab:

1. `00_dataset_audit.ipynb`: Automated inspection of WoundTissue image-mask pairs, color distributions, dimensions, and manifest generation.
2. `01_preprocessing_eda.ipynb`: Visual validation of transforms, categorical mask resizing, color space conversions (RGB, HSV, CIELAB), and illumination adjustments.
3. `02_baseline_training.ipynb`: Self-contained Colab notebook for training the baseline U-Net / FPN segmentation model on GPU.
4. `03_ablation_experiments.ipynb`: Systematic comparison of color spaces and illumination normalization pipelines.
5. `04_evaluation_error_analysis.ipynb`: Test-set evaluation, confusion matrix generation, Dice/IoU per-class calculations, and visual error analysis.

### Colab Execution Mode
- Each notebook begins with standard environment bootstrap cells:
  ```python
  # Clone repository or pull latest feature branch
  !git clone https://github.com/Lokeshkuma31/WoundMapAI.git
  %cd WoundMapAI
  !pip install -r requirements.txt
  ```
- Checkpoints and large artifacts are designated for saving to Google Drive (`/content/drive/MyDrive/WoundMapAI/checkpoints/`).

