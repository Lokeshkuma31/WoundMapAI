# WoundMap AI

**Biophysics & Biomedical Computer-Vision Research Prototype for Wound-Tissue Semantic Segmentation**

---

## 1. Project Overview
WoundMap AI is an academic research prototype developed as part of an undergraduate project (UGP) in biophysics and biomedical computer vision. The core objective is to evaluate multi-class semantic segmentation models on clinical wound photographs (using the WoundTissue dataset), with specific emphasis on how colour representations (RGB, HSV, CIELAB) and illumination normalisation impact segmentation robustness and tissue delineation.

### Non-Clinical Disclaimer
> [!WARNING]
> **Research Prototype Only**: This software is built strictly for scientific research and educational investigation into computer-vision techniques on biomedical imagery. It is **not** a medical device, is **not** validated for clinical diagnosis, prognosis, perfusion assessment, or treatment planning, and must never be used to direct patient care.

---

## 2. Research Focus & Scientific Safeguards
The project investigates:
- Multi-class tissue classification (e.g., granulation, slough, necrotic/eschar, epithelializing, background).
- Colour space invariance: Comparing input representations in standard RGB, Hue-Saturation-Value (HSV), and CIE $L^*a^*b^*$ under varying illumination conditions.
- Methodological rigor: Patient/wound-level splitting to prevent data leakage, strict evaluation on untouched test partitions, and per-class Dice/IoU reporting with explicit absent-class handling.

---

## 3. Repository Architecture
```text
WoundMapAI/
├── AGENTS.md                  # Persistent agent rules, safeguards, and autonomy boundaries
├── README.md                  # Setup, reproduction instructions, and project documentation
├── PLAN.md                    # Multi-phase execution dashboard, milestones, and gate criteria
├── CHANGELOG.md               # Chronological log of changes, commits, and handoffs
├── requirements.txt           # Python dependency specification
├── .gitignore                 # Exclusion rules for raw patient data, checkpoints, and secrets
├── configs/                   # Declarative YAML experiment configs
│   ├── dataset.yaml
│   ├── split.yaml
│   ├── baseline.yaml
│   └── ablations.yaml
├── data/
│   ├── README.md              # Dataset origin, licence, and expected layout
│   └── manifests/             # Versioned split manifests (CSV metadata only)
├── notebooks/                 # Runnable notebooks for Colab and local execution
│   ├── 00_dataset_audit.ipynb
│   ├── 01_preprocessing_eda.ipynb
│   ├── 02_baseline_training.ipynb
│   ├── 03_ablation_experiments.ipynb
│   └── 04_evaluation_error_analysis.ipynb
├── src/woundmap/              # Modular, reusable source package
│   ├── __init__.py
│   ├── data.py                # Dataset loaders & paired reading
│   ├── masks.py               # Mask colour decoding & categorical mapping
│   ├── preprocessing.py       # Geometric transforms, colour conversions, normalisation
│   ├── models.py              # Segmentation architectures (U-Net / FPN)
│   ├── train.py               # Training, loss computation, and checkpointing
│   ├── metrics.py             # Dice, IoU, confusion matrix calculation
│   └── inference.py           # Single-image inference pipeline
├── tests/                     # Automated unit and sanity test suite
│   ├── test_mask_mapping.py
│   ├── test_pairing.py
│   ├── test_transforms.py
│   └── test_metrics.py
├── app/                       # Streamlit web demonstration
│   └── streamlit_app.py
├── reports/                   # Academic writeups and protocol logs
│   ├── literature_review.md
│   ├── experiment_protocol.md
│   ├── results.md
│   └── limitations.md
└── presentation/              # Presentation slides and visual assets
```

---

## 4. Execution Workflow
1. **Local Environment**:
   - Platform: Windows 11 / Python 3.14.
   - Purpose: Code development, unit testing, data manifest generation, lightweight CPU smoke tests, Streamlit demo.
2. **Google Colab Environment**:
   - Hardware: High-memory GPU runtime (T4 / A100).
   - Purpose: End-to-end dataset training, hyperparameter sweeps, and ablation runs.
   - Workflow: Self-contained Jupyter notebooks with automated environment setup and Drive checkpoint preservation.

---

## 5. Getting Started
```powershell
# Clone the repository
git clone https://github.com/Lokeshkuma31/WoundMapAI.git
cd WoundMapAI

# Install dependencies
pip install -r requirements.txt

# Run automated tests
pytest tests/
```
