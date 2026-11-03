# Changelog

All notable changes to the WoundMap AI research prototype will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased] - Phase 3: Preprocessing & Exploratory Data Analysis
### Added
- Implemented pure NumPy vectorized CLAHE (`apply_clahe_channel`) with zero external C++ dependencies for localized contrast normalisation on luminance channels ($L^*$ and $V$).
- Implemented mathematical color conversions (`convert_color_space`, `preprocess_image`) supporting sRGB, HSV, and CIELAB normalized into $[0.0, 1.0]$ float32.
- Implemented deterministic paired resizing (`paired_resize`) using bicubic resampling for continuous images and strict nearest-neighbour resampling for categorical discrete masks.
- Built PyTorch dataset and data loader modules (`src/woundmap/data.py`: `WoundDataset`, `build_dataloader`) with paired geometric augmentations (flips, $90^\circ$ rotations) and image-only color jitter.
- Authored comprehensive, cell-by-cell pedagogical walkthrough notebook (`notebooks/01_preprocessing_eda.ipynb`) documenting every preprocessing stage.
- Generated high-resolution EDA figures: `reports/figures/color_spaces_eda.png`, `reports/figures/clahe_comparison.png`, and `reports/figures/augmentation_overlay.png`.
- Expanded automated unit test suite (`tests/test_transforms.py`), verifying 13/13 passing tests.

## [0.3.0] - Phase 2: Literature Review & Experiment Protocol
### Added
- Authored comprehensive literature review (`reports/literature_review.md`) citing clinical wound scoring (RYKW, PUSH, BWAT), hemoglobin/melanin biophysics, prior deep learning benchmarks (Kabir et al. 2025, FUSC, Medetec), and mathematical color models (sRGB, HSV, CIELAB, CLAHE).
- Developed formal experiment protocol (`reports/experiment_protocol.md`) defining research questions, formal hypotheses ($H_1, H_2, H_0$), controlled ablation matrix (6 runs), dual macro-averaging, and explicit absent-class handling policies.
- Generated locked dataset partition manifests (`data/manifests/train.csv` [9 images], `val.csv` [2 images], `test.csv` [2 images]) ensuring balanced class representation and leakage prevention.
- Added automated split integrity test suite (`tests/test_split_integrity.py`), confirming 9/9 passing tests with zero partition overlap.
- Updated declarative experiment configurations (`configs/split.yaml`, `configs/baseline.yaml`, `configs/ablations.yaml`).

## [0.2.0] - Phase 1: Dataset Acquisition & Audit
### Added
- Downloaded and audited `akabircs/WoundTissue` dataset release into `data/raw/WoundTissue/` (git-ignored).
- Generated comprehensive pairing manifest `data/manifests/manifest.csv` with per-image dimensions, class pixel counts, and percentages.
- Created `reports/figures/contact_sheet.png` rendering 13 image/mask pairs with 40% blend overlays.
- Created `notebooks/00_dataset_audit.ipynb` reproducing data audit checks.
- Updated `src/woundmap/masks.py` with audited `WOUNDTISSUE_PALETTE` supporting RGBA masks.
- Updated `configs/dataset.yaml` with verified classes, RGB color values, and citation.
- Documented findings in `data/README.md`, including disclosure of 13 available sample pairs and absence of patient metadata.
- Unit test suite expanded to 7 tests, all passing (`test_pairing.py` and `test_mask_mapping.py` active).

## [0.1.0] - Phase 0: Bootstrap & Scaffolding
### Added
- Created persistent agent rules (`AGENTS.md`) with explicit scientific and engineering safeguards.
- Created project tracking plan and dashboard (`PLAN.md`) detailing Phases 0 through 8 and acceptance gates.
- Created initial project documentation (`README.md`) with biophysics scope, non-clinical warning, and architecture.
- Added comprehensive `.gitignore` safeguarding medical raw images, patient identifiers, credentials, and model checkpoints.
- Established repository directory hierarchy: `configs/`, `data/`, `notebooks/`, `src/woundmap/`, `tests/`, `app/`, `reports/`, `presentation/`.
- Configured baseline project dependencies in `requirements.txt`.
- Documented environment integration capabilities (GitHub CLI, Google Colab hand-off, local GPU status).
