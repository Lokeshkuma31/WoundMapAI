# Changelog

All notable changes to the WoundMap AI research prototype will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased] - Phase 5: Evaluation & Error Analysis
### Added
- Evaluated baseline U-Net checkpoint on strictly untouched test partition (`data/manifests/test.csv`, 131,072 pixels).
- Implemented comprehensive evaluation metrics engine (`src/woundmap/metrics.py`) computing confusion matrices, per-class Dice, IoU, Precision, Recall, Support, and Dual Macro Averages with explicit absent-class handling.
- Implemented inference latency benchmark measuring runtime and throughput on target resolution (42.61 ms, 23.5 FPS on CPU).
- Exported empirical test results to `reports/test_metrics_summary.json` and `reports/test_metrics_per_class.csv`, and updated `reports/results.md`.
- Generated normalized pixel confusion matrix heatmap (`reports/figures/test_confusion_matrix.png`), qualitative test panels (`reports/figures/test_prediction_panels.png`), and boundary error analysis maps (`reports/figures/test_boundary_error_analysis.png`).
- Authored test evaluation walkthrough notebook (`notebooks/04_evaluation_error_analysis.ipynb`).
- Identified low-intensity necrotic tissue confusion with shadows as primary baseline failure mode, motivating Phase 6 ablations. Expanded test suite to 19/19 passing tests.

## [0.5.0] - Phase 4: Baseline Model Implementation
### Added
- Implemented modular 7-class convolutional U-Net (`src/woundmap/models.py`) with configurable base features, contracting path, and expanding skip-connection decoder.
- Implemented composite loss function ($0.5 \times \text{Soft Dice} + 0.5 \times \text{Cross-Entropy}$) and complete training engine (`src/woundmap/train.py`) with metric tracking, early stopping, and validation checkpointing.
- Implemented Python 3.14 PyTorch optimizer compatibility shim (`src/woundmap/compat.py`) to resolve pre-release dynamo circular imports.
- Built mandatory automated test suite (`tests/test_model_output.py`) covering output tensor shapes, composite gradient flows, and one-batch overfit convergence (>50% loss drop over 25 steps). Expanded test suite to 16/16 passing tests.
- Executed local baseline CPU smoke training run (10 epochs, Val Macro Dice peaked at 0.7719), saving checkpoint `outputs/checkpoints/baseline/best_baseline_unet_rgb.pt` (10.1MB, git-ignored) and `training_history.json`.
- Generated visual training curves (`reports/figures/baseline_training_curves.png`) and validation prediction overlay panels (`reports/figures/baseline_prediction_overlays.png`).
- Authored Colab-ready baseline training notebook (`notebooks/02_baseline_training.ipynb`) with 1-click execution badge.

## [0.4.0] - Phase 3: Preprocessing & Exploratory Data Analysis
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
