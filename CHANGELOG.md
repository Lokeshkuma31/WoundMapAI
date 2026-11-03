# Changelog

All notable changes to the WoundMap AI research prototype will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased] - Phase 1: Dataset Acquisition & Audit
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
