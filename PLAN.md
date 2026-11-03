# WoundMap AI — Project Execution Plan (PLAN.md)

## 1. Project Control Dashboard

| Phase | Description | Status | Evidence Link | Blocker | Human Approval |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0 Bootstrap** | Tool audit, repo structure, rules, git init | **ACCEPTED** | [Phase 0 PR #1](https://github.com/Lokeshkuma31/WoundMapAI/pull/1) | None | Approved by User |
| **1 Dataset Audit** | Inspect WoundTissue tree, pairs, class mapping | **ACCEPTED** | [Phase 1 PR #2](https://github.com/Lokeshkuma31/WoundMapAI/pull/2) | None | Approved by User |
| **2 Research Protocol** | Literature review, hypotheses, split policy | **ACCEPTED** | [Phase 2 PR #3](https://github.com/Lokeshkuma31/WoundMapAI/pull/3) | None | Approved by User |
| **3 Preprocessing & EDA**| Transforms, colour conversions, EDA overlays | **ACCEPTED** | [Phase 3 PR #4](https://github.com/Lokeshkuma31/WoundMapAI/pull/4) | None | Approved by User |
| **4 Baseline Model** | U-Net/FPN baseline, loss, training loop | **ACCEPTED** | [Phase 4 PR #5](https://github.com/Lokeshkuma31/WoundMapAI/pull/5) | None | Approved by User |
| **5 Evaluation** | Metrics (Dice/IoU), error analysis | **READY FOR GATE APPROVAL** | [Results Doc](reports/results.md) / [Evaluation Notebook](notebooks/04_evaluation_error_analysis.ipynb) | None | Pending User Sign-Off |
| **6 Controlled Ablations**| RGB vs HSV vs CIELAB + Illumination normalisation | NOT STARTED | - | Pending Phase 5 | Gate 6 Required |
| **7 Streamlit Demo** | Interactive research inspection demo | NOT STARTED | - | Pending Phase 6 | Gate 7 Required |
| **8 Report & Handover** | Academic report, presentation slides, reproducibility package | NOT STARTED | - | Pending Phase 7 | Gate 8 Required |

---

## 2. Phase Summaries & Acceptance Criteria

### Phase 0: Bootstrap and Integration Validation
- **Agent Work**: Inspect environment/tools, create repository structure, `AGENTS.md`, `PLAN.md`, `README.md`, `CHANGELOG.md`, `requirements.txt`, `.gitignore`. Test Git and GitHub CLI capabilities. Document Colab & Drive remote access reality.
- **Deliverables**: Local repository initialized, feature branch `phase/00-bootstrap`, complete project scaffolding, Integration & Capability Report.
- **Human Gate**: Confirm repo setup, permissions, and GitHub repository creation.

### Phase 1: Dataset Acquisition and Audit
- **Agent Work**: Obtain official WoundTissue dataset source/archive. Audit folders, image-mask pairing, pixel dimensions, colour encoding, unique RGB/hex mask values, class distribution, licence, and patient/wound IDs.
- **Deliverables**: `00_dataset_audit.ipynb`, manifest CSV, contact sheet (at least 10 pairs), class mapping dictionary, pixel-count report, `data/README.md`.
- **Human Gate**: Verify pairing, label meaning, and patient/wound grouping policy.

### Phase 2: Literature and Experiment Protocol
- **Agent Work**: Review wound tissue segmentation literature and colour-space representation papers. Formalize research questions, hypotheses, split strategy (leakage-free), evaluation metrics, and ablation matrix.
- **Deliverables**: `reports/literature_review.md`, `reports/experiment_protocol.md`, `configs/baseline.yaml`, `configs/ablations.yaml`.
- **Human Gate**: User/supervisor approves hypotheses, split definition, and metrics.

### Phase 3: Preprocessing and EDA
- **Agent Work**: Implement deterministic resizing, label index encoding, colour space conversion (RGB, HSV, CIELAB with documented scaling), optional illumination normalisation, paired data loaders, and geometric augmentations.
- **Deliverables**: `src/woundmap/preprocessing.py`, `notebooks/01_preprocessing_eda.ipynb`, unit tests for transforms and mask discrete integrity, visual overlay samples.
- **Human Gate**: Inspect visual overlays and verify mask integrity under transforms.

### Phase 4: Baseline Model
- **Agent Work**: Implement U-Net / FPN segmentation architecture with configurable backbone. Integrate multiclass loss (Dice + Cross-Entropy), optimizer, scheduler, validation checkpoints, and tensor range assertions. One-batch overfit test.
- **Deliverables**: `src/woundmap/models.py`, `src/woundmap/train.py`, `notebooks/02_baseline_training.ipynb`, training loss/Dice curves, sanity test logs.
- **Human Gate**: Sanity test passing, training curves non-trivial, baseline checkpoint ready.

### Phase 5: Evaluation and Error Analysis
- **Agent Work**: Evaluate baseline checkpoint on untouched test set. Compute per-class Dice, IoU, precision, recall, and support. Handle absent classes explicitly. Generate confusion matrix, boundary error analysis, and qualitative panel comparisons.
- **Deliverables**: `src/woundmap/metrics.py`, `notebooks/04_evaluation_error_analysis.ipynb`, metrics CSV/JSON, qualitative confusion panels.
- **Human Gate**: Verify test set isolation and metrics sanity.

### Phase 6: Controlled Ablations
- **Agent Work**: Execute controlled comparative runs: RGB, HSV, CIELAB, and illumination-normalised inputs under strict identical training budget, seeds, splits, and evaluation.
- **Deliverables**: `notebooks/03_ablation_experiments.ipynb`, comparison table with per-class Dice/IoU differences, discussion of findings.
- **Human Gate**: Confirm fair comparisons and validate observations vs interpretations.

### Phase 7: Streamlit Research Demo
- **Agent Work**: Build interactive Streamlit demo (`app/streamlit_app.py`) for uploading wound photos, running inference, displaying segment overlays, reporting tissue percentage breakdown, and providing non-clinical disclaimers.
- **Deliverables**: Streamlit application, test suite with 5+ valid/invalid inputs, UI screenshots.
- **Human Gate**: Test live app with sample cases and confirm clinical disclaimers.

### Phase 8: Report and Handover
- **Agent Work**: Compile final academic technical report, 6-8 minute presentation slides outline, updated README, and Colab execution guide.
- **Deliverables**: `reports/results.md`, `reports/limitations.md`, presentation assets, final reproducibility guide.
- **Human Gate**: Final sign-off on scientific integrity, claims, and reproduction packaging.

---

## 3. Decision Log
- **2026-11-03**: Adopted *WoundMap AI Antigravity Autonomous Execution Blueprint* as the governing specification.
- **2026-11-03**: Validated local runtime: Windows 11, Python 3.14.2, PyTorch CPU, NVIDIA GeForce GTX 1650 (Driver 592.82, CUDA 13.1 capability). Unit testing pipeline verified via `python -m unittest discover tests` (6 tests executed, 1 skipped by contract).
- **2026-11-03**: Created private remote repository `https://github.com/Lokeshkuma31/WoundMapAI`. Pushed `main` and feature branch `phase/00-bootstrap`.
- **2026-11-03**: Confirmed execution model: Antigravity writes 100% runnable self-contained Colab notebooks with 1-click execution badges, checkpoints saved to Drive/local. User provides dataset archive/link for Phase 1.
- **2026-11-03**: Phase 1 Audit completed on `akabircs/WoundTissue` (commit 557d2f4). 13 image-mask pairs verified with 100% 1-to-1 pairing and 0 unknown mask pixels across 766,243 total pixels. Disclosed that public release contains 13 sample images with arbitrary sequential filenames and no patient metadata. Manifest CSV, audit notebook, and visual contact sheet generated.
- **2026-11-03**: Phase 2 Protocol completed. Drafted comprehensive literature review (`reports/literature_review.md`) integrating biophysics of hemoglobin/melanin, clinical RYKW/PUSH scoring, prior deep learning benchmarks (Kabir et al. arXiv:2502.10652 FPN 82.25% Dice), and color space mathematics (sRGB, HSV, CIELAB, CLAHE). Formalized hypotheses H1/H2/H0 and established locked class-stratified split manifests (`train.csv` 9, `val.csv` 2, `test.csv` 2). Automated split integrity test added (9/9 tests passing).
- **2026-11-03**: Phase 3 Preprocessing & EDA completed. Implemented pure NumPy vectorized CLAHE (`apply_clahe_channel`), paired deterministic resizing (Bicubic image vs Nearest-Neighbor categorical mask), mathematical color conversions (sRGB, HSV, CIELAB with $[0.0, 1.0]$ bounds), and PyTorch `WoundDataset`/`build_dataloader`. Generated cell-by-cell pedagogical walkthrough notebook (`notebooks/01_preprocessing_eda.ipynb`) and high-resolution EDA figures (`color_spaces_eda.png`, `clahe_comparison.png`, `augmentation_overlay.png`). 13/13 unit tests passing.
- **2026-11-03**: Phase 4 Baseline Model completed. Implemented modular 7-class convolutional U-Net (`src/woundmap/models.py`), composite Dice + Cross-Entropy loss, training engine with validation checkpointing and early stopping (`src/woundmap/train.py`), and Python 3.14 optimizer compatibility shim (`src/woundmap/compat.py`). Verified mandatory model output tests and one-batch overfit sanity test (16/16 unit tests passing). Successfully executed local CPU smoke training (10 epochs, Val Macro Dice reached 0.7719). Checkpoint `best_baseline_unet_rgb.pt` (10.1MB, git-ignored), training curves, and validation prediction overlays generated. Colab-ready notebook created (`notebooks/02_baseline_training.ipynb`).
- **2026-11-03**: Phase 5 Evaluation & Error Analysis completed. Evaluated baseline checkpoint on strictly untouched test set (`test.csv`, 131,072 pixels). Computed per-class Dice, IoU, Precision, Recall, Support, and Dual Macro Averages (Present Classes Macro Dice: 0.3006; All 7 Classes Macro Dice: 0.5005). Benchmarked CPU latency: 42.61ms (23.5 FPS). Generated normalized confusion matrix heatmap, qualitative test prediction panels, boundary error maps, `test_metrics_summary.json`, `test_metrics_per_class.csv`, and `reports/results.md`. Identified low-reflectance necrotic tissue / shadow confusion as primary failure mode motivating Phase 6 ablations. Expanded test suite to 19/19 passing tests.
