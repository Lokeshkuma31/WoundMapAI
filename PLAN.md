# WoundMap AI — Project Execution Plan (PLAN.md)

## 1. Project Control Dashboard

| Phase | Description | Status | Evidence Link | Blocker | Human Approval |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0 Bootstrap** | Tool audit, repo structure, rules, git init | **IN PROGRESS** | [Phase 0 Setup Evidence](#phase-0-bootstrap) | None | Pending Confirmation |
| **1 Dataset Audit** | Inspect WoundTissue tree, pairs, class mapping | NOT STARTED | - | Pending Phase 0 | Gate 1 Required |
| **2 Research Protocol** | Literature review, hypotheses, split policy | NOT STARTED | - | Pending Phase 1 | Gate 2 Required |
| **3 Preprocessing & EDA**| Transforms, colour conversions, EDA overlays | NOT STARTED | - | Pending Phase 2 | Gate 3 Required |
| **4 Baseline Model** | U-Net/FPN baseline, loss, training loop | NOT STARTED | - | Pending Phase 3 | Gate 4 Required |
| **5 Evaluation** | Metrics (Dice/IoU), error analysis | NOT STARTED | - | Pending Phase 4 | Gate 5 Required |
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
- **2026-11-03**: Validated local runtime: Windows 11, Python 3.14.2, PyTorch CPU, NVIDIA GeForce GTX 1650 (CUDA not hooked into 3.14 venv). Colab designated as primary GPU compute target via runnable notebooks + checklist.
