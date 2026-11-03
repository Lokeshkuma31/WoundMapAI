# WoundTissue Dataset Documentation & Audit Record

## 1. Provenance & Citation
- **Dataset Name**: WoundTissue
- **Source Repository**: [https://github.com/akabircs/WoundTissue](https://github.com/akabircs/WoundTissue)
- **Primary Paper**: *Deep Learning for Wound Tissue Segmentation: A Comprehensive Evaluation using A Novel Dataset*, Muhammad Ashad Kabir, Nidita Roy, Md. Ekramul Hossain, Jill Featherston, Sayed Ahmed (2025). [arXiv:2502.10652](https://arxiv.org/abs/2502.10652).
- **Access Date**: 2026-11-03
- **Licence**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0). Strictly restricted to non-commercial research purposes.
- **Git Commit Cloned**: `557d2f431769f002876776c3693b7a4c49286964`

---

## 2. Dataset Structure & Pairing Audit
- **Total images in repository**: 13 PNG images (`img (1).png` to `img (31).png`) and 1 empty marker file (`test.txt`).
- **Total masks in repository**: 13 matching PNG masks.
- **Pairing verification**: 100% 1-to-1 exact filename correspondence between `data/raw/WoundTissue/image/` and `data/raw/WoundTissue/label/`.
- **Dimensions**: Variable square dimensions ranging from $128 \times 128$ to $467 \times 467$ pixels.
- **Color Format**:
  - Raw Images: 3-channel 8-bit RGB (`uint8`).
  - Raw Ground Truth Masks: 4-channel RGBA (`uint8`), with alpha channel uniformly equal to 255.

---

## 3. Audited Label Mapping Palette

| Class ID | Class Name | Mask RGB Color | Color Name | Pixel Count | Dataset Proportion (%) |
| :---: | :--- | :---: | :--- | :---: | :---: |
| 0 | **Background** | `(0, 0, 255)` | Blue | 430,172 | 56.14% |
| 1 | **Granulation** | `(255, 0, 0)` | Red | 225,248 | 29.40% |
| 2 | **Slough** | `(255, 255, 0)` | Yellow | 34,440 | 4.49% |
| 3 | **Necrosis** | `(0, 0, 0)` | Black | 36,398 | 4.75% |
| 4 | **Maceration** | `(255, 255, 255)` | White | 37,273 | 4.86% |
| 5 | **Bone** | `(255, 200, 124)` | Cream / Peach | 2,712 | 0.35% |
| 6 | **Tendon** | `(0, 191, 255)` | Sky Blue | 0 | 0.00% (absent in 13-subset) |
| - | **Total** | - | - | **766,243** | **100.00%** |

*Verification: Exactly 0 unknown or stray pixels across all 13 masks.*

---

## 4. Patient & Grouping ID Disclosure (Scientific Safeguard 5)
- **Observation**: Image filenames are indexed non-consecutively (`img (1)`, `img (12)`, `img (2)`, `img (20)`, `img (23)`, `img (27)`, `img (28)`, `img (29)`, `img (30)`, `img (31)`, `img (5)`, `img (6)`, `img (8)`).
- **Metadata Finding**: No patient identifiers, hospital tags, anatomical wound locations, or wound session group IDs are provided in the repository.
- **Mandated Safeguard Action**: Because true patient IDs are absent from the public release, we disclose that patient-grouped splitting cannot be guaranteed. To prevent unrealistic generalization claims, splits must be image-stratified by tissue class presence, and this limitation will be explicitly documented throughout the experimental protocol and research report.
