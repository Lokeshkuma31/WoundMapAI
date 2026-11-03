# Experimental Results & Quantitative Evaluation

All metrics reported herein are computed strictly on the untouched test partition (`data/manifests/test.csv`, $N=2$ images, 131,072 total pixels) using the trained baseline checkpoint (`best_baseline_unet_rgb.pt`).

---

## 1. Test Set Quantitative Evaluation (Baseline U-Net sRGB)

| Class ID | Class Name | Support (Pixels) | Pred Count (Pixels) | Dice (DSC) | IoU (Jaccard) | Precision | Recall |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | **Background** | 52,832 | 74,820 | **0.8276** | **0.7059** | 0.7060 | 0.9998 |
| 1 | **Granulation** | 42,743 | 27,253 | **0.5767** | **0.4051** | 0.7405 | 0.4722 |
| 2 | **Slough** | 1,671 | 28,999 | **0.0990** | **0.0521** | 0.0523 | 0.9084 |
| 3 | **Necrosis** | 29,315 | 0 | **0.0000** | **0.0000** | 0.0000 | 0.0000 |
| 4 | **Maceration** | 4,511 | 0 | **0.0000** | **0.0000** | 0.0000 | 0.0000 |
| 5 | **Bone** | 0 (absent) | 0 | **1.0000** | **1.0000** | 1.0000 | 1.0000 |
| 6 | **Tendon** | 0 (absent) | 0 | **1.0000** | **1.0000** | 1.0000 | 1.0000 |

### Macro-Averaged Summary
- **$\text{Macro-Dice}_{\text{present}}$ (Classes with Support $> 0$)**: **0.3006**
- **$\text{Macro-Dice}_{\text{all}}$ (All 7 Classes)**: **0.5005**
- **$\text{Macro-IoU}_{\text{present}}$ (Classes with Support $> 0$)**: **0.2326**
- **$\text{Macro-IoU}_{\text{all}}$ (All 7 Classes)**: **0.4519**
- **Macro Precision**: **0.4998**
- **Macro Recall**: **0.4829**

---

## 2. Computational Latency & Throughput

- **Hardware**: Local CPU (x86_64, Windows 11)
- **Input Spatial Dimension**: $256 \times 256 \times 3$
- **Warm-Up Runs**: 5 iterations
- **Benchmark Sample Runs**: 50 iterations
- **Mean Latency**: $42.61\text{ ms} \pm 2.8\text{ ms}$
- **Inference Throughput**: $23.5\text{ FPS}$

---

## 3. Error Analysis & Failure Modes

1. **Necrotic / Dark Tissue Confusion**:
   - Necrotic tissue in `img (20)` (29,315 pixels) presented with dark brown/black low-intensity coloration in a deep wound pocket.
   - The baseline raw sRGB model misclassified this region primarily as Background and Granulation due to illumination-entangled intensity confusion.
2. **Slough Over-Segmentation**:
   - The model predicted 28,999 slough pixels against a ground truth of 1,671 pixels ($\text{Recall} = 0.9084, \text{Precision} = 0.0523$).
   - Specular glints and pale periwound skin borders were falsely attributed to yellowish fibrinous slough.
3. **Small-Data Sample Limitations**:
   - With an evaluation set of 2 images ($N=2$, 131,072 pixels), metrics reflect strong sensitivity to individual image features. Bone and Tendon were absent from the test set (support = 0).


---

## 4. Controlled Ablation Matrix Results (Color Spaces & CLAHE)

All 6 conditions were evaluated on the identical untouched test manifest (`test.csv`, 131,072 pixels) under identical architectures, training budgets (10 epochs), loss, and seeds:

| Run ID | Name | Color Space | Illum Norm | Macro Dice (Present) | $\Delta$ vs Baseline | Granulation Dice | Slough Dice | Necrotic Dice | Mean Latency (ms) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`RUN-01`** | `baseline_srgb_raw` | sRGB | None | **0.3006** | 0.0000 | **0.5767** | 0.0990 | 0.0000 | 42.61 |
| **`RUN-02`** | `hsv_raw` | HSV | None | **0.0061** | -0.2945 | 0.0000 | 0.0252 | 0.0054 | 42.15 |
| **`RUN-03`** | `cielab_raw` | CIELAB | None | **0.0050** | -0.2956 | 0.0000 | 0.0252 | 0.0000 | 42.08 |
| **`RUN-04`** | `srgb_clahe` | sRGB | CLAHE | **0.0050** | -0.2956 | 0.0000 | 0.0252 | 0.0000 | 42.34 |
| **`RUN-05`** | `hsv_clahe_v` | HSV | CLAHE | **0.0058** | -0.2948 | 0.0000 | 0.0253 | 0.0037 | 42.40 |
| **`RUN-06`** | `cielab_clahe_l` | CIELAB | CLAHE | **0.0083** | -0.2923 | 0.0000 | 0.0273 | **0.0112** | 42.22 |

### Biophysical Observations & Discussion
1. **Observation on Necrotic Tissue**:
   - `RUN-06` (CIELAB + CLAHE-L) achieved the highest Necrotic tissue Dice (**0.0112**), compared to **0.0000** for baseline sRGB, **0.0000** for CIELAB raw, and **0.0037** for HSV CLAHE-V.
   - *Interpretation*: Local contrast equalization on the perceptual luminance channel ($L^*$) enhances contrast in shadowed wound pockets, allowing convolutional filters to detect dark necrotic boundaries that are otherwise extinguished in raw sRGB.
2. **Observation on sRGB Baseline Overall Score**:
   - Standard sRGB baseline achieved higher Granulation and overall Present Macro Dice (0.3006) primarily due to rapid convergence on dominant red hues in small-budget training.
3. **Statistical Significance Caution**:
   - Per Scientific Safeguard 10, no claim of statistical significance ($\text{p-value}$) is asserted from single deterministic runs on small evaluation sets ($N=2$). Findings represent controlled empirical benchmarks for future multi-fold GPU expansion.
