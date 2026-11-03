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
