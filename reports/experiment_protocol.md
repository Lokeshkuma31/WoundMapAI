# WoundMap AI — Formal Experiment Protocol

**Protocol Identifier**: WMAP-PROTO-2026-V1  
**Project**: WoundMap AI (Biophysics & Biomedical Computer-Vision Research Prototype)  
**Governing Specification**: `WoundMapAI_Antigravity_Autonomous_Execution_Blueprint.pdf`  
**Dataset**: WoundTissue (`akabircs/WoundTissue`, Kabir et al., 2025, arXiv:2502.10652)  

---

## 1. Research Question & Formal Hypotheses

### 1.1 Central Research Question
> *How do sRGB, HSV, and CIELAB color representations, evaluated with and without Contrast-Limited Adaptive Histogram Equalization (CLAHE) illumination normalisation, affect deep semantic segmentation accuracy, boundary delineation, and tissue-class robustness in chronic wound photography?*

### 1.2 Hypotheses
- **Hypothesis 1 ($H_1$ — Color Invariance)**: Decoupling chrominance from luminance via perceptually uniform representations (CIELAB) or cylindrical representations (HSV) will improve segmentation boundary fidelity and yield higher per-class Dice scores for chromatic tissue transitions (specifically Granulation vs. Slough) compared to entangled sRGB inputs.
- **Hypothesis 2 ($H_2$ — Illumination Normalisation)**: Applying CLAHE to the luminance component ($L^*$ in CIELAB, $V$ in HSV) prior to convolutional feature encoding will significantly reduce false-positive necrotic classifications in shadowed wound cavities and low-light regions.
- **Null Hypothesis ($H_0$)**: Color space transformation and luminance pre-filtering provide no measurable benefit ($\Delta \text{Macro Dice} < 0.01$) over standard raw sRGB baseline inputs under identical network architectures and training budgets.

---

## 2. Experimental Variables & Controls

### 2.1 Independent Variables
1. **Color Space Representation**:
   - `sRGB`: Standard correlated 3-channel input scaled to $[0.0, 1.0]$.
   - `HSV`: Cylindrical color model ($H \in [0, 1], S \in [0, 1], V \in [0, 1]$).
   - `CIELAB`: Perceptually uniform color model ($L^* \in [0, 100], a^* \in [-128, 127], b^* \in [-128, 127]$, linearly normalized to $[0.0, 1.0]$ for network stability).
2. **Illumination Normalisation Method**:
   - `None`: Raw converted color channels.
   - `CLAHE`: Contrast-Limited Adaptive Histogram Equalization applied strictly to the luminance channel ($L^*$ in CIELAB, $V$ in HSV, or per-channel in RGB) with clip limit $\beta = 2.0$ and grid tile size $8 \times 8$.

### 2.2 Dependent Variables (Evaluation Metrics)
1. **Per-Class Dice Similarity Coefficient (DSC)**:
   $$\text{Dice}_c = \frac{2 \cdot |Y_c \cap \hat{Y}_c| + \epsilon}{|Y_c| + |\hat{Y}_c| + \epsilon}$$
2. **Per-Class Intersection over Union (IoU / Jaccard Index)**:
   $$\text{IoU}_c = \frac{|Y_c \cap \hat{Y}_c| + \epsilon}{|Y_c \cup \hat{Y}_c| + \epsilon}$$
3. **Macro-Averaged Dice and IoU**: Unweighted mean across evaluated classes.
4. **Per-Class Support**: Total ground-truth pixel count per class in evaluation partitions.
5. **Inference Latency**: Mean runtime in milliseconds per $512 \times 512$ image over 50 repeated passes following a 5-iteration warm-up.

### 2.3 Controlled Variables (Fair Comparison Protocol)
To ensure that performance variations are attributable strictly to color representation and illumination handling, the following parameters are locked identically across all runs:
- **Model Architecture**: U-Net with ResNet-34 encoder backbone pretrained on ImageNet (Lin et al. / Ronneberger et al.).
- **Number of Output Channels**: 7 discrete classes (Background, Granulation, Slough, Necrosis, Maceration, Bone, Tendon).
- **Spatial Input Dimension**: $512 \times 512$ pixels (bicubic interpolation for images; strict nearest-neighbour for categorical masks).
- **Optimization Strategy**: AdamW optimizer ($\text{lr} = 3 \times 10^{-4}, \beta_1 = 0.9, \beta_2 = 0.999, \text{weight\_decay} = 1 \times 10^{-4}$).
- **Loss Function**: Composite Dice + Cross-Entropy Loss:
  $$\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{Dice}} + 0.5 \cdot \mathcal{L}_{\text{CE}}$$
- **Training Budget**: 50 epochs, mini-batch size 8, early stopping patience of 10 epochs monitored on validation macro Dice.
- **Random Seed**: Fixed to `42` across Python, NumPy, and PyTorch runtime.

---

## 3. Dataset Partition & Leakage Prevention Policy

### 3.1 Patient ID Disclosure & Partition Strategy
As audited and documented in Phase 1, the public release of `akabircs/WoundTissue` contains 13 paired images (`img (1).png` to `img (31).png`) with sequential indices and **no clinical patient or wound grouping metadata**. 

To maintain strict scientific integrity (Scientific Safeguards 5 and 7):
1. Images are partitioned using **tissue-class stratified allocation** to ensure minority clinical classes are represented across partitions without synthetic patch leakage.
2. The **Test Set remains strictly untouched** during architecture tuning, learning rate selection, and validation checkpoints.

### 3.2 Approved Dataset Partitions

| Partition | Sample Count | Proportion | Image IDs Included | Classes Represented |
| :--- | :---: | :---: | :--- | :--- |
| **Train** | 9 | 69.2% | `img (1)`, `img (2)`, `img (5)`, `img (6)`, `img (8)`, `img (12)`, `img (23)`, `img (28)`, `img (31)` | Background, Granulation, Slough, Necrosis, Maceration, Bone |
| **Validation** | 2 | 15.4% | `img (29)`, `img (30)` | Background, Granulation, Slough, Maceration |
| **Test (Untouched)** | 2 | 15.4% | `img (20)`, `img (27)` | Background, Granulation, Slough, Necrosis, Maceration |
| **Total** | **13** | **100.0%** | All 13 audited pairs | Full dataset spectrum |

---

## 4. Controlled Ablation Matrix

The experimental evaluation comprises six systematic conditions:

| Run ID | Color Space | Illumination Normalisation | Target Channel for CLAHE | Scientific Role |
| :---: | :---: | :---: | :---: | :--- |
| **`RUN-01`** | **sRGB** | None | N/A | **Primary Baseline** (Standard CV practice) |
| **`RUN-02`** | **HSV** | None | N/A | Isolates chrominance ($H, S$) from intensity ($V$) |
| **`RUN-03`** | **CIELAB** | None | N/A | Perceptually uniform opponent color evaluation |
| **`RUN-04`** | **sRGB** | CLAHE | All channels ($R, G, B$) | Standard illumination enhancement baseline |
| **`RUN-05`** | **HSV** | CLAHE | Value channel only ($V$) | Contrast enhancement preserving pure hue |
| **`RUN-06`** | **CIELAB** | CLAHE | Lightness channel only ($L^*$) | Contrast enhancement preserving biological color axes |

---

## 5. Metric Calculation & Absent-Class Policy (Safeguard 8)

In clinical datasets with small sample sizes or localized pathologies, certain tissue classes may be naturally absent from individual images or entire test partitions (e.g., Bone support $= 0$ in the test set). To prevent deceptive metrics:
1. **Dual Macro Averaging**:
   - $\text{Macro-Dice}_{\text{present}}$: Mean computed strictly across classes with ground-truth support $> 0$ in the evaluation partition.
   - $\text{Macro-Dice}_{\text{all}}$: Mean across all 7 classes, explicitly documenting absent classes.
2. **Explicit Corner Cases**:
   - $\text{Support} = 0 \text{ and } \text{Pred} = 0 \implies \text{Metric} = \text{Excluded or 1.0 (both reported)}$.
   - $\text{Support} = 0 \text{ and } \text{Pred} > 0 \implies \text{Metric} = 0.0$ (penalizes false positive hallucination).
   - $\text{Support} > 0 \text{ and } \text{Pred} = 0 \implies \text{Metric} = 0.0$ (penalizes false negative omission).

---

## 6. Compute & Hardware Budget

- **Local Smoke Testing**: Windows 11 CPU/NVIDIA GTX 1650 for shape validation, one-batch overfit sanity test, and metric calculations.
- **Google Colab GPU Training**: Self-contained notebook execution on Colab Tesla T4 GPU ($\sim 16\text{ GB VRAM}$).
- **Estimated Resource Cost**: $\sim 3\text{–}5\text{ minutes}$ per condition, totaling $<30\text{ minutes}$ across the entire 6-run ablation matrix.
