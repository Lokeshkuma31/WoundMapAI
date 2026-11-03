# WoundMap AI — Comprehensive Literature Review

## Multi-Class Wound Tissue Semantic Segmentation & Color Space Invariance

---

## 1. Clinical Context & Biophysical Motivation

### 1.1 The Clinical Challenge of Chronic Wounds
Chronic wounds—predominantly diabetic foot ulcers (DFUs), venous leg ulcers (VLUs), and pressure injuries—represent a major global health burden with high morbidity, amputation risk, and treatment costs. Clinical assessment relies heavily on serial monitoring of the wound bed to determine healing trajectory. 

Under the widely adopted **RYKW (Red-Yellow-Black-White)** tissue classification framework (van Rijswijk, 1993) and validated scoring instruments such as the **Pressure Ulcer Scale for Healing (PUSH)** and the **Bates-Jensen Wound Assessment Tool (BWAT)**, clinicians evaluate the proportionate surface area of specific tissue morphologies:
1. **Granulation Tissue (Red)**: Proliferative, highly vascularized connective tissue containing budding capillary loops and fibroblasts. Its presence signifies active healing and proliferative-phase tissue reconstruction.
2. **Slough Tissue (Yellow)**: Non-viable, moist devitalized tissue composed of fibrinous exudate, degenerating leukocytes, cellular debris, and bacteria. Excessive slough impedes epithelial migration and prolongs inflammatory arrest.
3. **Necrotic Eschar (Black / Brown)**: Dehydrated, avascular dead tissue and denatured protein matrix forming a hard or leathery crust. Necrosis prevents wound contraction and necessitates surgical or enzymatic debridement.
4. **Macerated / Epithelial Tissue (White / Pale)**: Softened, waterlogged tissue caused by prolonged exposure to wound exudate or moisture, disrupting periwound skin barrier integrity.
5. **Deep Exposed Structures (Bone / Tendon)**: Osseous or fibrous structures exposed in advanced stage IV ulcers, indicating extreme depth and high osteomyelitis risk.

### 1.2 Biophysics of Tissue Coloration in Ordinary Photography
The observed coloration in standard RGB photographs is governed by tissue-specific optical absorption and scattering properties:
- **Oxygenated Hemoglobin ($\text{HbO}_2$)**: Exhibits absorption minima in the red spectrum ($\sim 650\text{–}700\text{ nm}$) and strong Soret/Q-band absorption ($\sim 415\text{ nm}, 542\text{ nm}, 577\text{ nm}$), giving granulation its characteristic vivid red appearance.
- **Fibrin & Lipid Debris**: Reflects broadly across the yellow-green spectrum ($\sim 570\text{–}590\text{ nm}$) due to structural light scattering from denatured collagen matrices.
- **Melanin & Oxidized Hemoglobin / Necrotic Protein**: Broadband optical absorption across the entire visible spectrum ($400\text{–}700\text{ nm}$), resulting in a dark, achromatic or low-reflectance black/brown presentation.

However, standard smartphone or ward photographs suffer from non-standardized illumination, variable color temperature (e.g. fluorescent lighting vs tungsten vs flash), specular highlights from wound exudate, and shadow gradients. These environmental perturbations alter raw RGB channel values without changes in underlying tissue biology.

---

## 2. Deep Learning in Wound Tissue Segmentation

### 2.1 Foundational Architectures
Semantic segmentation in chronic wound analysis has evolved from manual feature engineering and classical thresholding (e.g., k-means, active contours) to deep convolutional neural networks:
- **U-Net** (Ronneberger et al., 2015): Features an encoder-decoder architecture with symmetric skip connections that combine high-level semantic context with fine-grained spatial localization. Widely regarded as the benchmark standard in medical image segmentation.
- **Feature Pyramid Network (FPN)** (Lin et al., 2017): Employs a top-down pathway and lateral connections to construct multi-scale feature representations at every spatial resolution, particularly effective for delineating irregular, multi-scale tissue boundaries.
- **DeepLabV3+** (Chen et al., 2018): Utilizes atrous spatial pyramid pooling (ASPP) to capture multi-scale context without reducing spatial feature resolution.

### 2.2 Prior Benchmarks & The WoundTissue Dataset
Historically, wound segmentation research was fragmented across disparate, single-center datasets:
- **Medetec Database** (Chino et al., 2020): Primarily provided wound contour / boundary annotations rather than multi-class tissue breakdowns.
- **Foot Ulcer Segmentation Challenge (FUSC 2021)** (Wang et al., 2020): Focused on binary wound boundary segmentation across 1,210 diabetic foot ulcer images, leaving internal tissue composition unaddressed.
- **AZH Wound Database** (Wound Care Services, 2019): Limited multi-class availability.

To overcome the lack of open, multi-class tissue segmentation benchmarks, **Kabir et al. (2025)** curated and published the **WoundTissue** dataset:
- **Citation**: Kabir, M. A., Roy, N., Hossain, M. E., Featherston, J., & Ahmed, S. (2025). *Deep Learning for Wound Tissue Segmentation: A Comprehensive Evaluation using A Novel Dataset*. arXiv preprint [arXiv:2502.10652](https://arxiv.org/abs/2502.10652).
- **Dataset Properties**: 147 wound images annotated across 6 tissue classes (granulation, slough, necrosis, maceration, bone, tendon) plus background.
- **Key Empirical Finding**: Kabir et al. comprehensively evaluated 82 segmentation and classification setups. Their findings demonstrated that an **FPN with VGG16 backbone** achieved the top segmentation performance with a macro Dice score of **82.25%**, outperforming standard vanilla U-Net, SegNet, and conditional GANs.

---

## 3. Color Representations & Illumination Normalisation

### 3.1 Color Space Formulations
The standard digital imaging pipeline encodes colors in sRGB, where luminance and chrominance are deeply entangled across all three channels ($R, G, B$):

$$\text{Intensity} \approx 0.299R + 0.587G + 0.114B$$

When ambient lighting changes, all three channels scale non-linearly, corrupting convolutional feature activations. Alternative color spaces decouple intensity from chromatic information:

1. **HSV (Hue, Saturation, Value)**:
   - **Hue ($H \in [0, 360^\circ)$)**: Represents pure dominant spectral wavelength.
   - **Saturation ($S \in [0, 1]$)**: Purity or colorfulness relative to brightness.
   - **Value ($V \in [0, 1]$)**: Maximum color intensity component: $V = \max(R, G, B)$.
   - *Limitation*: Hue possesses a circular angular discontinuity at $0^\circ / 360^\circ$ (red wraps around the origin).

2. **CIE $L^*a^*b^*$ (CIELAB)**:
   - Designed by the Commission Internationale de l'Éclairage (CIE) to approximate human visual perception and perceptual uniformity.
   - **$L^*$ (Luminance)**: Perceptual lightness ($0 = \text{black}$, $100 = \text{diffuse white}$).
   - **$a^*$ (Green-Red Axis)**: Negative values indicate green; positive values indicate red.
   - **$b^*$ (Blue-Yellow Axis)**: Negative values indicate blue; positive values indicate yellow.
   - The Euclidean distance in CIELAB, $\Delta E^* = \sqrt{(\Delta L^*)^2 + (\Delta a^*)^2 + (\Delta b^*)^2}$, closely correlates with perceived biological color difference.

### 3.2 Illumination Normalisation: Contrast-Limited Adaptive Histogram Equalization (CLAHE)
Global histogram equalization frequently causes over-saturation and amplified background noise in medical photographs. **CLAHE** (Zuiderveld, 1994) partitions the image into contextual contextual grid tiles ($8 \times 8$ pixels):
1. Computes local histogram for each grid tile.
2. Clips histogram bins exceeding a specified contrast limit $\beta$, redistributing the clipped excess uniformly across all bins.
3. Computes cumulative distribution function (CDF) for mapping gray levels.
4. Uses bilinear interpolation across adjacent tile borders to eliminate artificial boundary artifacts.

By applying CLAHE specifically to the luminance component ($L^*$ in CIELAB or $V$ in HSV) while preserving opponent chrominance channels ($a^*, b^*$ or $H, S$), illumination gradients and deep shadowing can be attenuated without distorting biological hue signatures.

---

## 4. Traceable Literature Citations

1. **Kabir, M. A., Roy, N., Hossain, M. E., Featherston, J., & Ahmed, S.** (2025). *Deep Learning for Wound Tissue Segmentation: A Comprehensive Evaluation using A Novel Dataset*. arXiv preprint [arXiv:2502.10652](https://arxiv.org/abs/2502.10652).
2. **Ronneberger, O., Fischer, P., & Brox, T.** (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), pp. 234–241.
3. **Lin, T. Y., Dollár, P., Girshick, R., He, K., Hariharan, B., & Belongie, S.** (2017). *Feature Pyramid Networks for Object Detection*. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2117–2125.
4. **Chen, L. C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H.** (2018). *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation*. European Conference on Computer Vision (ECCV), pp. 801–818.
5. **Wang, C. et al.** (2020). *Fully Automated Wound Segmentation with Deep Convolutional Neural Networks*. Scientific Reports, 10(1), 21897.
6. **Zuiderveld, K.** (1994). *Contrast Limited Adaptive Histogram Equalization*. Graphics Gems IV, Academic Press, pp. 474–485.
7. **van Rijswijk, L.** (1993). *Full-Thickness Leg Ulcers: Patient Characteristics and Predictors of Healing*. Journal of Family Practice, 36(6), 625–632.
8. **Bates-Jensen, B. M.** (1997). *The Pressure Sore Status Tool a Few Thousand Assessments Later*. Advances in Wound Care, 10(5), 65–73.
