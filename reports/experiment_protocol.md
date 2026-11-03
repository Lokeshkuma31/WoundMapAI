# Experiment Protocol: Controlled Illumination & Color Space Ablation

## Research Question
*How do RGB, HSV, and CIELAB color representations, with and without adaptive illumination normalization, affect deep semantic segmentation robustness for multi-class chronic wound tissue?*

## Hypotheses
1. **$H_1$ (Color Invariance)**: Decoupled color representations (HSV and CIELAB) will demonstrate higher boundary stability and Dice scores on tissue transitions (granulation vs slough) under non-uniform illumination compared to standard RGB.
2. **$H_2$ (Illumination Normalization)**: Preprocessing with contrast-limited adaptive histogram equalization (CLAHE) on the luminance/value channel will reduce false positive necrotic classification in shadowed wound pockets.

## Protocol Controls
- Fixed dataset split (patient/wound isolated) across all runs.
- Identical random seeds, architectures (U-Net with ResNet-34 backbone), learning rates, losses, and epoch budgets.
- Test set strictly held out until model training and validation selection are complete.
