# Literature Review: Multi-Class Wound Tissue Segmentation & Color Representations

## 1. Clinical Context & Tissue Classes
Chronic wounds (diabetic foot ulcers, pressure ulcers, venous leg ulcers) present heterogenous tissue beds comprising:
- **Granulation**: Proliferative, highly vascular pink/red tissue indicative of normal healing.
- **Slough**: Yellow/white avascular fibrinous debris and moist necrotic tissue.
- **Necrotic Eschar**: Hard, dehydrated black/brown devitalized tissue requiring debridement.
- **Epithelialization / Periwound**: Migrating epidermal cells at wound borders.

Accurate surface area quantification of these tissue types is a key clinical proxy for healing rate (PUSH tool, Bates-Jensen Assessment Tool).

## 2. Computer Vision Challenges in Wound Photography
- Uncontrolled ambient illumination (fluorescent ward lighting vs flash vs natural light).
- Varied patient skin tones and erythema confounding red/pink hues.
- Specular highlights caused by moisture, wound exudate, or transparent dressings.

## 3. Color Space Representations
- **RGB**: Correlated color channels sensitive to illumination intensity variations.
- **HSV**: Decouples chrominance (Hue) and saturation from intensity (Value), potentially improving invariance to light level.
- **CIE $L^*a^*b^*$**: Perceptually uniform color representation separating luminance ($L^*$) from opponent color axes ($a^*$, $b^*$).
