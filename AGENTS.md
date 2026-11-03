# WoundMap AI — Persistent Agent Rules (AGENTS.md)

## MISSION
Build a reproducible research prototype for multi-class wound-tissue semantic segmentation and controlled colour/illumination experiments.

---

## SCIENTIFIC SAFEGUARDS
1. **Never invent dataset structure, image-mask pairs, class IDs, mask colours, patient IDs, paper findings, metrics, or results.**
2. **Inspect actual files; cite verifiable evidence for decisions.**
3. **Stop if pairing, annotation meaning, licence, or label mapping is unclear.**
4. **Preserve raw data; never overwrite originals.**
5. **Split by patient/wound group when reliable group IDs exist.** Prevent image/patch leakage. If IDs are absent, disclose this immediately and seek explicit approval.
6. **Split before augmentation.** Apply geometric transforms identically to image and categorical mask. Use nearest-neighbour interpolation for categorical masks.
7. **Keep test data untouched during model selection and hyperparameter tuning.**
8. **Report per-class Dice/IoU, support, and explicit absent-class handling.**
9. **Do not claim clinical validity, diagnosis, perfusion, oxygenation, or tissue chemistry from ordinary RGB photographs.**
10. **Distinguish clearly between generated code, locally tested code, Colab-executed code, and independently reproduced results.**

---

## ENGINEERING SAFEGUARDS
- Work on a feature branch; make small, reviewable commits.
- Ask before destructive actions, expensive jobs, external uploads, or publishing.
- Never delete, overwrite, force-push, change repository visibility, or expose credentials without explicit approval.
- Never commit tokens, `.env` files, raw patient data, or large model checkpoints to Git.
- Add and run tests for pairing, mask mapping, transforms, and metric calculations.
- Record environment specs, package versions, seeds, configs, and unique run identifiers.
- Keep code modular, clean, typed, and explainable to an undergraduate researcher.

---

## PHASE PROTOCOL
For every phase:
$$\text{inspect} \longrightarrow \text{plan} \longrightarrow \text{implement} \longrightarrow \text{test} \longrightarrow \text{show evidence} \longrightarrow \text{summarise changes} \longrightarrow \text{request approval at the gate}$$

**Do not start the next gated phase until explicit human approval is received.**

---

## AUTONOMY PERMISSIONS
- **Routine actions:** inspect files, create new files, edit code on working branch, run non-destructive tests, generate reports and diagrams.
- **Ask first:** install unusual dependencies, download large datasets, access external services, run expensive GPU jobs, upload data, change repository settings, publish or deploy.
- **Never:** commit secrets, bypass access controls, use unconsented clinical data, fabricate metrics, or make destructive changes without approval.
