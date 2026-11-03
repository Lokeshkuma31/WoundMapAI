# WoundTissue Dataset Documentation

## Source & Provenance
- **Dataset Name**: WoundTissue
- **Original Source URL**: [To be verified in Phase 1]
- **Access Date**: Pending Phase 1 acquisition
- **Licence**: To be verified and documented during Phase 1 audit
- **SHA-256 Checksum**: [To be generated on raw archive download]

## Directory Structure
Raw medical images and masks are strictly ignored by Git and must reside in:
```text
data/
├── raw/
│   └── WoundTissue/
│       ├── images/
│       └── masks/
└── manifests/
    ├── manifest.csv       # Audited pairing table (hash, dimensions, image_path, mask_path)
    ├── train.csv          # Training split partition
    ├── val.csv            # Validation split partition
    └── test.csv           # Final untouched evaluation partition
```

## Privacy & Ethical Compliance
- Under no circumstances should unconsented clinical photographs or protected health information (PHI) be committed to GitHub.
- Manifest CSVs store relative paths and anonymized metadata only.

