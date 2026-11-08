# CMF-Net: Cross-Modal Feature Fusion Network for Volumetric Medical Image Segmentation

Official benchmark repository for CMF-Net, a cross-modal feature fusion architecture for multi-organ CT and MRI segmentation.

## Quick Start
```bash
pip install -r requirements.txt
pip install -e .

# Download benchmark datasets
bash scripts/download_data.sh ./data

# Run full reproduction pipeline
bash scripts/reproduce_all.sh
```

## Supported Benchmarks
- Synapse Multi-Organ CT
- ACDC Cardiac MRI
- BraTS Brain Tumor Multimodal MRI
