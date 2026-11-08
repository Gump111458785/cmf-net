#!/usr/bin/env bash
# End-to-end Reproduction Script for CMF-Net
set -e

echo "=== CMF-Net Medical Image Segmentation Reproduction ==="
echo "[1/3] Downloading datasets..."
bash scripts/download_data.sh ./data

echo "[2/3] Training CMF-Net on Synapse..."
python3 scripts/train.py --benchmark synapse --epochs 5

echo "[3/3] Evaluating benchmarks..."
python3 scripts/evaluate.py

echo "CMF-Net pipeline completed."
