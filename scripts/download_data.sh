#!/usr/bin/env bash
# Download all required benchmark datasets for CMF-Net
# Usage: bash scripts/download_data.sh [TARGET_DIR]

set -e

DEST_DIR="${1:-./data}"
mkdir -p "$DEST_DIR"

echo "=== CMF-Net Medical Segmentation Benchmark Downloader ==="
echo "Target directory: $DEST_DIR"

python3 data/download_data.py --dest_dir "$DEST_DIR"

echo "Dataset preparation completed."
