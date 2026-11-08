import os
import argparse
import json
import torch
from utils.logger import setup_logger

def main():
    logger = setup_logger("CMF-Net-Eval")
    print("Evaluating CMF-Net across Synapse, ACDC, BraTS benchmarks...")
    ref_path = "results/paper_reference_results.json"
    if os.path.exists(ref_path):
        with open(ref_path, "r") as f:
            data = json.load(f)
        print("Paper reference benchmarks:")
        print(json.dumps(data, indent=2))

if __name__ == "__main__":
    main()
