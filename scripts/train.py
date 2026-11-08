import os
import sys
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import MedicalSegmentationDataset
from models.cmf_net import CMFNet
from models.baselines import VanillaUNet
from utils.logger import setup_logger
from utils.metrics import compute_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="CMF-Net Medical Segmentation Training")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--benchmark", type=str, default="synapse", choices=["synapse", "acdc", "brats"])
    parser.add_argument("--model", type=str, default="cmf_net", choices=["cmf_net", "unet"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    return parser.parse_args()

def check_hardware():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "\n"
            "======================================================================\n"
            "HARDWARE ERROR: CMF-Net requires an NVIDIA CUDA-enabled GPU to run.\n"
            "torch.cuda.is_available() returned False.\n"
            "CPU execution is disabled for volumetric cross-modal feature operations.\n"
            "======================================================================\n"
        )
    print(f"[Hardware Check Passed] Using CUDA Device: {torch.cuda.get_device_name(0)}")

def main():
    args = parse_args()
    logger = setup_logger("CMF-Net-Train")
    logger.info("Initializing CMF-Net Training Pipeline...")
    logger.info(f"Benchmark: {args.benchmark}, Model: {args.model}")

    check_hardware()
    device = torch.device("cuda")

    data_file = os.path.join(args.data_dir, f"{args.benchmark}.pkl")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Missing {data_file}. Run scripts/download_data.sh first.")

    train_set = MedicalSegmentationDataset(data_file, split="train")
    val_set = MedicalSegmentationDataset(data_file, split="val")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    if args.model == "cmf_net":
        model = CMFNet(in_channels=2, num_classes=4).to(device)
    else:
        model = VanillaUNet(in_channels=2, num_classes=4).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        start = time.time()
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        elapsed = time.time() - start
        logger.info(f"Epoch [{epoch:02d}/{args.epochs:02d}] Loss: {total_loss/len(train_loader):.4f} | Time: {elapsed:.2f}s")

    # Evaluation
    model.eval()
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            metrics = compute_metrics(logits, masks, num_classes=4)
            logger.info(f"Validation Metrics on {args.benchmark}: {metrics}")
            break

if __name__ == "__main__":
    main()
