import os
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np

class MedicalSegmentationDataset(Dataset):
    """
    Volumetric 2D slice dataset for cross-modal medical image segmentation.
    """
    def __init__(self, data_path, split="train"):
        self.split = split
        if not os.path.exists(data_path):
            try:
                from .download_data import generate_benchmark_slices
                name = os.path.splitext(os.path.basename(data_path))[0]
                dest_dir = os.path.dirname(data_path) or "./data"
                os.makedirs(dest_dir, exist_ok=True)
                generate_benchmark_slices(dest_dir, name)
            except Exception:
                pass

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Benchmark file '{data_path}' not found! Run scripts/download_data.sh first.")

        with open(data_path, "rb") as f:
            self.samples = pickle.load(f)

        n = len(self.samples)
        if split == "train":
            self.data = self.samples[:int(n * 0.7)]
        elif split == "val":
            self.data = self.samples[int(n * 0.7):int(n * 0.85)]
        else:
            self.data = self.samples[int(n * 0.85):]

    def __len__(self):
        return max(len(self.data), 1)

    def __getitem__(self, idx):
        item = self.data[idx % len(self.data)]
        _, h, w = item["shape"]
        
        ch0 = np.array(item["img"][0], dtype=np.float32).reshape(h, w)
        ch1 = np.array(item["img"][1], dtype=np.float32).reshape(h, w)
        img = np.stack([ch0, ch1], axis=0) # [2, H, W]
        
        mask = np.array(item["mask"], dtype=np.int64).reshape(h, w) # [H, W]

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.long)
