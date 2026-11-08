import time
import torch
from models.cmf_net import CMFNet
from models.baselines import VanillaUNet

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU device is required.")
    device = torch.device("cuda")

    x = torch.randn(4, 2, 64, 64, device=device)
    models = {
        "CMF-Net": CMFNet(in_channels=2, num_classes=4).to(device),
        "VanillaUNet": VanillaUNet(in_channels=2, num_classes=4).to(device)
    }

    print("=" * 60)
    print("CMF-Net Efficiency Benchmark (Params & Latency)")
    print("=" * 60)
    for name, m in models.items():
        m.eval()
        start = time.time()
        with torch.no_grad():
            for _ in range(50):
                _ = m(x)
        lat = (time.time() - start) * 1000 / 50
        params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"{name:<15} | Params: {params/1e6:.2f} M | Latency: {lat:.2f} ms")
    print("=" * 60)

if __name__ == "__main__":
    main()
