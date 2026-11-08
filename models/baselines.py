import torch
import torch.nn as nn
from .cmf_net import DoubleConv

class VanillaUNet(nn.Module):
    def __init__(self, in_channels=2, num_classes=4, base_channels=32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_channels)
        self.down = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base_channels, base_channels * 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec = DoubleConv(base_channels * 2 + base_channels, base_channels)
        self.outc = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down(x1)
        x3 = self.bottleneck(x2)
        x4 = self.up(x3)
        x5 = self.dec(torch.cat([x4, x1], dim=1))
        return self.outc(x5)
