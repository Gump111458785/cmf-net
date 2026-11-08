import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttentionBlock(nn.Module):
    """
    Cross-modal attention module between complementary imaging modalities.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, m1, m2):
        # m1: modality 1 [B, C, H, W], m2: modality 2 [B, C, H, W]
        B, C, H, W = m1.shape
        proj_query = self.query_conv(m1).view(B, -1, H * W).permute(0, 2, 1) # [B, N, C']
        proj_key = self.key_conv(m2).view(B, -1, H * W) # [B, C', N]
        energy = torch.bmm(proj_query, proj_key) # [B, N, N]
        attention = F.softmax(energy / ((C // 4) ** 0.5), dim=-1)
        
        proj_value = self.value_conv(m2).view(B, -1, H * W) # [B, C, N]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)).view(B, C, H, W)
        return m1 + self.gamma * out

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class CMFNet(nn.Module):
    """
    CMF-Net: Cross-Modal Feature Fusion Network for Medical Image Segmentation.
    """
    def __init__(self, in_channels=2, num_classes=4, base_channels=32):
        super().__init__()
        self.num_classes = num_classes

        # Modality encoders
        self.enc1 = DoubleConv(1, base_channels)
        self.enc2 = DoubleConv(1, base_channels)

        # Cross-modal fusion
        self.fusion = CrossModalAttentionBlock(base_channels)

        # Bottleneck & Decoder
        self.down = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base_channels, base_channels * 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec = DoubleConv(base_channels * 2 + base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # x: [B, 2, H, W]
        m1 = x[:, 0:1, :, :]
        m2 = x[:, 1:2, :, :]

        f1 = self.enc1(m1)
        f2 = self.enc2(m2)

        fused = self.fusion(f1, f2)
        b = self.bottleneck(self.down(fused))
        u = self.up(b)
        out = self.dec(torch.cat([u, fused], dim=1))
        return self.head(out)
