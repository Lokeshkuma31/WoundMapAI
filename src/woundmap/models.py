"""Neural network architectures for wound tissue semantic segmentation."""
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Two 3x3 convolutions with BatchNorm and ReLU activations."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class UNet(nn.Module):
    """Modular U-Net architecture for multi-class tissue semantic segmentation.
    
    Self-contained pure PyTorch implementation with skip connections.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 7, base_features: int = 32):
        super().__init__()
        f = base_features
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Encoder (Contracting Path)
        self.inc = DoubleConv(in_channels, f)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f, f * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 2, f * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 4, f * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 8, f * 8))

        # Decoder (Expanding Path with Skip Connections)
        self.up1 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(f * 8 + f * 4, f * 4)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(f * 4 + f * 2, f * 2)

        self.up3 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(f * 2 + f, f)

        self.up4 = nn.ConvTranspose2d(f, f, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(f * 2, f)

        # Output Classifier (1x1 Conv producing raw unnormalized logits)
        self.outc = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assert valid input dimensions
        assert x.ndim == 4, f"Expected 4D tensor (B, C, H, W), got shape {x.shape}"
        assert x.shape[1] == self.in_channels, f"Expected {self.in_channels} channels, got {x.shape[1]}"

        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with concatenations
        u1 = self.up1(x5)
        u1 = torch.cat([u1, x4], dim=1)
        d1 = self.conv_up1(u1)

        u2 = self.up2(d1)
        u2 = torch.cat([u2, x3], dim=1)
        d2 = self.conv_up2(u2)

        u3 = self.up3(d2)
        u3 = torch.cat([u3, x2], dim=1)
        d3 = self.conv_up3(u3)

        u4 = self.up4(d3)
        u4 = torch.cat([u4, x1], dim=1)
        d4 = self.conv_up4(u4)

        logits = self.outc(d4)
        return logits

def build_model(
    model_name: str = "unet",
    in_channels: int = 3,
    num_classes: int = 7,
    base_features: int = 32
) -> nn.Module:
    """Model factory returning configured neural network instance."""
    name = model_name.lower()
    if name == "unet":
        return UNet(in_channels=in_channels, num_classes=num_classes, base_features=base_features)
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")
