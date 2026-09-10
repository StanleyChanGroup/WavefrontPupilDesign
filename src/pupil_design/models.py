"""Wavefront estimation networks.

Layer names and channel widths are fixed by the published checkpoints. Renaming
any attribute breaks ``load_state_dict`` on released weights.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

__all__ = ["UNet", "PhaseMLP", "build_model"]


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with batch norm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """Three-scale U-Net mapping intensity to phase. 7.7M parameters.

    Args:
        in_ch: 2 when the pupil mask is concatenated to the PSF, else 1.
        narrow: Halve every channel width. Named ``fight_overfitting`` in the
            original scripts; all published models use the wide variant.
    """

    def __init__(self, in_ch: int = 2, out_ch: int = 1, narrow: bool = False):
        super().__init__()
        c1, c2, c3, c4 = (32, 64, 128, 256) if narrow else (64, 128, 256, 512)

        self.enc1 = ConvBlock(in_ch, c1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(c1, c2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(c2, c3)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c3, c4)

        self.up3 = nn.ConvTranspose2d(c4, c3, 2, 2)
        self.dec3 = ConvBlock(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, 2)
        self.dec2 = ConvBlock(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, 2)
        self.dec1 = ConvBlock(c1 * 2, c1)
        self.final = nn.Conv2d(c1, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.final(d1)


class PhaseMLP(nn.Module):
    """Fully connected baseline. 269M parameters at 128x128.

    Included to show the asymmetry trend is a property of the imaging problem
    rather than of convolutional inductive bias.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 4096, depth: int = 5):
        super().__init__()
        self.output_dim_side = int(np.sqrt(output_dim))

        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(depth):
            layers += [nn.Linear(dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)]
            dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x.view(x.shape[0], -1))
        return out.view(-1, 1, self.output_dim_side, self.output_dim_side)


def build_model(
    model_type: str,
    bounding_box_size: int,
    with_pupil: bool = True,
    narrow: bool = False,
) -> nn.Module:
    """Construct the network named by ``model_type`` ('unet' or 'mlp')."""
    in_ch = 2 if with_pupil else 1
    kind = model_type.lower()
    if kind == "mlp":
        return PhaseMLP(
            input_dim=in_ch * bounding_box_size * bounding_box_size,
            output_dim=bounding_box_size * bounding_box_size,
        )
    if kind == "unet":
        return UNet(in_ch=in_ch, out_ch=1, narrow=narrow)
    raise ValueError(f"Unknown model_type {model_type!r}; expected 'unet' or 'mlp'")
