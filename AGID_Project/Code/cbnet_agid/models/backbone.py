"""Dual-stream backbone for CBNet-AGID.

Two streams operate in parallel on the same input image:

1. **Pixel stream** — standard torchvision ResNet-50 pretrained on ImageNet.
   Captures high-level semantic and texture features.

2. **Signal stream** — a lightweight CNN operating on the NPR-style residual
   (image minus interpolate-then-downsample). Captures low-level noise / signal
   patterns that are diagnostic of AI generation.

The two feature maps are concatenated channel-wise at the final spatial resolution
(7x7 for 224x224 input), yielding a [B, C_pixel + C_signal, 7, 7] feature tensor.

Memory note (8GB VRAM constraint): the signal stream is intentionally lightweight
(~512 output channels via a 5-layer CNN) so that the total feature width stays
manageable. The CBL on top uses 1x1 convolutions which are cheap.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class NPRResidual(nn.Module):
    """The Neighborhood Pixel Relationship residual from Tan et al. CVPR 2024.

    residual = x - interpolate(interpolate(x, 0.5), 2.0)

    Where `interpolate(_, 0.5)` is nearest-neighbor downsampling by 2 and
    `interpolate(_, 2.0)` is nearest-neighbor upsampling by 2. The composition
    leaves CNN-generated images with a much smaller residual than real photos
    because clean generation lacks pixel-to-pixel sensor noise.
    """

    def __init__(self, downscale: float = 0.5):
        super().__init__()
        self.downscale = downscale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure dims are even (NPR's original code crops to even sizes).
        n, c, h, w = x.shape
        if h % 2 == 1:
            x = x[:, :, :-1, :]
        if w % 2 == 1:
            x = x[:, :, :, :-1]
        # Downsample then upsample with NEAREST interpolation
        x_lo = F.interpolate(x, scale_factor=self.downscale, mode="nearest",
                             recompute_scale_factor=True)
        x_re = F.interpolate(x_lo, scale_factor=1.0 / self.downscale, mode="nearest",
                             recompute_scale_factor=True)
        return x - x_re


class SignalStreamCNN(nn.Module):
    """Small CNN backbone for the signal stream.

    Input:  NPR-residual map of the image, shape [B, 3, H, W].
    Output: feature map [B, C_out, H', W'] where (H', W') matches the pixel stream.

    For 224x224 input, after 5 stride-2 downsamples the spatial size is 7x7.

    Total params: ~3M (vs ResNet-50's 23M). Designed to fit comfortably under 8GB VRAM
    alongside the pixel stream.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )  # 224 -> 112
        self.layer1 = self._block(32, 64, stride=2)      # 112 -> 56
        self.layer2 = self._block(64, 128, stride=2)      # 56 -> 28
        self.layer3 = self._block(128, 256, stride=2)      # 28 -> 14
        self.layer4 = self._block(256, out_channels, stride=2)  # 14 -> 7
        self.out_channels = out_channels

    @staticmethod
    def _block(in_c: int, out_c: int, stride: int = 1) -> nn.Module:
        """Two-conv residual-like block (lightweight)."""
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class DualStreamBackbone(nn.Module):
    """Pixel-stream ResNet-50 + Signal-stream small-CNN, fused by channel concat.

    Args:
        pretrained:   whether to load ImageNet-pretrained ResNet-50 weights.
        signal_channels: output channels of the signal stream (default 512).
        frozen_stem:  if True, freeze the early layers of the pixel stream (conv1, bn1,
                       layer1). Useful for fine-tuning with limited compute.
    """

    def __init__(self, pretrained: bool = True, signal_channels: int = 512,
                 frozen_stem: bool = False):
        super().__init__()
        # Pixel stream: ResNet50 minus the final FC layer
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
        else:
            weights = None
        resnet = models.resnet50(weights=weights)
        self.pixel_stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.pixel_layer1 = resnet.layer1
        self.pixel_layer2 = resnet.layer2
        self.pixel_layer3 = resnet.layer3
        self.pixel_layer4 = resnet.layer4
        self.pixel_channels = 2048

        # Signal stream: NPR residual + small CNN
        self.npr = NPRResidual(downscale=0.5)
        self.signal_cnn = SignalStreamCNN(in_channels=3, out_channels=signal_channels)
        self.signal_channels = signal_channels

        # Total output channels
        self.out_channels = self.pixel_channels + self.signal_channels

        if frozen_stem:
            for module in [self.pixel_stem, self.pixel_layer1]:
                for p in module.parameters():
                    p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] input image, normalized with ImageNet stats.

        Returns:
            features: [B, pixel_C + signal_C, H', W']  fused feature map.
        """
        # Pixel stream
        pixel = self.pixel_stem(x)
        pixel = self.pixel_layer1(pixel)
        pixel = self.pixel_layer2(pixel)
        pixel = self.pixel_layer3(pixel)
        pixel = self.pixel_layer4(pixel)  # [B, 2048, 7, 7] for 224x224

        # Signal stream
        residual = self.npr(x)
        signal = self.signal_cnn(residual)  # [B, signal_channels, 7, 7]

        # Ensure spatial sizes match (in case of off-by-one due to NPR cropping)
        if signal.shape[-2:] != pixel.shape[-2:]:
            signal = F.interpolate(signal, size=pixel.shape[-2:], mode="bilinear",
                                   align_corners=False)

        return torch.cat([pixel, signal], dim=1)
