"""No-bottleneck baseline for Stage-4 R-1 (peer-review response, 2026-06-03).

Same DualStreamBackbone as CBNet-AGID, but WITHOUT the concept bottleneck: the fused
[B, 2560, H', W'] features are global-average-pooled and fed directly to a linear
classifier. This isolates the reviewer's question "what does the K=6 concept
bottleneck add?" — the baseline matches CBNet-AGID's backbone exactly and differs
only by removing the bottleneck (so any difference is attributable to the bottleneck).

NEW FILE — reuses (does NOT modify) models/backbone.py. It is deliberately separate
from cbnet.py so the audited architecture remains untouched.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .backbone import DualStreamBackbone


class BaselinePlain(nn.Module):
    """Dual-stream backbone -> GAP -> Linear(2560 -> 1). No concept bottleneck."""

    def __init__(self, pretrained: bool = True, signal_channels: int = 512,
                 frozen_stem: bool = False):
        super().__init__()
        self.backbone = DualStreamBackbone(
            pretrained=pretrained,
            signal_channels=signal_channels,
            frozen_stem=frozen_stem,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Linear head on the FULL pooled feature vector (2560-d) — no 6-channel bottleneck.
        self.classifier = nn.Linear(self.backbone.out_channels, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)                    # [B, 2560, H', W']
        pooled = self.pool(features).flatten(1)        # [B, 2560]
        logit = self.classifier(pooled).squeeze(-1)    # [B]
        prob = torch.sigmoid(logit)
        return {
            "logit": logit,
            "prob": prob,
            "features": features,   # retained for post-hoc attribution (Grad-CAM etc.)
            "pooled": pooled,
        }
