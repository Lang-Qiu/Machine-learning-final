"""CBNet-AGID main model: DualStreamBackbone → ConceptBottleneckLayer → Linear classifier.

The pipeline (forward pass):

    image [B, 3, 224, 224]
        ↓ DualStreamBackbone (ResNet50 + NPR-CNN)
    features [B, 2560, 7, 7]
        ↓ ConceptBottleneckLayer (K=6 concept heads)
    concept_vector [B, K=6]      concept_heatmaps [B, K=6, 7, 7]
        ↓ Linear (K → 1)
    logit [B]
        ↓ sigmoid
    P(AI) [B]

Returns a dict of outputs so loss code and visualization code can both retrieve
what they need without re-running the forward.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..concepts.base import ConceptBottleneckLayer
from .backbone import DualStreamBackbone


class CBNetAGID(nn.Module):
    """Full CBNet-AGID model.

    Args:
        n_concepts:    K, number of concept heads (default 6).
        pretrained:    whether to use ImageNet-pretrained ResNet-50 (default True).
        signal_channels: output channels for the signal stream (default 512).
        frozen_stem:   whether to freeze early ResNet layers.
    """

    def __init__(self, n_concepts: int = 6, pretrained: bool = True,
                 signal_channels: int = 512, frozen_stem: bool = False):
        super().__init__()
        self.backbone = DualStreamBackbone(
            pretrained=pretrained,
            signal_channels=signal_channels,
            frozen_stem=frozen_stem,
        )
        self.cbl = ConceptBottleneckLayer(
            in_channels=self.backbone.out_channels,
            n_concepts=n_concepts,
        )
        # Linear classifier on the concept vector — NO skip connection from features
        # to here. This is the architectural guarantee of interpretability.
        self.classifier = nn.Linear(n_concepts, 1)
        self.n_concepts = n_concepts

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W] input image.

        Returns dict with keys:
            'logit':          [B]        un-sigmoided classifier output
            'prob':           [B]        sigmoid(logit) = P(AI)
            'concepts':       [B, K]     concept vector (sigmoid-activated)
            'concept_maps':   [B, K, H', W']  per-concept spatial heatmaps
            'features':       [B, C, H', W']  raw backbone features (for inspection)
        """
        features = self.backbone(x)                          # [B, C, H', W']
        concepts, heatmaps = self.cbl(features)              # [B, K], [B, K, H', W']
        logit = self.classifier(concepts).squeeze(-1)        # [B]
        prob = torch.sigmoid(logit)
        return {
            "logit": logit,
            "prob": prob,
            "concepts": concepts,
            "concept_maps": heatmaps,
            "features": features,
        }

    def get_concept_contributions(self, x: torch.Tensor) -> torch.Tensor:
        """Get per-concept contribution to the final prediction: w_k * c_k.

        Useful for interpretability visualization. Returns [B, K] tensor.
        """
        out = self.forward(x)
        concepts = out["concepts"]                    # [B, K]
        w = self.classifier.weight.squeeze(0)         # [K]
        return concepts * w  # broadcast multiply → [B, K]
