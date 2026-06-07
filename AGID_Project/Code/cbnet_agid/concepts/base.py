"""Parametric concept heads and the Concept Bottleneck Layer.

A ConceptHead is a small CNN that takes backbone features as input and produces:
  - a scalar concept activation in [0, 1]
  - a spatial heatmap with the same spatial size as the backbone features

The *semantic identity* of each concept (BitPlane-LSB vs Freq-Radial vs ...) does
not come from the architecture (which is identical across all heads) — it comes
from the heuristic label that supervises the head during training, defined in
`concepts.heuristics`.

This design choice is deliberate: by sharing architecture and only differentiating
via supervision, we keep the model lightweight and ensure all concepts compete on a
level playing field. The bottleneck (the K-vector of concept activations) is the
*sole* path from features to prediction — architecturally enforced interpretability.
"""
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptHead(nn.Module):
    """A single concept head: feature → spatial heatmap → scalar.

    Args:
        in_channels: number of channels in backbone feature input (e.g., 2048 for ResNet50).
        hidden_channels: hidden width of the head (default in_channels // 4).
    """

    def __init__(self, in_channels: int, hidden_channels: int = None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels // 4
        self.spatial_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [B, C, H', W'] backbone feature map.

        Returns:
            concept_score:   [B]                  scalar in [0, 1] (sigmoid-activated mean)
            spatial_heatmap: [B, H', W']          per-spatial-location activation, post-sigmoid
        """
        h = self.spatial_head(features)            # [B, 1, H', W']
        h = h.squeeze(1)                            # [B, H', W']
        h_sigmoid = torch.sigmoid(h)                # spatial heatmap, in [0, 1]
        # Scalar concept: mean of the spatial heatmap (interpretable: average activation)
        concept_score = h_sigmoid.mean(dim=(1, 2))  # [B]
        return concept_score, h_sigmoid


class ConceptBottleneckLayer(nn.Module):
    """The Concept Bottleneck Layer: K parallel ConceptHeads producing a K-dim concept vector.

    The classifier downstream is a linear layer on this K-vector. There is no skip
    connection from features → classifier — the K-vector is the SOLE information path.
    This is the structural guarantee that makes CBNet-AGID interpretable by design.

    Args:
        in_channels: number of channels in backbone feature input.
        n_concepts:  K, number of concept heads.
    """

    def __init__(self, in_channels: int, n_concepts: int):
        super().__init__()
        self.n_concepts = n_concepts
        self.heads = nn.ModuleList([
            ConceptHead(in_channels) for _ in range(n_concepts)
        ])

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [B, C, H', W'] backbone feature map.

        Returns:
            concept_vector:  [B, K]                concept activations in [0, 1]
            concept_heatmaps:[B, K, H', W']        per-concept spatial heatmaps in [0, 1]
        """
        scores: List[torch.Tensor] = []
        heatmaps: List[torch.Tensor] = []
        for head in self.heads:
            c, h = head(features)
            scores.append(c)
            heatmaps.append(h)
        concept_vector = torch.stack(scores, dim=1)         # [B, K]
        concept_heatmaps = torch.stack(heatmaps, dim=1)     # [B, K, H', W']
        return concept_vector, concept_heatmaps
