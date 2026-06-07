"""Loss components for CBNet-AGID.

L_total = L_task + λ_c · L_concept + λ_g · L_gen_consistency + λ_s · L_sparsity

Implemented:
  - task_loss:              Binary cross-entropy on real/AI label
  - concept_loss:           MSE between concept activations and heuristic labels
  - gen_consistency_loss:   ‖c(I^g1) - c(I^g2)‖_1 averaged over generator pairs
  - sparsity_loss:          - sum_k c_k * log(c_k) — high entropy ⇒ diverse concepts
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def task_loss(logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Standard BCE-with-logits loss on the binary AI/real label.

    Args:
        logit: [B]   classifier output (un-sigmoided).
        label: [B]   0 = real, 1 = AI.

    Returns:
        scalar BCE loss.
    """
    return F.binary_cross_entropy_with_logits(logit, label.float())


def concept_loss(concepts: torch.Tensor, target_concepts: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """MSE between concept-bottleneck activations and heuristic concept labels.

    Args:
        concepts:        [B, K]   model concept vector (sigmoid-activated, in [0, 1]).
        target_concepts: [B, K]   pre-computed heuristic labels (normalized to [0, 1]).
        mask:            [B, K]   optional binary mask: 1 = include in loss, 0 = ignore.
                                    Useful when only some concepts have valid labels.

    Returns:
        scalar MSE.
    """
    if mask is not None:
        diff = (concepts - target_concepts) ** 2 * mask.float()
        return diff.sum() / mask.sum().clamp(min=1.0)
    return F.mse_loss(concepts, target_concepts)


def gen_consistency_loss(concepts_a: torch.Tensor, concepts_b: torch.Tensor) -> torch.Tensor:
    """Cross-generator consistency: concept activations should be similar for AI images
    that have similar content but were produced by different generators.

    Args:
        concepts_a: [B, K] concept vector from one generator (or one image variant).
        concepts_b: [B, K] concept vector from another generator (or augmented variant).

    Returns:
        scalar L1 loss between the two concept vectors.
    """
    return (concepts_a - concepts_b).abs().mean()


def sparsity_loss(concepts: torch.Tensor) -> torch.Tensor:
    """NEGATIVE entropy of the concept distribution per sample → low value ⇒ diverse activations.

    We use this as a *regularizer*: minimizing -entropy = maximizing entropy = encouraging
    the model NOT to collapse to using a single concept. Entropy of the concept distribution
    (treated as probabilities normalized to sum to 1):

        H = - sum_k p_k * log(p_k),   where p_k = c_k / sum(c)

    We return  -H,  so minimizing this loss maximizes H. Bounded in [-log K, 0].

    Args:
        concepts: [B, K] concept vector (sigmoid-activated, in [0, 1]).

    Returns:
        scalar.
    """
    p = concepts / (concepts.sum(dim=1, keepdim=True) + 1e-9)
    entropy = -(p * (p.clamp(min=1e-9)).log()).sum(dim=1)  # [B]
    return -entropy.mean()


def content_pairing_loss(
    concepts: torch.Tensor,
    generator_ids: torch.Tensor,
    class_idxs: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """L1 between concept vectors of content-paired cross-generator images.

    Pairs (i, j) are valid when:
        class_idxs[i] == class_idxs[j]    (same ImageNet class → same content)
        labels[i] == labels[j]             (both ai or both nature)
        generator_ids[i] != generator_ids[j]  (different generator)
        class_idxs[i] >= 0                 (filename parse succeeded)
    Upper-triangle only to avoid double-counting each pair.

    Args:
        concepts:       [B, K] concept vectors from model forward pass.
        generator_ids:  [B]    int tensor, each in [0, n_gens).
        class_idxs:     [B]    int tensor, ImageNet class or -1 if unparseable.
        labels:         [B]    int tensor, 0=real, 1=ai.

    Returns:
        scalar L1 loss (0.0 if no valid pairs exist in this batch).
    """
    B = concepts.size(0)
    device = concepts.device

    ci = class_idxs.unsqueeze(1).expand(B, B)
    cj = class_idxs.unsqueeze(0).expand(B, B)
    li = labels.unsqueeze(1).expand(B, B)
    lj = labels.unsqueeze(0).expand(B, B)
    gi = generator_ids.unsqueeze(1).expand(B, B)
    gj = generator_ids.unsqueeze(0).expand(B, B)
    upper = torch.triu(torch.ones(B, B, dtype=torch.bool, device=device), diagonal=1)

    mask = (ci == cj) & (li == lj) & (gi != gj) & (ci >= 0) & upper  # [B, B]

    if not mask.any():
        return torch.tensor(0.0, device=device, requires_grad=False)

    diff = (concepts.unsqueeze(1) - concepts.unsqueeze(0)).abs()  # [B, B, K]
    return diff[mask].mean()


class CBNetLoss(nn.Module):
    """Combined loss for CBNet-AGID training.

    Args:
        lambda_concept:     weight for L_concept (default 0.5).
        lambda_gen:         weight for L_gen_consistency (default 0.2).
        lambda_sparsity:    weight for L_sparsity (default 0.05).
    """

    def __init__(self, lambda_concept: float = 0.5, lambda_gen: float = 0.2,
                 lambda_sparsity: float = 0.05):
        super().__init__()
        self.lambda_concept = lambda_concept
        self.lambda_gen = lambda_gen
        self.lambda_sparsity = lambda_sparsity

    def forward(self, model_out: Dict[str, torch.Tensor],
                label: torch.Tensor,
                target_concepts: Optional[torch.Tensor] = None,
                paired_out: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Compute the combined loss and individual components for logging.

        Args:
            model_out:        dict returned by CBNetAGID.forward (must contain 'logit', 'concepts').
            label:            [B]     real/AI label (0/1).
            target_concepts:  [B, K]  optional pre-computed heuristic concept labels.
            paired_out:       optional dict for a paired (generator-varied) batch, for
                              cross-generator consistency. Must contain 'concepts' if provided.

        Returns:
            dict with keys 'total', 'task', 'concept', 'gen_consistency', 'sparsity'.
        """
        l_task = task_loss(model_out["logit"], label)
        l_concept = (
            concept_loss(model_out["concepts"], target_concepts)
            if target_concepts is not None
            else torch.tensor(0.0, device=l_task.device)
        )
        l_gen = (
            gen_consistency_loss(model_out["concepts"], paired_out["concepts"])
            if paired_out is not None
            else torch.tensor(0.0, device=l_task.device)
        )
        l_sparse = sparsity_loss(model_out["concepts"])

        total = (
            l_task
            + self.lambda_concept * l_concept
            + self.lambda_gen * l_gen
            + self.lambda_sparsity * l_sparse
        )

        return {
            "total": total,
            "task": l_task.detach(),
            "concept": l_concept.detach(),
            "gen_consistency": l_gen.detach(),
            "sparsity": l_sparse.detach(),
        }


class CBNetMultiGenLoss(nn.Module):
    """Combined loss for Route B multi-generator training.

    Extends CBNetLoss with a content-pairing consistency term that enforces
    similar concept activations for images from different generators that share
    the same ImageNet class index (i.e., same visual content, different generation
    process).

    L_total = L_task + λ_c · L_concept + λ_pair · L_content_pair + λ_s · L_sparsity

    Args:
        lambda_concept:     weight for L_concept MSE (default 0.5).
        lambda_pair:        weight for L_content_pair (default 0.2).
        lambda_sparsity:    weight for L_sparsity entropy regularizer (default 0.05).
    """

    def __init__(self, lambda_concept: float = 0.5, lambda_pair: float = 0.2,
                 lambda_sparsity: float = 0.05):
        super().__init__()
        self.lambda_concept = lambda_concept
        self.lambda_pair = lambda_pair
        self.lambda_sparsity = lambda_sparsity

    def forward(
        self,
        model_out: Dict[str, torch.Tensor],
        label: torch.Tensor,
        generator_ids: torch.Tensor,
        class_idxs: torch.Tensor,
        target_concepts: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the combined Route B loss.

        Args:
            model_out:       dict from CBNetAGID.forward ('logit', 'concepts').
            label:           [B]  real/AI label (0/1).
            generator_ids:   [B]  generator index per sample (from InterleavedBatchSampler).
            class_idxs:      [B]  ImageNet class index per sample (from filename parsing).
            target_concepts: [B, K] optional precomputed heuristic concept labels.

        Returns:
            dict: 'total', 'task', 'concept', 'content_pair', 'sparsity'.
        """
        l_task = task_loss(model_out["logit"], label)
        l_concept = (
            concept_loss(model_out["concepts"], target_concepts)
            if target_concepts is not None
            else torch.tensor(0.0, device=l_task.device)
        )
        l_pair = content_pairing_loss(
            model_out["concepts"], generator_ids, class_idxs, label
        )
        l_sparse = sparsity_loss(model_out["concepts"])

        total = (
            l_task
            + self.lambda_concept * l_concept
            + self.lambda_pair * l_pair
            + self.lambda_sparsity * l_sparse
        )

        return {
            "total": total,
            "task": l_task.detach(),
            "concept": l_concept.detach(),
            "content_pair": l_pair.detach(),
            "sparsity": l_sparse.detach(),
        }
