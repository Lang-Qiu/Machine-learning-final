"""Loss functions for CBNet-AGID training.

Total loss = L_task + λ_c·L_concept + λ_g·L_gen_consistency + λ_s·L_sparsity

Each component lives in its own function, and `CBNetLoss` provides a single Module
that computes the weighted sum.
"""
from .losses import (  # noqa: F401
    CBNetLoss,
    task_loss,
    concept_loss,
    gen_consistency_loss,
    sparsity_loss,
)
