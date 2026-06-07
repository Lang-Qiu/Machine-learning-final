"""Concept module — concept heads (parametric) + heuristic label generators (non-parametric).

The 6 concepts in the final set:
  1. BitPlane-LSB-Pattern
  2. Freq-Radial-Profile
  3. Color-Manifold-Deviation
  4. HF-Noise-Anomaly
  5. JPEG-QuantTrace-Absence    (borderline; included pending GenImage retest)
  6. Texture-Geometry-Drift     (borderline; included pending GenImage retest)

Each concept has two ingredients:
  - `ConceptHead` — a small parametric head that maps backbone features to (scalar, heatmap)
  - `heuristic_label_*` — a non-parametric function that maps raw images to a soft scalar
                          label in [0, 1] used to supervise the concept head via MSE
"""
from .base import ConceptHead, ConceptBottleneckLayer  # noqa: F401
from .heuristics import (  # noqa: F401
    heuristic_bitplane_lsb,
    heuristic_freq_radial,
    heuristic_color_manifold,
    heuristic_hf_noise,
    heuristic_jpeg_quant,
    heuristic_texture_geometry,
    compute_concept_labels,
    CONCEPT_NAMES,
)
