"""Heuristic concept-label generators (non-parametric).

These compute one scalar concept label per image, normalized to [0, 1]. They are
used as WEAK SUPERVISION targets in the concept-MSE loss during training.

Each function operates on a single image (or a batch). For batch operation, the
caller can vectorize via Python loop — heuristics here are not on the hot path
since they are pre-computed once per image and cached. See `compute_concept_labels`
for the batched convenience wrapper.

Algorithmic content matches `Code/scripts/day5_concept_extended.py` for reproducibility;
this module just ports the implementation to PyTorch-tensor-friendly operations
where possible (some still use NumPy / OpenCV under the hood).

Day 5 validation effect sizes on a 20+20 sample (camera-JPEG real / Bing-WebP fake):
  bitplane_lsb     d = -2.77  (anchor)
  freq_radial      d = -1.83
  color_manifold   d = -1.63
  hf_noise         d = -0.99
  jpeg_quant       d = +0.64  (borderline)
  texture_geometry d = +0.58  (borderline)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor

# Order is fixed and matches the K=6 concepts in the bottleneck.
CONCEPT_NAMES: List[str] = [
    "bitplane_lsb",
    "freq_radial",
    "color_manifold",
    "hf_noise",
    "jpeg_quant",
    "texture_geometry",
]


# --------------------------------------------------------------------------- #
# Per-image heuristic computations (NumPy)
# --------------------------------------------------------------------------- #

def heuristic_bitplane_lsb(img_uint8: np.ndarray) -> float:
    """|1-lag horizontal autocorrelation of grayscale LSB|.

    Real photos: ~0 (sensor noise → independent LSBs).
    AI photos:   higher (clean generation → correlated LSBs).
    """
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    lsb = (gray & 0x01).astype(np.float64)
    a = lsb[:, :-1].flatten()
    b = lsb[:, 1:].flatten()
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(abs(np.corrcoef(a, b)[0, 1]))


def heuristic_freq_radial(img_uint8: np.ndarray) -> float:
    """|slope of log-log radial PSD of full-image FFT|.

    Real photos: slope ~ -2 (1/f^2 spectrum).
    AI photos:   often flatter slope (less negative).
    """
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    max_dim = 512
    if max(gray.shape) > max_dim:
        scale = max_dim / max(gray.shape)
        gray = cv2.resize(gray, (int(gray.shape[1] * scale), int(gray.shape[0] * scale)))
    h, w = gray.shape
    f = np.fft.fft2(gray - gray.mean())
    psd = np.abs(np.fft.fftshift(f)) ** 2
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(np.int32)
    rmax = min(cy, cx)
    if rmax < 10:
        return 2.0  # neutral
    radial_psd = np.bincount(r.ravel(), psd.ravel())[1:rmax]
    radial_count = np.bincount(r.ravel())[1:rmax]
    radial_avg = radial_psd / np.maximum(radial_count, 1)
    rr = np.arange(1, rmax)
    log_r = np.log(rr)
    log_p = np.log(np.maximum(radial_avg, 1e-12))
    keep = slice(int(rmax * 0.1), int(rmax * 0.7))
    coef = np.polyfit(log_r[keep], log_p[keep], 1)
    return float(abs(coef[0]))


_REAL_COLOR_PRIOR: Optional[np.ndarray] = None
_COLOR_NBINS: int = 16


def set_real_color_prior(real_images_uint8: List[np.ndarray]) -> None:
    """Build the global real-image color prior (call once at training-data prep)."""
    global _REAL_COLOR_PRIOR
    if not real_images_uint8:
        _REAL_COLOR_PRIOR = None
        return
    hist_sum = np.zeros((_COLOR_NBINS,) * 3, dtype=np.float64)
    for img in real_images_uint8:
        h, _ = np.histogramdd(
            img.reshape(-1, 3),
            bins=(_COLOR_NBINS,) * 3,
            range=((0, 256), (0, 256), (0, 256)),
        )
        hist_sum += h / max(h.sum(), 1.0)
    _REAL_COLOR_PRIOR = hist_sum / len(real_images_uint8)


def load_real_color_prior(prior: np.ndarray) -> None:
    """Set the global real-image color prior directly from a precomputed array.

    Used to share a single prior across multiple generators (Route B). The shape must
    match (_COLOR_NBINS,) * 3.
    """
    global _REAL_COLOR_PRIOR
    expected_shape = (_COLOR_NBINS,) * 3
    if prior.shape != expected_shape:
        raise ValueError(
            f"Prior shape {prior.shape} != expected {expected_shape}; "
            f"_COLOR_NBINS={_COLOR_NBINS}. Did you build the prior with a different bin count?"
        )
    _REAL_COLOR_PRIOR = prior.astype(np.float64)


def get_real_color_prior() -> Optional[np.ndarray]:
    """Return a copy of the current global real-image color prior, or None if unset."""
    if _REAL_COLOR_PRIOR is None:
        return None
    return _REAL_COLOR_PRIOR.copy()


def heuristic_color_manifold(img_uint8: np.ndarray) -> float:
    """KL(image RGB histogram || real-image color prior)."""
    if _REAL_COLOR_PRIOR is None:
        # No prior yet → return neutral value. Will be near-zero discriminative.
        return 0.0
    h, _ = np.histogramdd(
        img_uint8.reshape(-1, 3),
        bins=(_COLOR_NBINS,) * 3,
        range=((0, 256), (0, 256), (0, 256)),
    )
    h = h.astype(np.float64) / max(h.sum(), 1.0)
    p = _REAL_COLOR_PRIOR + 1e-9
    q = h + 1e-9
    return float((q * np.log(q / p)).sum())


def heuristic_hf_noise(img_uint8: np.ndarray) -> float:
    """variance(Laplacian(grayscale))."""
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY).astype(np.float64)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def heuristic_jpeg_quant(img_uint8: np.ndarray, grid: float = 10.0) -> float:
    """variance of DCT-coefficient residuals after re-quantization (borderline concept)."""
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32) - 128.0
    h, w = gray.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    if h8 < 8 or w8 < 8:
        return 0.0
    gray = gray[:h8, :w8]
    blocks = gray.reshape(h8 // 8, 8, w8 // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    dcts = np.zeros_like(blocks)
    for i in range(blocks.shape[0]):
        dcts[i] = cv2.dct(blocks[i])
    residual = dcts - grid * np.round(dcts / grid)
    return float(residual.var())


def heuristic_texture_geometry(img_uint8: np.ndarray, patch: int = 32) -> float:
    """1 - Pearson(per-patch edge density, per-patch grayscale variance).

    Lower coupling (high return value) → AI-like.
    """
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY).astype(np.float64)
    edges = np.sqrt(cv2.Sobel(gray, cv2.CV_64F, 1, 0) ** 2 +
                    cv2.Sobel(gray, cv2.CV_64F, 0, 1) ** 2)
    h, w = gray.shape
    h_p, w_p = h // patch, w // patch
    if h_p < 2 or w_p < 2:
        return 0.5
    g = gray[:h_p * patch, :w_p * patch]
    e = edges[:h_p * patch, :w_p * patch]
    edge_dens = e.reshape(h_p, patch, w_p, patch).mean(axis=(1, 3)).flatten()
    tex_var = g.reshape(h_p, patch, w_p, patch).var(axis=(1, 3)).flatten()
    if edge_dens.std() == 0 or tex_var.std() == 0:
        return 0.5
    r = float(np.corrcoef(edge_dens, tex_var)[0, 1])
    return 1.0 - r


# --------------------------------------------------------------------------- #
# Batched API
# --------------------------------------------------------------------------- #

_HEURISTIC_FUNCS = {
    "bitplane_lsb":     heuristic_bitplane_lsb,
    "freq_radial":      heuristic_freq_radial,
    "color_manifold":   heuristic_color_manifold,
    "hf_noise":         heuristic_hf_noise,
    "jpeg_quant":       heuristic_jpeg_quant,
    "texture_geometry": heuristic_texture_geometry,
}


def compute_concept_labels(img_uint8: np.ndarray,
                           concept_names: Optional[List[str]] = None) -> Dict[str, float]:
    """Compute raw (non-normalized) heuristic concept labels for one image.

    Args:
        img_uint8:     [H, W, 3] uint8 RGB image.
        concept_names: which concepts to compute (default: all 6).
    """
    if concept_names is None:
        concept_names = CONCEPT_NAMES
    out = {}
    for name in concept_names:
        fn = _HEURISTIC_FUNCS[name]
        out[name] = float(fn(img_uint8))
    return out


def normalize_concept_labels(values: np.ndarray, low: float = None, high: float = None,
                              method: str = "minmax") -> np.ndarray:
    """Normalize raw concept values to [0, 1].

    Args:
        values: [N] array of raw concept values for a dataset.
        low, high: if provided, used directly (e.g., for test set with train statistics).
        method: 'minmax' (default), 'percentile' (uses 2nd / 98th), 'sigmoid'.

    Returns the normalized [N] array AND a (low, high) tuple if computed from data.
    """
    if method == "minmax":
        if low is None:
            low = float(np.min(values))
        if high is None:
            high = float(np.max(values))
    elif method == "percentile":
        if low is None:
            low = float(np.percentile(values, 2))
        if high is None:
            high = float(np.percentile(values, 98))
    elif method == "sigmoid":
        # Z-score then sigmoid
        if low is None:
            low = float(np.mean(values))
        if high is None:
            high = float(np.std(values) + 1e-9)
        return 1.0 / (1.0 + np.exp(-(values - low) / high))
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    span = max(high - low, 1e-9)
    return np.clip((values - low) / span, 0.0, 1.0)
