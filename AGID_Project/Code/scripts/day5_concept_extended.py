"""
Day 5+ Extended Concept Sanity — test all 8 candidate concepts (vs Day 5's first 3).

Extends `day5_concept_sanity.py` with 5 additional candidate concepts from
Stage1_Research/02_Methodology_Blueprint.md §4. Once GenImage val subset is available,
this script will tell us which 6 of the 8 candidates earn a slot in CBNet-AGID's
concept bottleneck layer.

Concepts (8 total):
  EXISTING (validated in Day 5):
    1. HF-Noise-Anomaly         (Laplacian variance)
    2. BitPlane-LSB-Pattern     (LSB autocorrelation)
    3. Freq-Subband-Energy      (8x8 block DCT mid+high ratio)
  NEW (Day 5+ additions):
    4. Freq-Radial-Profile      (alternative frequency analysis: full-image FFT log-radial slope)
    5. EdgeSharpness-Inconsistency  (variance of Sobel-magnitude across patches)
    6. Color-Manifold-Deviation     (RGB color histogram KL vs real-prior — uses sample-derived prior)
    7. JPEG-QuantTrace-Absence      (variance of DCT-coefficient quantization residuals)
    8. Texture-Geometry-Drift       (correlation between edge density and texture variance per patch)

Usage:
    python day5_concept_extended.py --images_dir <path-to-Data/sanity-or-GenImage-val>
    python day5_concept_extended.py --images_dir <path> --concepts 4,5,6  # subset
"""

import argparse
import os
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from scipy import stats


# ---------------------------------------------------------------------------
# Concept implementations (numpy-only)
# ---------------------------------------------------------------------------

def concept_hf_noise(img_rgb: np.ndarray) -> float:
    """[1] HF-Noise-Anomaly: Laplacian variance."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def concept_bitplane_lsb(img_rgb: np.ndarray) -> float:
    """[2] BitPlane-LSB-Pattern: |1-lag horizontal autocorrelation of LSB|."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lsb = (gray & 0x01).astype(np.float64)
    a = lsb[:, :-1].flatten()
    b = lsb[:, 1:].flatten()
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(abs(np.corrcoef(a, b)[0, 1]))


def concept_freq_subband(img_rgb: np.ndarray) -> float:
    """[3] Freq-Subband-Energy: 8x8 block DCT mid+high freq ratio."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) - 128.0
    h, w = gray.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    if h8 < 8 or w8 < 8:
        return float("nan")
    gray = gray[:h8, :w8]
    blocks = gray.reshape(h8 // 8, 8, w8 // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    dcts = np.zeros_like(blocks)
    for i in range(blocks.shape[0]):
        dcts[i] = cv2.dct(blocks[i])
    energy = dcts ** 2
    mask = np.zeros((8, 8), dtype=bool)
    for r in range(8):
        for c in range(8):
            if r + c >= 3:
                mask[r, c] = True
    total = energy.sum()
    if total <= 1e-9:
        return float("nan")
    return float(energy[:, mask].sum() / total)


def concept_freq_radial(img_rgb: np.ndarray) -> float:
    """[4] Freq-Radial-Profile: slope of log-log radial frequency spectrum.

    Natural images follow 1/f^alpha; AI images may deviate. We compute the slope
    of log(power) vs log(radial frequency) and return its absolute value.
    Real photographs: slope around -2 (1/f^2 spectrum).
    AI images: often flatter (less negative slope) due to upsampling/synthesis.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # Pad to power of 2 for efficient FFT; downsample if too big
    max_dim = 512
    if max(gray.shape) > max_dim:
        scale = max_dim / max(gray.shape)
        gray = cv2.resize(gray, (int(gray.shape[1] * scale), int(gray.shape[0] * scale)))
    h, w = gray.shape
    f = np.fft.fft2(gray - gray.mean())
    fshift = np.fft.fftshift(f)
    psd = np.abs(fshift) ** 2
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(np.int32)
    rmax = min(cy, cx)
    if rmax < 10:
        return float("nan")
    radial_psd = np.bincount(r.ravel(), psd.ravel())[1:rmax]
    radial_count = np.bincount(r.ravel())[1:rmax]
    radial_avg = radial_psd / np.maximum(radial_count, 1)
    # Linear fit on log-log
    rr = np.arange(1, rmax)
    log_r = np.log(rr)
    log_p = np.log(np.maximum(radial_avg, 1e-12))
    # Take middle 60% to skip DC and aliasing edge
    keep = slice(int(rmax * 0.1), int(rmax * 0.7))
    coef = np.polyfit(log_r[keep], log_p[keep], 1)
    return float(abs(coef[0]))  # absolute slope


def concept_edge_inconsistency(img_rgb: np.ndarray) -> float:
    """[5] EdgeSharpness-Inconsistency: across-patch variance of mean edge magnitude.

    Real photos have spatially varying sharpness (e.g., focal point sharp, background
    blurred). AI images often have homogeneous "everywhere-sharp" rendering — leading
    to LOWER variance of edge sharpness across patches.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = np.sqrt(cv2.Sobel(gray, cv2.CV_64F, 1, 0) ** 2 +
                    cv2.Sobel(gray, cv2.CV_64F, 0, 1) ** 2)
    h, w = edges.shape
    patch = 32
    h_p, w_p = h // patch, w // patch
    if h_p < 2 or w_p < 2:
        return float("nan")
    edges_crop = edges[: h_p * patch, : w_p * patch]
    patch_means = edges_crop.reshape(h_p, patch, w_p, patch).mean(axis=(1, 3))
    return float(patch_means.var())


# Global REAL color prior — built lazily from the first batch of "real" images we see.
_REAL_COLOR_PRIOR = None
_REAL_PRIOR_NBINS = 16


def _color_histogram(img_rgb: np.ndarray, nbins: int = _REAL_PRIOR_NBINS) -> np.ndarray:
    """Normalised RGB 3D histogram."""
    h, _ = np.histogramdd(
        img_rgb.reshape(-1, 3),
        bins=(nbins,) * 3,
        range=((0, 256), (0, 256), (0, 256)),
    )
    h = h.astype(np.float64) / max(h.sum(), 1.0)
    return h


def set_real_color_prior(real_images: list):
    """Build a global real-image color prior from a list of RGB images."""
    global _REAL_COLOR_PRIOR
    if not real_images:
        _REAL_COLOR_PRIOR = None
        return
    hist_sum = np.zeros((_REAL_PRIOR_NBINS,) * 3, dtype=np.float64)
    for img in real_images:
        hist_sum += _color_histogram(img)
    _REAL_COLOR_PRIOR = hist_sum / len(real_images)


def concept_color_manifold(img_rgb: np.ndarray) -> float:
    """[6] Color-Manifold-Deviation: KL(image_hist || real_prior_hist)."""
    if _REAL_COLOR_PRIOR is None:
        return float("nan")
    h = _color_histogram(img_rgb)
    p = _REAL_COLOR_PRIOR + 1e-9
    q = h + 1e-9
    kl = float((q * np.log(q / p)).sum())
    return kl


def concept_quant_trace(img_rgb: np.ndarray) -> float:
    """[7] JPEG-QuantTrace-Absence: variance of DCT-coefficient residuals after re-quantization.

    JPEG-compressed real photos have DCT coefficients near integer multiples of
    quantization steps (~quantization grid). AI-generated images often LACK this
    grid because the generator outputs continuous-valued pixels that are then
    encoded to JPEG/WebP after rendering. Measure: variance of (DCT_coef - rounded_DCT_coef)
    in mid-frequency band.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) - 128.0
    h, w = gray.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    if h8 < 8 or w8 < 8:
        return float("nan")
    gray = gray[:h8, :w8]
    blocks = gray.reshape(h8 // 8, 8, w8 // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    dcts = np.zeros_like(blocks)
    for i in range(blocks.shape[0]):
        dcts[i] = cv2.dct(blocks[i])
    # Estimate quantization step per coefficient as the median of |dct| / round(|dct|)
    # Simpler: compute residual after rounding to nearest quantization estimate
    abs_dcts = np.abs(dcts)
    # Use a heuristic quantization grid of 10 (typical for mid-quality JPEG)
    grid = 10.0
    residual = dcts - grid * np.round(dcts / grid)
    return float(residual.var())


def concept_texture_geometry(img_rgb: np.ndarray) -> float:
    """[8] Texture-Geometry-Drift: per-patch correlation between edge density and texture variance.

    In real photos: edges and texture co-vary (high-texture regions also have many edges).
    In AI images: this coupling can be broken because AI generators may produce textured
    regions WITHOUT consistent edges, or sharp edges WITHOUT consistent texture.
    Measure: Pearson r between per-patch edge-density and per-patch grayscale variance.
    A LOWER r suggests this coupling is broken (AI-like).
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    edges = np.sqrt(cv2.Sobel(gray, cv2.CV_64F, 1, 0) ** 2 +
                    cv2.Sobel(gray, cv2.CV_64F, 0, 1) ** 2)
    h, w = gray.shape
    patch = 32
    h_p, w_p = h // patch, w // patch
    if h_p < 2 or w_p < 2:
        return float("nan")
    gray_c = gray[:h_p * patch, :w_p * patch]
    edges_c = edges[:h_p * patch, :w_p * patch]
    edge_dens = edges_c.reshape(h_p, patch, w_p, patch).mean(axis=(1, 3)).flatten()
    tex_var = gray_c.reshape(h_p, patch, w_p, patch).var(axis=(1, 3)).flatten()
    if edge_dens.std() == 0 or tex_var.std() == 0:
        return float("nan")
    r = float(np.corrcoef(edge_dens, tex_var)[0, 1])
    # Return 1 - r so that AI-like (lower coupling) yields higher value
    return 1.0 - r


CONCEPTS = [
    ("HF-Noise-Anomaly", concept_hf_noise),
    ("BitPlane-LSB-Pattern", concept_bitplane_lsb),
    ("Freq-Subband-Energy", concept_freq_subband),
    ("Freq-Radial-Profile", concept_freq_radial),
    ("EdgeSharpness-Inconsistency", concept_edge_inconsistency),
    ("Color-Manifold-Deviation", concept_color_manifold),
    ("JPEG-QuantTrace-Absence", concept_quant_trace),
    ("Texture-Geometry-Drift", concept_texture_geometry),
]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def load_images_from_dir(dir_path: Path) -> list:
    images = []
    for p in dir_path.rglob("*"):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            continue
        try:
            img = np.array(Image.open(p).convert("RGB"))
            images.append((p, img))
        except Exception as e:
            print(f"[WARN] failed to load {p}: {e}")
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir",
                        default="E:/LQiu/lab_folder/Machine_learning/AGID_Project/Data/sanity")
    parser.add_argument("--concepts", default="1,2,3,4,5,6,7,8",
                        help="Comma-separated 1-indexed concept IDs to run.")
    parser.add_argument("--save_plots", action="store_true")
    args = parser.parse_args()

    selected_ids = [int(x) for x in args.concepts.split(",") if x.strip()]
    selected = [CONCEPTS[i - 1] for i in selected_ids]
    print(f"[INFO] running {len(selected)} concepts: {[n for n, _ in selected]}")

    base = Path(args.images_dir)
    real_dir = base / "0_real"
    fake_dir = base / "1_fake"
    for d in [real_dir, fake_dir]:
        if not d.exists():
            raise FileNotFoundError(d)

    real_imgs = load_images_from_dir(real_dir)
    fake_imgs = load_images_from_dir(fake_dir)
    print(f"[INFO] real: {len(real_imgs)}, fake: {len(fake_imgs)}")

    # Build real color prior if Concept 6 selected (uses ONLY real images, NOT fake)
    if any(name == "Color-Manifold-Deviation" for name, _ in selected):
        set_real_color_prior([img for _, img in real_imgs])
        print(f"[INFO] real-color prior built from {len(real_imgs)} real images")

    # Compute concept values
    results = {name: {"real": [], "fake": []} for name, _ in selected}
    for path, img in real_imgs:
        for name, fn in selected:
            v = fn(img)
            if not np.isnan(v):
                results[name]["real"].append(v)
    for path, img in fake_imgs:
        for name, fn in selected:
            v = fn(img)
            if not np.isnan(v):
                results[name]["fake"].append(v)

    # Summary
    print("\n" + "=" * 92)
    print("Extended Concept Sanity Results")
    print("=" * 92)
    print(f"{'Concept':<30s} {'Real μ±σ':<20s} {'AI μ±σ':<20s} {'t':>7s} {'p':>10s} {'d':>6s}")
    print("-" * 92)
    summary = []
    csv_rows = ["concept,real_mean,real_std,real_n,fake_mean,fake_std,fake_n,t_stat,p_value,cohens_d"]
    for name, _ in selected:
        r = np.array(results[name]["real"])
        f = np.array(results[name]["fake"])
        if len(r) < 2 or len(f) < 2:
            continue
        rm, rs = r.mean(), r.std()
        fm, fs = f.mean(), f.std()
        if rs > 0 or fs > 0:
            t, p = stats.ttest_ind(r, f, equal_var=False)
        else:
            t, p = float("nan"), float("nan")
        psd = np.sqrt((rs ** 2 + fs ** 2) / 2)
        d = (rm - fm) / psd if psd > 0 else float("nan")
        print(f"{name:<30s} {rm:>9.3g}±{rs:<10.3g} {fm:>9.3g}±{fs:<10.3g} {t:>+6.2f} {p:>10.3g} {d:>+6.2f}")
        summary.append((name, p, d))
        csv_rows.append(f"{name},{rm:.6g},{rs:.6g},{len(r)},{fm:.6g},{fs:.6g},{len(f)},{t:.6g},{p:.6g},{d:.6g}")

    print("\n" + "=" * 92)
    print("Verdict — sorted by effect size")
    print("=" * 92)
    summary.sort(key=lambda x: -abs(x[2]) if not np.isnan(x[2]) else 0)
    for name, p, d in summary:
        if np.isnan(p):
            v = "INSUFFICIENT"
        elif p < 0.001 and abs(d) >= 0.5:
            v = "✅ STRONG"
        elif p < 0.05 and abs(d) >= 0.3:
            v = "🟡 MODERATE"
        elif p < 0.05:
            v = "🟡 WEAK"
        else:
            v = "🔴 NO SIGNAL"
        print(f"  d={d:>+6.2f}  p={p:>10.3g}  {v:<14s} {name}")

    results_dir = Path("E:/LQiu/lab_folder/Machine_learning/AGID_Project/Results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "day5_concept_extended.csv"
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(csv_rows))
    print(f"\n[INFO] saved {csv_path}")

    if args.save_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            n = len(selected)
            cols = min(4, n)
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
            axes = np.array(axes).flatten() if rows * cols > 1 else [axes]
            for ax, (name, _) in zip(axes, selected):
                r = results[name]["real"]
                f = results[name]["fake"]
                if not r or not f:
                    ax.set_title(f"{name}\n(insufficient data)")
                    continue
                ax.hist(r, alpha=0.6, label="Real", bins=15, color="C0")
                ax.hist(f, alpha=0.6, label="AI", bins=15, color="C3")
                ax.set_title(name, fontsize=10)
                ax.legend(fontsize=8)
            for ax in axes[len(selected):]:
                ax.axis("off")
            fig.tight_layout()
            plot_path = results_dir / "day5_concept_extended.png"
            fig.savefig(plot_path, dpi=110)
            print(f"[INFO] saved {plot_path}")
        except ImportError as e:
            print(f"[WARN] matplotlib unavailable: {e}")


if __name__ == "__main__":
    main()
