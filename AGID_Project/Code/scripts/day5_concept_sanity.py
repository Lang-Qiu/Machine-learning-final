"""
Day 5 — Concept signal sanity check (CBNet-AGID pre-run validation).

Goal: BEFORE implementing the learnable Concept Bottleneck Layer, verify that the 3
candidate concept signals can ACTUALLY discriminate real from AI-generated images
using simple non-learned heuristics.

If the heuristic signals show significant statistical differences between real and AI
images (per t-test + Cohen's d), then a learnable version of the same concept has at
least a fighting chance. If NOT, the concept is broken and must be revised.

Three candidate concepts implemented (matching `Stage1_Research/02_Methodology_Blueprint.md`):
  1. HF-Noise-Anomaly        — Laplacian variance (proxy for high-frequency noise content)
  2. BitPlane-LSB-Pattern    — LSB plane entropy + spatial autocorrelation
  3. Freq-Subband-Energy     — ratio of mid/high-freq DCT energy to total

Inputs: a directory with subfolders `0_real/` and `1_fake/`
Output: console summary + concept_sanity_results.csv + matplotlib histograms

Usage:
    python day5_concept_sanity.py --images_dir <path-to-Data/sanity>
    # default: E:/LQiu/lab_folder/Machine_learning/AGID_Project/Data/sanity
"""

import argparse
import os
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from scipy import stats


DEFAULT_DATA_DIR = Path("E:/LQiu/lab_folder/Machine_learning/AGID_Project/Data/sanity")


# ---------------------------------------------------------------------------
# Concept implementations (numpy-only, no PyTorch needed for sanity check)
# ---------------------------------------------------------------------------

def concept_hf_noise(img_rgb: np.ndarray) -> float:
    """HF-Noise-Anomaly: Laplacian variance of grayscale image.

    Natural photographs typically have HIGH Laplacian variance due to sensor noise +
    real-world texture details. AI-generated images often have LOWER Laplacian
    variance (smoother high frequencies) because generators tend to under-produce
    high-frequency noise.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def concept_bitplane_lsb(img_rgb: np.ndarray) -> float:
    """BitPlane-LSB-Pattern: spatial autocorrelation of LSB plane.

    Real photos from cameras have near-random LSB (sensor noise → independent bits).
    AI images may have systematic structure in LSB (clean generation → spatially
    correlated bits). Measure: 1-lag autocorrelation of the LSB plane.

    Returns value in [-1, 1]: near 0 means random (real-like), away from 0 means
    structured (AI-like).
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lsb = (gray & 0x01).astype(np.float64)
    # Horizontal 1-lag autocorrelation
    a = lsb[:, :-1].flatten()
    b = lsb[:, 1:].flatten()
    if a.std() == 0 or b.std() == 0:
        return 0.0
    # Pearson correlation
    corr = np.corrcoef(a, b)[0, 1]
    return float(abs(corr))  # use abs since direction doesn't matter


def concept_freq_subband(img_rgb: np.ndarray) -> float:
    """Freq-Subband-Energy: ratio of mid/high-freq energy in 8x8-block DCT.

    Real photographs typically follow 1/f^alpha frequency spectra — energy concentrates
    in low frequencies. AI images often have RELATIVELY higher energy in mid/high
    frequency bands due to upsampling artifacts (GANs) or noise injection (diffusion).

    Returns: fraction of total DCT energy in mid+high freq bands (rows/cols ≥ 3).
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) - 128.0
    h, w = gray.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    gray = gray[:h8, :w8]
    if h8 < 8 or w8 < 8:
        return float("nan")
    # 8x8 block DCT
    blocks = gray.reshape(h8 // 8, 8, w8 // 8, 8).transpose(0, 2, 1, 3)  # (Nh, Nw, 8, 8)
    blocks = blocks.reshape(-1, 8, 8)  # (N, 8, 8)
    # Apply DCT per block
    dcts = np.zeros_like(blocks)
    for i in range(blocks.shape[0]):
        dcts[i] = cv2.dct(blocks[i])
    # Energy = sum of squared DCT coefficients per block
    energy = dcts ** 2
    # Define mid/high freq mask: row + col >= 3 (DC is at [0,0])
    mask_midhigh = np.zeros((8, 8), dtype=bool)
    for r in range(8):
        for c in range(8):
            if r + c >= 3:
                mask_midhigh[r, c] = True
    total_energy = energy.sum()
    if total_energy <= 1e-9:
        return float("nan")
    midhigh_energy = energy[:, mask_midhigh].sum()
    return float(midhigh_energy / total_energy)


CONCEPTS = {
    "HF-Noise-Anomaly":     concept_hf_noise,
    "BitPlane-LSB-Pattern": concept_bitplane_lsb,
    "Freq-Subband-Energy":  concept_freq_subband,
}


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
    parser.add_argument("--images_dir", default=str(DEFAULT_DATA_DIR),
                        help="Path to directory containing 0_real/ and 1_fake/ subdirs.")
    parser.add_argument("--save_plots", action="store_true",
                        help="Save matplotlib histograms to Results/ (default off — matplotlib may not be installed).")
    args = parser.parse_args()

    base = Path(args.images_dir)
    real_dir = base / "0_real"
    fake_dir = base / "1_fake"
    for d in [real_dir, fake_dir]:
        if not d.exists():
            raise FileNotFoundError(f"Missing directory: {d}")

    print(f"[INFO] loading images from {base}")
    real_imgs = load_images_from_dir(real_dir)
    fake_imgs = load_images_from_dir(fake_dir)
    print(f"[INFO] real: {len(real_imgs)}, fake: {len(fake_imgs)}")
    if not real_imgs or not fake_imgs:
        raise RuntimeError("Need at least 1 image in each class.")

    # Compute concept values
    print(f"\n[INFO] computing 3 concept signals...")
    results = {name: {"real": [], "fake": []} for name in CONCEPTS}
    for path, img in real_imgs:
        for name, fn in CONCEPTS.items():
            val = fn(img)
            if not np.isnan(val):
                results[name]["real"].append(val)
    for path, img in fake_imgs:
        for name, fn in CONCEPTS.items():
            val = fn(img)
            if not np.isnan(val):
                results[name]["fake"].append(val)

    # Summary statistics
    print("\n" + "=" * 78)
    print("Concept Sanity Check Results")
    print("=" * 78)
    print(f"{'Concept':<25s} {'Real mean±std':<22s} {'AI mean±std':<22s} {'t-test':<14s} {'Cohen d':<8s}")
    print("-" * 78)

    csv_rows = ["concept,real_mean,real_std,real_n,fake_mean,fake_std,fake_n,t_stat,p_value,cohens_d"]
    summary_rows = []
    for name in CONCEPTS:
        r = np.array(results[name]["real"])
        f = np.array(results[name]["fake"])
        rm, rs = r.mean(), r.std()
        fm, fs = f.mean(), f.std()
        # Welch's t-test (unequal variance)
        if len(r) >= 2 and len(f) >= 2 and (rs > 0 or fs > 0):
            t_stat, p_value = stats.ttest_ind(r, f, equal_var=False)
        else:
            t_stat, p_value = float("nan"), float("nan")
        # Cohen's d (pooled SD)
        pooled_sd = np.sqrt((rs ** 2 + fs ** 2) / 2)
        cohens_d = (rm - fm) / pooled_sd if pooled_sd > 0 else float("nan")
        verdict = "✓ significant" if p_value < 0.05 else "✗ NOT significant"
        print(f"{name:<25s} {rm:>8.3g}±{rs:<10.3g} {fm:>8.3g}±{fs:<10.3g} t={t_stat:>+.2f} p={p_value:.3g}  d={cohens_d:+.2f}")
        summary_rows.append((name, p_value, cohens_d))
        csv_rows.append(f"{name},{rm:.6g},{rs:.6g},{len(r)},{fm:.6g},{fs:.6g},{len(f)},{t_stat:.6g},{p_value:.6g},{cohens_d:.6g}")

    # Verdict
    print("\n" + "=" * 78)
    print("Verdict")
    print("=" * 78)
    for name, p_value, d in summary_rows:
        if np.isnan(p_value):
            v = "INSUFFICIENT DATA"
        elif p_value < 0.001 and abs(d) >= 0.5:
            v = "✅ STRONG SIGNAL (use in CBNet-AGID)"
        elif p_value < 0.05 and abs(d) >= 0.3:
            v = "🟡 MODERATE SIGNAL (consider keeping with caveats)"
        elif p_value < 0.05:
            v = "🟡 WEAK SIGNAL (statistically detected but small effect; need more samples)"
        else:
            v = "🔴 NO SIGNAL (concept likely won't work — revise design)"
        print(f"  {name:<25s} {v}")
    print()
    print("Effect size interpretation (Cohen's d):")
    print("  |d| < 0.2  trivial    |d| < 0.5  small    |d| < 0.8  medium    |d| ≥ 0.8  large")
    print()

    # Save CSV
    results_dir = Path("E:/LQiu/lab_folder/Machine_learning/AGID_Project/Results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "day5_concept_sanity.csv"
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(csv_rows))
    print(f"[INFO] results saved to {csv_path}")

    # Optional plots
    if args.save_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for ax, name in zip(axes, CONCEPTS):
                r = results[name]["real"]
                f = results[name]["fake"]
                ax.hist(r, alpha=0.6, label="Real", bins=15, color="C0")
                ax.hist(f, alpha=0.6, label="AI", bins=15, color="C3")
                ax.set_title(name)
                ax.legend()
                ax.set_xlabel("Concept value")
                ax.set_ylabel("Count")
            fig.tight_layout()
            plot_path = results_dir / "day5_concept_sanity.png"
            fig.savefig(plot_path, dpi=120)
            print(f"[INFO] plot saved to {plot_path}")
        except ImportError as e:
            print(f"[WARN] matplotlib unavailable, skipping plots: {e}")


if __name__ == "__main__":
    main()
