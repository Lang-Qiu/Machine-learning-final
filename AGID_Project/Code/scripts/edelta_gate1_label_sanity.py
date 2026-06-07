"""E-Delta GATE 1 — label-level sanity check + JPEG q-sweep (NO TRAINING).

Question (HΔ1): at what JPEG quality (if any) does re-encoding collapse the
`jpeg_quant` heuristic's real-vs-fake separation? This distinguishes:
  (1) jpeg_quant is JPEG-container-sensitive but needs strong compression
      (d collapses only at low q) → Q96 debias is too mild; OR
  (2) jpeg_quant is content/frequency-based, NOT container
      (d survives even at q30) → the "container leak" label is wrong, and the
      concept is closer to a genuine generative-trace channel.

Mirrors the real training pipeline:
  - heuristics via cbnet_agid.concepts.heuristics (prior-free 5 of 6 concepts)
  - debias via in-memory JPEG re-encode (same op as confound_sweep._JpegQuality)
  - input = Resize(N+32)+CenterCrop(N), identical to precompute center_crop=N

Reports Cohen's d (fake vs real) for raw + each swept q, pooled across the 4
training generators (and per generator for jpeg_quant).

Usage (from Code/):
    python scripts/edelta_gate1_label_sanity.py \
        --root ../Data/GenImage \
        --out  Results/edelta/gate1_label_sanity.json \
        --n_per_class 150 --q_sweep 96 90 75 50 30
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

_PKG_ROOT = Path(__file__).resolve().parent.parent  # Code/
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from cbnet_agid.concepts.heuristics import compute_concept_labels  # noqa: E402

CONCEPTS = ["jpeg_quant", "freq_radial", "bitplane_lsb", "hf_noise", "texture_geometry"]
GENS = ["Stable_Diffusion_v1.4", "BigGAN", "ADM", "Midjourney"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(d: Path):
    if not d.exists():
        return []
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def jpeg_recompress(img: Image.Image, q: int) -> Image.Image:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def center_crop_arr(img: Image.Image, n: int) -> np.ndarray:
    from torchvision import transforms as T
    img = T.Resize((n + 32, n + 32))(img)
    img = T.CenterCrop(n)(img)
    return np.array(img)


def cohens_d(fake, real) -> float:
    fake = np.asarray(fake, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)
    nf, nr = len(fake), len(real)
    if nf < 2 or nr < 2:
        return float("nan")
    vf, vr = fake.var(ddof=1), real.var(ddof=1)
    sp = np.sqrt(((nf - 1) * vf + (nr - 1) * vr) / (nf + nr - 2))
    if sp == 0:
        return float("nan")
    return float((fake.mean() - real.mean()) / sp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_per_class", type=int, default=150)
    ap.add_argument("--center_crop", type=int, default=256)
    ap.add_argument("--q_sweep", type=int, nargs="+", default=[96, 90, 75, 50, 30])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    rng = random.Random(args.seed)
    conditions = ["raw"] + [f"q{q}" for q in args.q_sweep]
    print(f"[INFO] GATE 1 q-sweep | n_per_class={args.n_per_class} | crop={args.center_crop} "
          f"| conditions={conditions} | seed={args.seed}")

    # store[cond][concept][class] = list of values, pooled across generators
    def empty():
        return {c: {k: {"real": [], "fake": []} for k in CONCEPTS} for c in conditions}

    pooled = empty()
    per_gen_jq = {}  # per-gen jpeg_quant d per condition

    for gen in GENS:
        ai = list_images(root / gen / "train" / "ai")
        na = list_images(root / gen / "train" / "nature")
        if not ai or not na:
            print(f"[WARN] {gen}: ai={len(ai)} nature={len(na)} — SKIP")
            continue
        ai = rng.sample(ai, min(args.n_per_class, len(ai)))
        na = rng.sample(na, min(args.n_per_class, len(na)))
        g = empty()
        for paths, cls in [(ai, "fake"), (na, "real")]:
            for p in paths:
                try:
                    img = Image.open(p).convert("RGB")
                except Exception as e:
                    print(f"[WARN] open {p}: {e}")
                    continue
                variants = [("raw", img)] + [(f"q{q}", jpeg_recompress(img, q)) for q in args.q_sweep]
                for cond, im2 in variants:
                    try:
                        arr = center_crop_arr(im2, args.center_crop)
                        labs = compute_concept_labels(arr, CONCEPTS)
                    except Exception as e:
                        print(f"[WARN] heuristic {p} [{cond}]: {e}")
                        continue
                    for k in CONCEPTS:
                        pooled[cond][k][cls].append(labs[k])
                        g[cond][k][cls].append(labs[k])
        per_gen_jq[gen] = {cond: cohens_d(g[cond]["jpeg_quant"]["fake"],
                                          g[cond]["jpeg_quant"]["real"]) for cond in conditions}
        line = "  ".join(f"{cond}={per_gen_jq[gen][cond]:+.2f}" for cond in conditions)
        print(f"  [{gen:<22s}] jpeg_quant d:  {line}")

    pooled_d = {cond: {k: cohens_d(pooled[cond][k]["fake"], pooled[cond][k]["real"])
                       for k in CONCEPTS} for cond in conditions}

    print("\n=== POOLED Cohen's d (fake - real) across 4 train gens, by condition ===")
    header = f"{'concept':<18s}" + "".join(f"{c:>9s}" for c in conditions)
    print(header)
    for k in CONCEPTS:
        row = f"{k:<18s}" + "".join(f"{pooled_d[c][k]:>+9.3f}" for c in conditions)
        print(row + ("   <-- HΔ1 target" if k == "jpeg_quant" else ""))

    jq = {c: pooled_d[c]["jpeg_quant"] for c in conditions}
    # find lowest q at which |d| < 0.3 (collapse)
    collapse_q = None
    for q in args.q_sweep:
        if abs(jq[f"q{q}"]) < 0.3:
            collapse_q = q  # keep last (lowest) that collapses; loop natural order high->low
    interp = ("CONTENT-BASED (survives all q → not a container detector)"
              if collapse_q is None else
              f"JPEG-SENSITIVE but needs q<= {collapse_q} to collapse")
    print(f"\n[HΔ1] jpeg_quant pooled |d|: raw={abs(jq['raw']):.3f}  "
          + "  ".join(f"q{q}={abs(jq[f'q{q}']):.3f}" for q in args.q_sweep))
    print(f"[INTERPRETATION] {interp}")
    print(f"[Q96 verdict] q96 |d|={abs(jq.get('q96', float('nan'))):.3f} "
          f"=> {'PASS' if abs(jq.get('q96', 9)) < 0.3 else 'FAIL (Q96 debias does NOT neutralize label)'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": vars(args),
        "conditions": conditions,
        "concepts": CONCEPTS,
        "pooled_cohens_d": pooled_d,
        "per_generator_jpeg_quant_d": per_gen_jq,
        "jpeg_quant_d_by_condition": jq,
        "collapse_q_threshold0.3": collapse_q,
        "interpretation": interp,
    }, indent=2))
    print(f"\n[SAVED] {out}")


if __name__ == "__main__":
    main()
