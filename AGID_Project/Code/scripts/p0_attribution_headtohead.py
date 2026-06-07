"""P0-#1 (peer-review response): post-hoc attribution head-to-head.

Reviewer question: does the concept bottleneck's *causal* per-channel ablation reveal
the compression reliance that post-hoc attribution on the no-bottleneck baseline cannot,
or does attribution recover it just as well (in which case the bottleneck is a
convenience, not a unique capability)?

Experiment (CPU-only; does NOT touch the GPU training):
  1. Run BaselinePlain (backbone -> GAP 2560-d -> Linear) over the 7-generator eval set;
     collect pooled features f (2560-d), logit, label, generator.
  2. For each image compute the 6 *dataset-level heuristic* concept statistics (raw image
     functions in concepts.heuristics) -- the same supervision targets, but independent of
     any model. These are the post-hoc attribution targets.
  3. For each concept: fit a linear probe f -> concept on a train split (ridge), report
     held-out R^2 (= is the concept linearly decodable from the baseline's features?).
  4. CAUSAL post-hoc ablation: project the probe direction out of f and re-run the baseline
     classifier:  logit' = w.f' + b = logit - (f . vhat)(w . vhat).
     Report Delta-acc on the held-out split, and cos(vhat, what) -- the alignment that
     governs the effect.
  5. Random-direction control (20 unit vectors) for a null Delta-acc.
  6. Compare baseline post-hoc Delta-acc to the bottleneck's causal jpeg-quant zero-out
     (-49.5 pp, from the existing audit).

Outputs Results/p0_attribution_headtohead.json + a printed summary.

Usage (from Code/):
  $env:PYTHONNOUSERSITE=1; python -m scripts.p0_attribution_headtohead \
      --root ../Data/GenImage \
      --ckpt Logs/cbnet_baseline_nobottleneck_s42/ckpt_epoch20.pth \
      --n_per_class 120
"""
from __future__ import annotations

import argparse, json, os, random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sys
_PKG = Path(__file__).resolve().parent.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from cbnet_agid.data.transforms import get_eval_transform
from cbnet_agid.models.baseline_plain import BaselinePlain
from cbnet_agid.concepts.heuristics import compute_concept_labels, CONCEPT_NAMES

TRAIN_GENS = ["Stable_Diffusion_v1.4", "BigGAN", "ADM", "Midjourney"]
OOD_GENS = ["GLIDE", "Wukong", "VQDM"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(d: Path):
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTS) if d.exists() else []


def build_samples(root: Path, gen: str, n: int, seed: int, shared_real):
    rng = random.Random(seed)
    if gen == "Stable_Diffusion_v1.4":
        ai_pool = list_images(root / gen / "val" / "ai")
        real_pool = list_images(root / gen / "val" / "nature")
    elif gen in ("BigGAN", "ADM", "Midjourney"):
        train_ai = list_images(root / gen / "train" / "ai")
        train_real = list_images(root / gen / "train" / "nature")
        subset = root / gen / "train_25k"
        if subset.exists():
            ua = {p.name for p in list_images(subset / "ai")}
            ur = {p.name for p in list_images(subset / "nature")}
            ai_pool = [p for p in train_ai if p.name not in ua]
            real_pool = [p for p in train_real if p.name not in ur]
        else:
            ai_pool, real_pool = train_ai, train_real
    else:  # OOD
        ai_pool = list_images(root / gen / "val" / "ai")
        real_pool = shared_real
    if not ai_pool or not real_pool:
        print(f"  [WARN] {gen}: empty pool (ai={len(ai_pool)}, real={len(real_pool)}) -- skip")
        return []
    ai_sel = rng.sample(ai_pool, min(n, len(ai_pool)))
    real_sel = rng.sample(real_pool, min(n, len(real_pool)))
    return [(p, 1, gen) for p in ai_sel] + [(p, 0, gen) for p in real_sel]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_per_class", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=100.0, help="ridge regularization")
    ap.add_argument("--out", default="Results/p0_attribution_headtohead.json")
    args = ap.parse_args()

    torch.set_num_threads(4)  # leave cores for the concurrent GPU-training dataloaders
    device = torch.device("cpu")  # GPU reserved for training
    root = Path(args.root)
    rng = np.random.RandomState(args.seed)

    print(f"[1] loading baseline: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = BaselinePlain(pretrained=False, signal_channels=512).to(device).eval()
    model.load_state_dict(state)
    w = model.classifier.weight.detach().cpu().numpy().squeeze().astype(np.float64)  # [2560]
    b = float(model.classifier.bias.detach().cpu().item())
    what = w / (np.linalg.norm(w) + 1e-12)

    print("[2] building eval samples")
    shared_real = list_images(root / "Stable_Diffusion_v1.4" / "val" / "nature")
    samples = []
    for g in TRAIN_GENS + OOD_GENS:
        s = build_samples(root, g, args.n_per_class, args.seed, shared_real)
        print(f"  {g:<24s} n={len(s)}")
        samples += s
    print(f"  total images: {len(samples)}")

    tf = get_eval_transform(256)
    F = np.zeros((len(samples), w.shape[0]), dtype=np.float32)
    logit = np.zeros(len(samples), dtype=np.float64)
    y = np.zeros(len(samples), dtype=np.int64)
    gens = []
    H = np.zeros((len(samples), len(CONCEPT_NAMES)), dtype=np.float32)  # raw heuristics

    print("[3] forward pass (CPU) + heuristic concepts")
    from tqdm import tqdm
    with torch.no_grad():
        for i, (p, lab, g) in enumerate(tqdm(samples)):
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                img = Image.new("RGB", (256, 256))
            x = tf(img).unsqueeze(0).to(device)
            out = model(x)
            F[i] = out["pooled"].cpu().numpy().squeeze()
            logit[i] = float(out["logit"].cpu().item())
            y[i] = lab
            gens.append(g)
            arr = np.asarray(img.resize((256, 256)), dtype=np.uint8)
            hd = compute_concept_labels(arr)
            H[i] = [hd[c] for c in CONCEPT_NAMES]
    gens = np.array(gens)

    # sanity: reconstruct logit from w.f+b
    recon = F.astype(np.float64) @ w + b
    print(f"  logit recon max|err| = {np.max(np.abs(recon - logit)):.4f}")
    base_pred = (logit > 0).astype(int)
    base_acc = float((base_pred == y).mean())
    print(f"  baseline overall acc = {base_acc*100:.2f}%")

    # ----- train/test split (probe fit on train, ablation measured on test) -----
    n = len(samples)
    idx = rng.permutation(n)
    ntr = int(0.7 * n)
    tr, te = idx[:ntr], idx[ntr:]

    # percentile-normalize each heuristic using train stats
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    def norm_col(v, lo, hi):
        return np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)

    Ftr, Fte = F[tr].astype(np.float64), F[te].astype(np.float64)
    te_acc = float(((logit[te] > 0).astype(int) == y[te]).mean())

    results = {"concepts": {}}
    print("\n[4] per-concept post-hoc probe + causal ablation (test split)\n")
    print(f"  {'concept':<18s} {'probeR2':>8s} {'cos(v,w)':>9s} {'dAcc_pp':>9s}")
    for k, cname in enumerate(CONCEPT_NAMES):
        lo, hi = np.percentile(H[tr, k], 2), np.percentile(H[tr, k], 98)
        ytr = norm_col(H[tr, k], lo, hi)
        yte = norm_col(H[te, k], lo, hi)
        probe = Ridge(alpha=args.alpha).fit(Ftr, ytr)
        r2 = float(r2_score(yte, probe.predict(Fte)))
        beta = probe.coef_.astype(np.float64)
        vhat = beta / (np.linalg.norm(beta) + 1e-12)
        cos_vw = float(vhat @ what)
        # ablate vhat from test features -> new logit
        proj = (Fte @ vhat)
        logit_abl = logit[te] - proj * float(w @ vhat)
        acc_abl = float(((logit_abl > 0).astype(int) == y[te]).mean())
        dacc = (acc_abl - te_acc) * 100.0
        results["concepts"][cname] = {
            "probe_r2": r2, "cos_v_w": cos_vw, "delta_acc_pp": dacc,
            "acc_after": acc_abl * 100.0,
        }
        print(f"  {cname:<18s} {r2:>8.3f} {cos_vw:>9.3f} {dacc:>9.2f}")

    # ----- random-direction control -----
    rand_dacc = []
    for _ in range(20):
        u = rng.randn(w.shape[0]); u /= np.linalg.norm(u)
        proj = (Fte @ u)
        la = logit[te] - proj * float(w @ u)
        rand_dacc.append((float(((la > 0).astype(int) == y[te]).mean()) - te_acc) * 100.0)
    results["random_control_delta_acc_pp"] = {
        "mean": float(np.mean(rand_dacc)), "std": float(np.std(rand_dacc)),
        "min": float(np.min(rand_dacc)), "max": float(np.max(rand_dacc)),
    }

    results["meta"] = {
        "n_images": n, "n_test": len(te), "baseline_acc_overall": base_acc * 100.0,
        "baseline_acc_test": te_acc * 100.0, "alpha": args.alpha,
        "bottleneck_jpeg_quant_zeroout_pp": -49.54,  # from existing seed-42 audit
        "ckpt": args.ckpt,
    }

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(outp, "w"), indent=2)
    rc = results["random_control_delta_acc_pp"]
    print(f"\n[5] random-direction control: dAcc = {rc['mean']:.2f} +/- {rc['std']:.2f} pp "
          f"(range {rc['min']:.2f}..{rc['max']:.2f})")
    jq = results["concepts"]["jpeg_quant"]
    print(f"\n[SUMMARY] baseline post-hoc jpeg_quant-direction ablation: "
          f"dAcc = {jq['delta_acc_pp']:.2f} pp (probe R2={jq['probe_r2']:.3f}, "
          f"cos(v,w)={jq['cos_v_w']:.3f})")
    print(f"          bottleneck causal jpeg_quant zero-out: -49.54 pp")
    print(f"[SAVED] {outp}")


if __name__ == "__main__":
    main()
