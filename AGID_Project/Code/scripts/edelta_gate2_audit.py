"""E-Delta GATE 2 — smoke model audit (A2 weights + A3 ablation).

Reads the smoke model checkpoint and runs the same audit battery as Batch 1/2:
  - A2: classifier weights w_k + bias
  - A3: per-concept zero-out ablation (which concept carries reliance now?)
  - Per-concept Cohen's d (real vs fake separation in model's concept activations)

Usage (from Code/):
    python scripts/edelta_gate2_audit.py \
        --ckpt ../Logs/cbnet_debiased_smoke/ckpt_epoch6.pth \
        --root  ../Data/GenImage_debiased_smoke \
        --out   Results/edelta/gate2_smoke_audit.json \
        --n_per_class 500
"""
from __future__ import annotations

import argparse, json, random, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from cbnet_agid.data.transforms import get_eval_transform
from cbnet_agid.models import CBNetAGID

CONCEPT_NAMES = ["bitplane_lsb", "freq_radial", "color_manifold", "hf_noise", "jpeg_quant", "texture_geometry"]
GENS = ["Stable_Diffusion_v1.4", "BigGAN", "ADM", "Midjourney"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(d: Path):
    if not d.exists(): return []
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def cohens_d(fake, real):
    fake, real = np.asarray(fake, float), np.asarray(real, float)
    nf, nr = len(fake), len(real)
    if nf < 2 or nr < 2: return float("nan")
    vf, vr = fake.var(ddof=1), real.var(ddof=1)
    sp = np.sqrt(((nf-1)*vf + (nr-1)*vr) / (nf+nr-2))
    return float((fake.mean()-real.mean())/sp) if sp > 0 else float("nan")


def zero_out(model, k: int):
    """Return a modified model state where concept k is zeroed at the bottleneck."""
    state = model.state_dict()
    # The classifier weight: shape [1, 6] — zero the k-th column
    w_key = None
    for key in state:
        if "classifier" in key and "weight" in key:
            w_key = key; break
    if w_key is None:
        raise RuntimeError("classifier weight not found in state_dict")
    new_state = {k2: v.clone() for k2, v in state.items()}
    new_state[w_key] = new_state[w_key].clone()
    new_state[w_key][:, k] = 0.0
    return new_state


def eval_model_on_gen(model_or_state, samples, transform, device, name, k_zero=None):
    """Eval on a list of (path, label). If k_zero is set, zero that concept."""
    model2 = CBNetAGID(n_concepts=6, pretrained=False, signal_channels=512).to(device)
    if isinstance(model_or_state, dict):
        model2.load_state_dict(model_or_state)
    else:
        model2.load_state_dict(model_or_state.state_dict())
    model2.eval()

    class _DS(Dataset):
        def __init__(self, s, tf): self.s = s; self.tf = tf
        def __len__(self): return len(self.s)
        def __getitem__(self, i):
            p, l = self.s[i]
            try:
                im = Image.open(p).convert("RGB")
            except Exception:
                im = Image.new("RGB", (256, 256))
            return self.tf(im), torch.tensor(l, dtype=torch.long)

    ds = _DS(samples, transform)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    n = len(samples)
    probs    = np.zeros(n, dtype=np.float32)
    labels   = np.zeros(n, dtype=np.int64)
    concepts = np.zeros((n, 6), dtype=np.float32)
    offset = 0
    with torch.no_grad():
        for x, y in tqdm(dl, desc=f"  {name}", leave=False):
            x = x.to(device)
            out = model2(x)
            bs = len(y)
            probs[offset:offset+bs]    = out["prob"].cpu().numpy().flatten()
            concepts[offset:offset+bs] = out["concepts"].cpu().numpy()
            labels[offset:offset+bs]   = y.numpy()
            offset += bs

    from sklearn.metrics import roc_auc_score
    preds = (probs > 0.5).astype(int)
    acc = float((preds == labels).mean())
    try: auc = float(roc_auc_score(labels, probs))
    except ValueError: auc = float("nan")
    rm, fm = labels == 0, labels == 1
    real_acc = float((preds[rm] == 0).mean()) if rm.any() else float("nan")
    fake_acc = float((preds[fm] == 1).mean()) if fm.any() else float("nan")
    d_per_c = {name: cohens_d(concepts[fm, k], concepts[rm, k])
               for k, name in enumerate(CONCEPT_NAMES)}
    return {"acc": acc, "auc": auc, "real_acc": real_acc, "fake_acc": fake_acc,
            "cohens_d": d_per_c, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_per_class", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_eval_transform(256)

    print(f"[INFO] GATE 2 audit | ckpt={args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = CBNetAGID(n_concepts=6, pretrained=False, signal_channels=512).to(device)
    model.load_state_dict(ckpt["model"])

    # ---- A2: classifier weights ----
    w = ckpt["model"]["classifier.weight"].cpu().numpy().flatten()
    b = float(ckpt["model"]["classifier.bias"].cpu().numpy())
    print("\n[A2 — classifier weights]")
    ranked = sorted(zip(CONCEPT_NAMES, w), key=lambda x: -abs(x[1]))
    for name, val in ranked:
        print(f"  w[{name:<16s}] = {val:+.3f}")
    print(f"  bias = {b:+.3f}")
    jq_rank = next(i for i, (n, _) in enumerate(ranked) if n == "jpeg_quant")
    print(f"  jpeg_quant |w| rank: #{jq_rank+1}/6")

    # ---- Sample test images ----
    rng = random.Random(args.seed)
    samples = {}
    for gen in GENS:
        ai = list_images(root / gen / "train" / "ai")
        na = list_images(root / gen / "train" / "nature")
        ai_s = rng.sample(ai, min(args.n_per_class, len(ai)))
        na_s = rng.sample(na, min(args.n_per_class, len(na)))
        samples[gen] = [(p, 1) for p in ai_s] + [(p, 0) for p in na_s]

    # ---- Eval baseline (no ablation) ----
    print("\n=== BASELINE (no ablation) ===")
    baseline = {}
    all_accs = []
    for gen in GENS:
        r = eval_model_on_gen(model, samples[gen], transform, device, gen)
        baseline[gen] = r
        all_accs.append(r["acc"])
        print(f"  {gen:<22s} acc={r['acc']*100:5.2f}%  jq|d|={abs(r['cohens_d']['jpeg_quant']):.3f}  "
              f"fr|d|={abs(r['cohens_d']['freq_radial']):.3f}")
    print(f"  MEAN acc = {np.mean(all_accs)*100:.2f}%")

    # ---- Concept Cohen's d (pooled) ----
    print("\n[Concept separation — pooled Cohen's d (fake vs real)]")
    for k, name in enumerate(CONCEPT_NAMES):
        ds = [baseline[g]["cohens_d"][name] for g in GENS]
        print(f"  {name:<16s} per-gen |d|: {' '.join(f'{abs(d):.3f}' for d in ds)}")

    # ---- A3: Per-concept zero-out ablation ----
    print("\n[A3 — per-concept zero-out ablation]")
    ablation = {}
    for k, name in enumerate(CONCEPT_NAMES):
        z_state = zero_out(model, k)
        row = {}
        for gen in GENS:
            r = eval_model_on_gen(z_state, samples[gen], transform, device, f"{gen}_z{name}")
            row[gen] = r
        accs = [row[g]["acc"] for g in GENS]
        mean_acc = float(np.mean(accs))
        delta = mean_acc - np.mean(all_accs)
        ablation[name] = {"mean_acc": mean_acc, "delta": delta, "per_gen": row}
        print(f"  zero({name:<16s}) mean_acc={mean_acc*100:5.2f}%  Δ={delta*100:+.2f}pp")

    # ---- Verdict ----
    jq_zero_delta = ablation["jpeg_quant"]["delta"] * 100
    jq_w = float(w[CONCEPT_NAMES.index("jpeg_quant")])
    fr_w = float(w[CONCEPT_NAMES.index("freq_radial")])
    top_name, top_w = ranked[0]
    print(f"\n=== GATE 2 SMOKE VERDICT ===")
    print(f"  jpeg_quant |w| rank: #{jq_rank+1}/6  w={jq_w:+.3f}")
    print(f"  top concept: {top_name}  w={top_w:+.3f}")
    print(f"  zero(jpeg_quant) Δacc = {jq_zero_delta:+.1f}pp")
    print(f"  (Route B raw: w[jq]=-13.53 #1, zero-out Δacc = -49.5pp)")
    migrated = jq_rank >= 2  # jpeg_quant is no longer top-|w|
    reliance_drop = abs(jq_zero_delta) < 15  # ablation impact dropped from 50pp to <15pp
    print(f"  MIGRATION: {'YES (jpeg_quant lost top rank + ablation impact << 50pp)' if migrated and reliance_drop else 'CHECK'}")
    print(f"  ===> SMOKE: {'POTENTIAL RELIANCE MIGRATION — present to user' if migrated or reliance_drop else 'jpeg_quant still dominant — discuss'}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "ckpt": args.ckpt,
        "classifier_weight": {n: float(w[k]) for k, n in enumerate(CONCEPT_NAMES)},
        "classifier_bias": b,
        "jpeg_quant_rank": jq_rank + 1,
        "top_concept": top_name,
        "baseline": {g: {kk: vv for kk, vv in r.items() if kk != "cohens_d"} for g, r in baseline.items()},
        "cohens_d": {g: r["cohens_d"] for g, r in baseline.items()},
        "ablation": {n: {"mean_acc": v["mean_acc"], "delta": v["delta"]} for n, v in ablation.items()},
        "migration_detected": migrated and reliance_drop,
        "jpeg_quant_zero_delta_pp": round(jq_zero_delta, 1),
    }, indent=2))
    print(f"\n[SAVED] {out}")


if __name__ == "__main__":
    main()
