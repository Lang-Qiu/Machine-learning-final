"""E-Delta GATE 1b — what does the MODEL's jpeg_quant concept respond to? (read-only)

Reads the existing full_inference_dump.npz (model concept activations on RAW data)
and reports, per generator and pooled:
  - the MODEL's jpeg_quant concept Cohen's d (fake vs real)
  - all 6 classifier weights + bias
so we can compare the model's learned concept against the GATE-1 heuristic-label
result (which was content-based, not container).

No training, no GPU — pure numpy on the saved dump.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

CONCEPTS = ["bitplane_lsb", "freq_radial", "color_manifold", "hf_noise", "jpeg_quant", "texture_geometry"]


def cohens_d(fake, real):
    fake = np.asarray(fake, float); real = np.asarray(real, float)
    nf, nr = len(fake), len(real)
    if nf < 2 or nr < 2: return float("nan")
    vf, vr = fake.var(ddof=1), real.var(ddof=1)
    sp = np.sqrt(((nf-1)*vf + (nr-1)*vr) / (nf+nr-2))
    return float((fake.mean()-real.mean())/sp) if sp > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = np.load(args.dump, allow_pickle=True)
    print("[keys]", list(d.keys()))
    C = d["concepts"]; lab = d["labels"]; gens = d["generators"]
    w = d["classifier_weight"].reshape(-1) if "classifier_weight" in d else None
    b = float(d["classifier_bias"]) if "classifier_bias" in d else None
    jq = CONCEPTS.index("jpeg_quant")

    print("\n[classifier weights]")
    if w is not None:
        for k, name in enumerate(CONCEPTS):
            print(f"  w[{name:<16s}] = {w[k]:+.3f}")
        print(f"  bias = {b:+.3f}" if b is not None else "")

    print("\n[MODEL jpeg_quant concept Cohen's d (fake vs real), by generator]")
    rows = {}
    for g in sorted(set(gens.tolist())):
        m = gens == g
        cg = C[m, jq]; lg = lab[m]
        dd = cohens_d(cg[lg == 1], cg[lg == 0])
        rows[g] = dd
        print(f"  {g:<24s} d={dd:+.3f}  (mean_real={cg[lg==0].mean():.3f} mean_fake={cg[lg==1].mean():.3f})")
    pooled = cohens_d(C[lab == 1, jq], C[lab == 0, jq])
    print(f"  {'POOLED':<24s} d={pooled:+.3f}")

    # also pooled d for every concept for context
    print("\n[MODEL all-concept pooled Cohen's d]")
    allc = {}
    for k, name in enumerate(CONCEPTS):
        dd = cohens_d(C[lab == 1, k], C[lab == 0, k])
        allc[name] = dd
        print(f"  {name:<16s} d={dd:+.3f}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "classifier_weight": {n: float(w[k]) for k, n in enumerate(CONCEPTS)} if w is not None else None,
        "classifier_bias": b,
        "model_jpeg_quant_d_by_gen": rows,
        "model_jpeg_quant_d_pooled": pooled,
        "model_all_concept_d_pooled": allc,
    }, indent=2))
    print(f"\n[SAVED] {out}")


if __name__ == "__main__":
    main()
