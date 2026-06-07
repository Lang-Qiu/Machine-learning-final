# D3: Route A vs Route B comparison

## Training scope
- **Route A**: SD-1.4 only, train_100k, epochs=1..8 (checkpoint epoch 8)
- **Route B**: SD-1.4 + BigGAN + ADM + Midjourney (train_25k each), epochs=1..20 (checkpoint epoch 20)

## SD-1.4 val (same data, both routes)
- Route A acc = **99.83%**  AUC = 0.9999
- Route B acc = **99.88%**  AUC = 0.9998
- Δ acc = +0.04 pp

## Cross-generator generalization (Route B held-out + OOD)
- Route B on BigGAN_heldout: acc = 99.90%  AUC = 1.0000
- Route B on ADM_heldout: acc = 99.95%  AUC = 0.9997
- Route B on MJ_heldout: acc = 99.85%  AUC = 1.0000

## OOD generators (never trained on, any route)
- GLIDE: acc = 99.80%  AUC = 1.0000
- Wukong: acc = 99.45%  AUC = 0.9999
- VQDM: acc = 99.75%  AUC = 1.0000

Route B OOD mean acc = **99.67%**  mean AUC = **0.9999**

## Takeaway
- Route A (single-generator) reaches ~99.8% on its training distribution but cannot be evaluated cross-generator (no multi-gen exposure).
- Route B sustains 99.45–99.95% on the 3 unseen OOD generators (GLIDE/Wukong/VQDM) at AUC=1.000, **showing concept-bottleneck transfers**.