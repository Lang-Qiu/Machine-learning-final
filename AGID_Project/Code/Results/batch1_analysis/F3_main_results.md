# F3: Main results — CBNet-AGID Route A vs Route B

| Generator             | Type                   | Route | n     | Accuracy (%) | AUC    | Real Acc (%) | Fake Acc (%) |
| --------------------- | ---------------------- | ----- | ----- | ------------ | ------ | ------------ | ------------ |
| SD-1.4 (val)          | ID — single-gen train  | A     | 12000 | 99.83        | 0.9999 | 99.70        | 99.97        |
| SD-1.4 (val)          | ID — multi-gen train   | B     | 12000 | 99.88        | 0.9998 | 99.80        | 99.95        |
| BigGAN (held-out)     | ID — multi-gen train   | B     | 2000  | 99.90        | 1.0000 | 99.80        | 100.00       |
| ADM (held-out)        | ID — multi-gen train   | B     | 2000  | 99.95        | 0.9997 | 99.90        | 100.00       |
| Midjourney (held-out) | ID — multi-gen train   | B     | 2000  | 99.85        | 1.0000 | 99.90        | 99.80        |
| GLIDE                 | OOD — unseen generator | B     | 2000  | 99.80        | 1.0000 | 99.60        | 100.00       |
| Wukong                | OOD — unseen generator | B     | 2000  | 99.45        | 0.9999 | 99.60        | 99.30        |
| VQDM                  | OOD — unseen generator | B     | 2000  | 99.75        | 1.0000 | 99.60        | 99.90        |