# Day 0-1 Setup Report — Environment Validation Complete

**Date:** 2026-05-24
**Status:** ✅ Complete (no blockers)

---

## What was set up

### Environment (`agid` conda env at `E:\LQiu\conda_envs\agid`)
- Python 3.11.15
- PyTorch 2.1.2+cu121 (CUDA 12.1)
- torchvision 0.16.2+cu121
- numpy 1.26.4 (downgraded from 2.x for PyTorch 2.1 compat)
- opencv-python 4.10.0.84 + opencv-python-headless 4.10.0.84
- Pillow 12.2.0
- scipy 1.17.1
- scikit-learn 1.8.0
- timm 1.0.27
- einops 0.8.2
- datasets 4.4.1, huggingface-hub 0.36.0, transformers 4.46.0
- Plus transitive deps: typing_extensions, pyyaml, requests, filelock, fsspec, packaging, jinja2, sympy, etc.

### Verified working
- ✅ `torch.cuda.is_available() == True`
- ✅ RTX 4060 Laptop GPU detected, 8.6 GB VRAM
- ✅ CUDA Conv2d forward pass works on GPU
- ✅ torch ↔ numpy interop works
- ✅ torchvision transforms work
- ✅ NPR inference smoke test passes (load weights → forward on sample image → sigmoid output)

### External repos cloned
- ✅ `Code/external/LOTA/` (https://github.com/hongsong-wang/LOTA) — ICCV 2025 baseline; **needs weights from Baidu Pan**
- ✅ `Code/external/NPR-DeepfakeDetection/` (https://github.com/chuangchuangtan/NPR-DeepfakeDetection) — CVPR 2024 baseline; **weights `NPR.pth` and `model_epoch_last_3090.pth` ship with repo**

### Scripts created
- `Code/scripts/test_npr_inference.py` — NPR single-image or directory inference
- `Code/scripts/test_lota_inference.py` — LOTA single-image or directory inference (needs weights)
- `Code/scripts/download_test_samples.py` — HuggingFace streaming download of test images
- `activate_agid.ps1` — environment activation helper

---

## Lessons learned / pitfalls (relevant for paper's "Implementation Details" section)

### Pitfall 1: NumPy 2.x vs PyTorch 2.1.2
PyTorch 2.1.2 was compiled against NumPy 1.x; importing NumPy 2.x triggers a runtime warning ("Failed to initialize NumPy: _ARRAY_API not found") and breaks downstream tensor↔numpy interop in some code paths. **Fix:** pin `numpy<2.0` in this environment (used `numpy==1.26.4`).

### Pitfall 2: Conda env contamination from user-site-packages
On Windows, the `agid` conda env was implicitly seeing packages from `C:\Users\LQiu\AppData\Roaming\Python\Python311\site-packages` (system Python's user-site). This caused (a) Pip to skip installing transitive deps that "looked already installed" and (b) the wrong versions of matplotlib etc. to be picked up. **Fix:** set `PYTHONNOUSERSITE=1` to disable user-site lookup; explicitly install all needed deps in the conda env.

### Pitfall 3: NPR's `NPR.pth` has `module.` prefix
Saved from a `torch.nn.DataParallel` wrapper. Loading into a non-DataParallel model requires stripping the prefix. **Fix:** handled in `test_npr_inference.py:load_npr_model()`.

### Pitfall 4: NPR's `resnet50` is a truncated variant
Only has `layer1` + `layer2` (no layer3/4); FC head is named `fc1`; input convolution is 3×3 (not 7×7); forward includes a unique "Neighborhood Pixel Relationship" residual `NPR = x - interpolate(x, 0.5)`. **Implication:** when we propose CBNet-AGID with NPR backbone, this truncated structure is what we'd extend with our concept bottleneck — keeping the model lightweight.

### Pitfall 5: LOTA weights are on Baidu Pan only
SD-v1.4 weights at https://pan.baidu.com/s/1H0IceEHzpB_ADh5J487bkA?pwd=imjw
SD-v1.5 weights at https://pan.baidu.com/s/1h9qN-tWjZrXT1wQsHhZBpw?pwd=a942
**Implication:** user must download manually (Baidu Pan requires account). No alternative mirror found.

---

## How to use the environment (going forward)

```powershell
# In a fresh PowerShell session in the project root:
cd E:\LQiu\lab_folder\Machine_learning\AGID_Project
. .\activate_agid.ps1

# Verify:
python -c "import torch; print(torch.cuda.is_available())"
```

Or for one-off command:
```powershell
$env:PYTHONNOUSERSITE = "1"
& "E:\LQiu\conda_envs\agid\python.exe" your_script.py
```

---

## Hardware-driven defaults updated (for Stage 1.2 Methodology Blueprint §6.2 — applied later in writing)

| Setting | Original spec | Hardware-adjusted |
|---|---|---|
| Batch size | 64 | **16** |
| Gradient accumulation | none | **4 steps** (effective bs=64) |
| Mixed precision | optional | **required (bfloat16, AMP)** |
| Image size | 256 | **256** (keep) |
| Backbone | dual-stream ResNet-50 | dual-stream **lightweight** (NPR's truncated ResNet on signal stream may save VRAM) |

---

## Next: Day 2-3 plan

1. **Get a real labeled test set** (~100-200 images) — three options remain:
   - HF stream (script ready: `download_test_samples.py`)
   - User downloads from NPR's CNNDetection Baidu Pan link (faster but needs auth)
   - User downloads small subset from any of the awesome-AGID-list mirrors
2. **Run NPR inference on labeled set** → verify ≥ 95% accuracy on at least 1 generator (e.g., ProGAN)
3. **Get LOTA SD-v1.4 weight from Baidu Pan** → run LOTA inference
4. **Time both inferences** → calibrate compute budget for Day 6-7

If Day 2-3 NPR accuracy verifies, the **NPR backbone for our CBNet-AGID is confirmed viable**.

---

## Decision request

For Day 2 test data download, my recommendation is still **HuggingFace streaming** (script ready), but if you have time/willingness, downloading the CNNDetection Baidu Pan subset (Table 1 Test, only ~1-2 GB) would be cleaner because it matches NPR's published numbers exactly.
