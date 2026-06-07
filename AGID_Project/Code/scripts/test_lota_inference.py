"""
Sanity test for LOTA (Wang et al., ICCV 2025) inference.

Goal: verify LOTA's pretrained weight (needs to be downloaded from Baidu Pan) loads correctly,
runs forward on GPU, and produces sensible output (sigmoid score per image).

Note: LOTA's preprocessing is non-trivial. It extracts the lowest 3 bit-planes per RGB channel
(masked with 0x07), tiles into patches, selects the patch with MAX gradient (heuristic), and
resizes that single patch back to image dimensions. The classifier (ResNet50 with 1-class head)
sees this bit-plane noise representation, NOT the raw image.

Usage:
    python test_lota_inference.py --weight <path-to-LOTA-weights.pth> --images_dir <path>
    python test_lota_inference.py --weight <path-to-LOTA-weights.pth> --single <path/to/image.jpg>
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Make LOTA's modules importable
LOTA_ROOT = Path(__file__).resolve().parents[1] / "external" / "LOTA"
sys.path.insert(0, str(LOTA_ROOT))
from model import model as LOTAModel  # noqa: E402
from bit_patch import bit_patch  # noqa: E402


def load_lota_model(weight_path: str, device: torch.device) -> torch.nn.Module:
    """Load LOTA's ResNet50 with the provided checkpoint."""
    model = LOTAModel(pretrain=True)
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def get_normalize_transform() -> transforms.Compose:
    """LOTA's normalization (assumed ImageNet stats since they use pretrained ResNet50)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def preprocess_lota(image_path: str,
                    img_height: int = 256,
                    patch_size: int = 32,
                    bit_mode: str = "scaling",
                    patch_mode: str = "max") -> torch.Tensor:
    """LOTA-style preprocessing: bit-plane extraction + max-gradient patch."""
    img = Image.open(image_path).convert("RGB").resize((img_height, img_height))
    patch_np = bit_patch(img, img_height, bit_mode, patch_size, patch_mode)
    # patch_np is HWC uint8 image at (img_height, img_height)
    pil = Image.fromarray(patch_np)
    tensor = get_normalize_transform()(pil).unsqueeze(0)
    return tensor


def infer_single(model: torch.nn.Module,
                 image_path: str,
                 device: torch.device,
                 img_height: int = 256,
                 patch_size: int = 32,
                 bit_mode: str = "scaling",
                 patch_mode: str = "max") -> float:
    """Run LOTA inference on a single image.

    NOTE: LOTA's label convention is INVERTED vs NPR/CNNDetection:
      - LOTA training labels:   natural=1, AI=0
      - LOTA sigmoid output:    >0.5 → REAL, <0.5 → AI
    We return P(AI) = 1 - sigmoid(logit) to match the AGID convention used by NPR
    (i.e., P(AI) > 0.5 → AI). This makes cross-method comparison consistent.
    """
    tensor = preprocess_lota(image_path, img_height, patch_size, bit_mode, patch_mode).to(device)
    with torch.no_grad():
        logit = model(tensor)
        p_real = torch.sigmoid(logit).item()
    return 1.0 - p_real  # convert LOTA's P(real) → P(AI)


def infer_dir(model: torch.nn.Module,
              images_dir: str,
              device: torch.device,
              img_height: int = 256,
              patch_size: int = 32) -> dict:
    """Run inference on all images in a directory."""
    image_dir = Path(images_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Directory not found: {image_dir}")

    labelled = (image_dir / "0_real").exists() and (image_dir / "1_fake").exists()
    files = []
    if labelled:
        for label_name, label in [("0_real", 0), ("1_fake", 1)]:
            for p in (image_dir / label_name).rglob("*"):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                    files.append((p, label))
    else:
        for p in image_dir.rglob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                files.append((p, None))

    print(f"[INFO] found {len(files)} images. labelled={labelled}")

    probs, labels, paths = [], [], []
    t0 = time.time()
    for path, label in files:
        try:
            p = infer_single(model, str(path), device, img_height, patch_size)
            probs.append(p)
            labels.append(label)
            paths.append(path)
        except Exception as e:
            print(f"[WARN] failed on {path}: {e}")
    elapsed = time.time() - t0

    probs = np.array(probs)
    result = {
        "n": len(probs),
        "elapsed_sec": elapsed,
        "ms_per_image": 1000 * elapsed / max(len(probs), 1),
        "mean_prob": float(probs.mean()) if len(probs) else float("nan"),
    }

    if labelled and any(l is not None for l in labels):
        labels_arr = np.array([l for l in labels if l is not None])
        valid_probs = np.array([p for p, l in zip(probs, labels) if l is not None])
        preds = (valid_probs > 0.5).astype(int)
        result["accuracy"] = float((preds == labels_arr).mean())
        result["real_acc"] = float((preds[labels_arr == 0] == 0).mean()) if (labels_arr == 0).any() else float("nan")
        result["fake_acc"] = float((preds[labels_arr == 1] == 1).mean()) if (labels_arr == 1).any() else float("nan")
        result["n_real"] = int((labels_arr == 0).sum())
        result["n_fake"] = int((labels_arr == 1).sum())

    return result, list(zip(paths, probs, labels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", required=True,
                        help="Path to LOTA weight file (download from Baidu Pan).")
    parser.add_argument("--images_dir", default=None,
                        help="Directory with images (uses 0_real/1_fake subdir convention).")
    parser.add_argument("--single", default=None, help="Path to single image.")
    parser.add_argument("--img_height", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--bit_mode", default="scaling", choices=["scaling", "thresholding"])
    parser.add_argument("--patch_mode", default="max", choices=["max", "min", "random"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")
    if device.type == "cuda":
        print(f"[INFO] gpu: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(args.weight):
        sys.exit(f"[ERROR] weight not found: {args.weight}")
    print(f"[INFO] loading weights: {args.weight}")
    model = load_lota_model(args.weight, device)

    if args.single:
        prob = infer_single(model, args.single, device,
                            args.img_height, args.patch_size,
                            args.bit_mode, args.patch_mode)
        verdict = "AI-GENERATED" if prob > 0.5 else "REAL"
        print(f"\n{args.single}")
        print(f"  P(AI) = {prob:.4f}  →  {verdict}")
        return

    if args.images_dir:
        result, per_image = infer_dir(model, args.images_dir, device,
                                       args.img_height, args.patch_size)
        print("\n=== Summary ===")
        for k, v in result.items():
            print(f"  {k}: {v}")
        print("\n=== Per-image (first 20) ===")
        for path, prob, label in per_image[:20]:
            label_str = "real" if label == 0 else ("fake" if label == 1 else "?")
            verdict = "AI" if prob > 0.5 else "real"
            print(f"  [gt={label_str:4s}] [P(AI)={prob:.3f}] [pred={verdict:4s}] {path.name}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
