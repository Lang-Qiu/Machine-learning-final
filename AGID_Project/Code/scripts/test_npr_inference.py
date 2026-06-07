"""
Sanity test for NPR (Tan et al., CVPR 2024) inference.

Goal: verify that NPR's pretrained weight (NPR.pth) loads correctly on our agid env,
runs forward on GPU, and produces sensible output (sigmoid score per image).

Usage:
    python test_npr_inference.py --images_dir <path>
    # or
    python test_npr_inference.py --single <path/to/image.jpg>

Expected behaviour:
    - Real photos:           sigmoid output close to 0.0
    - AI-generated images:   sigmoid output close to 1.0
    - On NPR's reported numbers, ProGAN test: ~99.9% real-acc + fake-acc
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

# Make NPR's networks module importable
NPR_ROOT = Path(__file__).resolve().parents[1] / "external" / "NPR-DeepfakeDetection"
sys.path.insert(0, str(NPR_ROOT))
from networks.resnet import resnet50  # noqa: E402


def load_npr_model(weight_path: str, device: torch.device) -> torch.nn.Module:
    """Load NPR's ResNet50 with the provided checkpoint.

    NPR uses a truncated ResNet-50 (only layer1 + layer2) with a custom NPR residual
    preprocessing in forward(). Weights are saved from DataParallel wrapper, so keys
    have 'module.' prefix that must be stripped.
    """
    model = resnet50(num_classes=1)
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
    # Some checkpoints wrap the dict; unwrap if needed.
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    # Strip "module." prefix from DataParallel-saved weights
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned[k[len("module."):]] = v
        else:
            cleaned[k] = v
    model.load_state_dict(cleaned, strict=True)
    model.to(device)
    model.eval()
    return model


def get_transform() -> transforms.Compose:
    """NPR's standard preprocessing (per their data loader defaults).
    no_resize=False, no_crop=True → resize to 256 then center crop 224.
    Normalisation uses ImageNet statistics.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def infer_single(model: torch.nn.Module,
                 tf: transforms.Compose,
                 image_path: str,
                 device: torch.device) -> float:
    """Run inference on a single image, return P(AI) ∈ [0,1]."""
    img = Image.open(image_path).convert("RGB")
    tensor = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(tensor)
        prob = torch.sigmoid(logit).item()
    return prob


def infer_dir(model: torch.nn.Module,
              tf: transforms.Compose,
              images_dir: str,
              device: torch.device) -> dict:
    """Run inference on all images in a directory.

    If directory has subfolders 0_real and 1_fake (NPR convention), labels are derived
    from folder name; otherwise labels are unknown.
    """
    image_dir = Path(images_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Directory not found: {image_dir}")

    # Try labelled mode (0_real, 1_fake)
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

    if not files:
        raise RuntimeError(f"No images found under {image_dir}")

    print(f"[INFO] found {len(files)} images. labelled={labelled}")

    probs, labels, paths = [], [], []
    t0 = time.time()
    for path, label in files:
        try:
            p = infer_single(model, tf, str(path), device)
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
        "min_prob": float(probs.min()) if len(probs) else float("nan"),
        "max_prob": float(probs.max()) if len(probs) else float("nan"),
    }

    if labelled and any(l is not None for l in labels):
        labels_arr = np.array([l for l in labels if l is not None])
        valid_probs = np.array([p for p, l in zip(probs, labels) if l is not None])
        preds = (valid_probs > 0.5).astype(int)
        accuracy = (preds == labels_arr).mean()
        real_mask = labels_arr == 0
        fake_mask = labels_arr == 1
        result["accuracy"] = float(accuracy)
        result["real_acc"] = float((preds[real_mask] == 0).mean()) if real_mask.any() else float("nan")
        result["fake_acc"] = float((preds[fake_mask] == 1).mean()) if fake_mask.any() else float("nan")
        result["n_real"] = int(real_mask.sum())
        result["n_fake"] = int(fake_mask.sum())

    return result, list(zip(paths, probs, labels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight",
                        default=str(NPR_ROOT / "NPR.pth"),
                        help="Path to NPR weight file.")
    parser.add_argument("--images_dir",
                        default=None,
                        help="Directory containing images. If subdirs 0_real/1_fake exist, labels are read.")
    parser.add_argument("--single",
                        default=None,
                        help="Path to a single image for one-shot inference.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")
    if device.type == "cuda":
        print(f"[INFO] gpu: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] vram: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    if not os.path.exists(args.weight):
        sys.exit(f"[ERROR] weight not found: {args.weight}")
    print(f"[INFO] loading weights: {args.weight}")
    model = load_npr_model(args.weight, device)
    tf = get_transform()

    if args.single:
        prob = infer_single(model, tf, args.single, device)
        verdict = "AI-GENERATED" if prob > 0.5 else "REAL"
        print(f"\n{args.single}")
        print(f"  P(AI) = {prob:.4f}  →  {verdict}")
        return

    if args.images_dir:
        result, per_image = infer_dir(model, tf, args.images_dir, device)
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
