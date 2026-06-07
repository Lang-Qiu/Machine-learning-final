"""GenImage dataset loader.

Expected folder layout (per GenImage README):
    <root>/Stable_Diffusion_v1.4/
        ├── train/
        │   ├── ai/         # AI-generated images
        │   └── nature/     # corresponding ImageNet real images
        └── val/
            ├── ai/
            └── nature/

Other generators follow the same pattern (BigGAN, Midjourney, Wukong, Stable_Diffusion_v1.5,
ADM, GLIDE, VQDM). For OOD / cross-generator evaluation, the val splits of all 8 generators
are loaded as separate datasets.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# All 8 GenImage generator subset names (folder names)
GENIMAGE_GENERATORS: List[str] = [
    "BigGAN",
    "Midjourney",
    "Wukong",
    "Stable_Diffusion_v1.4",
    "Stable_Diffusion_v1.5",
    "ADM",
    "GLIDE",
    "VQDM",
]

# Heuristic-label cache structure: per-image dict mapping concept name → raw value
# Stored alongside images as <imagepath>.heuristics.npy for fast loading


class GenImageDataset(Dataset):
    """A single GenImage subset (e.g., Stable_Diffusion_v1.4 train OR val).

    Args:
        root:           path to the GenImage root.
        generator:      one of GENIMAGE_GENERATORS (e.g., 'Stable_Diffusion_v1.4').
        split:          'train' or 'val'.
        transform:      torchvision transform applied to PIL images.
        return_path:    if True, also return image path (useful for debugging).
        return_concept_labels: if True, also return pre-computed heuristic concept labels [K].
        concept_label_path: path to .npy file with [N, K] heuristic labels (or None to skip).
        max_samples:    if not None, randomly subsample to this many images per class.
    """

    def __init__(self,
                 root: str,
                 generator: str = "Stable_Diffusion_v1.4",
                 split: str = "train",
                 transform: Optional[Callable] = None,
                 return_path: bool = False,
                 return_concept_labels: bool = False,
                 concept_label_path: Optional[str] = None,
                 max_samples: Optional[int] = None):
        self.root = Path(root)
        self.generator = generator
        self.split = split
        self.transform = transform
        self.return_path = return_path
        self.return_concept_labels = return_concept_labels

        base = self.root / generator / split
        if not base.exists():
            raise FileNotFoundError(f"GenImage path not found: {base}")

        ai_dir = base / "ai"
        nature_dir = base / "nature"
        all_ai = self._list_images(ai_dir) if ai_dir.exists() else []
        all_real = self._list_images(nature_dir) if nature_dir.exists() else []

        if max_samples is not None:
            random.seed(42)  # reproducible subsample
            # Track sorted indices so concept labels can be sliced in the same order.
            ai_idx = sorted(random.sample(range(len(all_ai)), min(max_samples, len(all_ai))))
            real_idx = sorted(random.sample(range(len(all_real)), min(max_samples, len(all_real))))
            ai_paths = [all_ai[i] for i in ai_idx]
            real_paths = [all_real[i] for i in real_idx]
        else:
            ai_paths = all_ai
            real_paths = all_real
            ai_idx = list(range(len(all_ai)))
            real_idx = list(range(len(all_real)))

        # AI label = 1, real label = 0 (standard NPR/CNNDetection convention)
        self.samples: List[Tuple[Path, int]] = (
            [(p, 1) for p in ai_paths] + [(p, 0) for p in real_paths]
        )
        if not self.samples:
            raise RuntimeError(f"No images found under {base}")

        # Concept labels (optional) — index-aligned even when max_samples is active.
        # concept_labels.npy layout: [all_ai | all_real] in sorted order.
        self.concept_labels: Optional[np.ndarray] = None
        if return_concept_labels and concept_label_path and os.path.exists(concept_label_path):
            full = np.load(concept_label_path)
            n_ai_full = len(all_ai)
            selected = np.array(ai_idx + [n_ai_full + j for j in real_idx])
            if len(selected) != len(full):
                # subsampling: slice the concept labels to match
                self.concept_labels = full[selected]
            else:
                self.concept_labels = full
            assert len(self.concept_labels) == len(self.samples), (
                f"Concept labels size {len(self.concept_labels)} != samples {len(self.samples)}"
            )

    @staticmethod
    def _list_images(directory: Path) -> List[Path]:
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        paths = []
        for p in directory.rglob("*"):
            if p.suffix.lower() in exts:
                paths.append(p)
        return sorted(paths)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Robust to corrupted images: fall back to previous index
            img = Image.new("RGB", (256, 256), (0, 0, 0))
        if self.transform is not None:
            img = self.transform(img)

        out = {"image": img, "label": torch.tensor(label, dtype=torch.long)}
        if self.return_path:
            out["path"] = str(path)
        if self.return_concept_labels and self.concept_labels is not None:
            out["concept_labels"] = torch.tensor(self.concept_labels[idx], dtype=torch.float32)
        return out


def build_genimage_loaders(root: str,
                            train_generator: str = "Stable_Diffusion_v1.4",
                            val_generators: Optional[List[str]] = None,
                            train_transform: Optional[Callable] = None,
                            eval_transform: Optional[Callable] = None,
                            batch_size: int = 32,
                            num_workers: int = 4,
                            max_train_samples: Optional[int] = None,
                            max_val_samples: Optional[int] = None,
                            train_split: str = "train") -> dict:
    """Convenience: build train loader + per-generator val loaders.

    Returns dict with keys:
        'train': DataLoader  (train_generator, train split)
        'val_<generator>': DataLoader  (each generator's val split)
    """
    loaders: dict = {}
    concept_label_path = (Path(root) / train_generator / train_split / "concept_labels.npy")
    if not concept_label_path.exists():
        concept_label_path = None
    train_ds = GenImageDataset(
        root=root, generator=train_generator, split=train_split,
        transform=train_transform, max_samples=max_train_samples,
        return_concept_labels=concept_label_path is not None,
        concept_label_path=str(concept_label_path) if concept_label_path else None,
    )
    loaders["train"] = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )

    if val_generators is None:
        val_generators = GENIMAGE_GENERATORS
    for g in val_generators:
        try:
            ds = GenImageDataset(
                root=root, generator=g, split="val",
                transform=eval_transform, max_samples=max_val_samples,
            )
            loaders[f"val_{g}"] = DataLoader(
                ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True,
            )
        except FileNotFoundError:
            print(f"[WARN] generator {g} val split not found, skipping")
            continue

    return loaders
