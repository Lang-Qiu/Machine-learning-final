"""Image transforms for training and evaluation.

Following CNNDetection / NPR / LOTA conventions: ImageNet normalization, optional
augmentations during training (flip, JPEG, blur — NOT color jitter, which would
interfere with the Color-Manifold-Deviation concept).
"""
from __future__ import annotations

from io import BytesIO

from PIL import Image
from torchvision import transforms


class RandomJPEGCompression:
    """Randomly re-encode image as JPEG with quality in [q_min, q_max]."""

    def __init__(self, q_min: int = 70, q_max: int = 100, p: float = 0.5):
        self.q_min = q_min
        self.q_max = q_max
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        import random
        if random.random() > self.p:
            return img
        q = random.randint(self.q_min, self.q_max)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomGaussianBlur:
    """Random Gaussian blur with sigma in [s_min, s_max]."""

    def __init__(self, s_min: float = 0.0, s_max: float = 2.0, p: float = 0.5):
        self.s_min = s_min
        self.s_max = s_max
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        import random
        if random.random() > self.p:
            return img
        from PIL import ImageFilter
        s = random.uniform(self.s_min, self.s_max)
        if s <= 0.01:
            return img
        return img.filter(ImageFilter.GaussianBlur(radius=s))


def get_train_transform(image_size: int = 256,
                        disable_destructive: bool = False) -> transforms.Compose:
    """Training transform: random crop + flip + (optionally) JPEG + blur + normalize.

    Note: NO color jitter — would interfere with Color-Manifold-Deviation concept.

    Args:
        image_size: Side length of the final crop.
        disable_destructive: If True, omit RandomJPEGCompression and RandomGaussianBlur.
            These two augmentations destroy the very signals that the heuristic concept
            labels (bitplane_lsb, jpeg_quant, hf_noise) are computed from, creating a
            ~30% label-noise rate on those concepts when precompute is run on the raw
            image. Set True when training with concept-MSE supervision; False to
            reproduce the Route A debug/main runs (default).
    """
    ops: list = [
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
    ]
    if not disable_destructive:
        ops.append(RandomJPEGCompression(q_min=70, q_max=100, p=0.3))
        ops.append(RandomGaussianBlur(s_min=0.0, s_max=2.0, p=0.3))
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]))
    return transforms.Compose(ops)


def get_eval_transform(image_size: int = 256) -> transforms.Compose:
    """Evaluation transform: center crop + normalize (deterministic)."""
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
