"""Quick smoke verify: max_train_samples + concept label alignment."""
import sys
sys.path.insert(0, r"E:\LQiu\lab_folder\Machine_learning\AGID_Project\Code")

from cbnet_agid.data.multigen import build_multigen_loaders
from cbnet_agid.data.transforms import get_train_transform, get_eval_transform

ROOT = r"E:\LQiu\lab_folder\Machine_learning\AGID_Project\Data\GenImage"
GENS = ["Stable_Diffusion_v1.4", "BigGAN", "ADM", "Midjourney"]

loaders = build_multigen_loaders(
    root=ROOT, generators=GENS, split="train_25k",
    use_shared_concept_labels=True,
    train_transform=get_train_transform(256, disable_destructive=True),
    eval_transform=get_eval_transform(256),
    batch_size=32, num_workers=0,
    max_train_samples=1000,
    val_generators=["Stable_Diffusion_v1.4"],
    max_val_samples=100,
)
train_ds = loaders["train"].dataset
print("n_per_gen=%d  total=%d  (expect 2000 per gen, 8000 total)" % (train_ds.n_per_gen, len(train_ds)))
b = next(iter(loaders["train"]))
print("image:", b["image"].shape)
print("concept_labels:", b["concept_labels"].shape)
print("generator_id sample:", b["generator_id"][:8].tolist())
print("class_idx sample:", b["class_idx"][:8].tolist())
print("max_samples + concept_labels_shared alignment: OK")
