"""Multi-generator interleaved dataset and loader for Route B.

Architecture
------------
InterleavedMultiGenDataset
    Wraps N GenImageDatasets and routes index i to generator (i % n_gens), local item
    (i // n_gens).  All sub-datasets are truncated to min(len) so the dataset is
    perfectly balanced.  Each item carries two extra keys beyond GenImageDataset:
      - 'generator_id': int in [0, n_gens)
      - 'class_idx':    ImageNet class index parsed from filename (for content pairing)

InterleavedBatchSampler
    Yields batches with exactly batch_size // n_gens items from each generator, with
    independent per-generator shuffling each epoch.  Guarantees generator diversity
    inside every batch, which is required for the content-pairing consistency loss.

build_multigen_loaders
    Convenience factory: assembles the interleaved train loader plus per-generator
    val loaders (same pattern as build_genimage_loaders).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .genimage import GENIMAGE_GENERATORS, GenImageDataset


class InterleavedMultiGenDataset(Dataset):
    """Round-robin interleaved view of N balanced GenImageDatasets.

    Index mapping:
        generator_idx = global_idx % n_gens
        local_idx     = global_idx // n_gens

    All sub-datasets are logically truncated to n_per_gen = min(len(ds)) so every
    generator contributes equally to the training epoch.
    """

    def __init__(self, datasets: List[GenImageDataset], generator_names: List[str]):
        assert len(datasets) == len(generator_names), "datasets and names must have equal length"
        assert len(datasets) >= 2, "need at least 2 generators for multi-gen training"
        self.datasets = datasets
        self.generator_names = generator_names
        self.n_gens = len(datasets)
        self.n_per_gen = min(len(ds) for ds in datasets)

    def __len__(self) -> int:
        return self.n_per_gen * self.n_gens

    def __getitem__(self, idx: int) -> dict:
        gen_idx = idx % self.n_gens
        item_idx = idx // self.n_gens
        item = self.datasets[gen_idx][item_idx]
        item["generator_id"] = gen_idx
        # Parse ImageNet class from filename: format {class}_{gen}_{img_id}.{ext}
        path = Path(str(self.datasets[gen_idx].samples[item_idx][0]))
        try:
            item["class_idx"] = int(path.stem.split("_")[0])
        except (ValueError, IndexError):
            item["class_idx"] = -1
        return item


class InterleavedBatchSampler(Sampler):
    """Yields batches of size batch_size with exactly batch_size // n_gens items per generator.

    Within each generator, items are independently shuffled per epoch.  Batch order is
    also shuffled when shuffle=True.  Drops the last incomplete batch.

    Args:
        n_per_gen:   number of items per generator in the dataset (dataset.n_per_gen).
        n_gens:      number of generators.
        batch_size:  must be divisible by n_gens (e.g., 32 = 8+8+8+8).
        shuffle:     shuffle within each generator and shuffle batch order.
    """

    def __init__(self, n_per_gen: int, n_gens: int, batch_size: int, shuffle: bool = True):
        if batch_size % n_gens != 0:
            raise ValueError(
                f"batch_size {batch_size} must be divisible by n_gens {n_gens} "
                f"for equal generator representation per batch."
            )
        self.n_per_gen = n_per_gen
        self.n_gens = n_gens
        self.batch_size = batch_size
        self.per_gen = batch_size // n_gens
        self.shuffle = shuffle
        self.n_batches = n_per_gen // self.per_gen  # drop_last

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator[List[int]]:
        # Build shuffled global-index lists per generator.
        # Global index for (gen_idx, local_idx): local_idx * n_gens + gen_idx
        gen_global: List[List[int]] = []
        for gen_idx in range(self.n_gens):
            local = list(range(self.n_per_gen))
            if self.shuffle:
                random.shuffle(local)
            gen_global.append([loc * self.n_gens + gen_idx for loc in local])

        batch_order = list(range(self.n_batches))
        if self.shuffle:
            random.shuffle(batch_order)

        for b in batch_order:
            batch: List[int] = []
            start = b * self.per_gen
            for gen_idx in range(self.n_gens):
                batch.extend(gen_global[gen_idx][start: start + self.per_gen])
            yield batch


def build_multigen_loaders(
    root: str,
    generators: List[str],
    split: str = "train_25k",
    use_shared_concept_labels: bool = True,
    train_transform: Optional[Callable] = None,
    eval_transform: Optional[Callable] = None,
    batch_size: int = 32,
    num_workers: int = 2,
    val_generators: Optional[List[str]] = None,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
) -> Dict[str, DataLoader]:
    """Build the interleaved multi-generator train loader and per-generator val loaders.

    Train loader:
        Uses InterleavedMultiGenDataset + InterleavedBatchSampler so each batch of
        batch_size contains exactly batch_size // len(generators) items per generator.
        Concept labels loaded from concept_labels_shared.npy when use_shared_concept_labels=True
        (falls back to concept_labels.npy if the shared file is absent).

    Val loaders:
        Standard single-generator DataLoaders for OOD evaluation.

    Returns:
        dict with keys:
            'train':          DataLoader (multi-gen interleaved)
            'val_<gen>':      DataLoader (per-generator val split)
    """
    root_path = Path(root)

    print("[build_multigen_loaders] assembling train datasets:")
    train_datasets: List[GenImageDataset] = []
    for gen in generators:
        clabel_fname = "concept_labels_shared.npy" if use_shared_concept_labels else "concept_labels.npy"
        clp = root_path / gen / split / clabel_fname
        if not clp.exists() and use_shared_concept_labels:
            print(f"  [WARN] concept_labels_shared.npy not found for {gen}, trying concept_labels.npy")
            clp = root_path / gen / split / "concept_labels.npy"
        has_labels = clp.exists()

        ds = GenImageDataset(
            root=root,
            generator=gen,
            split=split,
            transform=train_transform,
            return_path=False,
            return_concept_labels=has_labels,
            concept_label_path=str(clp) if has_labels else None,
            max_samples=max_train_samples,
        )
        label_status = "shared" if "shared" in clp.name else ("per-gen" if has_labels else "NONE")
        print(f"  {gen:<35s}  n={len(ds):>6d}  labels={label_status}")
        train_datasets.append(ds)

    train_ds = InterleavedMultiGenDataset(train_datasets, generators)
    n_batches = train_ds.n_per_gen // (batch_size // train_ds.n_gens)
    print(f"  InterleavedMultiGenDataset: {len(train_ds)} items, "
          f"{train_ds.n_per_gen} per gen, {n_batches} batches/epoch "
          f"(each batch = {batch_size//train_ds.n_gens} × {train_ds.n_gens} gens)")

    batch_sampler = InterleavedBatchSampler(
        n_per_gen=train_ds.n_per_gen,
        n_gens=train_ds.n_gens,
        batch_size=batch_size,
        shuffle=True,
    )
    loaders: Dict[str, DataLoader] = {
        "train": DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    }

    if val_generators is None:
        val_generators = GENIMAGE_GENERATORS
    print("\n[build_multigen_loaders] assembling val datasets:")
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
            print(f"  val_{g:<28s}  n={len(ds)}")
        except FileNotFoundError:
            print(f"  [WARN] {g} val split not found, skipping")

    return loaders
