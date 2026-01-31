from __future__ import annotations
from typing import Dict
import numpy as np

def make_split_indices(n: int, seed: int, split: Dict[str, float]) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(split["train"] * n)
    n_val = int(split["val"] * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]
    return {"train": train_idx, "val": val_idx, "test": test_idx}
