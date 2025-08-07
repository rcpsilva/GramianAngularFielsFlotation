"""
data_splits.py – build chronological train/val/test loaders
Usage:
    from data_splits import make_loaders

    loaders = make_loaders(
        path="data_gaf.npz",        # or "data_raw.npz"
        dataset_kind="gaf",         # or "raw"
        batch_size=32,
        split=(0.6, 0.2, 0.2),      # train / val / test
        num_workers=4,
        shuffle_train=True,
    )
    train_loader, val_loader, test_loader = loaders
"""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

# --- Existing GAF dataset wrapper (reuse) ------------------
from gaf import GAFForecastDataset                        # already defined

# --- Raw-series dataset wrapper ----------------------------
class RawWindowDataset(Dataset):
    """1-D sliding-window dataset: X (N, 1, W), y (N,)"""
    def __init__(self, X, y, starts):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.starts = starts

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return {"x": self.X[idx], "y": self.y[idx], "start": self.starts[idx]}

# -----------------------------------------------------------
def _blocked_indices(n, ratios):
    """Return three monotonically increasing index lists."""
    n_train = int(ratios[0] * n)
    n_val   = int(ratios[1] * n)
    train_idx = list(range(            0,             n_train))
    val_idx   = list(range(  n_train,  n_train+n_val ))
    test_idx  = list(range( n_train+n_val, n))
    return train_idx, val_idx, test_idx

def make_loaders(path: str,
                 dataset_kind: str = "gaf",
                 batch_size: int = 32,
                 split=(0.6, 0.2, 0.2),
                 num_workers: int = 0,
                 shuffle_train: bool = True):
    """Return (train_loader, val_loader, test_loader)."""
    data = np.load(Path(path))
    X, y, starts = data["X"], data["y"], data["starts"]

    if dataset_kind == "gaf":
        ds = GAFForecastDataset(X, y, starts)
    elif dataset_kind == "raw":
        ds = RawWindowDataset(X, y, starts)
    else:
        raise ValueError("dataset_kind must be 'gaf' or 'raw'")

    train_idx, val_idx, test_idx = _blocked_indices(len(ds), split)

    loaders = []
    for idxs, is_train in zip((train_idx, val_idx, test_idx),
                              (True, False, False)):
        loader = DataLoader(
            Subset(ds, idxs),
            batch_size=batch_size,
            shuffle=shuffle_train and is_train,   # shuffle **inside** train only
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        loaders.append(loader)
    return tuple(loaders)
