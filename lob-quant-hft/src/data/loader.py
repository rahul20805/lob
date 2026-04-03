"""
LOB Data Loader
===============
Supports FI-2010 benchmark dataset and generic tick-data CSVs.

FI-2010 format (Ntakaris et al., 2018):
    Each column is a feature; rows are tick observations.
    Features: 10 price/volume levels per side => 40 raw features.
    Labels: horizon {1,2,3,5,10} mid-price changes => {0,1,2} = {down,stat,up}.

Generic CSV format expected columns (at minimum):
    timestamp, bid_p1..bid_p10, bid_v1..bid_v10,
    ask_p1..ask_p10, ask_v1..ask_v10
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column name helpers
# ---------------------------------------------------------------------------

def _lob_columns(n_levels: int = 10) -> list[str]:
    cols = []
    for i in range(1, n_levels + 1):
        cols += [f"ask_p{i}", f"ask_v{i}", f"bid_p{i}", f"bid_v{i}"]
    return cols


# ---------------------------------------------------------------------------
# FI-2010 loader
# ---------------------------------------------------------------------------

def load_fi2010(
    data_dir: str | Path,
    horizon: int = 10,
    n_levels: int = 10,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the FI-2010 benchmark LOB dataset.

    Parameters
    ----------
    data_dir : path to directory containing the .txt files
    horizon  : prediction horizon — one of {1, 2, 3, 5, 10}
    n_levels : number of price levels to use (max 10)
    normalize: whether to z-score normalise features

    Returns
    -------
    X : float32 array of shape (N, 4*n_levels)
    y : int64  array of shape (N,) with values in {0, 1, 2}
    """
    data_dir = Path(data_dir)
    horizon_map = {1: 0, 2: 1, 3: 2, 5: 3, 10: 4}
    if horizon not in horizon_map:
        raise ValueError(f"horizon must be one of {list(horizon_map)}, got {horizon}")

    files = sorted(data_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")

    X_list, y_list = [], []
    for f in files:
        raw = np.loadtxt(f, delimiter=",")
        # FI-2010: first 40 cols are features, last 5 cols are labels per horizon
        n_feature_cols = 4 * n_levels
        X_list.append(raw[:, :n_feature_cols])
        # label column for chosen horizon (0-indexed)
        label_col = 40 + horizon_map[horizon]
        labels = raw[:, label_col].astype(int)
        # remap {1,2,3} -> {0,1,2}
        labels = labels - 1
        y_list.append(labels)

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.int64)

    if normalize:
        mean = X.mean(axis=0)
        std  = X.std(axis=0) + 1e-8
        X = (X - mean) / std

    logger.info("FI-2010 loaded: X=%s y=%s classes=%s", X.shape, y.shape,
                np.bincount(y))
    return X, y


# ---------------------------------------------------------------------------
# Generic CSV loader
# ---------------------------------------------------------------------------

def load_csv_lob(
    path: str | Path,
    n_levels: int = 10,
    normalize: bool = True,
    threshold: float = 0.0002,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a generic tick-data CSV into LOB arrays.

    Parameters
    ----------
    path      : CSV file path
    n_levels  : number of price levels
    normalize : z-score normalise features
    threshold : fractional change in mid-price for up/down classification

    Returns
    -------
    X : float32 (N, 4*n_levels)
    y : int64   (N,)  {0=down, 1=stationary, 2=up}
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    cols = _lob_columns(n_levels)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[cols].values.astype(np.float32)

    # Compute mid-price
    mid = (df["ask_p1"].values + df["bid_p1"].values) / 2.0
    # Future mid-price (shift by 1)
    mid_next = np.roll(mid, -1)
    mid_next[-1] = mid[-1]
    ret = (mid_next - mid) / (mid + 1e-8)

    y = np.where(ret > threshold, 2,
        np.where(ret < -threshold, 0, 1)).astype(np.int64)

    if normalize:
        mean = X.mean(axis=0)
        std  = X.std(axis=0) + 1e-8
        X = (X - mean) / std

    logger.info("CSV LOB loaded: X=%s y=%s classes=%s", X.shape, y.shape,
                np.bincount(y))
    return X, y


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class LOBDataset(Dataset):
    """
    Sliding-window LOB dataset for sequence models.

    Each sample is a (window_size, n_features) tensor with a scalar label.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        window_size: int = 100,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.window_size = window_size
        self.stride = stride

        # Pre-compute valid indices
        self._indices = list(range(window_size - 1, len(X), stride))

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        end = self._indices[idx] + 1
        start = end - self.window_size
        x = self.X[start:end]           # (window_size, features)
        label = self.y[self._indices[idx]]
        return x, label


# ---------------------------------------------------------------------------
# Train / Val / Test splitter
# ---------------------------------------------------------------------------

def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    window_size: int = 100,
    stride: int = 1,
) -> Tuple[LOBDataset, LOBDataset, LOBDataset]:
    """
    Chronological split into train / val / test LOBDatasets.
    """
    n = len(X)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val,   y_val   = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test,  y_test  = X[n_train + n_val:], y[n_train + n_val:]

    train_ds = LOBDataset(X_train, y_train, window_size, stride)
    val_ds   = LOBDataset(X_val,   y_val,   window_size, stride)
    test_ds  = LOBDataset(X_test,  y_test,  window_size, stride)

    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloaders(
    train_ds: LOBDataset,
    val_ds:   LOBDataset,
    test_ds:  LOBDataset,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    balance_classes: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders.  Training loader optionally uses
    WeightedRandomSampler to handle class imbalance.
    """
    if balance_classes:
        labels = train_ds.y[torch.tensor(train_ds._indices)].numpy()
        class_counts = np.bincount(labels)
        weights = 1.0 / class_counts[labels]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights).float(),
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
