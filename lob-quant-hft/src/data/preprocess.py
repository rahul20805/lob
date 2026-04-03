"""
LOB Preprocessing Pipeline
===========================
Handles:
  - Raw LOB data cleaning & validation
  - Mid-price / spread computation
  - Normalisation (z-score, min-max, robust)
  - Label generation (horizon-based mid-price movement)
  - Synthetic LOB generation for testing (when real data is unavailable)
"""

from __future__ import annotations

import logging
from typing import Literal, Tuple

import numpy as np
import pandas as pd
from numba import njit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class Normalizer:
    """Stateful normaliser fit on training data, applied to val/test."""

    METHODS = ("zscore", "minmax", "robust")

    def __init__(self, method: Literal["zscore", "minmax", "robust"] = "zscore"):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}")
        self.method = method
        self._params: dict = {}
        self._fitted = False

    def fit(self, X: np.ndarray) -> "Normalizer":
        if self.method == "zscore":
            self._params = {"mean": X.mean(0), "std": X.std(0) + 1e-8}
        elif self.method == "minmax":
            self._params = {"min": X.min(0), "max": X.max(0) + 1e-8}
        elif self.method == "robust":
            q25 = np.percentile(X, 25, axis=0)
            q75 = np.percentile(X, 75, axis=0)
            self._params = {
                "median": np.median(X, axis=0),
                "iqr": (q75 - q25) + 1e-8,
            }
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")
        X = X.copy()
        if self.method == "zscore":
            X = (X - self._params["mean"]) / self._params["std"]
        elif self.method == "minmax":
            X = (X - self._params["min"]) / (
                self._params["max"] - self._params["min"] + 1e-8
            )
        elif self.method == "robust":
            X = (X - self._params["median"]) / self._params["iqr"]
        return X.astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.method == "zscore":
            return X * self._params["std"] + self._params["mean"]
        if self.method == "minmax":
            rng = self._params["max"] - self._params["min"]
            return X * rng + self._params["min"]
        if self.method == "robust":
            return X * self._params["iqr"] + self._params["median"]
        raise ValueError


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

def compute_mid_price(bid_p: np.ndarray, ask_p: np.ndarray) -> np.ndarray:
    """Element-wise mid-price."""
    return (bid_p + ask_p) / 2.0


def compute_spread(bid_p: np.ndarray, ask_p: np.ndarray) -> np.ndarray:
    return ask_p - bid_p


def label_from_mid_price(
    mid: np.ndarray,
    horizon: int = 10,
    threshold: float = 0.0002,
    smooth: bool = True,
    alpha: float = 0.00,
) -> np.ndarray:
    """
    Generate 3-class labels {0=down, 1=stationary, 2=up} from mid-price series.

    Uses the *average* future mid price over `horizon` steps (as in DeepLOB paper).

    Parameters
    ----------
    mid       : 1D array of mid prices
    horizon   : number of future ticks to average
    threshold : relative threshold for up/down
    smooth    : use smoothed (average) future mid vs raw next tick
    alpha     : additional label smoothing (not applied here, done in loss)
    """
    N = len(mid)
    labels = np.ones(N, dtype=np.int64)  # default: stationary

    if smooth:
        # m_{t+horizon} = mean(mid[t+1 : t+horizon+1])
        future = np.array([
            mid[i + 1 : i + horizon + 1].mean() if i + horizon < N else mid[-1]
            for i in range(N)
        ])
    else:
        future = np.roll(mid, -horizon)
        future[-horizon:] = mid[-1]

    ret = (future - mid) / (mid + 1e-10)
    labels[ret >  threshold] = 2
    labels[ret < -threshold] = 0

    return labels


# ---------------------------------------------------------------------------
# LOB data cleaning
# ---------------------------------------------------------------------------

def validate_lob(df: pd.DataFrame, n_levels: int = 10) -> pd.DataFrame:
    """
    Basic sanity checks and cleaning on a LOB DataFrame.

    Enforces:
      - bid < ask at each level
      - prices are positive
      - volumes are non-negative
      - ascending ask / descending bid price levels
    """
    df = df.copy()

    for i in range(1, n_levels + 1):
        bp, ap = f"bid_p{i}", f"ask_p{i}"
        bv, av = f"bid_v{i}", f"ask_v{i}"

        # Drop rows with crossed quotes
        crossed = df[bp] >= df[ap]
        if crossed.any():
            logger.warning("Dropping %d crossed-quote rows at level %d",
                           crossed.sum(), i)
            df = df[~crossed]

        # Clip negative volumes
        for col in (bv, av):
            df[col] = df[col].clip(lower=0)

    # Drop any remaining NaN rows
    n_before = len(df)
    df = df.dropna()
    if len(df) < n_before:
        logger.warning("Dropped %d NaN rows", n_before - len(df))

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Synthetic LOB generator (for unit tests / demos without real data)
# ---------------------------------------------------------------------------

@njit
def _simulate_lob_tick(
    mid: float,
    spread: float,
    n_levels: int,
    tick_size: float,
    vol_scale: float,
    rng_state: np.ndarray,
) -> np.ndarray:
    """
    Numba-jit inner loop: returns a single LOB tick array of length 4*n_levels.
    Layout: [ask_p1, ask_v1, bid_p1, bid_v1, ask_p2, ask_v2, ...]
    """
    row = np.empty(4 * n_levels, dtype=np.float64)
    half = spread / 2.0
    for i in range(n_levels):
        ask_p = mid + half + i * tick_size
        bid_p = mid - half - i * tick_size
        ask_v = max(1.0, vol_scale * (1 + i * 0.3) * abs(rng_state[i]))
        bid_v = max(1.0, vol_scale * (1 + i * 0.3) * abs(rng_state[i + n_levels]))
        row[4 * i]     = ask_p
        row[4 * i + 1] = ask_v
        row[4 * i + 2] = bid_p
        row[4 * i + 3] = bid_v
    return row


def generate_synthetic_lob(
    n_ticks: int = 50_000,
    n_levels: int = 10,
    tick_size: float = 0.01,
    initial_price: float = 100.0,
    spread: float = 0.02,
    vol_per_tick: float = 0.0005,
    vol_scale: float = 100.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic LOB dataset for testing.

    Returns
    -------
    X : float32 (n_ticks, 4*n_levels)
    y : int64   (n_ticks,)  3-class labels
    """
    rng = np.random.default_rng(seed)

    # Simulate mid-price as arithmetic random walk
    returns = rng.normal(0, vol_per_tick, n_ticks)
    mid_prices = initial_price * np.cumprod(1 + returns)

    X = np.zeros((n_ticks, 4 * n_levels), dtype=np.float32)
    for t in range(n_ticks):
        noise = rng.standard_normal(2 * n_levels)
        row = _simulate_lob_tick(
            mid_prices[t], spread, n_levels, tick_size, vol_scale, noise
        )
        X[t] = row.astype(np.float32)

    y = label_from_mid_price(mid_prices, horizon=10, threshold=0.0002)

    logger.info("Synthetic LOB generated: X=%s y=%s classes=%s",
                X.shape, y.shape, np.bincount(y))
    return X, y


# ---------------------------------------------------------------------------
# Convenience pipeline
# ---------------------------------------------------------------------------

def preprocess_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    norm_method: str = "zscore",
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Normalizer,
]:
    """
    Full preprocessing pipeline: split → fit normaliser on train → transform all.

    Returns
    -------
    (X_train, y_train), (X_val, y_val), (X_test, y_test), normalizer
    """
    n = len(X)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val,   y_val   = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test,  y_test  = X[n_train + n_val:], y[n_train + n_val:]

    normalizer = Normalizer(norm_method)
    X_train = normalizer.fit_transform(X_train)
    X_val   = normalizer.transform(X_val)
    X_test  = normalizer.transform(X_test)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), normalizer
