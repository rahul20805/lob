"""
Market Microstructure Features
================================
Computes commonly used LOB-derived features for ML models:

  - Order Flow Imbalance  (OFI)
  - Volume Order Imbalance (VOI)
  - Depth Imbalance
  - Weighted Mid-Price
  - Queue Imbalance per level
  - Trade Intensity / Arrival Rate
  - Kyle's Lambda (price impact)
  - Amihud Illiquidity
  - Roll Spread Estimator
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit


# ---------------------------------------------------------------------------
# Low-level helpers (Numba-accelerated)
# ---------------------------------------------------------------------------

@njit
def _ofi_single(
    bid_p_prev: float, bid_v_prev: float,
    ask_p_prev: float, ask_v_prev: float,
    bid_p_curr: float, bid_v_curr: float,
    ask_p_curr: float, ask_v_curr: float,
) -> float:
    """
    Order Flow Imbalance (Cont et al., 2014) for a single tick transition.

    OFI = ΔBid_V - ΔAsk_V  (with sign convention for price level changes)
    """
    # Bid side
    if bid_p_curr > bid_p_prev:
        d_bid = bid_v_curr
    elif bid_p_curr == bid_p_prev:
        d_bid = bid_v_curr - bid_v_prev
    else:
        d_bid = -bid_v_prev

    # Ask side
    if ask_p_curr < ask_p_prev:
        d_ask = ask_v_curr
    elif ask_p_curr == ask_p_prev:
        d_ask = ask_v_curr - ask_v_prev
    else:
        d_ask = -ask_v_prev

    return d_bid - d_ask


@njit
def _compute_ofi_series(
    bid_p: np.ndarray, bid_v: np.ndarray,
    ask_p: np.ndarray, ask_v: np.ndarray,
) -> np.ndarray:
    n = len(bid_p)
    ofi = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        ofi[t] = _ofi_single(
            bid_p[t - 1], bid_v[t - 1],
            ask_p[t - 1], ask_v[t - 1],
            bid_p[t],     bid_v[t],
            ask_p[t],     ask_v[t],
        )
    return ofi


# ---------------------------------------------------------------------------
# Feature computation functions
# ---------------------------------------------------------------------------

def order_flow_imbalance(
    df: pd.DataFrame,
    level: int = 1,
    window: int = 1,
) -> np.ndarray:
    """
    Order Flow Imbalance at a given price level.

    Parameters
    ----------
    df     : DataFrame with columns bid_p{level}, bid_v{level}, ask_p{level}, ask_v{level}
    level  : LOB depth level (1 = best quote)
    window : rolling window to smooth OFI (1 = raw tick-by-tick)
    """
    bp = df[f"bid_p{level}"].values.astype(np.float64)
    bv = df[f"bid_v{level}"].values.astype(np.float64)
    ap = df[f"ask_p{level}"].values.astype(np.float64)
    av = df[f"ask_v{level}"].values.astype(np.float64)

    ofi = _compute_ofi_series(bp, bv, ap, av)

    if window > 1:
        ofi = pd.Series(ofi).rolling(window, min_periods=1).sum().values

    return ofi.astype(np.float32)


def volume_order_imbalance(
    df: pd.DataFrame,
    n_levels: int = 10,
) -> np.ndarray:
    """
    Volume Order Imbalance (VOI) = total bid volume − total ask volume
    across all levels.
    """
    total_bid = sum(df[f"bid_v{i}"].values for i in range(1, n_levels + 1))
    total_ask = sum(df[f"ask_v{i}"].values for i in range(1, n_levels + 1))
    voi = (total_bid - total_ask) / (total_bid + total_ask + 1e-8)
    return voi.astype(np.float32)


def depth_imbalance(
    df: pd.DataFrame,
    n_levels: int = 10,
    weights: str = "uniform",
) -> np.ndarray:
    """
    Depth imbalance = weighted sum of (bid_v - ask_v) / (bid_v + ask_v) per level.

    Parameters
    ----------
    weights : 'uniform' | 'linear' | 'exp'
               linear  → level 1 gets weight n, level n gets weight 1
               exp     → level i gets weight exp(-i)
    """
    w = _level_weights(n_levels, weights)
    imbalance = np.zeros(len(df), dtype=np.float64)
    for i, wi in enumerate(w, start=1):
        bv = df[f"bid_v{i}"].values
        av = df[f"ask_v{i}"].values
        imbalance += wi * (bv - av) / (bv + av + 1e-8)
    return (imbalance / w.sum()).astype(np.float32)


def weighted_mid_price(
    df: pd.DataFrame,
    n_levels: int = 1,
) -> np.ndarray:
    """
    Volume-weighted mid price (VWMP).

    For n_levels=1 this reduces to the standard mid-price weighted by
    best-bid and best-ask volumes.
    """
    num = np.zeros(len(df), dtype=np.float64)
    den = np.zeros(len(df), dtype=np.float64)
    for i in range(1, n_levels + 1):
        bv = df[f"bid_v{i}"].values
        av = df[f"ask_v{i}"].values
        bp = df[f"bid_p{i}"].values
        ap = df[f"ask_p{i}"].values
        num += ap * bv + bp * av
        den += bv + av
    return (num / (den + 1e-8)).astype(np.float32)


def queue_imbalance(
    df: pd.DataFrame,
    level: int = 1,
) -> np.ndarray:
    """Queue imbalance at a single level: (bid_v - ask_v) / (bid_v + ask_v)."""
    bv = df[f"bid_v{level}"].values
    av = df[f"ask_v{level}"].values
    qi = (bv - av) / (bv + av + 1e-8)
    return qi.astype(np.float32)


def relative_spread(df: pd.DataFrame) -> np.ndarray:
    """Relative bid–ask spread = (ask_p1 - bid_p1) / mid_p1."""
    mid = (df["ask_p1"].values + df["bid_p1"].values) / 2.0
    spread = df["ask_p1"].values - df["bid_p1"].values
    return (spread / (mid + 1e-8)).astype(np.float32)


def log_return(df: pd.DataFrame, window: int = 1) -> np.ndarray:
    """Log mid-price return over *window* ticks."""
    mid = (df["ask_p1"].values + df["bid_p1"].values) / 2.0
    ret = np.log(mid[window:] + 1e-10) - np.log(mid[:-window] + 1e-10)
    return np.concatenate([np.zeros(window, dtype=np.float32),
                           ret.astype(np.float32)])


def realized_volatility(df: pd.DataFrame, window: int = 20) -> np.ndarray:
    """Rolling realized volatility of log mid-price returns."""
    lr = log_return(df, window=1)
    rv = pd.Series(lr).rolling(window, min_periods=2).std().values
    rv = np.nan_to_num(rv, nan=0.0)
    return rv.astype(np.float32)


def amihud_illiquidity(
    df: pd.DataFrame,
    price_col: str = "ask_p1",
    volume_col: str = "ask_v1",
    window: int = 20,
) -> np.ndarray:
    """
    Amihud (2002) illiquidity ratio: |r_t| / volume_t  (rolling mean).
    """
    price = df[price_col].values
    vol   = df[volume_col].values
    ret   = np.abs(np.diff(np.log(price + 1e-10)))
    ret   = np.concatenate([[0.0], ret])
    illiq = ret / (vol + 1e-8)
    illiq = pd.Series(illiq).rolling(window, min_periods=1).mean().values
    return illiq.astype(np.float32)


def roll_spread(df: pd.DataFrame, window: int = 20) -> np.ndarray:
    """
    Roll (1984) implied spread estimator: 2 * sqrt(max(-Cov(Δp_t, Δp_{t-1}), 0)).
    """
    mid = (df["ask_p1"].values + df["bid_p1"].values) / 2.0
    dp  = np.diff(mid)
    dp  = np.concatenate([[0.0], dp])

    spreads = np.zeros(len(df), dtype=np.float32)
    for t in range(window, len(df)):
        d  = dp[t - window : t]
        d1 = dp[t - window + 1 : t + 1]
        cov = np.cov(d, d1)[0, 1]
        spreads[t] = 2.0 * np.sqrt(max(-cov, 0.0))

    return spreads


# ---------------------------------------------------------------------------
# Full feature matrix builder
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df: pd.DataFrame,
    n_levels: int = 10,
    include_ofi: bool = True,
    include_voi: bool = True,
    include_depth: bool = True,
    include_spread: bool = True,
    include_volatility: bool = True,
    vol_window: int = 20,
    ofi_window: int = 5,
) -> np.ndarray:
    """
    Build a full microstructure feature matrix from a LOB DataFrame.

    Returns
    -------
    F : float32 array of shape (N, n_features)
    """
    parts = []

    # Raw LOB features (ask_p, ask_v, bid_p, bid_v per level)
    for i in range(1, n_levels + 1):
        parts.append(df[f"ask_p{i}"].values[:, None])
        parts.append(df[f"ask_v{i}"].values[:, None])
        parts.append(df[f"bid_p{i}"].values[:, None])
        parts.append(df[f"bid_v{i}"].values[:, None])

    if include_ofi:
        ofi = order_flow_imbalance(df, level=1, window=ofi_window)
        parts.append(ofi[:, None])

    if include_voi:
        voi = volume_order_imbalance(df, n_levels=n_levels)
        parts.append(voi[:, None])

    if include_depth:
        di = depth_imbalance(df, n_levels=n_levels, weights="linear")
        parts.append(di[:, None])

    if include_spread:
        rs = relative_spread(df)
        parts.append(rs[:, None])

    if include_volatility:
        rv = realized_volatility(df, window=vol_window)
        parts.append(rv[:, None])

    F = np.concatenate(parts, axis=1).astype(np.float32)
    return F


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _level_weights(n_levels: int, mode: str) -> np.ndarray:
    if mode == "uniform":
        return np.ones(n_levels)
    if mode == "linear":
        return np.arange(n_levels, 0, -1, dtype=float)
    if mode == "exp":
        return np.exp(-np.arange(n_levels, dtype=float))
    raise ValueError(f"Unknown weight mode: {mode}")
