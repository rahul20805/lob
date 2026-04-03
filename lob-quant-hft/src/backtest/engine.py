"""
High-Frequency Backtest Engine
================================
Simulates limit order execution using LOB data with:
  - Configurable transaction costs and slippage
  - Queue-based fill probability model
  - Inventory and position tracking
  - Integration with ML model for signal generation
  - Integration with AvellanedaStoikovMM for quoting
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn as nn

from src.strategy.market_maker import AvellanedaStoikovMM, Fill, Quote

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fill probability model
# ---------------------------------------------------------------------------

def fill_probability(
    quote_price: float,
    mid_price:   float,
    spread:      float,
    kappa:       float = 1.5,
    dt:          float = 0.001,
    side:        str   = "bid",
) -> float:
    """
    Probability that a limit order is filled in one time step.

    Derived from Poisson arrival model (Avellaneda & Stoikov, 2008):
        P(fill) ≈ exp(-kappa * delta)
    where delta = |quote_price - mid_price|.
    """
    delta = abs(quote_price - mid_price)
    prob  = np.exp(-kappa * delta) * dt
    return float(np.clip(prob, 0.0, 1.0))


# ---------------------------------------------------------------------------
# LOB tick iterator
# ---------------------------------------------------------------------------

@dataclass
class LOBTick:
    timestamp:  int
    mid:        float
    bid_p1:     float
    ask_p1:     float
    spread:     float
    features:   np.ndarray   # raw LOB feature vector for the model


def lob_tick_iter(
    X: np.ndarray,
    window_size: int = 100,
    step: int = 1,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """
    Yield (timestamp, window_X, current_row) tuples from a LOB feature matrix.

    Parameters
    ----------
    X           : (N, F) feature matrix; first 4 columns = ask_p1, ask_v1, bid_p1, bid_v1
    window_size : lookback window for model input
    step        : stride between ticks
    """
    N = len(X)
    for t in range(window_size, N, step):
        window = X[t - window_size : t]       # (window_size, F)
        row    = X[t]
        yield t, window, row


# ---------------------------------------------------------------------------
# Backtest result
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    pnl_series:       np.ndarray
    inventory_series: np.ndarray
    fill_series:      list[Fill]
    quote_series:     list[Quote]
    total_pnl:        float
    total_trades:     int
    metrics:          dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Backtest Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Event-driven HFT backtest engine.

    Parameters
    ----------
    model          : trained neural network (DeepLOB or LOBTransformer)
    strategy       : AvellanedaStoikovMM instance
    transaction_cost : fraction of trade value (e.g. 0.0001 = 1 bps)
    slippage         : additional slippage per trade (fraction of price)
    initial_capital  : starting cash
    device           : torch device for model inference
    """

    def __init__(
        self,
        model:            nn.Module,
        strategy:         AvellanedaStoikovMM,
        transaction_cost: float = 0.0001,
        slippage:         float = 0.00005,
        initial_capital:  float = 1_000_000.0,
        device:           torch.device | None = None,
        min_edge:         float = 0.0001,
    ) -> None:
        self.model            = model
        self.strategy         = strategy
        self.transaction_cost = transaction_cost
        self.slippage         = slippage
        self.initial_capital  = initial_capital
        self.device           = device or torch.device("cpu")
        self.min_edge         = min_edge

        self.model.eval()
        self.model.to(self.device)

    def run(
        self,
        X: np.ndarray,
        window_size: int = 100,
        step: int = 1,
        seed: int = 42,
    ) -> BacktestResult:
        """
        Run the full backtest over LOB data X.

        Parameters
        ----------
        X           : (N, F) feature array; columns [ask_p1, ask_v1, bid_p1, bid_v1, ...]
        window_size : model lookback
        step        : ticks between strategy updates

        Returns
        -------
        BacktestResult
        """
        rng = np.random.default_rng(seed)
        self.strategy.reset()
        self.strategy.state.cash = self.initial_capital

        pnl_series       = []
        inventory_series = []

        logger.info("Starting backtest: N=%d window=%d step=%d", len(X), window_size, step)

        for t, window, row in lob_tick_iter(X, window_size, step):
            # Extract prices from row (expected order: ask_p1, ask_v1, bid_p1, bid_v1, ...)
            ask_p1 = float(row[0])
            bid_p1 = float(row[2])
            mid    = (ask_p1 + bid_p1) / 2.0
            spread = ask_p1 - bid_p1

            # ── ML inference ────────────────────────────────────────────────
            with torch.inference_mode():
                x_t = torch.from_numpy(window[None]).float().to(self.device)
                probs = self.model.predict_proba(x_t).cpu().numpy()[0]

            # ── Compute quotes ───────────────────────────────────────────────
            quote = self.strategy.compute_quotes(mid, signal_probs=probs)

            # ── Simulate fills ───────────────────────────────────────────────
            # Bid fill
            p_bid = fill_probability(
                quote.bid_price, mid, spread,
                kappa=self.strategy.kappa, dt=self.strategy.dt, side="bid"
            )
            if rng.random() < p_bid and not self.strategy.is_inventory_hard_limit:
                fill_price = quote.bid_price * (1.0 + self.slippage)
                cost       = fill_price * quote.bid_size * self.transaction_cost
                self.strategy.on_fill("buy", fill_price, quote.bid_size, t)
                self.strategy.state.cash -= cost

            # Ask fill
            p_ask = fill_probability(
                quote.ask_price, mid, spread,
                kappa=self.strategy.kappa, dt=self.strategy.dt, side="ask"
            )
            if rng.random() < p_ask and not self.strategy.is_inventory_hard_limit:
                fill_price = quote.ask_price * (1.0 - self.slippage)
                cost       = fill_price * quote.ask_size * self.transaction_cost
                self.strategy.on_fill("sell", fill_price, quote.ask_size, t)
                self.strategy.state.cash -= cost

            self.strategy.step()

            # Mark-to-market
            mtm = self.strategy.mark_to_market(mid) - self.initial_capital
            pnl_series.append(mtm)
            inventory_series.append(self.strategy.state.inventory)

        result = BacktestResult(
            pnl_series       = np.array(pnl_series),
            inventory_series = np.array(inventory_series),
            fill_series      = list(self.strategy.state.fills),
            quote_series     = list(self.strategy.state.quotes_history),
            total_pnl        = pnl_series[-1] if pnl_series else 0.0,
            total_trades     = self.strategy.state.n_fills,
        )

        logger.info(
            "Backtest complete: PnL=%.2f  Trades=%d  FinalInventory=%.0f",
            result.total_pnl, result.total_trades,
            inventory_series[-1] if len(inventory_series) else 0,
        )
        return result
