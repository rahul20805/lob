"""
Avellaneda-Stoikov Market Making Strategy
==========================================
Implements the closed-form solution of Avellaneda & Stoikov (2008)
augmented with ML-predicted directional signals.

Key equations
-------------
Reservation price:
    r(s, q, t) = s - q * gamma * sigma^2 * (T - t)

Optimal spread:
    delta^a + delta^b = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/kappa)

where:
    s     = mid-price
    q     = current inventory
    gamma = risk aversion coefficient
    sigma = volatility
    T     = time horizon
    t     = current time
    kappa = order arrival rate

ML augmentation:
    - If the model predicts UP with high confidence,
      skew the reservation price upward (reduce ask offset, widen bid offset)
    - If the model predicts DOWN, do the opposite
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Order / Fill dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Quote:
    bid_price:  float
    ask_price:  float
    bid_size:   float
    ask_size:   float
    timestamp:  int = 0


@dataclass
class Fill:
    side:       str   # 'buy' | 'sell'
    price:      float
    size:       float
    timestamp:  int = 0
    pnl:        float = 0.0


@dataclass
class StrategyState:
    inventory:      float = 0.0
    cash:           float = 0.0
    realized_pnl:   float = 0.0
    n_fills:        int   = 0
    fills:          list[Fill] = field(default_factory=list)
    quotes_history: list[Quote] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Avellaneda-Stoikov Market Maker
# ---------------------------------------------------------------------------

class AvellanedaStoikovMM:
    """
    Risk-averse market maker with optional ML signal integration.

    Parameters
    ----------
    gamma        : risk aversion (higher = narrower inventory limit)
    kappa        : order arrival rate intensity
    sigma        : asset volatility (per unit time)
    T            : total time horizon
    dt           : time step size
    max_inventory: hard inventory limit (units)
    order_size   : default quote size
    min_spread   : minimum allowed quoted spread
    signal_weight: weight of ML signal in price skew [0, 1]
    tick_size    : minimum price increment
    """

    def __init__(
        self,
        gamma:          float = 0.1,
        kappa:          float = 1.5,
        sigma:          float = 0.02,
        T:              float = 1.0,
        dt:             float = 0.001,
        max_inventory:  float = 1000.0,
        order_size:     float = 100.0,
        min_spread:     float = 0.0001,
        signal_weight:  float = 0.3,
        tick_size:      float = 0.01,
    ) -> None:
        self.gamma         = gamma
        self.kappa         = kappa
        self.sigma         = sigma
        self.T             = T
        self.dt            = dt
        self.max_inventory = max_inventory
        self.order_size    = order_size
        self.min_spread    = min_spread
        self.signal_weight = signal_weight
        self.tick_size     = tick_size

        self.state = StrategyState()
        self._t    = 0.0

    # ── Core A-S formulas ───────────────────────────────────────────────────

    def reservation_price(self, mid: float) -> float:
        """r = s - q * gamma * sigma^2 * (T - t)"""
        tau = max(self.T - self._t, 1e-8)
        return mid - self.state.inventory * self.gamma * self.sigma ** 2 * tau

    def optimal_spread(self) -> float:
        """delta^a + delta^b = gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/kappa)"""
        tau = max(self.T - self._t, 1e-8)
        s = (
            self.gamma * self.sigma ** 2 * tau
            + (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.kappa)
        )
        return max(s, self.min_spread)

    def compute_quotes(
        self,
        mid: float,
        signal_probs: np.ndarray | None = None,
    ) -> Quote:
        """
        Compute bid and ask quotes.

        Parameters
        ----------
        mid          : current mid-price
        signal_probs : (3,) softmax probabilities [P(down), P(stat), P(up)]
                       from the ML model — optional

        Returns
        -------
        Quote with bid_price and ask_price
        """
        r     = self.reservation_price(mid)
        half  = self.optimal_spread() / 2.0

        # Inventory skew: reduce size on the side we're already long/short
        inv_ratio = self.state.inventory / (self.max_inventory + 1e-8)
        inv_skew  = inv_ratio * half * 0.5      # shift quotes toward offloading

        bid_delta = half + inv_skew
        ask_delta = half - inv_skew

        # ML signal skew
        if signal_probs is not None and self.signal_weight > 0:
            signal_skew = self._signal_skew(signal_probs, half)
            bid_delta += signal_skew
            ask_delta -= signal_skew

        # Final prices
        bid_price = self._round_tick(r - bid_delta)
        ask_price = self._round_tick(r + ask_delta)

        # Ensure minimum spread
        if ask_price - bid_price < self.min_spread:
            mid_q = (bid_price + ask_price) / 2.0
            bid_price = self._round_tick(mid_q - self.min_spread / 2.0)
            ask_price = self._round_tick(mid_q + self.min_spread / 2.0)

        # Size: reduce on the constrained side
        bid_size = self.order_size * (1.0 - max(inv_ratio, 0.0))
        ask_size = self.order_size * (1.0 + min(inv_ratio, 0.0))
        bid_size = max(bid_size, self.order_size * 0.1)
        ask_size = max(ask_size, self.order_size * 0.1)

        q = Quote(bid_price=bid_price, ask_price=ask_price,
                  bid_size=bid_size,   ask_size=ask_size,
                  timestamp=int(self._t / self.dt))
        self.state.quotes_history.append(q)
        return q

    def _signal_skew(self, probs: np.ndarray, half_spread: float) -> float:
        """
        Return a price skew based on directional signal.

        Positive skew => raise reservation price => better ask, worse bid
                         (i.e. we think price is going up)
        """
        p_down, p_stat, p_up = probs
        directional = p_up - p_down       # in [-1, 1]
        return self.signal_weight * directional * half_spread

    # ── Fill simulation ─────────────────────────────────────────────────────

    def on_fill(
        self,
        side: str,
        price: float,
        size: float,
        timestamp: int = 0,
    ) -> None:
        """
        Called when a limit order is filled.

        Parameters
        ----------
        side  : 'buy' (bid filled) or 'sell' (ask filled)
        price : fill price
        size  : fill size
        """
        if side == "buy":
            self.state.inventory += size
            self.state.cash      -= price * size
        else:
            self.state.inventory -= size
            self.state.cash      += price * size

        self.state.n_fills += 1
        self.state.fills.append(Fill(side=side, price=price, size=size,
                                     timestamp=timestamp))
        logger.debug("Fill %s %.1f @ %.4f  inventory=%.0f",
                     side, size, price, self.state.inventory)

    def mark_to_market(self, mid: float) -> float:
        """Total P&L = cash + inventory * mid."""
        return self.state.cash + self.state.inventory * mid

    # ── Time step ───────────────────────────────────────────────────────────

    def step(self) -> None:
        """Advance time by one dt."""
        self._t = min(self._t + self.dt, self.T)

    def reset(self) -> None:
        self.state = StrategyState()
        self._t    = 0.0

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _round_tick(self, price: float) -> float:
        return round(price / self.tick_size) * self.tick_size

    @property
    def is_inventory_hard_limit(self) -> bool:
        return abs(self.state.inventory) >= self.max_inventory

    def inventory_pct(self) -> float:
        return self.state.inventory / self.max_inventory
