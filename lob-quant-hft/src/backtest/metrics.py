"""
Backtest Performance Metrics
=============================
  - Sharpe Ratio
  - Sortino Ratio
  - Calmar Ratio
  - Maximum Drawdown
  - Win Rate
  - Profit Factor
  - Average Fill P&L
  - Turnover / Inventory Metrics
"""

from __future__ import annotations

import numpy as np
from scipy.stats import skew, kurtosis

from src.backtest.engine import BacktestResult


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.05,
    periods_per_year: int = 252 * 390 * 60,   # tick-level assumption
) -> float:
    """Annualized Sharpe ratio from a series of period returns."""
    if len(returns) < 2 or returns.std() < 1e-10:
        return 0.0
    excess  = returns - risk_free_rate / periods_per_year
    return float(excess.mean() / excess.std() * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.05,
    periods_per_year: int = 252 * 390 * 60,
) -> float:
    """Sortino ratio: penalises only downside volatility."""
    excess   = returns - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    if len(downside) < 2 or downside.std() < 1e-10:
        return 0.0
    return float(excess.mean() / downside.std() * np.sqrt(periods_per_year))


def max_drawdown(pnl_series: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown (as a positive fraction)."""
    cumulative = pnl_series
    rolling_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - rolling_max
    mdd = float(drawdown.min())
    return mdd   # negative value; callers may negate for display


def calmar_ratio(
    pnl_series: np.ndarray,
    periods_per_year: int = 252 * 390 * 60,
) -> float:
    """Calmar ratio = annualised return / |max drawdown|."""
    mdd = abs(max_drawdown(pnl_series))
    if mdd < 1e-10:
        return float("inf")
    n = len(pnl_series)
    annualised_return = (pnl_series[-1] - pnl_series[0]) / n * periods_per_year
    return float(annualised_return / mdd)


def win_rate(fills) -> float:
    """Fraction of round-trip trades that are profitable."""
    # Pair up buys and sells (FIFO matching)
    buy_queue, sell_queue = [], []
    pnls = []
    for f in fills:
        if f.side == "buy":
            buy_queue.append((f.price, f.size))
        elif f.side == "sell" and buy_queue:
            buy_price, buy_size = buy_queue.pop(0)
            pnl = (f.price - buy_price) * min(f.size, buy_size)
            pnls.append(pnl)
    if not pnls:
        return float("nan")
    return float(np.mean(np.array(pnls) > 0))


def profit_factor(fills) -> float:
    """Gross profit / gross loss across all round-trip trades."""
    buy_queue = []
    gross_profit = gross_loss = 0.0
    for f in fills:
        if f.side == "buy":
            buy_queue.append((f.price, f.size))
        elif f.side == "sell" and buy_queue:
            buy_price, buy_size = buy_queue.pop(0)
            pnl = (f.price - buy_price) * min(f.size, buy_size)
            if pnl > 0:
                gross_profit += pnl
            else:
                gross_loss   += abs(pnl)
    if gross_loss < 1e-10:
        return float("inf")
    return float(gross_profit / gross_loss)


def turnover(inventory_series: np.ndarray) -> float:
    """Total absolute change in inventory (proxy for turnover)."""
    return float(np.abs(np.diff(inventory_series)).sum())


def inventory_stats(inventory_series: np.ndarray) -> dict:
    return {
        "mean":   float(inventory_series.mean()),
        "std":    float(inventory_series.std()),
        "min":    float(inventory_series.min()),
        "max":    float(inventory_series.max()),
        "abs_mean": float(np.abs(inventory_series).mean()),
    }


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def compute_metrics(
    result: BacktestResult,
    risk_free_rate: float = 0.05,
    periods_per_year: int = 252 * 390 * 60,
    initial_capital: float = 1_000_000.0,
) -> dict:
    """
    Compute a comprehensive set of backtest metrics from a BacktestResult.

    Returns
    -------
    dict of metric name → value
    """
    pnl = result.pnl_series

    # Convert P&L series to period returns
    returns = np.diff(pnl, prepend=pnl[0])

    sr = sharpe_ratio(returns, risk_free_rate, periods_per_year)
    so = sortino_ratio(returns, risk_free_rate, periods_per_year)
    mdd = max_drawdown(pnl)
    cal = calmar_ratio(pnl, periods_per_year)
    wr  = win_rate(result.fill_series)
    pf  = profit_factor(result.fill_series)
    tv  = turnover(result.inventory_series)
    inv = inventory_stats(result.inventory_series)

    total_return_pct = (result.total_pnl / (initial_capital + 1e-8)) * 100.0

    metrics = {
        "total_pnl":         result.total_pnl,
        "total_return_pct":  total_return_pct,
        "total_trades":      result.total_trades,
        "sharpe_ratio":      sr,
        "sortino_ratio":     so,
        "max_drawdown":      mdd,
        "calmar_ratio":      cal,
        "win_rate":          wr,
        "profit_factor":     pf,
        "turnover":          tv,
        "inventory":         inv,
        "return_skew":       float(skew(returns)),
        "return_kurtosis":   float(kurtosis(returns)),
        "volatility_annual": float(returns.std() * np.sqrt(periods_per_year)),
    }

    _print_metrics(metrics)
    return metrics


def _print_metrics(m: dict) -> None:
    print("\n" + "=" * 50)
    print("  BACKTEST PERFORMANCE METRICS")
    print("=" * 50)
    print(f"  Total P&L:          {m['total_pnl']:>12,.2f}")
    print(f"  Total Return (%):   {m['total_return_pct']:>12.4f}%")
    print(f"  Total Trades:       {m['total_trades']:>12,}")
    print(f"  Sharpe Ratio:       {m['sharpe_ratio']:>12.4f}")
    print(f"  Sortino Ratio:      {m['sortino_ratio']:>12.4f}")
    print(f"  Calmar Ratio:       {m['calmar_ratio']:>12.4f}")
    print(f"  Max Drawdown:       {m['max_drawdown']:>12,.2f}")
    print(f"  Win Rate:           {m['win_rate']:>12.4f}")
    print(f"  Profit Factor:      {m['profit_factor']:>12.4f}")
    print(f"  Annual Volatility:  {m['volatility_annual']:>12.6f}")
    print(f"  Return Skewness:    {m['return_skew']:>12.4f}")
    print(f"  Return Kurtosis:    {m['return_kurtosis']:>12.4f}")
    print(f"  Inventory (mean):   {m['inventory']['mean']:>12.2f}")
    print(f"  Inventory (|mean|): {m['inventory']['abs_mean']:>12.2f}")
    print("=" * 50 + "\n")
