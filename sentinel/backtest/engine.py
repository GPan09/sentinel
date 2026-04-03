"""
sentinel/backtest/engine.py
Phase 2 Deliverable 1 — Core Backtesting Engine

Given a price series and a signal series, simulates a long-only strategy and
returns the equity curve, trade log, and performance metrics.

Math primer
-----------
Daily log-return:         r_t  = log(P_t / P_{t-1})
Strategy return:          r̃_t  = s_{t-1} · r_t          (signal lagged 1 day → no lookahead bias)
Cumulative equity:        E_t  = E_0 · exp( Σ r̃_i )
Buy-and-hold equity:      B_t  = E_0 · exp( Σ r_i  )

Sharpe ratio (annualised): SR = (μ_strat - r_f) / σ_strat · √252
    where μ_strat, σ_strat are mean and std of *daily* strategy returns

Max drawdown:             MDD = min_t { (max_{s≤t} E_s − E_t) / max_{s≤t} E_s }
CAGR:                     CAGR = (E_T / E_0)^(252/T) − 1
Win rate:                 # profitable trades / total completed trades
"""

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd


# ── helpers ────────────────────────────────────────────────────────────────────

def _log_returns(prices: pd.Series) -> pd.Series:
    """Daily log returns, NaN on first row."""
    return np.log(prices / prices.shift(1))


def _equity_curve(log_returns: pd.Series, initial: float = 10_000.0) -> pd.Series:
    """Convert log-return series to dollar equity curve."""
    cum = log_returns.fillna(0).cumsum()
    return initial * np.exp(cum)


def _drawdown_series(equity: pd.Series) -> pd.Series:
    """Rolling drawdown (negative fraction) from peak."""
    peak = equity.cummax()
    return (equity - peak) / peak


def _max_drawdown(equity: pd.Series) -> float:
    return float(_drawdown_series(equity).min())


def _sharpe(daily_returns: pd.Series, risk_free_annual: float = 0.0) -> float:
    """Annualised Sharpe ratio from daily log returns."""
    rf_daily = risk_free_annual / 252
    excess = daily_returns - rf_daily
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(252))


def _cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    """Compound annual growth rate from equity curve."""
    n = len(equity)
    if n < 2 or equity.iloc[0] == 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (periods_per_year / n) - 1)


def _trade_log(signal: pd.Series) -> pd.DataFrame:
    """
    Extract individual trades from a 0/1 signal series.
    Returns DataFrame with columns: entry_date, exit_date, direction.
    """
    edges = signal.diff().fillna(0)
    entries = signal.index[edges == 1].tolist()
    exits   = signal.index[edges == -1].tolist()

    # handle open trade at end of series
    if len(entries) > len(exits):
        exits.append(signal.index[-1])

    trades = pd.DataFrame({"entry_date": entries[:len(exits)],
                           "exit_date":  exits})
    return trades


def _win_rate(prices: pd.Series, trades: pd.DataFrame) -> float:
    """Fraction of trades with positive return."""
    if trades.empty:
        return float("nan")
    results = []
    for _, row in trades.iterrows():
        p_entry = prices.loc[row["entry_date"]]
        p_exit  = prices.loc[row["exit_date"]]
        results.append(p_exit > p_entry)
    return float(np.mean(results))


def _rolling_sharpe(daily_returns: pd.Series, window: int = 60) -> pd.Series:
    """Rolling annualised Sharpe with given lookback window."""
    roll_mean = daily_returns.rolling(window).mean()
    roll_std  = daily_returns.rolling(window).std()
    return (roll_mean / roll_std.replace(0, np.nan)) * np.sqrt(252)


# ── main class ─────────────────────────────────────────────────────────────────

class Backtester:
    """
    Single-asset long-only backtester.

    Parameters
    ----------
    prices  : pd.Series  daily adjusted close prices (DatetimeIndex)
    signal  : pd.Series  0/1 position series (same index as prices)
    initial : float      starting capital in dollars (default 10 000)
    """

    def __init__(self, prices: pd.Series, signal: pd.Series,
                 initial: float = 10_000.0):
        self.prices  = prices.dropna()
        self.signal  = signal.reindex(self.prices.index).ffill().fillna(0)
        self.initial = initial
        self._ran    = False

    def run(self) -> "Backtester":
        """Simulate the strategy. Call before accessing results."""
        r = _log_returns(self.prices)

        # Lag signal by 1 day: trade on *next* open after signal fires
        strat_r = self.signal.shift(1).fillna(0) * r
        bh_r    = r.copy()

        self.returns_strat = strat_r.dropna()
        self.returns_bh    = bh_r.dropna()

        self.equity_strat  = _equity_curve(self.returns_strat, self.initial)
        self.equity_bh     = _equity_curve(self.returns_bh,    self.initial)

        self.drawdown      = _drawdown_series(self.equity_strat)
        self.rolling_sharpe = _rolling_sharpe(self.returns_strat)

        self.trades        = _trade_log(self.signal)
        self._ran          = True
        return self

    def metrics(self) -> Dict[str, float]:
        """Return summary statistics dict."""
        assert self._ran, "Call .run() first."
        return {
            "total_return":   float(self.equity_strat.iloc[-1] / self.initial - 1),
            "bh_return":      float(self.equity_bh.iloc[-1]    / self.initial - 1),
            "cagr":           _cagr(self.equity_strat),
            "sharpe":         _sharpe(self.returns_strat),
            "max_drawdown":   _max_drawdown(self.equity_strat),
            "win_rate":       _win_rate(self.prices, self.trades),
            "n_trades":       len(self.trades),
        }
