"""
sentinel/backtest/momentum.py
Phase 2 Deliverable 2 — Momentum Strategy Signals

Two momentum flavours are implemented:

1. Time-Series Momentum (TSMOM)
   ─────────────────────────────
   Moskovitz, Ooi & Pedersen (2012): "Time Series Momentum", JFE.
   Signal: buy ticker i if its trailing L-day return is positive, else flat.

       r_{t-L→t} = log( P_t / P_{t-L} )
       s_i(t) = 1  if  r_{t-L→t} > 0
                0  otherwise

   Intuition: assets that have been going up tend to keep going up (trend).

2. Cross-Sectional Momentum (CSMOM)
   ──────────────────────────────────
   Jegadeesh & Titman (1993): "Returns to buying winners and selling losers", JF.
   Each month, rank all N tickers by trailing L-day return.
   Go long the top-K tickers (equal weight), flat the rest.

       rank_i(t) = rank of ticker i by r_{t-L→t}   (highest = 1)
       s_i(t) = 1/K  if rank_i ≤ K
                0    otherwise

   Rebalancing is monthly (every ~21 trading days).

Both strategies use a 252-day (≈1 year) lookback minus 21-day (≈1 month) skip
to avoid the short-term reversal effect documented in the literature.
i.e. effective lookback = L_long - L_skip = 252 - 21 = 231 days
"""

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd


# ── helpers ────────────────────────────────────────────────────────────────────

def trailing_return(prices: pd.DataFrame,
                    lookback: int = 252,
                    skip: int = 21) -> pd.DataFrame:
    """
    Compute trailing log-return for each ticker, skipping most recent `skip` days
    to avoid short-term reversal contamination.

    ret(t) = log( P_{t - skip} / P_{t - lookback} )
    """
    shifted_end   = prices.shift(skip)            # price skip days ago
    shifted_start = prices.shift(lookback)        # price lookback days ago
    return np.log(shifted_end / shifted_start)


# ── 1. Time-Series Momentum ────────────────────────────────────────────────────

def tsmom_signals(prices: pd.DataFrame,
                  lookback: int = 252,
                  skip: int = 21) -> pd.DataFrame:
    """
    Time-series momentum signals (0/1) for each ticker.

    Returns
    -------
    signals : pd.DataFrame  same shape as prices, values in {0, 1}
    scores  : pd.DataFrame  trailing returns used to generate signals
    """
    scores  = trailing_return(prices, lookback, skip)
    signals = (scores > 0).astype(int)
    return signals, scores


# ── 2. Cross-Sectional Momentum ────────────────────────────────────────────────

def csmom_signals(prices: pd.DataFrame,
                  lookback: int = 252,
                  skip: int = 21,
                  top_k: int = 2,
                  rebal_freq: int = 21) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cross-sectional momentum: go long top_k tickers by trailing return,
    rebalancing every rebal_freq trading days.

    Returns
    -------
    signals : pd.DataFrame  fractional weights (0 or 1/top_k) same shape as prices
    scores  : pd.DataFrame  trailing returns used for ranking
    """
    scores  = trailing_return(prices, lookback, skip)
    signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    rebal_dates = prices.index[lookback::rebal_freq]   # monthly rebalance points

    prev_sig = pd.Series(0.0, index=prices.columns)
    for date in prices.index:
        if date in rebal_dates and not scores.loc[date].isna().all():
            row = scores.loc[date].dropna()
            if len(row) >= top_k:
                winners = row.nlargest(top_k).index
                prev_sig = pd.Series(0.0, index=prices.columns)
                prev_sig[winners] = 1.0 / top_k
        signals.loc[date] = prev_sig

    return signals, scores


# ── portfolio returns from fractional signal matrix ────────────────────────────

def portfolio_returns_from_signals(prices: pd.DataFrame,
                                   signals: pd.DataFrame) -> pd.Series:
    """
    Convert a (possibly fractional) signal DataFrame into a daily portfolio
    log-return series.

    r_port(t) = Σ_i  s_i(t-1) · log(P_i(t) / P_i(t-1))
    """
    log_rets = np.log(prices / prices.shift(1))
    # lag signals by 1 day to avoid lookahead
    lagged   = signals.shift(1).fillna(0)
    port_r   = (lagged * log_rets).sum(axis=1)
    return port_r.rename("portfolio_return")


def equity_curve_from_returns(returns: pd.Series,
                               initial: float = 10_000.0) -> pd.Series:
    return initial * np.exp(returns.fillna(0).cumsum())
