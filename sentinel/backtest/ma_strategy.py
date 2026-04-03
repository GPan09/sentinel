"""
sentinel/backtest/ma_strategy.py
Phase 2 Deliverable 1 — Moving Average Crossover Strategy

Generates long/flat signals for a single price series using two moving averages.

Strategy logic
--------------
Golden cross  → buy signal  (s = 1): fast MA crosses *above* slow MA
Death cross   → flat signal (s = 0): fast MA crosses *below* slow MA

Both SMA (Simple) and EMA (Exponential) variants are supported.

SMA_n(t)  = (1/n) Σ_{i=0}^{n-1} P_{t-i}          simple, equal-weight window
EMA_n(t)  = α · P_t + (1-α) · EMA_n(t-1)           exponentially decaying weights
            where α = 2 / (n + 1)                  (standard span formula)

Using EMA is more responsive to recent price moves; SMA is smoother and slower.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd


def sma(prices: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return prices.rolling(window, min_periods=window).mean()


def ema(prices: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average (standard span formula)."""
    return prices.ewm(span=span, adjust=False).mean()


def sma_crossover(
    prices: pd.Series,
    fast: int = 20,
    slow: int = 50,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    SMA crossover signal.

    Returns
    -------
    signal   : pd.Series  0/1 position (1 = long, 0 = flat)
    ma_fast  : pd.Series  fast SMA
    ma_slow  : pd.Series  slow SMA
    """
    ma_fast = sma(prices, fast)
    ma_slow = sma(prices, slow)
    signal  = (ma_fast > ma_slow).astype(int)
    # Zero out the warmup period where slow MA is NaN
    signal.iloc[:slow] = 0
    return signal, ma_fast, ma_slow


def ema_crossover(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    EMA crossover signal (similar to MACD crossover logic).

    Returns
    -------
    signal   : pd.Series  0/1 position
    ema_fast : pd.Series  fast EMA
    ema_slow : pd.Series  slow EMA
    """
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    signal   = (ema_fast > ema_slow).astype(int)
    signal.iloc[:slow] = 0
    return signal, ema_fast, ema_slow


def run_sma_universe(
    prices_df: pd.DataFrame,
    fast: int = 20,
    slow: int = 50,
) -> dict:
    """
    Apply SMA crossover to each column (ticker) in prices_df.

    Returns dict mapping ticker → (signal, ma_fast, ma_slow).
    """
    result = {}
    for ticker in prices_df.columns:
        result[ticker] = sma_crossover(prices_df[ticker].dropna(), fast, slow)
    return result
