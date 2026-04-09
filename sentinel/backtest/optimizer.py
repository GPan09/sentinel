"""
sentinel/backtest/optimizer.py
Phase 2 Deliverable 3 — Strategy Optimizer & Ensemble

1. SMA Parameter Grid Search
   ─────────────────────────
   Sweep (fast, slow) pairs over a grid, compute annualised Sharpe for each.
   This reveals which parameter combinations are robust vs. overfit.

   ⚠️ In-sample optimisation warning: the best grid cell WILL be overfit.
   A proper production system uses walk-forward validation (train on window t,
   test on window t+1). We compute IS Sharpe here purely for visualisation.

2. Ensemble Strategy
   ─────────────────
   Combine SMA crossover + time-series momentum signals by majority vote:

       s_ensemble(t) = 1  if  s_SMA(t-1) + s_TSMOM(t-1) ≥ 1   (either fires)
                       0  otherwise

   OR intersection (both must agree):
       s_ensemble(t) = s_SMA(t-1) AND s_TSMOM(t-1)

   Union reduces missed opportunities; intersection reduces false signals.
   We compute both and let the caller choose.

3. Calendar Returns
   ─────────────────
   Monthly P&L expressed as percentage, arranged in a year × month grid.
   Classic "quant tearsheet" format.

4. Performance Scorecard
   ──────────────────────
   Side-by-side comparison of all strategies on key metrics:
   Total Return, CAGR, Sharpe, Sortino, Max Drawdown, Win Rate, # Trades.

   Sortino ratio: like Sharpe but only penalises downside volatility.
       Sortino = (μ_strat - r_f) / σ_down · √252
   where σ_down = std of returns that are < 0 (downside deviation).
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from sentinel.backtest.engine import Backtester, _sharpe, _cagr, _max_drawdown
from sentinel.backtest.ma_strategy import sma_crossover
from sentinel.backtest.momentum import tsmom_signals, portfolio_returns_from_signals


# ── 1. SMA parameter grid search ──────────────────────────────────────────────

def sma_grid_search(
    prices: pd.Series,
    fast_range: range = range(5, 40, 5),
    slow_range: range = range(20, 120, 10),
) -> pd.DataFrame:
    """
    Compute annualised Sharpe for each (fast, slow) SMA pair.

    Returns
    -------
    results : pd.DataFrame  index=fast, columns=slow, values=Sharpe ratio
              NaN where fast >= slow (invalid combination)
    """
    rows = []
    for fast in fast_range:
        for slow in slow_range:
            if fast >= slow:
                rows.append({"fast": fast, "slow": slow, "sharpe": np.nan})
                continue
            sig, _, _ = sma_crossover(prices, fast, slow)
            bt = Backtester(prices, sig).run()
            sr = _sharpe(bt.returns_strat)
            rows.append({"fast": fast, "slow": slow, "sharpe": sr})

    df = pd.DataFrame(rows)
    return df.pivot(index="fast", columns="slow", values="sharpe")


def sma_grid_search_universe(
    prices_df: pd.DataFrame,
    fast_range: range = range(5, 40, 5),
    slow_range: range = range(20, 120, 10),
) -> pd.DataFrame:
    """Average Sharpe grid across all tickers."""
    grids = [sma_grid_search(prices_df[t], fast_range, slow_range)
             for t in prices_df.columns]
    combined = grids[0].copy() * 0
    for g in grids:
        combined = combined.add(g, fill_value=0)
    return combined / len(grids)


# ── 2. Ensemble strategy ───────────────────────────────────────────────────────

def ensemble_signals(
    prices: pd.DataFrame,
    sma_fast: int = 20,
    sma_slow: int = 50,
    mom_lookback: int = 252,
    mom_skip: int = 21,
    mode: str = "union",          # "union" | "intersection"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build ensemble from SMA crossover + TS-momentum signals.

    Parameters
    ----------
    mode : "union"        → long if *either* signal is active  (more trades)
           "intersection" → long only if *both* signals agree  (fewer trades, higher conviction)

    Returns
    -------
    ensemble_sig : pd.DataFrame  0/1 signals, same columns as prices
    sma_sig      : pd.DataFrame  raw SMA signals
    mom_sig      : pd.DataFrame  raw TS-mom signals
    """
    # SMA signals (per ticker)
    sma_sig = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for ticker in prices.columns:
        sig, _, _ = sma_crossover(prices[ticker], sma_fast, sma_slow)
        sma_sig[ticker] = sig

    # TS-mom signals
    mom_sig, _ = tsmom_signals(prices, mom_lookback, mom_skip)
    mom_sig = mom_sig.reindex(prices.index).fillna(0)

    if mode == "union":
        ensemble = ((sma_sig + mom_sig) >= 1).astype(int)
    else:   # intersection
        ensemble = (sma_sig * mom_sig).astype(int)

    return ensemble.astype(float), sma_sig.astype(float), mom_sig.astype(float)


def portfolio_equity(prices: pd.DataFrame,
                     signals: pd.DataFrame,
                     initial: float = 10_000.0) -> pd.Series:
    """Equal-weight portfolio equity from per-ticker signals."""
    log_r = np.log(prices / prices.shift(1))
    lagged = signals.shift(1).fillna(0)
    # equal weight among active tickers each day
    n_active = lagged.sum(axis=1).replace(0, np.nan)
    weighted = (lagged.div(n_active, axis=0) * log_r).sum(axis=1).fillna(0)
    return initial * np.exp(weighted.cumsum())


# ── 3. Monthly returns calendar ────────────────────────────────────────────────

def monthly_returns_calendar(daily_returns: pd.Series) -> pd.DataFrame:
    """
    Reshape daily log-returns into a Year × Month grid of total monthly returns.
    Values are simple % returns (not log) for readability.
    """
    monthly = daily_returns.resample("ME").sum()        # sum of log returns
    monthly_pct = (np.exp(monthly) - 1) * 100          # convert to simple %
    cal = monthly_pct.to_frame("ret")
    cal["year"]  = cal.index.year
    cal["month"] = cal.index.month
    pivot = cal.pivot(index="year", columns="month", values="ret")
    pivot.columns = [pd.Timestamp(2000, m, 1).strftime("%b")
                     for m in pivot.columns]
    pivot["Full Year"] = (np.exp(monthly.resample("YE").sum()) - 1) * 100
    return pivot


# ── 4. Performance scorecard ───────────────────────────────────────────────────

def _sortino(daily_returns: pd.Series, risk_free_annual: float = 0.0) -> float:
    """Annualised Sortino ratio (penalises downside vol only)."""
    rf_daily = risk_free_annual / 252
    excess = daily_returns - rf_daily
    downside = excess[excess < 0]
    if downside.std() == 0 or len(downside) == 0:
        return np.nan
    return float(excess.mean() / downside.std() * np.sqrt(252))


def build_scorecard(strategy_returns: Dict[str, pd.Series],
                    initial: float = 10_000.0) -> pd.DataFrame:
    """
    Compute full performance table for a dict of {name: daily_return_series}.

    Columns: Total Return, CAGR, Sharpe, Sortino, Max Drawdown, Calmar, Vol
    """
    rows = []
    for name, r in strategy_returns.items():
        r = r.dropna()
        equity = initial * np.exp(r.cumsum())
        mdd = _max_drawdown(equity)
        cagr = _cagr(equity)
        calmar = cagr / abs(mdd) if mdd != 0 else np.nan
        rows.append({
            "Strategy":     name,
            "Total Return": f"{(equity.iloc[-1]/initial - 1):+.1%}",
            "CAGR":         f"{cagr:+.1%}",
            "Sharpe":       f"{_sharpe(r):.2f}",
            "Sortino":      f"{_sortino(r):.2f}",
            "Max Drawdown": f"{mdd:.1%}",
            "Calmar":       f"{calmar:.2f}" if not np.isnan(calmar) else "—",
            "Ann. Vol":     f"{r.std() * np.sqrt(252):.1%}",
        })
    return pd.DataFrame(rows).set_index("Strategy")
