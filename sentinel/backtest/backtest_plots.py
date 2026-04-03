"""
sentinel/backtest/backtest_plots.py
Phase 2 Deliverable 1 — Backtester Chart Functions (Charts 13–16)

Chart 13 — Price + SMA20/SMA50 with buy/sell signal markers (one ticker)
Chart 14 — Equity curves: strategy vs buy-and-hold, one panel per ticker
Chart 15 — Rolling 60-day Sharpe ratio (strategy) per ticker
Chart 16 — Drawdown plot (strategy) per ticker
"""

from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

SENTINEL_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
CHART_STYLE = {
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#c9d1d9",
    "grid.color":        "#21262d",
    "grid.linewidth":    0.6,
    "legend.facecolor":  "#161b22",
    "legend.edgecolor":  "#30363d",
    "font.size":         11,
}


def _apply_spine_style(ax):
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#30363d")


# ── Chart 13 ───────────────────────────────────────────────────────────────────

def plot_price_signals(
    prices: pd.Series,
    ma_fast: pd.Series,
    ma_slow: pd.Series,
    signal: pd.Series,
    ticker: str = "",
    fast: int = 20,
    slow: int = 50,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Chart 13 — Price with SMA overlays and buy/sell markers.

    Buy  marker (▲) : day signal flips from 0 → 1 (golden cross)
    Sell marker (▼) : day signal flips from 1 → 0 (death cross)
    """
    with plt.rc_context(CHART_STYLE):
        fig, ax = plt.subplots(figsize=(12, 5.5))

        ax.plot(prices.index, prices, color="#8b949e", lw=1.0,
                alpha=0.6, label="Price", zorder=1)
        ax.plot(ma_fast.index, ma_fast, color=SENTINEL_PALETTE[0],
                lw=1.6, label=f"SMA{fast}", zorder=2)
        ax.plot(ma_slow.index, ma_slow, color=SENTINEL_PALETTE[1],
                lw=1.6, label=f"SMA{slow}", zorder=2)

        edges = signal.diff().fillna(0)
        buy_dates  = signal.index[edges == 1]
        sell_dates = signal.index[edges == -1]

        ax.scatter(buy_dates,  prices.reindex(buy_dates),
                   marker="^", color="#2ca02c", s=80, zorder=5,
                   label="Buy (golden cross)")
        ax.scatter(sell_dates, prices.reindex(sell_dates),
                   marker="v", color="#d62728", s=80, zorder=5,
                   label="Sell (death cross)")

        _apply_spine_style(ax)
        ax.set_title(f"Chart 13 — {ticker} Price + SMA{fast}/{slow} with Signals")
        ax.set_ylabel("Price (USD)")
        ax.legend(fontsize=9)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig


# ── Chart 14 ───────────────────────────────────────────────────────────────────

def plot_equity_curves(
    backtests: dict,          # ticker → Backtester (already .run())
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Chart 14 — Strategy vs Buy-and-Hold equity curves, one subplot per ticker.
    """
    tickers = list(backtests.keys())
    n = len(tickers)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    with plt.rc_context(CHART_STYLE):
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5.5 * ncols, 4 * nrows),
                                 squeeze=False)
        fig.suptitle("Chart 14 — Equity Curves: Strategy vs Buy-and-Hold",
                     fontsize=13, y=1.01)

        for idx, ticker in enumerate(tickers):
            row, col = divmod(idx, ncols)
            ax  = axes[row][col]
            bt  = backtests[ticker]
            m   = bt.metrics()

            ax.plot(bt.equity_strat.index, bt.equity_strat,
                    color=SENTINEL_PALETTE[0], lw=1.8,
                    label=f"Strategy  {m['total_return']:+.1%}")
            ax.plot(bt.equity_bh.index, bt.equity_bh,
                    color="#8b949e", lw=1.2, ls="--",
                    label=f"B&H  {m['bh_return']:+.1%}")

            _apply_spine_style(ax)
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
            ax.set_title(ticker, fontsize=11)
            ax.legend(fontsize=8)

        # hide empty subplots
        for idx in range(n, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


# ── Chart 15 ───────────────────────────────────────────────────────────────────

def plot_rolling_sharpe(
    backtests: dict,
    window: int = 60,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Chart 15 — Rolling annualised Sharpe ratio (strategy) per ticker.

    The rolling Sharpe shows *when* the strategy was risk-efficient vs not —
    dips below zero mean the strategy lost money net of risk during that window.
    """
    with plt.rc_context(CHART_STYLE):
        fig, ax = plt.subplots(figsize=(12, 5))

        for i, (ticker, bt) in enumerate(backtests.items()):
            rs = bt.rolling_sharpe.dropna()
            ax.plot(rs.index, rs,
                    color=SENTINEL_PALETTE[i % len(SENTINEL_PALETTE)],
                    lw=1.4, label=ticker)

        ax.axhline(0,   color="#8b949e", lw=0.9, ls="--")
        ax.axhline(1,   color="#2ca02c", lw=0.7, ls=":", alpha=0.6,
                   label="Sharpe = 1 (good)")
        ax.axhline(-1,  color="#d62728", lw=0.7, ls=":", alpha=0.6)

        _apply_spine_style(ax)
        ax.set_title(f"Chart 15 — Rolling {window}-Day Sharpe Ratio (Strategy)")
        ax.set_ylabel("Annualised Sharpe")
        ax.legend(fontsize=9)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig


# ── Chart 16 ───────────────────────────────────────────────────────────────────

def plot_drawdowns(
    backtests: dict,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Chart 16 — Drawdown over time per ticker (strategy).

    Drawdown at time t = (peak_equity - equity_t) / peak_equity
    The deeper the trough, the worse the strategy's worst losing streak.
    """
    tickers = list(backtests.keys())
    n = len(tickers)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    with plt.rc_context(CHART_STYLE):
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5.5 * ncols, 3.5 * nrows),
                                 squeeze=False)
        fig.suptitle("Chart 16 — Strategy Drawdown by Ticker",
                     fontsize=13, y=1.01)

        for idx, ticker in enumerate(tickers):
            row, col = divmod(idx, ncols)
            ax  = axes[row][col]
            bt  = backtests[ticker]
            dd  = bt.drawdown * 100   # convert to %

            ax.fill_between(dd.index, dd, 0,
                            color=SENTINEL_PALETTE[3], alpha=0.45)
            ax.plot(dd.index, dd, color=SENTINEL_PALETTE[3], lw=1.2)

            mdd = bt.metrics()["max_drawdown"] * 100
            ax.axhline(mdd, color=SENTINEL_PALETTE[1], lw=1.0, ls="--",
                       label=f"MDD {mdd:.1f}%")

            _apply_spine_style(ax)
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
            ax.set_title(ticker, fontsize=11)
            ax.legend(fontsize=8)

        for idx in range(n, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
