"""
sentinel/backtest/momentum_plots.py
Phase 2 Deliverable 2 — Momentum Chart Functions (Charts 17–20)

Chart 17 — Momentum score heatmap (trailing return per ticker × month)
Chart 18 — Cross-sectional momentum equity curve vs equal-weight B&H
Chart 19 — Time-series momentum equity curves per ticker
Chart 20 — Strategy correlation matrix (SMA vs TSMOM vs CSMOM)
"""

from __future__ import annotations
from typing import Optional, Dict
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors

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


def _apply_style(ax):
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#30363d")


# ── Chart 17 — Momentum score heatmap ─────────────────────────────────────────

def plot_momentum_heatmap(
    scores: pd.DataFrame,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Chart 17 — Monthly trailing-return heatmap per ticker.

    Each cell = trailing 12-1 month return for ticker i at month-end t.
    Green = positive momentum (buy signal), red = negative (flat).
    """
    # Resample to month-end
    monthly = scores.resample("ME").last()
    monthly_pct = monthly * 100   # convert to %

    with plt.rc_context(CHART_STYLE):
        fig, ax = plt.subplots(figsize=(14, max(3, len(scores.columns) * 0.9 + 1.5)))

        # diverging colormap centred at 0
        cmap = matplotlib.colormaps.get_cmap("RdYlGn")
        vmax = monthly_pct.abs().quantile(0.95).max()
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        im = ax.imshow(
            monthly_pct.T.values,
            aspect="auto",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )

        # axes labels
        ax.set_yticks(range(len(monthly_pct.columns)))
        ax.set_yticklabels(monthly_pct.columns, fontsize=10)

        # x-ticks: every 3 months
        step = max(1, len(monthly_pct) // 10)
        ax.set_xticks(range(0, len(monthly_pct), step))
        ax.set_xticklabels(
            [d.strftime("%b '%y") for d in monthly_pct.index[::step]],
            rotation=45, ha="right", fontsize=8,
        )

        # annotate cells with % value
        for row_i, ticker in enumerate(monthly_pct.columns):
            for col_j, date in enumerate(monthly_pct.index):
                val = monthly_pct.loc[date, ticker]
                if not np.isnan(val):
                    ax.text(col_j, row_i, f"{val:+.0f}%",
                            ha="center", va="center", fontsize=6.5,
                            color="black" if abs(val) < vmax * 0.6 else "white")

        cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
        cbar.set_label("Trailing 12-1m Return (%)", color="#c9d1d9", fontsize=9)
        cbar.ax.yaxis.set_tick_params(color="#8b949e")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#c9d1d9")

        ax.set_title("Chart 17 — Momentum Score Heatmap (12-1 Month Trailing Return)")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


# ── Chart 18 — CS momentum equity curve ───────────────────────────────────────

def plot_csmom_equity(
    csmom_equity: pd.Series,
    bh_equity: pd.Series,
    top_k: int = 2,
    n_total: int = 5,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Chart 18 — Cross-sectional momentum equity vs equal-weight buy-and-hold.
    """
    with plt.rc_context(CHART_STYLE):
        fig, ax = plt.subplots(figsize=(12, 5.5))

        csmom_ret = csmom_equity.iloc[-1] / csmom_equity.iloc[0] - 1
        bh_ret    = bh_equity.iloc[-1]    / bh_equity.iloc[0]    - 1

        ax.plot(csmom_equity.index, csmom_equity,
                color=SENTINEL_PALETTE[0], lw=2,
                label=f"CS-Mom top-{top_k}/{n_total}  {csmom_ret:+.1%}")
        ax.plot(bh_equity.index, bh_equity,
                color="#8b949e", lw=1.4, ls="--",
                label=f"Equal-weight B&H  {bh_ret:+.1%}")

        _apply_style(ax)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.set_title(f"Chart 18 — Cross-Sectional Momentum: Top-{top_k} vs Equal-Weight B&H")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend()
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig


# ── Chart 19 — TS momentum equity curves ──────────────────────────────────────

def plot_tsmom_equities(
    tsmom_equities: Dict[str, pd.Series],
    bh_equities: Dict[str, pd.Series],
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Chart 19 — Time-series momentum equity curve per ticker vs its own B&H.
    """
    tickers = list(tsmom_equities.keys())
    n = len(tickers)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    with plt.rc_context(CHART_STYLE):
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5.5 * ncols, 4 * nrows),
                                 squeeze=False)
        fig.suptitle("Chart 19 — Time-Series Momentum vs Buy-and-Hold per Ticker",
                     fontsize=13, y=1.01)

        for idx, ticker in enumerate(tickers):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            ts  = tsmom_equities[ticker]
            bh  = bh_equities[ticker]
            ts_ret = ts.iloc[-1] / ts.iloc[0] - 1
            bh_ret = bh.iloc[-1] / bh.iloc[0] - 1

            ax.plot(ts.index, ts, color=SENTINEL_PALETTE[0], lw=1.8,
                    label=f"TS-Mom  {ts_ret:+.1%}")
            ax.plot(bh.index, bh, color="#8b949e", lw=1.2, ls="--",
                    label=f"B&H  {bh_ret:+.1%}")

            _apply_style(ax)
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
            ax.set_title(ticker, fontsize=11)
            ax.legend(fontsize=8)

        for idx in range(n, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


# ── Chart 20 — Strategy correlation matrix ────────────────────────────────────

def plot_strategy_correlation(
    returns_dict: Dict[str, pd.Series],
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Chart 20 — Pearson correlation of daily returns across strategies.

    Low correlation between strategies = diversification benefit:
    combining uncorrelated alpha sources reduces portfolio vol without
    proportionally reducing expected return (Markowitz, 1952).
    """
    df = pd.DataFrame(returns_dict).dropna()
    corr = df.corr()

    with plt.rc_context(CHART_STYLE):
        fig, ax = plt.subplots(figsize=(7, 6))

        cmap = matplotlib.colormaps.get_cmap("coolwarm")
        im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1,
                       interpolation="nearest")

        n = len(corr)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(corr.columns, rotation=30, ha="right", fontsize=10)
        ax.set_yticklabels(corr.index, fontsize=10)

        for i in range(n):
            for j in range(n):
                val = corr.iloc[i, j]
                ax.text(j, i, f"{val:.2f}",
                        ha="center", va="center", fontsize=10,
                        color="white" if abs(val) > 0.5 else "#c9d1d9",
                        fontweight="bold" if i == j else "normal")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Pearson ρ", color="#c9d1d9")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#c9d1d9")

        ax.set_title("Chart 20 — Strategy Return Correlation Matrix")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig
