"""
sentinel/utils/plotting.py — Shared visualization utilities.

All plots save to the outputs/ directory AND display inline.
Every function takes a `save_path` parameter so you can control where
the file lands.

Design principle: each chart has exactly one job. Don't combine subplots
in this module — let the phase scripts compose them as needed.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats as scipy_stats

# ── Style ──────────────────────────────────────────────────────────────────────
# A clean, minimal style that looks professional in GitHub READMEs and reports.
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

COLORS = ["#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED"]


def _ensure_dir(path: str) -> None:
    """Create parent directories for a file path if they don't exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


# ── 1. Normalized price chart ──────────────────────────────────────────────────

def plot_prices(
    prices: pd.DataFrame,
    normalize: bool = True,
    save_path: str = "outputs/prices.png",
) -> None:
    """
    Plot price history for all tickers on a single chart.

    When normalize=True (default), we set each series to 100 at the start date.
    This lets you visually compare performance across assets that trade at
    wildly different price levels (e.g., AAPL at ~$180 vs SPY at ~$470).

    Formula: normalized_price_t = (P_t / P_0) × 100

    This is just the cumulative simple return + 1, scaled to start at 100.
    It answers: "If I invested $100 in each asset on day 0, what would it be worth?"
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    data = (prices / prices.iloc[0] * 100) if normalize else prices
    label_suffix = " (normalized to 100)" if normalize else " (USD)"

    for i, col in enumerate(data.columns):
        ax.plot(data.index, data[col], label=col, color=COLORS[i % len(COLORS)], linewidth=1.8)

    ax.set_title("Price History" + label_suffix, fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Value (base 100)" if normalize else "Adjusted Close Price (USD)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend(framealpha=0.8)
    fig.tight_layout()

    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()
    plt.close(fig)


# ── 2. Daily returns time series ───────────────────────────────────────────────

def plot_returns(
    returns: pd.DataFrame,
    save_path: str = "outputs/returns.png",
) -> None:
    """
    Plot daily log returns as a time series.

    This chart immediately reveals two important empirical facts about markets:
      1. Volatility clustering — calm periods and turbulent periods cluster
         together. You'll model this formally with GARCH in Phase 3.
      2. Fat tails — occasional extreme spikes far larger than you'd expect
         from a normal distribution. These are the events that blow up funds.

    Look for the March 2020 COVID crash spike. It should be unmistakable.
    """
    n = len(returns.columns)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, col in enumerate(returns.columns):
        ax = axes[i]
        ax.plot(returns.index, returns[col], color=COLORS[i % len(COLORS)],
                linewidth=0.7, alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
        ax.set_ylabel(f"{col}\nlog return")
        ax.set_title(f"{col} — Daily Log Returns", fontsize=12, fontweight="bold")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    fig.suptitle("Daily Log Returns", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()
    plt.close(fig)


# ── 3. Return distribution with normal overlay ─────────────────────────────────

def plot_return_distribution(
    returns: pd.DataFrame,
    save_path: str = "outputs/distribution.png",
) -> None:
    """
    Histogram of daily log returns with a fitted normal distribution overlay.

    This is one of the most important charts in quantitative finance.
    The gap between the histogram bars and the normal curve tells you
    everything about tail risk.

    The orange curve is a normal distribution with the SAME mean and
    standard deviation as your data. The red dashed lines mark ±2σ,
    which under a normal distribution would capture 95.4% of all days.

    If the histogram bars EXCEED the orange curve at the extremes (fat tails),
    your data has more extreme moves than normal. This almost always happens
    with real stock returns — it's called "excess kurtosis" or leptokurtosis.
    Phase 3 exists to model this phenomenon properly.
    """
    n = len(returns.columns)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, col in enumerate(returns.columns):
        ax = axes[i]
        r = returns[col].dropna()
        mu, sigma = r.mean(), r.std(ddof=1)

        # Histogram (density=True so area integrates to 1, matching the PDF)
        ax.hist(r, bins=80, density=True, alpha=0.55,
                color=COLORS[i % len(COLORS)], edgecolor="none", label="Actual")

        # Fitted normal PDF
        x = np.linspace(r.min(), r.max(), 500)
        ax.plot(x, scipy_stats.norm.pdf(x, mu, sigma),
                color="darkorange", linewidth=2, label="Normal fit")

        # ±2σ reference lines
        for sign in [-1, 1]:
            ax.axvline(mu + sign * 2 * sigma, color="red",
                       linestyle="--", linewidth=1.2, alpha=0.7)

        ax.set_title(f"{col}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Daily log return")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

        # Annotation: show excess kurtosis
        kurt = r.kurt()
        ax.text(0.98, 0.96, f"Excess kurtosis: {kurt:.2f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="dimgray")

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Return Distribution vs Normal", fontsize=14, fontweight="bold")
    fig.tight_layout()

    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()
    plt.close(fig)


# ── 4. Annualized volatility bar chart ─────────────────────────────────────────

def plot_volatility_bar(
    vol: pd.Series,
    save_path: str = "outputs/volatility.png",
) -> None:
    """
    Bar chart comparing annualized volatility across tickers.

    This is a first-pass risk comparison. A stock with σ = 0.30 (30%) means
    that in a typical year, its price moves ±30% around its expected value
    (one standard deviation). Higher vol → more risk → should demand
    higher expected return (this trade-off is the core of Phase 1).

    Rough benchmarks:
      SPY (S&P 500 ETF):        ~15–18% annual vol (low — it's diversified)
      Large-cap tech (AAPL):    ~25–30% annual vol
      Small/mid-cap growth:     ~35–50% annual vol
      Crypto (if you add it):   ~70–100%+ annual vol
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(vol.index, vol.values * 100,   # convert to percentage
                  color=COLORS[:len(vol)], width=0.5, edgecolor="none")

    # Value labels on top of bars
    for bar, v in zip(bars, vol.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{v * 100:.1f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_title("Annualized Volatility (σ)", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Annualized Volatility (%)")
    ax.set_ylim(0, vol.max() * 100 * 1.25)
    fig.tight_layout()

    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()
    plt.close(fig)
