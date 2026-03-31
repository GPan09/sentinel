"""sentinel/risk/mc_plots.py
Phase 1 · Deliverable 3 — Monte Carlo chart functions

Charts produced
---------------
09  P&L distribution   – histogram + KDE + VaR / CVaR vertical lines
10  Portfolio fan chart – 5th / 25th / 75th / 95th percentile bands + median
11  VaR convergence     – VaR & CVaR vs number of simulations
12  Weight pie          – equal-weight breakdown for the simulated portfolio
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

SENTINEL_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
CHART_STYLE = {"figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
               "axes.edgecolor": "#30363d", "axes.labelcolor": "#c9d1d9",
               "text.color": "#c9d1d9", "xtick.color": "#c9d1d9",
               "ytick.color": "#c9d1d9", "grid.color": "#21262d",
               "grid.alpha": 0.6}


def _apply_style(fig, ax_or_axes):
    fig.patch.set_facecolor(CHART_STYLE["figure.facecolor"])
    axes = ax_or_axes if isinstance(ax_or_axes, (list, np.ndarray)) else [ax_or_axes]
    for ax in np.array(axes).ravel():
        ax.set_facecolor(CHART_STYLE["axes.facecolor"])
        ax.tick_params(colors=CHART_STYLE["xtick.color"])
        ax.xaxis.label.set_color(CHART_STYLE["axes.labelcolor"])
        ax.yaxis.label.set_color(CHART_STYLE["axes.labelcolor"])
        ax.title.set_color(CHART_STYLE["text.color"])
        for spine in ax.spines.values():
            spine.set_edgecolor(CHART_STYLE["axes.edgecolor"])
        ax.grid(True, color=CHART_STYLE["grid.color"],
                alpha=CHART_STYLE["grid.alpha"], linewidth=0.5)


# ── Chart 09 ──────────────────────────────────────────────────────────────────
def plot_pnl_distribution(
    terminal_pnl: np.ndarray,
    var: float,
    cvar: float,
    confidence: float = 0.95,
    save_path=None,
) -> plt.Figure:
    """Histogram + KDE of simulated 1-day portfolio log-returns with VaR/CVaR."""
    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_style(fig, ax)

    # Histogram
    ax.hist(terminal_pnl, bins=100, density=True,
            color=SENTINEL_PALETTE[0], alpha=0.45, label="Simulated P&L")

    # KDE overlay
    kde = gaussian_kde(terminal_pnl, bw_method="scott")
    xs = np.linspace(terminal_pnl.min(), terminal_pnl.max(), 400)
    ax.plot(xs, kde(xs), color=SENTINEL_PALETTE[0], linewidth=2)

    # VaR / CVaR verticals
    ax.axvline(-var,  color="#d62728", linewidth=1.8, linestyle="--",
               label=f"VaR {confidence:.0%}  = {var:.4f}")
    ax.axvline(-cvar, color="#ff7f0e", linewidth=1.8, linestyle=":",
               label=f"CVaR {confidence:.0%} = {cvar:.4f}")

    # Shade tail
    tail_x = xs[xs <= -var]
    ax.fill_between(tail_x, kde(tail_x), color="#d62728", alpha=0.25)

    ax.set_xlabel("1-Day Portfolio Log-Return")
    ax.set_ylabel("Density")
    ax.set_title("Chart 09 · Monte Carlo P&L Distribution", fontweight="bold")
    ax.legend(framealpha=0.2, labelcolor="#c9d1d9")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


# ── Chart 10 ──────────────────────────────────────────────────────────────────
def plot_portfolio_fan(
    paths: np.ndarray,
    save_path=None,
) -> plt.Figure:
    """Fan chart of simulated cumulative portfolio log-return paths."""
    fig, ax = plt.subplots(figsize=(11, 5))
    _apply_style(fig, ax)

    days = np.arange(paths.shape[1]) + 1
    p5,  p25 = np.percentile(paths, 5,  axis=0), np.percentile(paths, 25, axis=0)
    p75, p95 = np.percentile(paths, 75, axis=0), np.percentile(paths, 95, axis=0)
    med       = np.median(paths, axis=0)

    ax.fill_between(days, p5,  p95, color=SENTINEL_PALETTE[0], alpha=0.15, label="5–95 pct")
    ax.fill_between(days, p25, p75, color=SENTINEL_PALETTE[0], alpha=0.30, label="25–75 pct")
    ax.plot(days, med, color=SENTINEL_PALETTE[0], linewidth=2, label="Median")
    ax.axhline(0, color="#c9d1d9", linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Cumulative Log-Return")
    ax.set_title("Chart 10 · Portfolio Simulation Fan Chart", fontweight="bold")
    ax.legend(framealpha=0.2, labelcolor="#c9d1d9")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


# ── Chart 11 ──────────────────────────────────────────────────────────────────
def plot_var_convergence(
    convergence_df: pd.DataFrame,
    confidence: float = 0.95,
    save_path=None,
) -> plt.Figure:
    """VaR and CVaR estimates as a function of number of simulations."""
    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_style(fig, ax)

    ax.plot(convergence_df["n_sims"], convergence_df["var"],
            color="#d62728", linewidth=2, label=f"VaR {confidence:.0%}")
    ax.plot(convergence_df["n_sims"], convergence_df["cvar"],
            color="#ff7f0e", linewidth=2, linestyle="--",
            label=f"CVaR {confidence:.0%}")

    # Asymptotic reference (last value)
    ax.axhline(convergence_df["var"].iloc[-1],  color="#d62728",
               alpha=0.3, linewidth=1, linestyle=":")
    ax.axhline(convergence_df["cvar"].iloc[-1], color="#ff7f0e",
               alpha=0.3, linewidth=1, linestyle=":")

    ax.set_xscale("log")
    ax.set_xlabel("Number of Simulations (log scale)")
    ax.set_ylabel("VaR / CVaR (log-return units)")
    ax.set_title("Chart 11 · VaR Convergence vs Simulation Count", fontweight="bold")
    ax.legend(framealpha=0.2, labelcolor="#c9d1d9")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


# ── Chart 12 ──────────────────────────────────────────────────────────────────
def plot_portfolio_weights(
    weights: np.ndarray,
    tickers: list,
    save_path=None,
) -> plt.Figure:
    """Pie chart of portfolio weights used in the Monte Carlo simulation."""
    fig, ax = plt.subplots(figsize=(7, 7))
    _apply_style(fig, ax)

    wedges, texts, autotexts = ax.pie(
        weights,
        labels=tickers,
        autopct="%1.1f%%",
        colors=SENTINEL_PALETTE * (len(tickers) // len(SENTINEL_PALETTE) + 1),
        startangle=90,
        wedgeprops={"linewidth": 0.5, "edgecolor": "#0d1117"},
    )
    for t in texts + autotexts:
        t.set_color("#c9d1d9")

    ax.set_title("Chart 12 · Portfolio Weights (Monte Carlo Simulation)",
                 fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig
