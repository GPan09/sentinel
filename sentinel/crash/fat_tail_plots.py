"""
sentinel/crash/fat_tail_plots.py
Phase 3 Deliverable 2 — Fat-Tail Chart Functions (Charts 29–32)

Chart 29 — Return distribution: empirical vs Normal vs Student-t fits (per ticker)
Chart 30 — QQ plots: Normal vs Student-t (shows which fits the tails better)
Chart 31 — Tail risk dashboard: excess kurtosis + skewness + tail ratios across tickers
Chart 32 — VaR underestimation: Normal VaR vs empirical VaR at 95% and 99%
"""

from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec

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


def _style(ax):
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#30363d")


# ── Chart 29 — Distribution fits ──────────────────────────────────────────────

def plot_distribution_fits(
    log_rets: pd.DataFrame,
    fits: Dict[str, Dict[str, dict]],
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Chart 29 — For each ticker: histogram of daily returns with Normal and
    Student-t PDF overlays. Fat tails visible where histogram exceeds Normal.
    """
    tickers = list(fits.keys())
    n = len(tickers)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    with plt.rc_context(CHART_STYLE):
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5.5 * ncols, 4 * nrows),
                                 squeeze=False)
        fig.suptitle("Chart 29 — Return Distributions: Empirical vs Normal vs Student-t",
                     fontsize=13, y=1.01)

        for idx, ticker in enumerate(tickers):
            row, col = divmod(idx, ncols)
            ax  = axes[row][col]
            r   = log_rets[ticker].dropna() * 100   # in %
            ticker_fits = fits[ticker]

            ax.hist(r, bins=80, density=True,
                    color=SENTINEL_PALETTE[0], alpha=0.45, label="Empirical")

            xs = np.linspace(r.quantile(0.001), r.quantile(0.999), 400)

            # Normal fit
            nf = ticker_fits["Normal"]
            ax.plot(xs, norm.pdf(xs, nf["mu"]*100, nf["sigma"]*100),
                    color="#8b949e", lw=1.8, ls="--", label="Normal")

            # Student-t fit
            tf = ticker_fits["Student-t"]
            d  = tf["scipy_dist"]
            ax.plot(xs, d.pdf(xs, tf["df"], tf["loc"]*100, tf["scale"]*100),
                    color=SENTINEL_PALETTE[1], lw=2.0,
                    label=f"Student-t  ν={tf['df']:.1f}")

            _style(ax)
            ax.set_title(ticker, fontsize=11)
            ax.set_xlabel("Daily Return (%)")
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)

            # stat box
            from scipy.stats import jarque_bera
            jb_p = jarque_bera(log_rets[ticker].dropna())[1]
            kurt = log_rets[ticker].dropna().kurtosis()
            ax.text(0.03, 0.97,
                    f"ex.kurt={kurt:.2f}\nJB p={jb_p:.3f}",
                    transform=ax.transAxes, fontsize=8, va="top",
                    color="#c9d1d9",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="#21262d", alpha=0.8))

        for idx in range(n, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


# ── Chart 30 — QQ plots ────────────────────────────────────────────────────────

def plot_qq(
    log_rets: pd.DataFrame,
    fits: Dict[str, Dict[str, dict]],
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Chart 30 — QQ plots comparing Normal and Student-t fits.
    Each ticker gets two panels: Normal QQ (left) and Student-t QQ (right).
    S-shaped deviations from the 45° line = fat tails not captured by Normal.
    """
    from sentinel.crash.fat_tails import qq_data
    tickers = list(fits.keys())
    n = len(tickers)

    with plt.rc_context(CHART_STYLE):
        fig, axes = plt.subplots(n, 2,
                                 figsize=(11, 3.5 * n),
                                 squeeze=False)
        fig.suptitle("Chart 30 — QQ Plots: Normal vs Student-t Fit",
                     fontsize=13, y=1.01)

        for idx, ticker in enumerate(tickers):
            r = log_rets[ticker].dropna()
            for col, dist_name in enumerate(["Normal", "Student-t"]):
                ax = axes[idx][col]
                th, emp = qq_data(r, fits[ticker][dist_name])

                ax.scatter(th, emp * 100, s=4,
                           color=SENTINEL_PALETTE[0], alpha=0.5)

                # 45° reference line
                lo = min(th.min(), (emp * 100).min())
                hi = max(th.max(), (emp * 100).max())
                ax.plot([lo, hi], [lo*100 if dist_name=="Normal" else lo,
                                   hi*100 if dist_name=="Normal" else hi],
                        color="#8b949e", lw=1.2, ls="--")

                # cleaner: just use the data range
                emp_pct = emp * 100
                lims = [min(th.min(), emp_pct.min()),
                        max(th.max(), emp_pct.max())]
                ax.plot(lims, lims, color="#8b949e", lw=1.2, ls="--",
                        label="Perfect fit")

                _style(ax)
                ax.set_xlabel(f"Theoretical ({dist_name})")
                ax.set_ylabel("Empirical (%)")
                ax.set_title(f"{ticker} — {dist_name} QQ", fontsize=10)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


# ── Chart 31 — Tail risk dashboard ────────────────────────────────────────────

def plot_tail_dashboard(
    metrics_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Chart 31 — 4-panel dashboard: excess kurtosis, skewness,
    3σ tail ratio, and 4σ tail ratio across all tickers.

    Tail ratio > 1 means crashes happen more often than Normal predicts.
    """
    with plt.rc_context(CHART_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Chart 31 — Tail Risk Dashboard", fontsize=13)

        tickers = metrics_df.index.tolist()
        x = np.arange(len(tickers))
        bar_kw = dict(width=0.6, edgecolor="#0d1117", linewidth=0.5)

        # excess kurtosis
        ax = axes[0][0]
        vals = metrics_df["excess_kurtosis"]
        colors = [SENTINEL_PALETTE[3] if v > 0 else SENTINEL_PALETTE[2] for v in vals]
        ax.bar(x, vals, color=colors, **bar_kw)
        ax.axhline(0, color="#8b949e", lw=0.8, ls="--")
        ax.set_xticks(x); ax.set_xticklabels(tickers)
        ax.set_title("Excess Kurtosis  (>0 = fat tails)")
        _style(ax)

        # skewness
        ax = axes[0][1]
        vals = metrics_df["skewness"]
        colors = [SENTINEL_PALETTE[3] if v < 0 else SENTINEL_PALETTE[2] for v in vals]
        ax.bar(x, vals, color=colors, **bar_kw)
        ax.axhline(0, color="#8b949e", lw=0.8, ls="--")
        ax.set_xticks(x); ax.set_xticklabels(tickers)
        ax.set_title("Skewness  (<0 = crash-prone)")
        _style(ax)

        # 3σ tail ratio
        ax = axes[1][0]
        vals = metrics_df["tail_ratio_3sigma"]
        ax.bar(x, vals, color=SENTINEL_PALETTE[1], **bar_kw)
        ax.axhline(1, color="#8b949e", lw=1.2, ls="--", label="Normal baseline")
        ax.set_xticks(x); ax.set_xticklabels(tickers)
        ax.set_title("3σ Tail Ratio  (empirical / Normal)")
        ax.legend(fontsize=8)
        _style(ax)

        # 4σ tail ratio
        ax = axes[1][1]
        vals = metrics_df["tail_ratio_4sigma"]
        ax.bar(x, vals, color=SENTINEL_PALETTE[4], **bar_kw)
        ax.axhline(1, color="#8b949e", lw=1.2, ls="--", label="Normal baseline")
        ax.set_xticks(x); ax.set_xticklabels(tickers)
        ax.set_title("4σ Tail Ratio  (empirical / Normal)")
        ax.legend(fontsize=8)
        _style(ax)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


# ── Chart 32 — VaR underestimation ────────────────────────────────────────────

def plot_var_underestimation(
    metrics_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Chart 32 — Normal VaR vs Empirical VaR at 95% and 99%.

    When Normal VaR < Empirical VaR → Normal is underestimating risk.
    The gap widens at 99% because fat tails matter most in the deep tail.
    """
    with plt.rc_context(CHART_STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Chart 32 — Normal VaR vs Empirical VaR: Underestimation of Tail Risk",
                     fontsize=13)

        tickers = metrics_df.index.tolist()
        x = np.arange(len(tickers))
        w = 0.35

        for ax, conf, norm_col, emp_col in [
            (ax1, "95%", "var_95_normal", "var_95_empirical"),
            (ax2, "99%", "var_99_normal", "var_99_empirical"),
        ]:
            norm_vals = metrics_df[norm_col] * 100
            emp_vals  = metrics_df[emp_col]  * 100

            ax.bar(x - w/2, norm_vals, w, label="Normal VaR",
                   color=SENTINEL_PALETTE[0], edgecolor="#0d1117", lw=0.5)
            ax.bar(x + w/2, emp_vals,  w, label="Empirical VaR",
                   color=SENTINEL_PALETTE[3], edgecolor="#0d1117", lw=0.5)

            # underestimation arrows
            for i, (nv, ev) in enumerate(zip(norm_vals, emp_vals)):
                if ev > nv:
                    ax.annotate("", xy=(i + w/2, ev), xytext=(i - w/2, nv),
                                arrowprops=dict(arrowstyle="->",
                                                color="#ff7f0e", lw=1.2))

            ax.set_xticks(x)
            ax.set_xticklabels(tickers)
            ax.set_ylabel("VaR (%)")
            ax.set_title(f"{conf} Confidence VaR")
            ax.legend(fontsize=9)
            _style(ax)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig
