"""
sentinel/crash/spike_plots.py
Phase 3 Deliverable 3 — Correlation Spike Chart Functions (Charts 33–36)

Chart 33 — APC over time with spike flags and regime shading
Chart 34 — Rolling pairwise correlation heatmap (month-end snapshots)
Chart 35 — Conditional VaR: low vs medium vs high correlation regime
Chart 36 — Crash warning dashboard (composite: APC + vol + drawdown)
"""

from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

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


# ── Chart 33 — APC with spikes ────────────────────────────────────────────────

def plot_apc_spikes(
    apc: pd.Series,
    spikes: pd.Series,
    regime: pd.Series,
    episodes: pd.DataFrame,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Chart 33 — Average Pairwise Correlation over time.
    Regime shading: green=low, yellow=medium, red=high.
    Spike days marked with vertical red lines.
    """
    with plt.rc_context(CHART_STYLE):
        fig, ax = plt.subplots(figsize=(13, 5.5))

        # regime background shading
        reg_colors = {0: "#2ca02c", 1: "#ff7f0e", 2: "#d62728"}
        reg_alpha  = {0: 0.08,      1: 0.08,      2: 0.15}
        prev_reg   = None
        seg_start  = apc.dropna().index[0]

        reg_aligned = regime.reindex(apc.dropna().index).ffill()
        for date, reg in reg_aligned.items():
            if reg != prev_reg:
                if prev_reg is not None:
                    ax.axvspan(seg_start, date,
                               color=reg_colors[prev_reg],
                               alpha=reg_alpha[prev_reg])
                seg_start = date
                prev_reg  = reg
        if prev_reg is not None:
            ax.axvspan(seg_start, apc.dropna().index[-1],
                       color=reg_colors[prev_reg],
                       alpha=reg_alpha[prev_reg])

        # APC line
        apc_clean = apc.dropna()
        ax.plot(apc_clean.index, apc_clean,
                color=SENTINEL_PALETTE[0], lw=1.8, label="APC (60-day rolling)")

        # spike markers
        spike_dates = spikes[spikes].index
        for sd in spike_dates:
            ax.axvline(sd, color=SENTINEL_PALETTE[3],
                       lw=0.6, alpha=0.5)
        if len(spike_dates):
            ax.axvline(spike_dates[0], color=SENTINEL_PALETTE[3],
                       lw=0.6, alpha=0.5, label=f"Spike ({len(spike_dates)} days)")

        # mean + threshold
        mu  = apc_clean.mean()
        sig = apc_clean.std()
        ax.axhline(mu,         color="#8b949e", lw=0.8, ls="--", label=f"Mean={mu:.2f}")
        ax.axhline(mu+1.5*sig, color=SENTINEL_PALETTE[3], lw=0.8, ls=":",
                   label=f"Spike threshold={mu+1.5*sig:.2f}")

        # legend patches for regime
        patches = [
            mpatches.Patch(color="#2ca02c", alpha=0.4, label="Low corr regime"),
            mpatches.Patch(color="#ff7f0e", alpha=0.4, label="Medium"),
            mpatches.Patch(color="#d62728", alpha=0.4, label="High corr regime ⚠️"),
        ]
        _style(ax)
        ax.set_title("Chart 33 — Average Pairwise Correlation + Spike Detection")
        ax.set_ylabel("Avg Pairwise Correlation")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles + patches, labels + [p.get_label() for p in patches],
                  fontsize=8, ncol=2)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


# ── Chart 34 — Rolling correlation heatmap snapshots ─────────────────────────

def plot_corr_heatmap_snapshots(
    prices: pd.DataFrame,
    window: int = 60,
    n_snapshots: int = 4,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Chart 34 — Correlation matrix at N evenly-spaced month-end dates.
    Shows how inter-asset correlations evolve over time.
    """
    log_rets  = np.log(prices / prices.shift(1)).dropna()
    month_ends = log_rets.resample("ME").last().index
    step       = max(1, len(month_ends) // n_snapshots)
    dates      = month_ends[::step][:n_snapshots]

    with plt.rc_context(CHART_STYLE):
        fig, axes = plt.subplots(1, n_snapshots,
                                 figsize=(4.5 * n_snapshots, 4.5),
                                 squeeze=False)
        fig.suptitle("Chart 34 — Correlation Matrix Snapshots Over Time",
                     fontsize=13)

        cmap = matplotlib.colormaps.get_cmap("RdYlGn")
        tickers = prices.columns.tolist()

        for col, date in enumerate(dates):
            ax   = axes[0][col]
            mask = (log_rets.index <= date) & \
                   (log_rets.index > date - pd.Timedelta(days=window * 2))
            window_rets = log_rets[mask]
            if len(window_rets) < 10:
                ax.set_visible(False)
                continue

            corr = window_rets.corr().values
            im   = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1,
                             interpolation="nearest")

            ax.set_xticks(range(len(tickers)))
            ax.set_yticks(range(len(tickers)))
            ax.set_xticklabels(tickers, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(tickers, fontsize=8)

            for i in range(len(tickers)):
                for j in range(len(tickers)):
                    ax.text(j, i, f"{corr[i,j]:.2f}",
                            ha="center", va="center", fontsize=7,
                            color="black" if abs(corr[i,j]) < 0.6 else "white")

            ax.set_title(date.strftime("%b %Y"), fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


# ── Chart 35 — Conditional VaR by regime ──────────────────────────────────────

def plot_conditional_var(
    cond_var_df: pd.DataFrame,
    confidence: float = 0.95,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Chart 35 — VaR and CVaR side-by-side for each correlation regime.
    High-regime VaR >> Low-regime VaR: regime matters enormously for risk.
    """
    with plt.rc_context(CHART_STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Chart 35 — Conditional VaR by Correlation Regime ({confidence:.0%})",
                     fontsize=13)

        regimes = cond_var_df.index.tolist()
        x = np.arange(len(regimes))
        reg_colors = [SENTINEL_PALETTE[2], SENTINEL_PALETTE[1], SENTINEL_PALETTE[3]]

        for ax, col, label in [(ax1, "var", "VaR"), (ax2, "cvar", "CVaR")]:
            vals = cond_var_df[col] * 100
            bars = ax.bar(x, vals, color=reg_colors[:len(regimes)],
                          edgecolor="#0d1117", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(regimes, fontsize=9)
            ax.set_ylabel(f"{label} (%)")
            ax.set_title(f"{label} per Regime")
            _style(ax)

            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.05,
                        f"{val:.2f}%", ha="center", va="bottom",
                        fontsize=9, color="#c9d1d9")

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig


# ── Chart 36 — Composite crash dashboard ─────────────────────────────────────

def plot_crash_dashboard(
    prices: pd.DataFrame,
    apc: pd.Series,
    spikes: pd.Series,
    garch_vols: Optional[Dict[str, pd.Series]] = None,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Chart 36 — 3-panel composite crash warning dashboard:
      Top:    SPY price + drawdown shading
      Middle: APC with spike flags
      Bottom: Equal-weight portfolio rolling 30-day volatility
    """
    log_rets = np.log(prices / prices.shift(1)).dropna()
    port_r   = log_rets.mean(axis=1)
    eq_cum   = np.exp(port_r.cumsum())
    drawdown = (eq_cum - eq_cum.cummax()) / eq_cum.cummax() * 100

    roll_vol = port_r.rolling(30).std() * np.sqrt(252) * 100

    with plt.rc_context(CHART_STYLE):
        fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True,
                                 gridspec_kw={"height_ratios": [1.5, 1, 1]})
        fig.suptitle("Chart 36 — Composite Crash Warning Dashboard", fontsize=13)

        # Panel 1: portfolio cumulative return + drawdown shading
        ax1 = axes[0]
        ax1.plot(eq_cum.index, eq_cum, color=SENTINEL_PALETTE[0], lw=1.6,
                 label="EW Portfolio (cumulative)")
        ax1b = ax1.twinx()
        ax1b.fill_between(drawdown.index, drawdown, 0,
                          color=SENTINEL_PALETTE[3], alpha=0.3)
        ax1b.set_ylabel("Drawdown (%)", color=SENTINEL_PALETTE[3])
        ax1b.tick_params(axis="y", colors=SENTINEL_PALETTE[3])
        ax1b.spines[["top"]].set_visible(False)
        _style(ax1)
        ax1.set_ylabel("Cumulative Return")
        ax1.legend(fontsize=9)

        # Panel 2: APC + spikes
        ax2 = axes[1]
        apc_clean = apc.dropna()
        ax2.plot(apc_clean.index, apc_clean,
                 color=SENTINEL_PALETTE[1], lw=1.6, label="APC")
        spike_days = spikes[spikes].reindex(apc_clean.index)
        ax2.scatter(spike_days.dropna().index,
                    apc_clean.reindex(spike_days.dropna().index),
                    color=SENTINEL_PALETTE[3], s=25, zorder=5,
                    label="Spike")
        mu = apc_clean.mean(); sig = apc_clean.std()
        ax2.axhline(mu + 1.5*sig, color=SENTINEL_PALETTE[3],
                    lw=0.8, ls=":", alpha=0.8)
        _style(ax2)
        ax2.set_ylabel("Avg Pairwise Corr")
        ax2.legend(fontsize=9)

        # Panel 3: rolling volatility
        ax3 = axes[2]
        ax3.fill_between(roll_vol.index, roll_vol, 0,
                         color=SENTINEL_PALETTE[4], alpha=0.4)
        ax3.plot(roll_vol.index, roll_vol,
                 color=SENTINEL_PALETTE[4], lw=1.4, label="30d Rolling Vol (ann. %)")
        _style(ax3)
        ax3.set_ylabel("Ann. Vol (%)")
        ax3.legend(fontsize=9)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
