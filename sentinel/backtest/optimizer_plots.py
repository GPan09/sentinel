"""
sentinel/backtest/optimizer_plots.py
Phase 2 Deliverable 3 — Optimizer Chart Functions (Charts 21–24)

Chart 21 — SMA parameter heatmap (Sharpe ratio over fast × slow grid)
Chart 22 — Ensemble strategy equity curve (union + intersection vs B&H)
Chart 23 — Monthly returns calendar heatmap
Chart 24 — Performance scorecard table (all strategies)
"""

from __future__ import annotations
from typing import Dict, Optional
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


# ── Chart 21 — SMA parameter heatmap ──────────────────────────────────────────

def plot_sma_heatmap(
    grid: pd.DataFrame,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Chart 21 — Sharpe ratio heatmap over (fast, slow) SMA parameter grid.

    Bright green = high Sharpe (good), dark red = negative Sharpe (bad).
    The star marks the best (fast, slow) pair.
    """
    with plt.rc_context(CHART_STYLE):
        fig, ax = plt.subplots(figsize=(11, 6))

        data = grid.values.astype(float)
        vmax = np.nanmax(np.abs(data))
        cmap = matplotlib.colormaps.get_cmap("RdYlGn")
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        im = ax.imshow(data, cmap=cmap, norm=norm,
                       aspect="auto", interpolation="nearest")

        # axes
        ax.set_xticks(range(len(grid.columns)))
        ax.set_yticks(range(len(grid.index)))
        ax.set_xticklabels([f"SL{c}" for c in grid.columns], fontsize=8, rotation=45)
        ax.set_yticklabels([f"FA{r}" for r in grid.index], fontsize=8)
        ax.set_xlabel("Slow MA window")
        ax.set_ylabel("Fast MA window")

        # annotate cells
        for i in range(len(grid.index)):
            for j in range(len(grid.columns)):
                val = data[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}",
                            ha="center", va="center", fontsize=7,
                            color="black" if abs(val) < vmax * 0.7 else "white")

        # star on best cell
        best_idx = np.unravel_index(np.nanargmax(data), data.shape)
        ax.scatter(best_idx[1], best_idx[0], marker="*",
                   color="white", s=200, zorder=5,
                   label=f"Best: FA{grid.index[best_idx[0]]}/SL{grid.columns[best_idx[1]]}")

        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Sharpe Ratio (in-sample)", color="#c9d1d9")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#c9d1d9")

        ax.set_title("Chart 21 — SMA Parameter Grid: Sharpe Ratio (Avg over All Tickers)")
        ax.legend(fontsize=9, loc="upper left")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


# ── Chart 22 — Ensemble equity curve ──────────────────────────────────────────

def plot_ensemble_equity(
    equities: Dict[str, pd.Series],
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Chart 22 — Ensemble strategy (union & intersection) vs SMA, TS-Mom, B&H.
    """
    colors = {
        "Ensemble (Union)":        SENTINEL_PALETTE[0],
        "Ensemble (Intersection)": SENTINEL_PALETTE[2],
        "SMA 20/50":               SENTINEL_PALETTE[1],
        "TS-Mom":                  SENTINEL_PALETTE[4],
        "EW B&H":                  "#8b949e",
    }
    styles = {
        "Ensemble (Union)":        (2.2, "-"),
        "Ensemble (Intersection)": (2.0, "-"),
        "SMA 20/50":               (1.4, "--"),
        "TS-Mom":                  (1.4, "--"),
        "EW B&H":                  (1.2, ":"),
    }

    with plt.rc_context(CHART_STYLE):
        fig, ax = plt.subplots(figsize=(13, 5.5))

        for name, eq in equities.items():
            ret = eq.iloc[-1] / eq.iloc[0] - 1
            lw, ls = styles.get(name, (1.4, "-"))
            ax.plot(eq.index, eq,
                    color=colors.get(name, "#c9d1d9"),
                    lw=lw, ls=ls,
                    label=f"{name}  {ret:+.1%}")

        _apply_style(ax)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.set_title("Chart 22 — Ensemble Strategy Equity Curves")
        ax.set_ylabel("Portfolio Value ($10,000 start)")
        ax.legend(fontsize=9)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig


# ── Chart 23 — Monthly returns calendar ───────────────────────────────────────

def plot_monthly_calendar(
    calendar: pd.DataFrame,
    strategy_name: str = "Strategy",
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Chart 23 — Calendar heatmap of monthly % returns.
    Rows = years, columns = months + full-year total.
    """
    with plt.rc_context(CHART_STYLE):
        fig, ax = plt.subplots(figsize=(14, max(3, len(calendar) * 0.7 + 1.5)))

        data = calendar.values.astype(float)
        vmax = np.nanpercentile(np.abs(data[~np.isnan(data)]), 95)
        cmap = matplotlib.colormaps.get_cmap("RdYlGn")
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        im = ax.imshow(data, cmap=cmap, norm=norm,
                       aspect="auto", interpolation="nearest")

        ax.set_xticks(range(len(calendar.columns)))
        ax.set_xticklabels(calendar.columns, fontsize=9)
        ax.set_yticks(range(len(calendar.index)))
        ax.set_yticklabels(calendar.index.astype(str), fontsize=9)

        for i in range(len(calendar.index)):
            for j in range(len(calendar.columns)):
                val = data[i, j]
                if not np.isnan(val):
                    txt = f"{val:+.1f}%"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=7.5,
                            color="black" if abs(val) < vmax * 0.6 else "white",
                            fontweight="bold" if j == len(calendar.columns) - 1 else "normal")

        # bold separator before "Full Year" column
        ax.axvline(len(calendar.columns) - 1.5, color="#c9d1d9", lw=1.2, alpha=0.5)

        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        cbar.set_label("Monthly Return (%)", color="#c9d1d9", fontsize=9)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#c9d1d9")

        ax.set_title(f"Chart 23 — {strategy_name}: Monthly Returns Calendar")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


# ── Chart 24 — Performance scorecard table ────────────────────────────────────

def plot_scorecard_table(
    scorecard: pd.DataFrame,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Chart 24 — Styled table comparing all strategies on all key metrics.
    """
    n_rows = len(scorecard)
    n_cols = len(scorecard.columns)

    with plt.rc_context(CHART_STYLE):
        fig, ax = plt.subplots(figsize=(max(10, n_cols * 1.5), n_rows * 0.7 + 1.5))
        ax.set_axis_off()

        col_widths = [0.22] + [0.12] * n_cols
        table = ax.table(
            cellText   = scorecard.reset_index().values,
            colLabels  = ["Strategy"] + list(scorecard.columns),
            cellLoc    = "center",
            loc        = "center",
            colWidths  = col_widths,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.0)

        # style header
        for j in range(n_cols + 1):
            cell = table[0, j]
            cell.set_facecolor("#21262d")
            cell.set_text_props(color="#c9d1d9", fontweight="bold")
            cell.set_edgecolor("#30363d")

        # style data rows
        for i in range(1, n_rows + 1):
            for j in range(n_cols + 1):
                cell = table[i, j]
                cell.set_facecolor("#161b22" if i % 2 == 0 else "#0d1117")
                cell.set_text_props(color="#c9d1d9")
                cell.set_edgecolor("#21262d")

                # highlight negative Sharpe in red
                if j > 0 and "-" in str(scorecard.columns[j - 1]):
                    pass
                txt = cell.get_text().get_text()
                if txt.startswith("-") and j > 0:
                    cell.set_text_props(color="#d62728")
                elif txt.startswith("+") and j > 0:
                    cell.set_text_props(color="#2ca02c")

        ax.set_title("Chart 24 — Full Strategy Performance Scorecard",
                     pad=12, fontsize=12, color="#c9d1d9")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
