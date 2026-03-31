"""
sentinel/risk/plots.py
======================
Plotting functions for Phase 1: VaR visualisations.

Charts produced
---------------
01_var_comparison.png    — grouped bar chart: 3 VaR methods x 5 tickers at 95%
02_return_histograms.png — return histograms with VaR / ES lines for each ticker
03_var_surface.png       — heatmap: VaR(ticker, confidence) for one method
04_es_vs_var.png         — scatter: Expected Shortfall vs VaR (shows tail severity)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.8,
    "font.family": "DejaVu Sans",
}
TICKERS_COLOR = {
    "AAPL": "#1f77b4",
    "GOOGL": "#d62728",
    "MSFT": "#ff7f0e",
    "BRK-B": "#2ca02c",
    "SPY": "#9467bd",
}
METHOD_COLOR = {
    "Historical": "#1f77b4",
    "Parametric": "#ff7f0e",
    "Monte Carlo": "#2ca02c",
}


def plot_var_comparison(
    var_df: pd.DataFrame,
    confidence: float = 0.95,
    save_path: str = "01_var_comparison.png",
) -> None:
    """
    Grouped bar chart comparing Historical, Parametric, and Monte Carlo VaR
    across all tickers at the given confidence level.
    """
    plt.rcParams.update(STYLE)
    subset = var_df[var_df["confidence"] == confidence].copy()
    tickers = subset["ticker"].tolist()
    n = len(tickers)
    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))

    bars_h = ax.bar(x - width, subset["hist_var"],  width, label="Historical",
                    color=METHOD_COLOR["Historical"],  alpha=0.85, edgecolor="white")
    bars_p = ax.bar(x,         subset["param_var"], width, label="Parametric",
                    color=METHOD_COLOR["Parametric"],  alpha=0.85, edgecolor="white")
    bars_m = ax.bar(x + width, subset["mc_var"],    width, label="Monte Carlo",
                    color=METHOD_COLOR["Monte Carlo"], alpha=0.85, edgecolor="white")

    for bars in (bars_h, bars_p, bars_m):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.0002,
                    f"{h:.2%}", ha="center", va="bottom", fontsize=8, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(tickers, fontsize=11)
    ax.set_ylabel("1-Day VaR (log return magnitude)", fontsize=11)
    ax.set_title(
        f"Value at Risk — Three Methods at {confidence:.0%} Confidence\n"
        "VaR = loss not exceeded with probability alpha",
        fontsize=13, pad=12,
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax.legend(fontsize=10)

    pct_diffs = subset["param_vs_hist_pct"].values
    worst_idx = int(np.argmin(pct_diffs))
    worst_pct = pct_diffs[worst_idx]
    if worst_pct < -2:
        ax.annotate(
            f"Parametric under-estimates\nhistorical by {abs(worst_pct):.1f}%\n(fat tails!)",
            xy=(x[worst_idx] - width, subset["hist_var"].iloc[worst_idx]),
            xytext=(x[worst_idx] - width - 0.6, subset["hist_var"].max() * 0.85),
            fontsize=8, color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_return_histograms(
    returns: pd.DataFrame,
    var_df: pd.DataFrame,
    confidence: float = 0.95,
    save_path: str = "02_return_histograms.png",
) -> None:
    """
    For each ticker: histogram of daily returns with normal fit overlay
    and vertical lines marking Historical VaR, Parametric VaR, and ES.
    """
    plt.rcParams.update(STYLE)
    tickers = returns.columns.tolist()
    n = len(tickers)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.5 * nrows))
    axes = axes.flatten()

    subset = var_df[var_df["confidence"] == confidence]

    for i, ticker in enumerate(tickers):
        ax = axes[i]
        r = returns[ticker].dropna().values
        row = subset[subset["ticker"] == ticker].iloc[0]

        ax.hist(r, bins=80, density=True,
                color=TICKERS_COLOR.get(ticker, "#888"),
                alpha=0.6, edgecolor="none", label="Observed returns")

        mu, sigma = r.mean(), r.std(ddof=1)
        x_grid = np.linspace(r.min(), r.max(), 400)
        ax.plot(x_grid, stats.norm.pdf(x_grid, mu, sigma),
                "k--", lw=1.5, alpha=0.7, label="Normal fit")

        ax.axvline(-row["hist_var"],  color="#1f77b4", lw=1.8, linestyle="-",
                   label=f"Hist VaR  {row['hist_var']:.2%}")
        ax.axvline(-row["param_var"], color="#ff7f0e", lw=1.8, linestyle="--",
                   label=f"Param VaR {row['param_var']:.2%}")
        ax.axvline(-row["hist_es"],   color="#1f77b4", lw=1.2, linestyle=":",
                   label=f"Hist ES   {row['hist_es']:.2%}")

        ax.set_title(ticker, fontsize=12, fontweight="bold")
        ax.set_xlabel("Daily log return", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Return Distributions with VaR Lines ({confidence:.0%} confidence)\n"
        "Fat tails: historical VaR > parametric VaR",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_var_surface(
    var_df: pd.DataFrame,
    method: str = "hist",
    save_path: str = "03_var_surface.png",
) -> None:
    """
    Heatmap showing VaR across tickers (rows) and confidence levels (columns).
    """
    plt.rcParams.update(STYLE)
    col_map = {"hist": "hist_var", "param": "param_var", "mc": "mc_var"}
    method_label = {"hist": "Historical", "param": "Parametric", "mc": "Monte Carlo"}

    col = col_map[method]
    pivot = var_df.pivot(index="ticker", columns="confidence", values=col)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.0%}" for c in pivot.columns], fontsize=11)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=11)

    for row_i in range(pivot.shape[0]):
        for col_j in range(pivot.shape[1]):
            val = pivot.values[row_i, col_j]
            text_color = "white" if val > pivot.values.max() * 0.7 else "black"
            ax.text(col_j, row_i, f"{val:.2%}", ha="center", va="center",
                    fontsize=10, color=text_color)

    plt.colorbar(im, ax=ax, label="1-Day VaR")
    ax.set_xlabel("Confidence Level", fontsize=11)
    ax.set_ylabel("Ticker", fontsize=11)
    ax.set_title(
        f"VaR Surface — {method_label[method]} Method\n"
        "Higher confidence level -> deeper tail cut -> larger VaR",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_es_vs_var(
    var_df: pd.DataFrame,
    confidence: float = 0.95,
    save_path: str = "04_es_vs_var.png",
) -> None:
    """
    Scatter of Expected Shortfall vs VaR (Historical method).
    ES/VaR ratio shows tail severity beyond the VaR threshold.
    """
    plt.rcParams.update(STYLE)
    subset = var_df[var_df["confidence"] == confidence].copy()
    subset["es_var_ratio"] = subset["hist_es"] / subset["hist_var"]

    fig, ax = plt.subplots(figsize=(8, 6))

    for _, row in subset.iterrows():
        ticker = row["ticker"]
        color = TICKERS_COLOR.get(ticker, "#888")
        ax.scatter(row["hist_var"], row["hist_es"], s=180,
                   color=color, zorder=5, edgecolors="white", linewidths=1)
        ax.annotate(
            f"{ticker}\nES/VaR={row['es_var_ratio']:.2f}",
            xy=(row["hist_var"], row["hist_es"]),
            xytext=(row["hist_var"] + 0.0003, row["hist_es"] + 0.0003),
            fontsize=9, color=color,
        )

    lim_min = subset["hist_var"].min() * 0.95
    lim_max = subset["hist_es"].max() * 1.1
    ref = np.linspace(lim_min, lim_max, 100)
    ax.plot(ref, ref, "k--", lw=1, alpha=0.4, label="ES = VaR (reference)")

    ax.set_xlabel("Historical VaR (1-day, log return)", fontsize=11)
    ax.set_ylabel("Historical Expected Shortfall (CVaR)", fontsize=11)
    ax.set_title(
        f"Expected Shortfall vs VaR — {confidence:.0%} confidence\n"
        "ES/VaR > 1.2 indicates significant tail mass beyond VaR threshold",
        fontsize=13,
    )
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
