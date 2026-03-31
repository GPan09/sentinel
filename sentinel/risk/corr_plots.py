"""
Sentinel – Phase 1, Deliverable 2: Correlation Visualisations
=============================================================
Four publication-quality charts:
  05  correlation heatmap
  06  rolling correlations vs SPY
  07  stress vs normal correlation comparison
  08  eigenvalue scree plot (PCA)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional


SENTINEL_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


# ── 05 ─ correlation heatmap ────────────────────────────────────

def plot_correlation_heatmap(
    corr: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Annotated heatmap of the full Pearson correlation matrix."""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    n = len(corr)
    cmap = plt.cm.RdYlGn
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    im = ax.imshow(corr.values, cmap=cmap, norm=norm, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Pearson ρ", fontsize=11)

    tickers = corr.columns.tolist()
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tickers, fontsize=11)
    ax.set_yticklabels(tickers, fontsize=11)

    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.7 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    ax.set_title("Correlation Matrix – 2-Year Daily Log Returns",
                 fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig


# ── 06 ─ rolling correlations ──────────────────────────────────

def plot_rolling_correlations(
    rolling: pd.DataFrame,
    window: int = 30,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Time-series of rolling correlation of each ticker vs SPY."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, col in enumerate(rolling.columns):
        ax.plot(rolling.index, rolling[col],
                label=col, color=SENTINEL_PALETTE[i % len(SENTINEL_PALETTE)],
                linewidth=1.2, alpha=0.85)

    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.set_ylabel("Correlation with SPY", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_title(f"{window}-Day Rolling Correlation vs SPY",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower left")
    ax.set_ylim(-0.3, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig


# ── 07 ─ stress vs normal ──────────────────────────────────────

def plot_stress_vs_normal(
    stress_corr: pd.DataFrame,
    normal_corr: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Grouped bar chart comparing off-diagonal correlations
    during stress days vs normal days.
    """
    tickers = stress_corr.columns.tolist()
    n = len(tickers)
    pairs, stress_vals, normal_vals = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(f"{tickers[i]}\n{tickers[j]}")
            stress_vals.append(stress_corr.iloc[i, j])
            normal_vals.append(normal_corr.iloc[i, j])

    x = np.arange(len(pairs))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - w / 2, stress_vals, w, label="Stress (SPY ≤ −2%)",
           color="#d62728", alpha=0.85)
    ax.bar(x + w / 2, normal_vals, w, label="Normal days",
           color="#2ca02c", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(pairs, fontsize=9)
    ax.set_ylabel("Pearson ρ", fontsize=11)
    ax.set_title("Pairwise Correlation: Stress vs Normal Days",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # annotate delta
    for k in range(len(pairs)):
        delta = stress_vals[k] - normal_vals[k]
        ymax = max(stress_vals[k], normal_vals[k])
        ax.text(x[k], ymax + 0.02, f"Δ{delta:+.3f}",
                ha="center", va="bottom", fontsize=8, color="#333")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig


# ── 08 ─ eigenvalue scree ──────────────────────────────────────

def plot_eigenvalue_scree(
    eigenvalues: np.ndarray,
    explained_var: np.ndarray,
    tickers: list,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Scree plot: eigenvalue bars + cumulative explained-variance line.
    """
    n = len(eigenvalues)
    x = np.arange(1, n + 1)

    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    bars = ax1.bar(x, eigenvalues, color="#1f77b4", alpha=0.8,
                   label="Eigenvalue")
    ax1.set_xlabel("Principal Component", fontsize=11)
    ax1.set_ylabel("Eigenvalue", fontsize=11, color="#1f77b4")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"PC{i}" for i in x], fontsize=10)
    ax1.axhline(1.0, color="grey", linewidth=0.8, linestyle="--",
                label="Kaiser criterion (λ=1)")

    ax2 = ax1.twinx()
    ax2.plot(x, explained_var * 100, "o-", color="#d62728",
             linewidth=2, markersize=7, label="Cumulative %")
    ax2.set_ylabel("Cumulative Explained Variance (%)",
                   fontsize=11, color="#d62728")
    ax2.set_ylim(0, 105)

    # annotate percentages
    for i in range(n):
        ax2.annotate(f"{explained_var[i]*100:.1f}%",
                     (x[i], explained_var[i]*100),
                     textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=9, color="#d62728")

    ax1.set_title("PCA of Correlation Matrix – Eigenvalue Scree Plot",
                  fontsize=13, fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=9, loc="center right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig
