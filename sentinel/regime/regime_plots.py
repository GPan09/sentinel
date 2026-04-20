"""
sentinel/regime/regime_plots.py
Phase 4 Deliverable 1 — HMM Regime Chart Functions (Charts 37–40)

Chart 37 — Smoothed regime posteriors over time (stacked area)
Chart 38 — Price with Viterbi regime shading (most-likely state per day)
Chart 39 — Return distributions per regime + fitted Gaussian overlays
Chart 40 — Transition matrix heatmap + stationary distribution / durations
"""

from __future__ import annotations
from typing import Optional, Sequence
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm


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

# Bear → red, Sideways → orange, Bull → green (sorted lowest-mean → highest-mean,
# so this matches the canonical state ordering used by hmm_detector.fit_hmm).
REGIME_COLOR_2 = ["#d62728", "#2ca02c"]
REGIME_COLOR_3 = ["#d62728", "#ff7f0e", "#2ca02c"]
REGIME_COLOR_4 = ["#7c1c1c", "#d62728", "#ff7f0e", "#2ca02c"]


def _regime_colors(k: int):
    return {2: REGIME_COLOR_2, 3: REGIME_COLOR_3, 4: REGIME_COLOR_4}.get(
        k, SENTINEL_PALETTE[:k]
    )


def _style(ax):
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#30363d")


# ── Chart 37 — Smoothed posterior probabilities ───────────────────────────────

def plot_regime_posteriors(
    posteriors: pd.DataFrame,
    ticker: str = "SPY",
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Chart 37 — Stacked-area plot of P(S_t = k | r_1, …, r_T).

    These are *smoothed* posteriors (forward-backward), so they use both past
    AND future returns to label day t. They're the cleanest historical view of
    regime, but you can't trade on them in real time — use the filtered
    posteriors  P(S_t | r_1,…,r_t)  for live decisions (Phase 4 D3).
    """
    labels = list(posteriors.columns)
    colors = _regime_colors(len(labels))

    with plt.rc_context(CHART_STYLE):
        fig, ax = plt.subplots(figsize=(13, 5.5))
        ax.stackplot(
            posteriors.index,
            posteriors.values.T,
            colors=colors,
            labels=labels,
            alpha=0.85,
            edgecolor="none",
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("P(regime | data)")
        ax.set_title(
            f"Chart 37 — {ticker} HMM Smoothed Regime Posteriors  "
            f"(K={len(labels)} states)"
        )
        _style(ax)
        ax.legend(loc="upper left", fontsize=9, ncol=len(labels))

        # annotate latest posterior
        last = posteriors.iloc[-1]
        annot = "  ".join(f"{lab}: {p:.0%}" for lab, p in last.items())
        ax.text(0.99, 0.05, f"Latest:  {annot}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, color="#c9d1d9",
                bbox=dict(facecolor="#161b22", edgecolor="#30363d"))

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


# ── Chart 38 — Viterbi path on price ──────────────────────────────────────────

def plot_viterbi_on_price(
    prices: pd.Series,
    viterbi_labeled: pd.Series,
    labels: Sequence[str],
    ticker: str = "SPY",
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Chart 38 — Two-panel view:
      Top    : price (log scale) with background shaded by Viterbi regime.
      Bottom : Viterbi state strip — a thin band coloured by regime per day.
    """
    colors = _regime_colors(len(labels))

    p = prices.dropna()
    v = viterbi_labeled.reindex(p.index).ffill().bfill().astype(int)

    # precompute contiguous regime segments once — used by both panels
    change_pts = v.ne(v.shift()).cumsum()
    segments = [(chunk.index[0], chunk.index[-1], int(chunk.iloc[0]))
                for _, chunk in v.groupby(change_pts)]

    with plt.rc_context(CHART_STYLE):
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(13, 7),
            sharex=True,
            gridspec_kw={"height_ratios": [4, 0.8]},
        )

        # — top: price + regime shading —
        for start, end, s in segments:
            ax1.axvspan(start, end, color=colors[s], alpha=0.18, zorder=0)
        ax1.plot(p.index, p.values, color="#c9d1d9", lw=1.4, label=ticker,
                 zorder=3)

        ax1.set_yscale("log")
        ax1.set_ylabel(f"{ticker} (log scale)")
        ax1.set_title(
            f"Chart 38 — {ticker} with HMM Viterbi Regime Path  "
            f"(K={len(labels)} states, sorted Bear → Bull)"
        )
        _style(ax1)
        patches = [mpatches.Patch(color=colors[i], alpha=0.5, label=lab)
                   for i, lab in enumerate(labels)]
        ax1.legend(handles=[*patches,
                            plt.Line2D([0], [0], color="#c9d1d9", lw=1.4,
                                       label=ticker)],
                   fontsize=9, loc="upper left")

        # — bottom: regime strip via filled spans (same date units as ax1) —
        for start, end, s in segments:
            ax2.axvspan(start, end, color=colors[s], alpha=0.95,
                        ymin=0.0, ymax=1.0)
        ax2.set_ylim(0, 1)
        ax2.set_yticks([])
        ax2.set_xlabel("Date")
        ax2.set_ylabel("State")

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


# ── Chart 39 — Return distributions per regime ────────────────────────────────

def plot_state_return_distributions(
    returns: pd.Series,
    viterbi_labeled: pd.Series,
    means: np.ndarray,
    variances: np.ndarray,
    labels: Sequence[str],
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Chart 39 — Histogram of returns conditional on state, with the fitted
    Gaussian PDF for that state overlaid.

    What you should see:
      • Bull state:  narrow histogram, mean slightly positive, low σ
      • Bear state:  wide histogram, mean negative, high σ
    The two distributions barely overlap if the HMM has separated cleanly.
    """
    K = len(labels)
    colors = _regime_colors(K)

    aligned_v = viterbi_labeled.reindex(returns.index).dropna().astype(int)
    r_aligned = returns.reindex(aligned_v.index)

    # global x-range so all subplots share scale
    lo, hi = float(r_aligned.quantile(0.001)), float(r_aligned.quantile(0.999))
    grid = np.linspace(lo, hi, 400)

    with plt.rc_context(CHART_STYLE):
        fig, axes = plt.subplots(1, K, figsize=(5 * K, 5), sharey=True,
                                 squeeze=False)
        fig.suptitle("Chart 39 — Return Distribution by HMM Regime",
                     fontsize=13)

        for i, lab in enumerate(labels):
            ax = axes[0][i]
            mask = aligned_v == i
            r_i  = r_aligned[mask].values
            if len(r_i) < 5:
                ax.text(0.5, 0.5, f"{lab}: too few obs",
                        transform=ax.transAxes, ha="center")
                _style(ax)
                continue

            ax.hist(r_i, bins=50, range=(lo, hi),
                    density=True, color=colors[i], alpha=0.55,
                    edgecolor="#0d1117")

            mu, sd = means[i], np.sqrt(variances[i])
            pdf    = norm.pdf(grid, loc=mu, scale=sd)
            ax.plot(grid, pdf, color="#c9d1d9", lw=1.6,
                    label=f"N(μ={mu:.4f}, σ={sd:.4f})")

            ax.axvline(mu, color="#c9d1d9", lw=0.8, ls="--", alpha=0.6)
            ax.axvline(0,   color="#8b949e", lw=0.8, ls=":",  alpha=0.6)

            ann_lines = [
                f"n = {len(r_i):,}",
                f"μ_ann = {mu*252:+.1%}",
                f"σ_ann = {sd*np.sqrt(252):.1%}",
                f"Sharpe ≈ {(mu*252)/(sd*np.sqrt(252)+1e-12):+.2f}",
            ]
            ax.text(0.02, 0.98, "\n".join(ann_lines),
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=9, color="#c9d1d9",
                    bbox=dict(facecolor="#161b22", edgecolor="#30363d"))

            ax.set_title(lab)
            ax.set_xlabel("Daily log return")
            ax.legend(fontsize=8, loc="upper right")
            _style(ax)

        axes[0][0].set_ylabel("Density")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


# ── Chart 40 — Transition matrix + stationary distribution ────────────────────

def plot_transition_matrix(
    trans_matrix: pd.DataFrame,
    stationary: pd.Series,
    durations: pd.Series,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Chart 40 — Two panels:
      Left  : KxK heatmap of transition probabilities A_{ij} = P(S_{t+1}=j | S_t=i)
      Right : Stationary distribution π and expected duration per regime,
              shown as paired bars (left axis = π, right axis = duration days).

    The diagonal of A is regime "stickiness" — values near 1 mean the model
    expects long persistent regimes (low whip-sawing).
    """
    labels = list(trans_matrix.index)
    K = len(labels)
    colors = _regime_colors(K)

    with plt.rc_context(CHART_STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5),
                                       gridspec_kw={"width_ratios": [1.1, 1.4]})
        fig.suptitle("Chart 40 — HMM Transition Matrix & Stationary Distribution",
                     fontsize=13)

        # — left: heatmap —
        A = trans_matrix.values
        im = ax1.imshow(A, cmap="Blues", vmin=0, vmax=1)
        ax1.set_xticks(range(K)); ax1.set_yticks(range(K))
        ax1.set_xticklabels(labels); ax1.set_yticklabels(labels)
        ax1.set_xlabel("To state (t+1)")
        ax1.set_ylabel("From state (t)")
        ax1.set_title("Transition Matrix  A_{ij} = P(S_{t+1}=j | S_t=i)")
        for i in range(K):
            for j in range(K):
                ax1.text(j, i, f"{A[i,j]:.2%}", ha="center", va="center",
                         color="black" if A[i, j] < 0.5 else "white",
                         fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

        # — right: stationary distribution + expected duration paired bars —
        x = np.arange(K)
        w = 0.38
        bars1 = ax2.bar(x - w/2, stationary.values * 100, width=w,
                        color=colors, edgecolor="#0d1117", linewidth=0.5,
                        label="Stationary π (%)")
        ax2.set_ylabel("Stationary probability (%)")
        ax2.set_xticks(x); ax2.set_xticklabels(labels)
        for bar, val in zip(bars1, stationary.values * 100):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 1.0,
                     f"{val:.1f}%", ha="center", va="bottom",
                     fontsize=9, color="#c9d1d9")
        _style(ax2)

        ax2b = ax2.twinx()
        bars2 = ax2b.bar(x + w/2, durations.values, width=w,
                         color="#c9d1d9", alpha=0.55, edgecolor="#0d1117",
                         linewidth=0.5, label="Expected duration (days)")
        ax2b.set_ylabel("Expected duration (days)")
        ax2b.spines[["top"]].set_visible(False)
        for bar, val in zip(bars2, durations.values):
            ax2b.text(bar.get_x() + bar.get_width()/2,
                      bar.get_height() + max(durations.values)*0.02,
                      f"{val:.0f}d", ha="center", va="bottom",
                      fontsize=9, color="#c9d1d9")

        ax2.set_title("Long-Run Behaviour")
        # combined legend
        h1, l1 = ax2.get_legend_handles_labels()
        h2, l2 = ax2b.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper left")

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


# ════════════════════════════════════════════════════════════════════════════
#  Phase 4 Deliverable 2 — Multivariate HMM + Model Selection (Charts 41–44)
# ════════════════════════════════════════════════════════════════════════════

# ── Chart 41 — BIC / AIC vs K ─────────────────────────────────────────────────

def plot_k_selection(
    comparison: pd.DataFrame,
    chosen_k: int,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Chart 41 — Information criteria across candidate state counts K.

    Lower BIC / AIC = better. BIC penalises complexity more heavily than AIC,
    so BIC is usually what we honour when they disagree. The chosen K (here the
    BIC minimum) is highlighted with a vertical dashed line.
    """
    with plt.rc_context(CHART_STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                       gridspec_kw={"width_ratios": [1.3, 1]})

        xs = comparison.index.values

        # left: BIC + AIC on shared y-axis
        ax1.plot(xs, comparison["bic"], color=SENTINEL_PALETTE[0],
                 marker="o", lw=1.8, label="BIC")
        ax1.plot(xs, comparison["aic"], color=SENTINEL_PALETTE[1],
                 marker="s", lw=1.8, label="AIC")
        ax1.axvline(chosen_k, color=SENTINEL_PALETTE[3], lw=1.2, ls="--",
                    label=f"Chosen K = {chosen_k}")
        ax1.set_xticks(xs)
        ax1.set_xlabel("Number of hidden states K")
        ax1.set_ylabel("Information criterion  (lower = better)")
        ax1.set_title("Chart 41 — Model Selection: BIC & AIC vs K")
        _style(ax1)
        ax1.legend(fontsize=9, loc="best")

        bic_at_chosen = comparison.loc[chosen_k, "bic"]
        ax1.annotate(f"BIC = {bic_at_chosen:.1f}",
                     xy=(chosen_k, bic_at_chosen),
                     xytext=(10, -25), textcoords="offset points",
                     color="#c9d1d9", fontsize=9,
                     arrowprops=dict(arrowstyle="->", color="#8b949e", lw=0.8))

        ax2.bar(xs - 0.2, comparison["log_likelihood"], width=0.4,
                color=SENTINEL_PALETTE[2], label="log L")
        ax2.set_ylabel("log-likelihood", color=SENTINEL_PALETTE[2])
        ax2.tick_params(axis="y", colors=SENTINEL_PALETTE[2])
        ax2.set_xticks(xs)
        ax2.set_xlabel("K")
        _style(ax2)

        ax2b = ax2.twinx()
        ax2b.bar(xs + 0.2, comparison["n_params"], width=0.4,
                 color="#c9d1d9", alpha=0.55, label="params")
        ax2b.set_ylabel("# free parameters", color="#c9d1d9")
        ax2b.spines[["top"]].set_visible(False)

        ax2.set_title("Fit quality vs complexity")
        h1, l1 = ax2.get_legend_handles_labels()
        h2, l2 = ax2b.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper left")

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


# ── Chart 42 — Price + multivariate posteriors + features ─────────────────────

def plot_mv_posteriors_with_features(
    prices: pd.Series,
    posteriors: pd.DataFrame,
    features: pd.DataFrame,
    ticker: str = "SPY",
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Chart 42 — Three stacked panels on a shared date axis:
      Top    : benchmark price (log) with Viterbi-implied regime shading
               (derived here as the argmax of the posteriors per day)
      Middle : stacked-area smoothed posteriors
      Bottom : rolling realised vol and APC (the non-return features the HMM used)

    This is the "show your work" panel — you can see exactly what information
    the model had, and where its confidence shifts.
    """
    labels = list(posteriors.columns)
    K = len(labels)
    colors = _regime_colors(K)

    v = posteriors.idxmax(axis=1).map({lab: i for i, lab in enumerate(labels)})
    v = v.reindex(posteriors.index).ffill().astype(int)

    p = prices.reindex(posteriors.index).dropna()
    feat = features.reindex(posteriors.index).ffill()

    vol_col = [c for c in feat.columns if c.startswith("vol_")]
    apc_col = [c for c in feat.columns if c.startswith("apc_")]
    vol_s = feat[vol_col[0]] if vol_col else None
    apc_s = feat[apc_col[0]] if apc_col else None

    change_pts = v.ne(v.shift()).cumsum()
    segments = [(chunk.index[0], chunk.index[-1], int(chunk.iloc[0]))
                for _, chunk in v.groupby(change_pts)]

    with plt.rc_context(CHART_STYLE):
        fig, axes = plt.subplots(
            3, 1, figsize=(13, 10), sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1.4, 1.4]},
        )
        ax1, ax2, ax3 = axes

        for start, end, s in segments:
            ax1.axvspan(start, end, color=colors[s], alpha=0.18, zorder=0)
        ax1.plot(p.index, p.values, color="#c9d1d9", lw=1.4, label=ticker, zorder=3)
        ax1.set_yscale("log")
        ax1.set_ylabel(f"{ticker} (log)")
        ax1.set_title(
            f"Chart 42 — {ticker} with Multivariate HMM Regimes  (K={K})"
        )
        _style(ax1)
        patches = [mpatches.Patch(color=colors[i], alpha=0.5, label=lab)
                   for i, lab in enumerate(labels)]
        ax1.legend(handles=[*patches,
                            plt.Line2D([0], [0], color="#c9d1d9", lw=1.4,
                                       label=ticker)],
                   fontsize=9, loc="upper left")

        ax2.stackplot(posteriors.index, posteriors.values.T,
                      colors=colors, labels=labels, alpha=0.85,
                      edgecolor="none")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("P(regime | data)")
        _style(ax2)
        ax2.legend(loc="upper left", fontsize=8, ncol=K)

        if vol_s is not None:
            ax3.plot(vol_s.index, vol_s.values * np.sqrt(252) * 100,
                     color=SENTINEL_PALETTE[1], lw=1.2,
                     label=f"{vol_col[0]}  (ann. %)")
            ax3.set_ylabel("Realised vol (ann. %)",
                           color=SENTINEL_PALETTE[1])
            ax3.tick_params(axis="y", colors=SENTINEL_PALETTE[1])
        _style(ax3)

        if apc_s is not None:
            ax3b = ax3.twinx()
            ax3b.plot(apc_s.index, apc_s.values,
                      color=SENTINEL_PALETTE[4], lw=1.2,
                      label=apc_col[0])
            ax3b.set_ylabel("Avg pairwise corr",
                            color=SENTINEL_PALETTE[4])
            ax3b.tick_params(axis="y", colors=SENTINEL_PALETTE[4])
            ax3b.spines[["top"]].set_visible(False)
            h1, l1 = ax3.get_legend_handles_labels()
            h2, l2 = ax3b.get_legend_handles_labels()
            ax3.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
        else:
            ax3.legend(fontsize=8, loc="upper left")

        ax3.set_xlabel("Date")

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


# ── Chart 43 — Feature scatter pairs, coloured by state ──────────────────────

def plot_feature_pairs_by_state(
    features: pd.DataFrame,
    viterbi_labeled: pd.Series,
    labels: Sequence[str],
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Chart 43 — Pairwise scatter plots of the features, points coloured by
    Viterbi regime.

    What you want to see: regimes occupy different blobs in feature space.
    The HMM is essentially drawing ellipses in this multivariate cloud — if
    the ellipses don't separate, the features aren't informative.
    """
    K = len(labels)
    colors = _regime_colors(K)
    feat_cols = list(features.columns)
    if len(feat_cols) < 2:
        raise ValueError("Need at least 2 features for scatter plots.")

    pairs = []
    for i in range(len(feat_cols)):
        for j in range(i + 1, len(feat_cols)):
            pairs.append((feat_cols[i], feat_cols[j]))
    pairs = pairs[:3]

    v = viterbi_labeled.reindex(features.index).dropna().astype(int)
    feat = features.loc[v.index]

    with plt.rc_context(CHART_STYLE):
        fig, axes = plt.subplots(1, len(pairs), figsize=(5.5 * len(pairs), 5),
                                 squeeze=False)
        fig.suptitle("Chart 43 — Feature-Pair Scatter by HMM Regime",
                     fontsize=13)

        for col, (fx, fy) in enumerate(pairs):
            ax = axes[0][col]
            for i, lab in enumerate(labels):
                mask = v == i
                ax.scatter(feat.loc[mask, fx], feat.loc[mask, fy],
                           s=10, alpha=0.55, color=colors[i],
                           edgecolors="none", label=lab)
            ax.set_xlabel(fx)
            ax.set_ylabel(fy)
            _style(ax)
            if col == 0:
                ax.legend(fontsize=9, markerscale=2)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


# ── Chart 44 — Per-feature per-state distribution boxplots ────────────────────

def plot_feature_boxplots_by_state(
    features: pd.DataFrame,
    viterbi_labeled: pd.Series,
    labels: Sequence[str],
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Chart 44 — For each feature, a box plot comparing its distribution across
    regimes.

    Reads as: "When the HMM says we're in Bear, what does the distribution of
    returns / vol / APC look like vs. Sideways vs. Bull?" This diagnoses
    whether each feature is genuinely helping the classifier.
    """
    K = len(labels)
    colors = _regime_colors(K)
    feat_cols = list(features.columns)
    D = len(feat_cols)

    v = viterbi_labeled.reindex(features.index).dropna().astype(int)
    feat = features.loc[v.index]

    with plt.rc_context(CHART_STYLE):
        fig, axes = plt.subplots(1, D, figsize=(5 * D, 5), squeeze=False)
        fig.suptitle("Chart 44 — Feature Distributions by HMM Regime",
                     fontsize=13)

        for col, fc in enumerate(feat_cols):
            ax = axes[0][col]
            data = [feat.loc[v == i, fc].values for i in range(K)]
            bp = ax.boxplot(
                data, tick_labels=labels, patch_artist=True,
                medianprops=dict(color="#c9d1d9", lw=1.5),
                whiskerprops=dict(color="#8b949e"),
                capprops=dict(color="#8b949e"),
                flierprops=dict(marker="o", markersize=3,
                                markerfacecolor="#8b949e",
                                markeredgecolor="#8b949e", alpha=0.5),
            )
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.70)
                patch.set_edgecolor("#0d1117")
            ax.set_title(fc)
            ax.set_ylabel(fc)
            _style(ax)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
