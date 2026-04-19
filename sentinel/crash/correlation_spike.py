"""
sentinel/crash/correlation_spike.py
Phase 3 Deliverable 3 — Correlation Spike Detection

In normal markets, assets have moderate correlations driven by shared fundamentals.
During crashes, correlations spike toward 1 — everything sells off together as
investors liquidate across the board. This "correlation contagion" is a reliable
early warning signal.

Key concepts
────────────
Rolling correlation:
    ρ_{ij}(t) = Corr(r_i, r_j) computed over a rolling window W
    Standard choice: W = 60 days (≈ 3 months)

Average pairwise correlation (APC):
    APC(t) = (2 / N(N-1)) Σ_{i<j} ρ_{ij}(t)
    A single number summarising "how correlated the market is right now"
    Rising APC → danger; falling APC → diversification returning

Spike detection:
    A correlation spike at time t is flagged when:
        APC(t) > μ_APC + k·σ_APC    (k=1.5 by default)
    where μ, σ are computed over the full history (or a training window).

Correlation regime:
    "Low correlation"  regime: APC < 33rd percentile
    "High correlation" regime: APC > 67th percentile
    This feeds directly into Phase 4's Hidden Markov Model.

Conditional VaR:
    VaR computed separately in high-correlation and low-correlation regimes.
    High-correlation VaR is typically 2-3× larger — regime-conditioning
    gives a more honest picture of tail risk than unconditional VaR.
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd


# ── rolling pairwise correlations ─────────────────────────────────────────────

def rolling_pairwise_corr(
    prices: pd.DataFrame,
    window: int = 60,
) -> Dict[str, pd.Series]:
    """
    Compute rolling Pearson correlation for every ticker pair.

    Returns dict mapping "TKR1_TKR2" → pd.Series of rolling correlation.
    """
    log_rets = np.log(prices / prices.shift(1)).dropna()
    tickers  = log_rets.columns.tolist()
    result   = {}
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            a, b = tickers[i], tickers[j]
            key  = f"{a}_{b}"
            result[key] = log_rets[a].rolling(window).corr(log_rets[b])
    return result


def average_pairwise_correlation(
    prices: pd.DataFrame,
    window: int = 60,
) -> pd.Series:
    """
    Average Pairwise Correlation (APC) — the market-wide correlation index.
    Values near 1 signal a crash/stress regime.
    """
    pairs = rolling_pairwise_corr(prices, window)
    df    = pd.DataFrame(pairs)
    return df.mean(axis=1).rename("APC")


# ── spike detection ────────────────────────────────────────────────────────────

def detect_spikes(
    apc: pd.Series,
    k: float = 1.5,
    min_periods: int = 120,
) -> pd.Series:
    """
    Flag correlation spikes: APC > rolling_mean + k * rolling_std.

    Returns boolean Series (True = spike day).
    Using expanding mean/std so there's no lookahead.
    """
    expanding_mean = apc.expanding(min_periods=min_periods).mean()
    expanding_std  = apc.expanding(min_periods=min_periods).std()
    threshold      = expanding_mean + k * expanding_std
    return (apc > threshold).rename("spike")


def spike_episodes(spike_series: pd.Series, min_gap: int = 5) -> pd.DataFrame:
    """
    Collapse consecutive spike days into episodes (start_date, end_date, duration).
    min_gap: days of non-spike required to end an episode.
    """
    s = spike_series.fillna(False).astype(bool)
    episodes = []
    in_ep    = False
    start    = None
    last_spike = None

    for date, val in s.items():
        if val:
            if not in_ep:
                in_ep = True
                start = date
            last_spike = date
        else:
            if in_ep and last_spike is not None:
                gap = (date - last_spike).days
                if gap > min_gap:
                    episodes.append({"start": start, "end": last_spike,
                                     "duration_days": (last_spike - start).days + 1})
                    in_ep = False

    if in_ep:
        episodes.append({"start": start, "end": last_spike,
                         "duration_days": (last_spike - start).days + 1})

    return pd.DataFrame(episodes)


# ── correlation regimes ────────────────────────────────────────────────────────

def correlation_regimes(
    apc: pd.Series,
    low_pct: float = 33,
    high_pct: float = 67,
) -> pd.Series:
    """
    Assign each day to a correlation regime:
        0 = Low correlation  (diversification working)
        1 = Medium
        2 = High correlation (crash warning)
    """
    lo = np.nanpercentile(apc.dropna(), low_pct)
    hi = np.nanpercentile(apc.dropna(), high_pct)

    regime = pd.Series(1, index=apc.index, name="corr_regime")
    regime[apc <= lo] = 0
    regime[apc >= hi] = 2
    return regime


# ── conditional VaR ───────────────────────────────────────────────────────────

def conditional_var(
    prices: pd.DataFrame,
    regime: pd.Series,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Compute equal-weight portfolio VaR separately for each regime.

    Returns DataFrame with columns: regime, var_95, cvar_95, n_days.
    """
    log_rets  = np.log(prices / prices.shift(1)).dropna()
    port_rets = log_rets.mean(axis=1)   # equal-weight

    rows = []
    for reg, label in [(0, "Low corr"), (1, "Medium corr"), (2, "High corr")]:
        mask = regime.reindex(port_rets.index) == reg
        r    = port_rets[mask].dropna()
        if len(r) < 10:
            continue
        alpha = 1 - confidence
        var   = float(-np.quantile(r, alpha))
        tail  = r[r <= -var]
        cvar  = float(-tail.mean()) if len(tail) > 0 else var
        rows.append({
            "regime":  label,
            "var":     var,
            "cvar":    cvar,
            "n_days":  len(r),
            "mean_ret": float(r.mean()),
            "vol":     float(r.std()),
        })
    return pd.DataFrame(rows).set_index("regime")


# ── full crash warning summary ────────────────────────────────────────────────

def crash_warning_summary(
    prices: pd.DataFrame,
    window: int = 60,
    k: float = 1.5,
    confidence: float = 0.95,
) -> dict:
    """
    Run full correlation spike detection pipeline.

    Returns dict with: apc, pairs, spikes, episodes, regime,
                       conditional_var_df, current_apc, current_regime
    """
    apc     = average_pairwise_correlation(prices, window)
    pairs   = rolling_pairwise_corr(prices, window)
    spikes  = detect_spikes(apc, k)
    eps     = spike_episodes(spikes)
    regime  = correlation_regimes(apc)
    cond_v  = conditional_var(prices, regime, confidence)

    current_apc    = float(apc.dropna().iloc[-1])
    current_regime = int(regime.reindex(apc.dropna().index).iloc[-1])

    regime_labels = {0: "Low (diversification active)",
                     1: "Medium",
                     2: "HIGH — crash warning ⚠️"}

    return {
        "apc":               apc,
        "pairs":             pairs,
        "spikes":            spikes,
        "episodes":          eps,
        "regime":            regime,
        "conditional_var":   cond_v,
        "current_apc":       current_apc,
        "current_regime":    regime_labels[current_regime],
        "n_spike_days":      int(spikes.sum()),
        "n_episodes":        len(eps),
    }
