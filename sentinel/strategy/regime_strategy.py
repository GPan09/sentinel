"""
sentinel/strategy/regime_strategy.py
Phase 4 Deliverable 3 — Regime-aware strategy, rolling alpha, and nowcast

Problem
-------
The multivariate HMM from D2 tells us, every single day, the posterior
probability distribution over K hidden regimes. That's information. The
question is: can we turn that information into trading decisions that
outperform buy-and-hold?

Strategy — "regime gate"
────────────────────────
Hold the benchmark when the model is confident we are in a *bullish-enough*
regime; otherwise sit in cash.

Concretely, given a posterior matrix γ_t(k) = P(S_t = k | data) and a chosen
set of bullish regime labels B ⊆ {0, …, K−1}, the position at time t is

    s_t = 1   if   Σ_{k ∈ B}  γ_t(k)  >  τ
    s_t = 0   otherwise

The strategy return is the position from *yesterday* applied to today's return:

    r̃_t = s_{t-1} · r_t

That one-day lag is critical — without it you would be using the posterior of
today's return to decide today's position, which is look-ahead bias.

Comparing to buy-and-hold
─────────────────────────
Buy-and-hold (B&H) is the trivial strategy s_t = 1 ∀ t. We compare two
equity curves:

    E^strat_t = E_0 · exp( Σ r̃_i )
    E^BH_t    = E_0 · exp( Σ r_i  )

and look at CAGR, Sharpe, max drawdown, Calmar, and — most importantly —
*where* the strategy's edge came from. Rolling 1-year alpha answers that:

    α_t = mean_{last 252 days}( r̃ − r ) · 252

A good regime-gated strategy has α ≈ 0 during bull markets (it just tracks
B&H because it is always invested) and α ≫ 0 during bear markets (B&H loses
money, the strategy is in cash).

Regime-change early warning
───────────────────────────
The L1 distance between the posterior today and the posterior N days ago is
a scalar "how much did the regime distribution shift" score:

    Δ_t = Σ_k  |γ_t(k) − γ_{t−N}(k)|           (∈ [0, 2])

Spikes in Δ_t flag days when the market is plausibly transitioning between
regimes — the model is saying "something has changed" before the labels flip.

Nowcast
───────
The final snapshot: current regime, posterior probabilities, expected days
until transition, implied strategy position (invested vs cash), and recent
strategy performance. This is the "what is the model saying about today"
dashboard.
"""

from __future__ import annotations
from typing import Dict, Sequence
import numpy as np
import pandas as pd

# Reuse the Phase 2 backtester so we don't reimplement equity/drawdown/Sharpe.
from sentinel.backtest.engine import Backtester


# ── signal construction ───────────────────────────────────────────────────────

def regime_gated_signal(
    posteriors: pd.DataFrame,
    bullish_labels: Sequence[str],
    threshold: float = 0.5,
) -> pd.Series:
    """
    Map per-day posterior probabilities to a 0/1 "invested vs cash" signal.

    Parameters
    ----------
    posteriors : pd.DataFrame   shape (T, K), columns = state labels
    bullish_labels : list[str]  regimes we want to be invested during
    threshold : float           posterior-mass cutoff (default 0.5)

    Returns
    -------
    pd.Series of int {0, 1}, same DatetimeIndex as `posteriors`.

    Notes
    -----
    The threshold is applied to the SUM of posterior probabilities across the
    bullish set. With K=3 and bullish_labels = ["Sideways", "Bull"], we hold
    whenever P(Sideways) + P(Bull) > 0.5 — i.e. when the model is more likely
    NOT in Bear than in Bear.
    """
    missing = [lab for lab in bullish_labels if lab not in posteriors.columns]
    if missing:
        raise ValueError(f"bullish_labels not found in posteriors: {missing}. "
                         f"Available: {list(posteriors.columns)}")
    bullish_prob = posteriors[list(bullish_labels)].sum(axis=1)
    signal = (bullish_prob > threshold).astype(int)
    signal.name = "signal"
    return signal


def classify_labels(labels: Sequence[str]) -> Dict[str, list]:
    """
    Partition regime labels into bullish vs bearish using the canonical naming
    convention (Bear, Severe Bear, Crash are bearish; everything else bullish).
    Used as a default — callers can override.
    """
    bearish_keywords = ("bear", "crash")
    bearish, bullish = [], []
    for lab in labels:
        low = lab.lower()
        if any(k in low for k in bearish_keywords):
            bearish.append(lab)
        else:
            bullish.append(lab)
    return {"bullish": bullish, "bearish": bearish}


# ── backtest wrapper ──────────────────────────────────────────────────────────

def run_strategy(
    prices: pd.Series,
    posteriors: pd.DataFrame,
    bullish_labels: Sequence[str],
    threshold: float = 0.5,
    initial: float = 10_000.0,
) -> dict:
    """
    Run the regime-gated strategy and return a dict of everything downstream
    code (plots, CSV writers, nowcast) will need.

    Returns
    -------
    {
      'signal':          pd.Series  0/1 signal,
      'backtester':      Backtester (post-run), has equity_strat/equity_bh etc.,
      'metrics_strat':   dict with CAGR, Sharpe, max_drawdown, …,
      'metrics_bh':      dict for buy-and-hold (same keys),
      'bullish_prob':    pd.Series P(bullish regimes) per day,
      'labels_used':     {'bullish': [...], 'bearish': [...]},
      'threshold':       float,
    }
    """
    # Align — posteriors may start later than prices (features need warmup).
    prices_aligned = prices.reindex(posteriors.index).dropna()
    signal = regime_gated_signal(posteriors, bullish_labels, threshold)
    signal = signal.reindex(prices_aligned.index).ffill().fillna(0).astype(int)

    bt = Backtester(prices_aligned, signal, initial=initial).run()

    strat_metrics = bt.metrics()
    bh_metrics    = _buy_hold_metrics(bt, initial)

    bullish_prob = posteriors[list(bullish_labels)].sum(axis=1)
    bullish_prob = bullish_prob.reindex(prices_aligned.index).ffill()

    return {
        "signal":        signal,
        "backtester":    bt,
        "metrics_strat": strat_metrics,
        "metrics_bh":    bh_metrics,
        "bullish_prob":  bullish_prob,
        "labels_used":   {"bullish": list(bullish_labels)},
        "threshold":     threshold,
    }


def _buy_hold_metrics(bt: Backtester, initial: float) -> dict:
    """Compute the same metric dict for buy-and-hold using the same period."""
    from sentinel.backtest.engine import (
        _cagr, _sharpe, _max_drawdown,
    )
    return {
        "total_return":   float(bt.equity_bh.iloc[-1] / initial - 1),
        "cagr":           _cagr(bt.equity_bh),
        "sharpe":         _sharpe(bt.returns_bh),
        "max_drawdown":   _max_drawdown(bt.equity_bh),
    }


# ── rolling alpha ─────────────────────────────────────────────────────────────

def rolling_alpha(
    returns_strat: pd.Series,
    returns_bh: pd.Series,
    window: int = 252,
) -> pd.Series:
    """
    Rolling annualised excess return of strategy over buy-and-hold.

    α_t = mean_{last `window` days}( r̃ − r ) · 252

    Expressed as a fraction (0.05 = 5% annualised alpha).
    """
    excess = (returns_strat - returns_bh).dropna()
    roll = excess.rolling(window, min_periods=window).mean() * 252
    roll.name = f"rolling_alpha_{window}d"
    return roll


# ── regime-change early warning ──────────────────────────────────────────────

def regime_change_score(
    posteriors: pd.DataFrame,
    lookback: int = 5,
) -> pd.Series:
    """
    L1 distance between posterior today and posterior `lookback` days ago.
    Ranges from 0 (identical distribution) to 2 (orthogonal one-hot shift).

    This is an *early-warning* signal: it spikes when the posterior starts
    shifting, often several days before the Viterbi hard-label flips.
    """
    shifted = posteriors.shift(lookback)
    diff = (posteriors - shifted).abs().sum(axis=1)
    diff.name = f"regime_change_{lookback}d"
    return diff


def detect_transition_days(
    posteriors: pd.DataFrame,
    lookback: int = 5,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Return a DataFrame of days where the regime_change_score exceeded
    `threshold`. For each such day, record the dominant regime before and
    after the shift (mode of posterior argmax in the surrounding window).
    """
    score = regime_change_score(posteriors, lookback)
    hits = score[score > threshold].dropna()

    rows = []
    k_argmax = posteriors.idxmax(axis=1)
    for dt in hits.index:
        idx_now = posteriors.index.get_loc(dt)
        if idx_now < lookback or idx_now + lookback >= len(posteriors):
            continue
        before_win = k_argmax.iloc[max(0, idx_now - 2 * lookback):idx_now]
        after_win  = k_argmax.iloc[idx_now:idx_now + 2 * lookback]
        from_state = before_win.mode().iloc[0] if len(before_win) else "—"
        to_state   = after_win.mode().iloc[0]  if len(after_win)  else "—"
        if from_state != to_state:
            rows.append({
                "date":       dt,
                "score":      float(hits.loc[dt]),
                "from":       from_state,
                "to":         to_state,
            })
    return pd.DataFrame(rows)


# ── nowcast ──────────────────────────────────────────────────────────────────

def regime_nowcast(
    summary: dict,
    strategy_results: dict,
    recent_days: int = 60,
) -> dict:
    """
    Single-snapshot summary of 'what is the model saying today?'.

    Fields returned
    ---------------
    as_of                   : str (latest date in ISO format)
    current_state           : str
    current_posterior       : dict[label → prob]
    expected_days_in_regime : float (1 / (1 − A_ii))
    stationary_dist         : dict[label → prob]
    invested                : bool (strategy position today)
    bullish_prob_today      : float (P(bullish regimes))
    strategy_metrics        : dict (full strat metrics)
    bh_metrics              : dict (buy-and-hold metrics)
    recent_60d_regime_split : dict[label → fraction of last N days in that state]
    """
    posteriors     = summary["posteriors"]
    viterbi        = summary["viterbi_labeled"]
    trans_matrix   = summary["trans_matrix"]
    labels         = summary["labels"]
    stationary     = summary["stationary"]
    current_state  = summary["current_state"]
    current_post   = summary["current_post"]

    # Expected days in current regime from its diagonal transition probability
    a_ii = float(trans_matrix.loc[current_state, current_state])
    exp_days_remaining = 1.0 / max(1e-9, 1.0 - a_ii)

    # Fraction of the last N days spent in each state (Viterbi-based)
    recent = viterbi.iloc[-recent_days:] if len(viterbi) >= recent_days else viterbi
    counts = recent.value_counts(normalize=True)
    # Map int -> label via labels list
    recent_split = {labels[int(i)]: float(counts.get(int(i), 0.0))
                    for i in range(len(labels))}

    signal = strategy_results["signal"]
    last_signal = int(signal.iloc[-1]) if len(signal) else 0
    bullish_prob_today = float(
        strategy_results["bullish_prob"].iloc[-1]
    ) if len(strategy_results["bullish_prob"]) else float("nan")

    return {
        "as_of":                   str(posteriors.index[-1].date()),
        "current_state":           current_state,
        "current_posterior":       {k: float(v) for k, v in current_post.items()},
        "expected_days_in_regime": float(exp_days_remaining),
        "stationary_dist":         {k: float(v) for k, v in stationary.items()},
        "invested":                bool(last_signal),
        "bullish_prob_today":      bullish_prob_today,
        "strategy_metrics":        strategy_results["metrics_strat"],
        "bh_metrics":              strategy_results["metrics_bh"],
        "recent_regime_split":     recent_split,
        "recent_window_days":      int(min(recent_days, len(viterbi))),
    }


# ── pretty formatter for the nowcast (console / CSV row) ─────────────────────

def nowcast_to_row(nowcast: dict) -> pd.DataFrame:
    """Flatten the nowcast dict into a one-row DataFrame for CSV export."""
    row = {
        "as_of":                     nowcast["as_of"],
        "current_state":             nowcast["current_state"],
        "bullish_prob":              nowcast["bullish_prob_today"],
        "expected_days_in_regime":   nowcast["expected_days_in_regime"],
        "invested":                  nowcast["invested"],
        "strategy_cagr":             nowcast["strategy_metrics"].get("cagr"),
        "strategy_sharpe":           nowcast["strategy_metrics"].get("sharpe"),
        "strategy_max_drawdown":     nowcast["strategy_metrics"].get("max_drawdown"),
        "bh_cagr":                   nowcast["bh_metrics"].get("cagr"),
        "bh_sharpe":                 nowcast["bh_metrics"].get("sharpe"),
        "bh_max_drawdown":           nowcast["bh_metrics"].get("max_drawdown"),
    }
    for k, v in nowcast["current_posterior"].items():
        row[f"post_{k}"] = v
    for k, v in nowcast["recent_regime_split"].items():
        row[f"recent_{k}"] = v
    return pd.DataFrame([row])
