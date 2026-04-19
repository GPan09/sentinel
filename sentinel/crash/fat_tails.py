"""
sentinel/crash/fat_tails.py
Phase 3 Deliverable 2 — Fat-Tail Distribution Analysis

The Gaussian assumption underlying most textbook finance is empirically wrong.
Real return distributions have:
  • Excess kurtosis (fat tails) — extreme events happen far more often than N(0,1) predicts
  • Negative skewness — crashes are larger than rallies of equal probability
  • Volatility clustering — GARCH addresses this, but even GARCH residuals have fat tails

We fit three distributions and compare:

1. Normal (Gaussian)          — baseline, usually wrong
   PDF: f(x) = (1/σ√2π) exp(−(x−μ)²/2σ²)

2. Student-t                  — fat tails controlled by degrees-of-freedom ν
   PDF: f(x;ν) ∝ (1 + x²/ν)^(−(ν+1)/2)
   As ν→∞, converges to Normal. ν≈3-5 is typical for daily equity returns.
   Excess kurtosis = 6/(ν−4) for ν>4 — lower ν = fatter tails.

3. Normal Inverse Gaussian (NIG) — captures both skewness AND fat tails
   More flexible but harder to interpret; included for completeness.

Key diagnostic: QQ plot
   Plot empirical quantiles vs theoretical quantiles.
   If points lie on the 45° line → good fit.
   Fat tails show as S-curve deviating upward at extremes.

Tail risk metrics
─────────────────
  Excess kurtosis: κ = E[(X−μ)⁴]/σ⁴ − 3    (0 for Normal, >0 for fat tails)
  Skewness:        γ = E[(X−μ)³]/σ³          (<0 = left-skewed = crash-prone)
  Jarque-Bera test: H₀ = Normality. p<0.05 → reject normality.
  Tail probability ratio: P(|r|>3σ) empirical vs Normal theoretical
"""

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, t as student_t, jarque_bera


# ── distribution fitting ───────────────────────────────────────────────────────

def fit_normal(returns: pd.Series) -> dict:
    """Fit Normal distribution by MLE (= sample mean & std)."""
    mu, sigma = norm.fit(returns.dropna())
    return {"dist": "Normal", "mu": mu, "sigma": sigma,
            "scipy_dist": norm, "params": (mu, sigma)}


def fit_student_t(returns: pd.Series) -> dict:
    """
    Fit Student-t distribution by MLE.
    Returns df (degrees of freedom), loc, scale.
    Low df (3–6) = very fat tails.
    """
    df, loc, scale = student_t.fit(returns.dropna())
    return {"dist": "Student-t", "df": df, "loc": loc, "scale": scale,
            "scipy_dist": student_t, "params": (df, loc, scale)}


def fit_all(returns: pd.Series) -> Dict[str, dict]:
    """Fit Normal and Student-t to a return series."""
    return {
        "Normal":    fit_normal(returns),
        "Student-t": fit_student_t(returns),
    }


def fit_universe(prices: pd.DataFrame) -> Dict[str, Dict[str, dict]]:
    """Fit distributions to every ticker. Returns {ticker: {dist_name: fit_dict}}."""
    log_rets = np.log(prices / prices.shift(1)).dropna()
    return {ticker: fit_all(log_rets[ticker]) for ticker in log_rets.columns}


# ── tail risk metrics ──────────────────────────────────────────────────────────

def tail_metrics(returns: pd.Series) -> dict:
    """
    Compute key tail risk statistics for a return series.

    Returns dict with: mean, std, skewness, excess_kurtosis,
                       jb_stat, jb_pvalue, tail_ratio_3sigma,
                       tail_ratio_4sigma, var_95_empirical,
                       var_99_empirical, var_95_normal, var_99_normal
    """
    r = returns.dropna()
    mu, sigma = r.mean(), r.std()

    jb_stat, jb_p = jarque_bera(r)

    # empirical vs normal tail probabilities
    def tail_ratio(n_sigma):
        empirical   = (np.abs(r - mu) > n_sigma * sigma).mean()
        theoretical = 2 * norm.sf(n_sigma)   # P(|Z|>n) for N(0,1)
        return empirical / theoretical if theoretical > 0 else np.nan

    return {
        "mean":               float(mu),
        "std":                float(sigma),
        "skewness":           float(r.skew()),
        "excess_kurtosis":    float(r.kurtosis()),   # scipy returns excess kurtosis
        "jb_stat":            float(jb_stat),
        "jb_pvalue":          float(jb_p),
        "tail_ratio_3sigma":  float(tail_ratio(3)),
        "tail_ratio_4sigma":  float(tail_ratio(4)),
        "var_95_empirical":   float(-r.quantile(0.05)),
        "var_99_empirical":   float(-r.quantile(0.01)),
        "var_95_normal":      float(-norm.ppf(0.05) * sigma + mu),
        "var_99_normal":      float(-norm.ppf(0.01) * sigma + mu),
    }


def metrics_universe(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute tail metrics for all tickers. Returns summary DataFrame."""
    log_rets = np.log(prices / prices.shift(1)).dropna()
    rows = []
    for ticker in log_rets.columns:
        m = tail_metrics(log_rets[ticker])
        rows.append({"ticker": ticker, **m})
    return pd.DataFrame(rows).set_index("ticker")


# ── QQ data ────────────────────────────────────────────────────────────────────

def qq_data(returns: pd.Series,
            dist_fit: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute theoretical vs empirical quantiles for a QQ plot.

    Returns (theoretical_quantiles, empirical_quantiles).
    Points on the 45° line = good fit.
    """
    r = np.sort(returns.dropna().values)
    n = len(r)
    probs = (np.arange(1, n + 1) - 0.5) / n   # plotting positions

    d = dist_fit["scipy_dist"]
    theoretical = d.ppf(probs, *dist_fit["params"])
    return theoretical, r
