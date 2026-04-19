"""
sentinel/risk/var.py
====================
Value at Risk (VaR) — three methods:
    1. Historical  (empirical quantile, no distributional assumptions)
    2. Parametric  (variance-covariance, normal assumption)
    3. Monte Carlo (simulation-based)

Mathematical foundations
------------------------

HISTORICAL VAR
--------------
    VaR_α = -Q_{1-α}(r)

where Q_{1-α} is the empirical (1−α)-quantile of the return sample.

Pros: nonparametric, captures the actual shape of the tail.
Cons: limited by sample length; rare events beyond history are invisible.

PARAMETRIC VAR (Normal)
-----------------------
    VaR_α = -(μ - z_α · σ)

where z_α = Φ^{-1}(1−α) is the standard-normal quantile.

For a portfolio with weight vector w and covariance matrix Σ:
    σ_p = sqrt(w^T Σ w)
    VaR_α = -(w^T μ - z_α · σ_p)

Pros: fast, closed-form, risk easily decomposed across assets.
Cons: underestimates tail risk when kurtosis > 3 (fat tails) — the
      normal assumption is almost always wrong for real equity returns.
      Phase 0 showed SPY excess kurtosis ≈ 13, so this matters a lot.

MONTE CARLO VAR
---------------
    r_sim ~ N(μ, Σ) for N simulations
    VaR_α = -Q_{1-α}(simulated P&L)

Draw N random log returns from the fitted distribution, compound them
over the holding horizon, then read off the empirical quantile.

Pros: flexible — can swap in non-normal distributions (e.g. Student-t)
      in Phase 3 without changing the framework.
Cons: variance from simulation noise; needs large N for tail stability.

EXPECTED SHORTFALL (CVaR)
--------------------------
    ES_α = -E[r | r ≤ -VaR_α]

The average loss in the worst (1−α) fraction of outcomes.
Unlike VaR, ES is a *coherent* risk measure — it satisfies subadditivity,
meaning ES(A + B) ≤ ES(A) + ES(B), which VaR can violate.

SQUARE-ROOT-OF-TIME RULE
------------------------
    VaR(T days) ≈ VaR(1 day) · sqrt(T)

Holds exactly when daily returns are iid normal. Good approximation for
short horizons (T ≤ 10). For longer horizons use simulation directly.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, List


# ---------------------------------------------------------------------------
# Single-asset VaR
# ---------------------------------------------------------------------------

def historical_var(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
) -> Tuple[float, float]:
    """
    Historical (empirical quantile) VaR and Expected Shortfall.

    Parameters
    ----------
    returns    : pd.Series  — daily log returns for one asset
    confidence : float      — e.g. 0.95 for 95% VaR
    horizon    : int        — holding period in days (sqrt-of-time scaling)

    Returns
    -------
    (var, es) both as positive numbers (loss magnitudes)
    """
    r = returns.dropna().values
    scale = np.sqrt(horizon)

    cutoff = np.percentile(r, (1 - confidence) * 100)
    var = -cutoff * scale
    tail = r[r <= cutoff]
    es = -tail.mean() * scale if len(tail) > 0 else var
    return float(var), float(es)


def parametric_var(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
) -> Tuple[float, float]:
    """
    Parametric (variance-covariance) VaR assuming normally distributed returns.

    Parameters
    ----------
    returns    : pd.Series  — daily log returns
    confidence : float
    horizon    : int        — holding period in days

    Returns
    -------
    (var, es) both as positive numbers
    """
    r = returns.dropna().values
    mu = r.mean()
    sigma = r.std(ddof=1)  # Bessel correction: unbiased sample std
    scale = np.sqrt(horizon)

    z = stats.norm.ppf(1 - confidence)
    var = -(mu + z * sigma) * scale

    # ES under normality: E[r | r < VaR] = mu - sigma * phi(z) / (1 - alpha)
    phi_z = stats.norm.pdf(-z)
    es = -(mu * scale - sigma * scale * phi_z / (1 - confidence))
    return float(var), float(es)


def monte_carlo_var(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
    n_simulations: int = 10_000,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Monte Carlo VaR via simulation from the fitted normal distribution.

    Parameters
    ----------
    returns       : pd.Series — daily log returns
    confidence    : float
    horizon       : int       — holding period in days (simulated, not sqrt-scaled)
    n_simulations : int       — number of paths
    seed          : int       — for reproducibility

    Returns
    -------
    (var, es) both as positive numbers
    """
    rng = np.random.default_rng(seed)
    r = returns.dropna().values
    mu = r.mean()
    sigma = r.std(ddof=1)

    # Simulate `horizon`-day compound log returns
    sim_daily = rng.normal(mu, sigma, size=(n_simulations, horizon))
    sim_total = sim_daily.sum(axis=1)   # sum of log returns = log compound return

    cutoff = np.percentile(sim_total, (1 - confidence) * 100)
    var = -cutoff
    tail = sim_total[sim_total <= cutoff]
    es = -tail.mean() if len(tail) > 0 else var
    return float(var), float(es)


# ---------------------------------------------------------------------------
# Portfolio VaR (parametric, full covariance matrix)
# ---------------------------------------------------------------------------

def portfolio_parametric_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float = 0.95,
    horizon: int = 1,
) -> Tuple[float, float]:
    """
    Parametric portfolio VaR using the full covariance matrix.

        sigma_p = sqrt(w^T Sigma w)
        VaR_alpha = -(w^T mu - z_alpha * sigma_p) * sqrt(horizon)

    Parameters
    ----------
    returns    : pd.DataFrame — daily log returns, one column per asset
    weights    : np.ndarray  — portfolio weights (must sum to 1)
    confidence : float
    horizon    : int

    Returns
    -------
    (var, es) both as positive numbers
    """
    r = returns.dropna()
    mu_vec = r.mean().values
    cov_mat = r.cov().values
    scale = np.sqrt(horizon)

    mu_p = float(weights @ mu_vec)
    sigma_p = float(np.sqrt(weights @ cov_mat @ weights))

    z = stats.norm.ppf(1 - confidence)
    var = -(mu_p + z * sigma_p) * scale

    phi_z = stats.norm.pdf(-z)
    es = -(mu_p * scale - sigma_p * scale * phi_z / (1 - confidence))
    return float(var), float(es)


def portfolio_mc_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float = 0.95,
    horizon: int = 1,
    n_simulations: int = 10_000,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Monte Carlo portfolio VaR using correlated multivariate normal draws.

    Draws from the joint return distribution using the Cholesky decomposition
    of the sample covariance matrix, preserving cross-asset correlations.

    Parameters
    ----------
    returns       : pd.DataFrame
    weights       : np.ndarray
    confidence    : float
    horizon       : int
    n_simulations : int
    seed          : int

    Returns
    -------
    (var, es) both as positive numbers
    """
    rng = np.random.default_rng(seed)
    r = returns.dropna()
    mu_vec = r.mean().values
    cov_mat = r.cov().values

    # Simulate correlated daily returns, compound over horizon
    sim_total = np.zeros(n_simulations)
    for _ in range(horizon):
        daily = rng.multivariate_normal(mu_vec, cov_mat, size=n_simulations)
        sim_total += daily @ weights  # portfolio log return each day

    cutoff = np.percentile(sim_total, (1 - confidence) * 100)
    var = -cutoff
    tail = sim_total[sim_total <= cutoff]
    es = -tail.mean() if len(tail) > 0 else var
    return float(var), float(es)


# ---------------------------------------------------------------------------
# Summary table across all tickers and confidence levels
# ---------------------------------------------------------------------------

def var_summary(
    returns: pd.DataFrame,
    confidence_levels: List[float] = [0.90, 0.95, 0.99],
    horizon: int = 1,
    n_simulations: int = 10_000,
) -> pd.DataFrame:
    """
    Compute VaR and ES for every ticker across all 3 methods and
    multiple confidence levels.

    Returns
    -------
    pd.DataFrame with columns:
        ticker, confidence, hist_var, hist_es, param_var, param_es,
        mc_var, mc_es, param_vs_hist_pct
    """
    records = []
    for ticker in returns.columns:
        r = returns[ticker].dropna()
        for conf in confidence_levels:
            h_var, h_es = historical_var(r, conf, horizon)
            p_var, p_es = parametric_var(r, conf, horizon)
            m_var, m_es = monte_carlo_var(r, conf, horizon, n_simulations)

            pct_diff = (p_var - h_var) / h_var * 100 if h_var != 0 else 0.0

            records.append({
                "ticker": ticker,
                "confidence": conf,
                "hist_var": round(h_var, 6),
                "hist_es": round(h_es, 6),
                "param_var": round(p_var, 6),
                "param_es": round(p_es, 6),
                "mc_var": round(m_var, 6),
                "mc_es": round(m_es, 6),
                "param_vs_hist_pct": round(pct_diff, 2),
            })

    df = pd.DataFrame(records)
    return df
