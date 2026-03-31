"""sentinel/risk/monte_carlo.py
Phase 1 · Deliverable 3 — Monte Carlo Portfolio Simulation

Core math
---------
Given T historical log-returns r_t ∈ ℝ^n, the sample covariance Σ is
Cholesky-decomposed as Σ = L Lᵀ.  Each simulated day's return vector is

    r_sim = μ + L z,   z ~ N(0, I)

where μ is the sample mean vector.  We simulate N paths of length H days
and compute the portfolio P&L distribution at the horizon.

Outputs
-------
- eigenvalues / explained variance (validation vs empirical corr)
- simulated terminal P&L distribution → MC VaR / CVaR
- convergence of VaR estimate as n_sims grows
- per-path portfolio value trajectories (for fan chart)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple


def _cholesky(returns: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean_vector, L) where Σ = L @ Lᵀ."""
    mu = returns.mean().values
    cov = returns.cov().values
    # Add a small regularisation term for numerical stability
    cov += np.eye(len(mu)) * 1e-8
    L = np.linalg.cholesky(cov)
    return mu, L


def simulate_paths(
    returns: pd.DataFrame,
    weights: np.ndarray | None = None,
    n_sims: int = 10_000,
    horizon: int = 1,
    seed: int = 42,
) -> Dict:
    """
    Simulate portfolio return paths via Cholesky-correlated Gaussians.

    Parameters
    ----------
    returns   : daily log-return DataFrame (T × n)
    weights   : portfolio weights (n,); equal-weight if None
    n_sims    : number of Monte Carlo paths
    horizon   : simulation horizon in trading days
    seed      : RNG seed for reproducibility

    Returns
    -------
    dict with keys:
        terminal_pnl  – (n_sims,) array of horizon portfolio log-returns
        paths         – (n_sims, horizon) cumulative portfolio log-return paths
        weights       – weights used
        n             – number of assets
        tickers       – list of ticker strings
    """
    n = returns.shape[1]
    if weights is None:
        weights = np.ones(n) / n
    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()

    mu, L = _cholesky(returns)

    rng = np.random.default_rng(seed)
    # shape: (n_sims, horizon, n_assets)
    z = rng.standard_normal((n_sims, horizon, n))
    # correlated daily returns: (n_sims, horizon, n)
    r_sim = mu + z @ L.T
    # portfolio daily log-return: (n_sims, horizon)
    port_daily = r_sim @ weights
    # cumulative log-return paths: (n_sims, horizon)
    paths = np.cumsum(port_daily, axis=1)
    terminal_pnl = paths[:, -1]

    return {
        "terminal_pnl": terminal_pnl,
        "paths": paths,
        "weights": weights,
        "n": n,
        "tickers": list(returns.columns),
    }


def mc_var_cvar(
    terminal_pnl: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Monte Carlo VaR and CVaR from the simulated terminal P&L distribution.

    VaR   = -quantile(terminal_pnl, 1 - confidence)
    CVaR  = -mean(terminal_pnl[terminal_pnl <= -VaR])
    """
    alpha = 1.0 - confidence
    var = -np.quantile(terminal_pnl, alpha)
    tail = terminal_pnl[terminal_pnl <= -var]
    cvar = -tail.mean() if len(tail) > 0 else var
    return float(var), float(cvar)


def var_convergence(
    returns: pd.DataFrame,
    weights: np.ndarray | None = None,
    max_sims: int = 10_000,
    steps: int = 40,
    confidence: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Track how VaR estimate changes as number of simulations grows.

    Returns a DataFrame with columns [n_sims, var, cvar].
    """
    sim_counts = np.unique(
        np.geomspace(100, max_sims, steps).astype(int)
    )
    result = simulate_paths(returns, weights, max_sims, horizon=1, seed=seed)
    terminal = result["terminal_pnl"]

    rows = []
    for k in sim_counts:
        v, c = mc_var_cvar(terminal[:k], confidence)
        rows.append({"n_sims": int(k), "var": v, "cvar": c})
    return pd.DataFrame(rows)


def mc_summary(
    returns: pd.DataFrame,
    weights: np.ndarray | None = None,
    n_sims: int = 10_000,
    horizon: int = 1,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict:
    """
    Full Monte Carlo summary dict.

    Keys: terminal_pnl, paths, weights, tickers, var, cvar,
          convergence_df, confidence, n_sims, horizon
    """
    sim = simulate_paths(returns, weights, n_sims, horizon, seed)
    var, cvar = mc_var_cvar(sim["terminal_pnl"], confidence)
    conv = var_convergence(returns, weights, n_sims,
                           confidence=confidence, seed=seed)
    return {**sim,
            "var": var,
            "cvar": cvar,
            "convergence_df": conv,
            "confidence": confidence,
            "n_sims": n_sims,
            "horizon": horizon}
