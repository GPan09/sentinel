"""
Sentinel – Phase 1, Deliverable 2: Covariance & Correlation Analysis
=====================================================================
Functions for computing covariance matrices, correlation matrices,
rolling correlations, stress-period correlations, and eigenvalue
decomposition (PCA) of the return covariance structure.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


# ── core matrices ────────────────────────────────────────────────

def covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Sample covariance matrix (Bessel-corrected, ddof=1)."""
    return returns.cov()


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix."""
    return returns.corr()


# ── rolling correlations ────────────────────────────────────────

def rolling_correlations(
    returns: pd.DataFrame,
    window: int = 30,
    base_ticker: str = "SPY",
) -> pd.DataFrame:
    """
    Rolling pairwise correlation of every ticker vs *base_ticker*.

    Returns DataFrame: index = date, columns = other tickers,
    values = rolling Pearson ρ over *window* trading days.
    """
    others = [c for c in returns.columns if c != base_ticker]
    rolling = {}
    for tkr in others:
        rolling[tkr] = (
            returns[tkr]
            .rolling(window)
            .corr(returns[base_ticker])
        )
    return pd.DataFrame(rolling, index=returns.index)


# ── stress vs normal correlations ───────────────────────────────

def stress_normal_correlations(
    returns: pd.DataFrame,
    market_col: str = "SPY",
    threshold: float = -0.02,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split returns into *stress* days (market_col return ≤ threshold)
    and *normal* days, then compute the correlation matrix for each.

    Returns (stress_corr, normal_corr).
    """
    mask_stress = returns[market_col] <= threshold
    stress_corr = returns.loc[mask_stress].corr()
    normal_corr = returns.loc[~mask_stress].corr()
    return stress_corr, normal_corr


# ── eigenvalue decomposition / PCA ──────────────────────────────

def eigen_decomposition(
    returns: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Eigenvalue decomposition of the correlation matrix.

    Returns
    -------
    eigenvalues : sorted descending
    eigenvectors : columns are eigenvectors (matching eigenvalue order)
    explained_var : cumulative explained-variance ratio
    """
    corr = returns.corr().values
    vals, vecs = np.linalg.eigh(corr)
    # sort descending
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    explained = np.cumsum(vals) / vals.sum()
    return vals, vecs, explained


# ── convenience summary ─────────────────────────────────────────

def correlation_summary(
    returns: pd.DataFrame,
    rolling_window: int = 30,
    market_col: str = "SPY",
    stress_threshold: float = -0.02,
) -> Dict:
    """
    One-call summary returning all correlation artefacts.

    Keys: cov, corr, rolling, stress_corr, normal_corr,
          eigenvalues, eigenvectors, explained_var, tickers.
    """
    cov = covariance_matrix(returns)
    corr = correlation_matrix(returns)
    roll = rolling_correlations(returns, rolling_window, market_col)
    stress_corr, normal_corr = stress_normal_correlations(
        returns, market_col, stress_threshold
    )
    vals, vecs, explained = eigen_decomposition(returns)
    return dict(
        cov=cov,
        corr=corr,
        rolling=roll,
        stress_corr=stress_corr,
        normal_corr=normal_corr,
        eigenvalues=vals,
        eigenvectors=vecs,
        explained_var=explained,
        tickers=list(returns.columns),
    )
