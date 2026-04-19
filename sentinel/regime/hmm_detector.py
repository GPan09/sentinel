"""
sentinel/regime/hmm_detector.py
Phase 4 Deliverable 1 — Gaussian Hidden Markov Model for Market Regimes

Why HMMs?
─────────
Markets are not a single "regime." Returns in 2017 (low vol, steady grind) and
returns in March 2020 (20% daily swings) are drawn from visibly different
distributions. A single Gaussian fit to the whole history is a lie — it averages
across regimes that behave very differently. An HMM lets us say:

    "The market is in one of K unobserved regimes. Returns depend on which
     regime we're in. The regime itself evolves over time as a Markov chain."

Formal setup
────────────
Observations:  r_1, r_2, …, r_T     (daily log returns)
Hidden states: S_1, S_2, …, S_T     (regime label ∈ {0, …, K−1})

Three ingredients parameterise the model (collectively called λ):

  1. Initial-state distribution
       π_k = P(S_1 = k)

  2. Transition matrix (Markov property — next state depends only on current)
       A_{ij} = P(S_{t+1} = j | S_t = i)
       Row i sums to 1. Diagonal A_{ii} is "regime stickiness".

  3. Emission distribution  (what returns look like GIVEN the state)
       r_t | S_t = k  ~  Normal(μ_k, σ_k²)
       Each regime has its own mean and variance.

Typical fitted values for a 2-state equity HMM:
   State "Bull":  μ ≈ +0.07% / day,  σ ≈ 0.8% / day,  A_ii ≈ 0.98
   State "Bear":  μ ≈ −0.10% / day,  σ ≈ 2.0% / day,  A_ii ≈ 0.92

Stickiness near 1 means the expected duration of a regime is long:
   Expected duration in state i = 1 / (1 − A_ii)
   e.g. A_ii = 0.98 → 50-day avg regime length.

Three algorithms you care about
───────────────────────────────
1. FORWARD–BACKWARD — computes the "smoothed posterior"
       γ_t(k) = P(S_t = k | r_1, …, r_T, λ)
   i.e. the probability that we were in regime k on day t, using ALL data
   (including future). This is what we plot — it's the cleanest view of history.

2. VITERBI — computes the single most likely state sequence
       S* = argmax_{S_{1:T}}  P(S_{1:T} | r_{1:T}, λ)
   Dynamic programming over the state trellis. Output is a hard label per day.
   Use Viterbi for "what regime was the market in?" decisions.

3. BAUM–WELCH (an EM algorithm) — fits the parameters λ = (π, A, μ, σ)
   by maximising likelihood P(r | λ). We don't implement it — hmmlearn does.

Label identifiability
─────────────────────
HMMs are invariant under permutation of state labels (state 0 and state 1 are
interchangeable from the model's perspective). After fitting we re-sort states
by mean return: lowest-mean → Bear, highest-mean → Bull. This keeps the
labelling stable across tickers, windows, and random seeds.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd

# hmmlearn emits deprecation/convergence warnings that are fine to ignore here.
warnings.filterwarnings("ignore", category=UserWarning,   module="hmmlearn")
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── default labels for K-state models ──────────────────────────────────────────
# Sorted from lowest-mean to highest-mean after fitting.
DEFAULT_LABELS: Dict[int, List[str]] = {
    2: ["Bear", "Bull"],
    3: ["Bear", "Sideways", "Bull"],
    4: ["Crash", "Bear", "Sideways", "Bull"],
}


# ── core fit ───────────────────────────────────────────────────────────────────

def fit_hmm(
    returns: pd.Series,
    n_states: int = 2,
    covariance_type: str = "full",
    n_iter: int = 500,
    seed: int = 42,
) -> dict:
    """
    Fit a Gaussian HMM to a univariate return series via Baum-Welch (EM).

    Parameters
    ----------
    returns : pd.Series
        Log returns, indexed by date. NaNs are dropped.
    n_states : int
        Number of hidden regimes K. 2 = bull/bear, 3 = adds sideways.
    covariance_type : str
        Irrelevant for 1-D (all types collapse to scalar variance) but kept for
        API consistency with multivariate extension in D2.
    n_iter : int
        Max EM iterations.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys:
        model            : fitted hmmlearn GaussianHMM
        returns          : cleaned return series used for fitting
        posteriors       : pd.DataFrame [T × K] — smoothed P(S_t=k | all data)
        viterbi          : pd.Series  [T]     — most likely state sequence (raw int)
        viterbi_labeled  : pd.Series  [T]     — re-indexed so 0=lowest-μ, K−1=highest-μ
        state_order      : list[int]          — raw indices sorted by mean
        means            : np.array [K]       — sorted state means
        variances        : np.array [K]       — sorted state variances
        trans_matrix     : pd.DataFrame [K×K] — relabeled transition matrix
        start_prob       : np.array [K]       — relabeled initial distribution
        log_likelihood   : float              — final log-likelihood
        n_params         : int                — free parameters (for BIC in D2)
        labels           : list[str]          — DEFAULT_LABELS[K] or generated
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as e:
        raise ImportError(
            "hmmlearn not installed. `pip install hmmlearn` (already in "
            "requirements.txt)."
        ) from e

    r = returns.dropna().astype(float)
    X = r.values.reshape(-1, 1)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=seed,
        tol=1e-4,
    )
    model.fit(X)

    # — posterior probabilities (forward-backward) and Viterbi path —
    post = model.predict_proba(X)       # shape (T, K)
    viterbi_raw = model.predict(X)      # shape (T,)

    # — sort states by mean return so 0 is always lowest, K-1 always highest —
    raw_means = model.means_.flatten()
    state_order = np.argsort(raw_means).tolist()
    index_map = {raw: new for new, raw in enumerate(state_order)}

    sorted_means = raw_means[state_order]
    sorted_vars  = model.covars_.reshape(n_states, -1).sum(axis=1)[state_order]

    # relabel Viterbi path and posteriors
    viterbi_labeled = pd.Series(
        [index_map[s] for s in viterbi_raw], index=r.index, name="state"
    )
    post_labeled = post[:, state_order]

    # relabel transition matrix: A'[i,j] = A[order[i], order[j]]
    trans_raw = model.transmat_
    trans_lab = trans_raw[np.ix_(state_order, state_order)]
    start_lab = model.startprob_[state_order]

    # — pick friendly labels —
    labels = DEFAULT_LABELS.get(n_states,
                                [f"S{i}" for i in range(n_states)])

    posteriors_df = pd.DataFrame(post_labeled, index=r.index, columns=labels)
    trans_df      = pd.DataFrame(trans_lab, index=labels, columns=labels)

    # — log-likelihood and free-param count (for BIC in D2) —
    log_lik = float(model.score(X))
    # params: K means + K variances + K(K-1) free transition probs + (K-1) start
    n_params = n_states + n_states + n_states * (n_states - 1) + (n_states - 1)

    return {
        "model":           model,
        "returns":         r,
        "posteriors":      posteriors_df,
        "viterbi":         pd.Series(viterbi_raw, index=r.index, name="state_raw"),
        "viterbi_labeled": viterbi_labeled,
        "state_order":     state_order,
        "means":           sorted_means,
        "variances":       sorted_vars,
        "trans_matrix":    trans_df,
        "start_prob":      start_lab,
        "log_likelihood":  log_lik,
        "n_params":        n_params,
        "labels":          labels,
    }


# ── regime analytics ───────────────────────────────────────────────────────────

def stationary_distribution(trans_matrix: pd.DataFrame) -> pd.Series:
    """
    Long-run proportion of time spent in each state.

    Solves  π A = π   subject to  Σπ = 1.
    Equivalently: π is the left eigenvector of A at eigenvalue 1,
    so we find the right eigenvector of A^T at eigenvalue 1.

    Intuition: if you simulate the Markov chain forward forever and count
    the fraction of time in state k, you get π_k. For a 2-state chain
    with A_00 = a, A_11 = b,
         π_0 = (1−b) / (2 − a − b)
         π_1 = (1−a) / (2 − a − b)
    """
    A = trans_matrix.values
    w, v = np.linalg.eig(A.T)
    # pick eigenvector closest to eigenvalue 1
    idx   = np.argmin(np.abs(w - 1.0))
    stat  = np.real(v[:, idx])
    stat  = stat / stat.sum()
    return pd.Series(stat, index=trans_matrix.index, name="stationary")


def expected_durations(trans_matrix: pd.DataFrame) -> pd.Series:
    """
    Expected time (in days) spent in each state before leaving.

    The time until first transition out of state i is Geometric(1 − A_ii),
    which has mean  1 / (1 − A_ii).

    A stickiness of 0.98 → ~50-day regimes. 0.95 → ~20 days. 0.80 → ~5 days.
    """
    diag = np.diag(trans_matrix.values)
    dur  = 1.0 / (1.0 - diag + 1e-12)
    return pd.Series(dur, index=trans_matrix.index, name="expected_duration_days")


def state_characteristics(result: dict) -> pd.DataFrame:
    """
    One-row-per-state summary: mean, std, annualized stats, stationary prob,
    expected duration, realized frequency in Viterbi path.

    Annualization uses the standard √252 rule for σ and ×252 for μ.
    Sharpe is computed ignoring risk-free rate (typical for regime comparison).
    """
    means      = result["means"]
    stds       = np.sqrt(result["variances"])
    labels     = result["labels"]
    stat       = stationary_distribution(result["trans_matrix"])
    dur        = expected_durations(result["trans_matrix"])
    viterbi_lb = result["viterbi_labeled"]

    freq = viterbi_lb.value_counts(normalize=True).reindex(range(len(labels))).fillna(0)
    freq.index = labels

    rows = []
    for i, lab in enumerate(labels):
        mu, s = means[i], stds[i]
        rows.append({
            "state":                    lab,
            "mean_daily_return":        mu,
            "daily_volatility":         s,
            "annualized_return":        mu * 252,
            "annualized_volatility":    s * np.sqrt(252),
            "sharpe_approx":            (mu * 252) / (s * np.sqrt(252) + 1e-12),
            "stationary_prob":          float(stat.iloc[i]),
            "expected_duration_days":   float(dur.iloc[i]),
            "realized_frequency":       float(freq.iloc[i]),
        })
    return pd.DataFrame(rows).set_index("state")


def regime_segments(viterbi_labeled: pd.Series, labels: Sequence[str]) -> pd.DataFrame:
    """
    Collapse a Viterbi path into contiguous regime episodes.

    Returns DataFrame with columns:
        start, end, state_idx, state, duration_days
    """
    v   = viterbi_labeled.dropna()
    if v.empty:
        return pd.DataFrame(columns=["start", "end", "state_idx",
                                     "state", "duration_days"])

    changes = v.ne(v.shift()).cumsum()
    rows = []
    for _, chunk in v.groupby(changes):
        s_idx = int(chunk.iloc[0])
        rows.append({
            "start":         chunk.index[0],
            "end":           chunk.index[-1],
            "state_idx":     s_idx,
            "state":         labels[s_idx],
            "duration_days": (chunk.index[-1] - chunk.index[0]).days + 1,
        })
    return pd.DataFrame(rows)


# ── end-to-end pipeline ───────────────────────────────────────────────────────

def regime_summary(
    returns: pd.Series,
    n_states: int = 2,
    seed: int = 42,
    n_iter: int = 500,
) -> dict:
    """
    Full HMM pipeline: fit + analytics + segments.

    Returns dict with everything `fit_hmm` provides PLUS:
        characteristics : per-state stats DataFrame
        segments        : regime episode DataFrame
        stationary      : stationary distribution Series
        durations       : expected durations Series
        current_state   : label of the most recent day's state (Viterbi)
        current_post    : posterior probabilities of the most recent day
        bic             : Bayesian Information Criterion (hooked for D2)
        aic             : Akaike Information Criterion
    """
    result = fit_hmm(returns, n_states=n_states, seed=seed, n_iter=n_iter)
    labels = result["labels"]

    chars  = state_characteristics(result)
    segs   = regime_segments(result["viterbi_labeled"], labels)
    stat   = stationary_distribution(result["trans_matrix"])
    dur    = expected_durations(result["trans_matrix"])

    T = len(result["returns"])
    k = result["n_params"]
    ll = result["log_likelihood"]
    bic = -2 * ll + k * np.log(T)
    aic = -2 * ll + 2 * k

    current_state = labels[int(result["viterbi_labeled"].iloc[-1])]
    current_post  = result["posteriors"].iloc[-1]

    return {
        **result,
        "characteristics": chars,
        "segments":        segs,
        "stationary":      stat,
        "durations":       dur,
        "current_state":   current_state,
        "current_post":    current_post,
        "bic":             float(bic),
        "aic":             float(aic),
    }
