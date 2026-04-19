r"""
Phase 4 Deliverable 1 — HMM Regime Detection Runner
Generates charts 37–40 and regime_summary.csv into sentinel/outputs/phase4/.

The HMM is fit to SPY log returns by default (2 states = Bull/Bear).
States are re-sorted after fitting so the canonical order is
lowest-mean → highest-mean (Bear → Bull for K=2, Bear → Sideways → Bull for K=3).

Usage
-----
    cd ~/Documents/Claude/Projects/Passion\ project\ finance/sentinel
    python3 scripts/phase4_hmm_run.py
    python3 scripts/phase4_hmm_run.py --ticker SPY --states 2 --period 5y
    python3 scripts/phase4_hmm_run.py --ticker QQQ --states 3
"""

import argparse, os, pathlib, subprocess, sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sentinel.regime.hmm_detector import regime_summary
from sentinel.regime.regime_plots import (
    plot_regime_posteriors,
    plot_viterbi_on_price,
    plot_state_return_distributions,
    plot_transition_matrix,
)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 4 D1 — Gaussian HMM Regime Detection")
    p.add_argument("--ticker",  default="SPY",
                   help="Ticker to fit the HMM on (univariate for D1).")
    p.add_argument("--states",  type=int,   default=2,
                   help="Number of hidden regimes K (2 or 3).")
    p.add_argument("--period",  default="5y",
                   help="yfinance period, e.g. 5y, 10y, max.")
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--n_iter",  type=int,   default=500)
    p.add_argument("--outdir",  default="outputs/phase4")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = REPO_ROOT / args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. data ────────────────────────────────────────────────────────────────
    print(f"Downloading prices: {args.ticker}  period={args.period} …")
    raw = yf.download(args.ticker, period=args.period,
                      auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.DataFrame):
        raw = raw.iloc[:, 0]
    raw = raw.rename(args.ticker)
    log_rets = np.log(raw / raw.shift(1)).dropna().rename("log_return")
    print(f"  {len(raw)} trading days  |  "
          f"{raw.index[0].date()} → {raw.index[-1].date()}")

    # ── 2. fit HMM + analytics ────────────────────────────────────────────────
    print(f"\nFitting Gaussian HMM (K={args.states} states, seed={args.seed}) …")
    summary = regime_summary(
        log_rets,
        n_states=args.states,
        seed=args.seed,
        n_iter=args.n_iter,
    )

    chars = summary["characteristics"]
    segs  = summary["segments"]

    print("\n  State characteristics:")
    print(chars.round(5).to_string())

    print("\n  Transition matrix  A_{ij} = P(S_{t+1}=j | S_t=i):")
    print(summary["trans_matrix"].round(4).to_string())

    print("\n  Stationary distribution:")
    print((summary["stationary"] * 100).round(2).to_string())

    print("\n  Expected regime durations (days):")
    print(summary["durations"].round(1).to_string())

    print(f"\n  Log-likelihood: {summary['log_likelihood']:.2f}")
    print(f"  BIC           : {summary['bic']:.2f}   "
          f"(smaller = better; used in D2 for K selection)")
    print(f"  AIC           : {summary['aic']:.2f}")
    print(f"\n  Regime episodes: {len(segs)}")
    print(f"  Current state : {summary['current_state']}  "
          f"(posterior: "
          f"{', '.join(f'{l}={p:.0%}' for l, p in summary['current_post'].items())})")

    # ── 3. chart 37 — posterior probabilities ──────────────────────────────────
    print("\nPlotting chart 37: smoothed regime posteriors …")
    fig37 = plot_regime_posteriors(
        posteriors = summary["posteriors"],
        ticker     = args.ticker,
        save_path  = str(out_dir / "37_regime_posteriors.png"),
    )
    plt.close(fig37)

    # ── 4. chart 38 — viterbi on price ────────────────────────────────────────
    print("Plotting chart 38: Viterbi regime path on price …")
    fig38 = plot_viterbi_on_price(
        prices          = raw,
        viterbi_labeled = summary["viterbi_labeled"],
        labels          = summary["labels"],
        ticker          = args.ticker,
        save_path       = str(out_dir / "38_viterbi_path.png"),
    )
    plt.close(fig38)

    # ── 5. chart 39 — return distributions per state ──────────────────────────
    print("Plotting chart 39: return distributions by regime …")
    fig39 = plot_state_return_distributions(
        returns         = summary["returns"],
        viterbi_labeled = summary["viterbi_labeled"],
        means           = summary["means"],
        variances       = summary["variances"],
        labels          = summary["labels"],
        save_path       = str(out_dir / "39_state_distributions.png"),
    )
    plt.close(fig39)

    # ── 6. chart 40 — transition matrix + stationary ──────────────────────────
    print("Plotting chart 40: transition matrix + stationary distribution …")
    fig40 = plot_transition_matrix(
        trans_matrix = summary["trans_matrix"],
        stationary   = summary["stationary"],
        durations    = summary["durations"],
        save_path    = str(out_dir / "40_transition_matrix.png"),
    )
    plt.close(fig40)

    # ── 7. regime_summary.csv ─────────────────────────────────────────────────
    chars.to_csv(out_dir / "regime_summary.csv")
    summary["trans_matrix"].to_csv(out_dir / "transition_matrix.csv")
    segs.to_csv(out_dir / "regime_episodes.csv", index=False)
    print(f"\nSaved regime_summary.csv, transition_matrix.csv, regime_episodes.csv")

    # ── 8. done ────────────────────────────────────────────────────────────────
    new_files = ["37_regime_posteriors.png",
                 "38_viterbi_path.png",
                 "39_state_distributions.png",
                 "40_transition_matrix.png",
                 "regime_summary.csv",
                 "transition_matrix.csv",
                 "regime_episodes.csv"]
    print(f"\n✅  Done! New files in {out_dir}:")
    for f in new_files:
        if (out_dir / f).exists():
            print(f"   {f}")

    # ── 9. git commit + push ──────────────────────────────────────────────────
    print("\nCommitting Phase 4 D1 files to GitHub …")
    os.chdir(REPO_ROOT)
    cmds = [
        ["git", "add",
         "sentinel/regime/hmm_detector.py",
         "sentinel/regime/regime_plots.py",
         "scripts/phase4_hmm_run.py"],
        ["git", "commit", "-m",
         "Add Phase4 D1: Gaussian HMM regime detection (charts 37-40)"],
        ["git", "push"],
    ]
    for cmd in cmds:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ⚠️  {' '.join(cmd[:2])}: {result.stderr.strip()}")
        else:
            print(f"  ✓  {' '.join(cmd[:3])}")


if __name__ == "__main__":
    main()
