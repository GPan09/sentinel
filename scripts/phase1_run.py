"""
scripts/phase1_run.py
=====================
Phase 1 entry point: Portfolio Risk Analyzer — VaR (three methods).

Usage
-----
    python scripts/phase1_run.py
    python scripts/phase1_run.py --tickers AAPL MSFT SPY --confidence 0.95 0.99

Output
------
    outputs/phase1/01_var_comparison.png
    outputs/phase1/02_return_histograms.png
    outputs/phase1/03_var_surface.png
    outputs/phase1/04_es_vs_var.png
    outputs/phase1/var_summary.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from sentinel.data.fetcher import fetch_prices, compute_log_returns
from sentinel.risk.var import var_summary, portfolio_parametric_var, portfolio_mc_var
from sentinel.risk.plots import (
    plot_var_comparison,
    plot_return_histograms,
    plot_var_surface,
    plot_es_vs_var,
)

DEFAULT_TICKERS = ["AAPL", "GOOGL", "MSFT", "BRK-B", "SPY"]
DEFAULT_START   = "2019-01-01"
DEFAULT_END     = "2024-12-31"
DEFAULT_CONFS   = [0.90, 0.95, 0.99]
OUTPUT_DIR      = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "phase1",
)


def parse_args():
    p = argparse.ArgumentParser(description="Sentinel Phase 1 — VaR")
    p.add_argument("--tickers",    nargs="+", default=DEFAULT_TICKERS)
    p.add_argument("--start",      default=DEFAULT_START)
    p.add_argument("--end",        default=DEFAULT_END)
    p.add_argument("--confidence", nargs="+", type=float, default=DEFAULT_CONFS)
    p.add_argument("--n-sim",      type=int,   default=10_000)
    p.add_argument("--horizon",    type=int,   default=1)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("SENTINEL  Phase 1 — Value at Risk")
    print("=" * 60)
    print(f"Tickers   : {args.tickers}")
    print(f"Period    : {args.start} -> {args.end}")
    print(f"Confidence: {[f'{c:.0%}' for c in args.confidence]}")
    print(f"Horizon   : {args.horizon} day(s)")
    print(f"MC sims   : {args.n_sim:,}")
    print()

    print("Fetching prices ...")
    prices  = fetch_prices(args.tickers, args.start, args.end)
    returns = compute_log_returns(prices)
    print(f"Returns shape: {returns.shape}\n")

    print("Computing VaR (Historical, Parametric, Monte Carlo) ...")
    vdf = var_summary(
        returns,
        confidence_levels=args.confidence,
        horizon=args.horizon,
        n_simulations=args.n_sim,
    )
    print(vdf.to_string(index=False))
    print()

    n_assets = len(args.tickers)
    weights  = np.full(n_assets, 1.0 / n_assets)
    conf95   = 0.95

    p_var, p_es = portfolio_parametric_var(returns, weights, conf95, args.horizon)
    m_var, m_es = portfolio_mc_var(returns, weights, conf95, args.horizon, args.n_sim)

    print(f"Equal-weight portfolio VaR at {conf95:.0%}:")
    print(f"  Parametric : {p_var:.4%}  (ES: {p_es:.4%})")
    print(f"  Monte Carlo: {m_var:.4%}  (ES: {m_es:.4%})\n")

    print("Generating charts ...")
    plot_var_comparison(
        vdf, confidence=conf95,
        save_path=os.path.join(OUTPUT_DIR, "01_var_comparison.png"),
    )
    plot_return_histograms(
        returns, vdf, confidence=conf95,
        save_path=os.path.join(OUTPUT_DIR, "02_return_histograms.png"),
    )
    plot_var_surface(
        vdf, method="hist",
        save_path=os.path.join(OUTPUT_DIR, "03_var_surface.png"),
    )
    plot_es_vs_var(
        vdf, confidence=conf95,
        save_path=os.path.join(OUTPUT_DIR, "04_es_vs_var.png"),
    )

    csv_path = os.path.join(OUTPUT_DIR, "var_summary.csv")
    vdf.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}\n")
    print("Phase 1 complete.")


if __name__ == "__main__":
    main()
