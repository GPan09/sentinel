"""
Sentinel – Phase 1 Correlation Runner
======================================
Entry-point script that fetches prices, computes log returns,
runs the full correlation analysis, and saves 4 PNGs + CSV.

Usage (from repo root):
    python scripts/phase1_corr_run.py --tickers AAPL GOOGL MSFT BRK-B SPY \
                                       --period 2y --window 30
"""

import argparse, pathlib, sys, os

# ensure repo root is on the path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
from sentinel.data.fetcher import fetch_prices, compute_log_returns
from sentinel.risk.correlation import correlation_summary
from sentinel.risk.corr_plots import (
    plot_correlation_heatmap,
    plot_rolling_correlations,
    plot_stress_vs_normal,
    plot_eigenvalue_scree,
)


def main():
    ap = argparse.ArgumentParser(description="Phase 1 – Correlation Analysis")
    ap.add_argument("--tickers", nargs="+",
                    default=["AAPL", "GOOGL", "MSFT", "BRK-B", "SPY"])
    ap.add_argument("--period", default="2y")
    ap.add_argument("--window", type=int, default=30,
                    help="Rolling-correlation window (trading days)")
    ap.add_argument("--outdir", default="outputs/phase1")
    args = ap.parse_args()

    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # ── data ────────────────────────────────────────────────────
    print(f"Fetching {args.tickers} ({args.period}) …")
    prices = fetch_prices(args.tickers, period=args.period)
    returns = compute_log_returns(prices)
    print(f"Returns: {returns.shape[0]} rows × {returns.shape[1]} tickers")

    # ── analysis ────────────────────────────────────────────────
    summary = correlation_summary(
        returns,
        rolling_window=args.window,
        market_col="SPY",
        stress_threshold=-0.02,
    )

    # ── charts ──────────────────────────────────────────────────
    plot_correlation_heatmap(summary["corr"],
                            save_path=out / "05_correlation_heatmap.png")
    print("✓ 05_correlation_heatmap.png")

    plot_rolling_correlations(summary["rolling"], window=args.window,
                              save_path=out / "06_rolling_correlations.png")
    print("✓ 06_rolling_correlations.png")

    plot_stress_vs_normal(summary["stress_corr"], summary["normal_corr"],
                          save_path=out / "07_stress_vs_normal.png")
    print("✓ 07_stress_vs_normal.png")

    plot_eigenvalue_scree(summary["eigenvalues"], summary["explained_var"],
                          summary["tickers"],
                          save_path=out / "08_eigenvalue_scree.png")
    print("✓ 08_eigenvalue_scree.png")

    # ── CSV summary ─────────────────────────────────────────────
    corr = summary["corr"]
    corr.to_csv(out / "correlation_matrix.csv")
    print(f"✓ correlation_matrix.csv")

    # print key findings
    print("\n══ Correlation Matrix ══")
    print(corr.to_string(float_format="{:.4f}".format))

    print(f"\n══ Eigenvalues ══")
    for i, (v, e) in enumerate(zip(summary["eigenvalues"],
                                    summary["explained_var"])):
        print(f"  PC{i+1}: λ={v:.4f}  cumulative={e*100:.1f}%")

    stress_mean = summary["stress_corr"].values[
        ~pd.np.eye(len(summary["tickers"]), dtype=bool)
    ].mean() if hasattr(pd, 'np') else summary["stress_corr"].values[
        ~np.eye(len(summary["tickers"]), dtype=bool)
    ].mean()
    normal_mean = summary["normal_corr"].values[
        ~np.eye(len(summary["tickers"]), dtype=bool)
    ].mean()
    print(f"\n══ Stress vs Normal ══")
    print(f"  Mean off-diag ρ (stress): {stress_mean:.4f}")
    print(f"  Mean off-diag ρ (normal): {normal_mean:.4f}")
    print(f"  Δ = {stress_mean - normal_mean:+.4f}  ← correlation spike in sell-offs")

    print("\nDone ✓")


if __name__ == "__main__":
    import numpy as np  # needed for the mask above
    main()
