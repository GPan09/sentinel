"""scripts/phase1_mc_run.py
Phase 1 · Deliverable 3 — Monte Carlo runner

Usage
-----
python scripts/phase1_mc_run.py \
    --tickers AAPL GOOGL MSFT BRK-B SPY \
    --period 2y \
    --n_sims 10000 \
    --horizon 1 \
    --confidence 0.95 \
    --outdir sentinel/outputs/phase1
"""

import argparse
import os
import numpy as np
import pandas as pd
import yfinance as yf

from sentinel.risk.monte_carlo import mc_summary
from sentinel.risk.mc_plots import (
    plot_pnl_distribution,
    plot_portfolio_fan,
    plot_var_convergence,
    plot_portfolio_weights,
)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1 D3: Monte Carlo simulation")
    p.add_argument("--tickers",    nargs="+", default=["AAPL","GOOGL","MSFT","BRK-B","SPY"])
    p.add_argument("--period",     default="2y")
    p.add_argument("--n_sims",     type=int,   default=10_000)
    p.add_argument("--horizon",    type=int,   default=1)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--outdir",     default="sentinel/outputs/phase1")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Downloading {args.tickers} ({args.period})...")
    prices = yf.download(args.tickers, period=args.period,
                         auto_adjust=True)["Close"]
    returns = np.log(prices / prices.shift(1)).dropna()

    print(f"Running Monte Carlo: {args.n_sims:,} sims, horizon={args.horizon}d ...")
    summary = mc_summary(
        returns,
        n_sims=args.n_sims,
        horizon=args.horizon,
        confidence=args.confidence,
        seed=args.seed,
    )

    var, cvar = summary["var"], summary["cvar"]
    print(f"  VaR  {args.confidence:.0%}: {var:.4f}")
    print(f"  CVaR {args.confidence:.0%}: {cvar:.4f}")

    # ── charts ────────────────────────────────────────────────────────────────
    plot_pnl_distribution(
        summary["terminal_pnl"], var, cvar, args.confidence,
        save_path=os.path.join(args.outdir, "09_pnl_distribution.png"),
    )
    plot_portfolio_fan(
        summary["paths"],
        save_path=os.path.join(args.outdir, "10_portfolio_fan.png"),
    )
    plot_var_convergence(
        summary["convergence_df"], args.confidence,
        save_path=os.path.join(args.outdir, "11_var_convergence.png"),
    )
    plot_portfolio_weights(
        summary["weights"], summary["tickers"],
        save_path=os.path.join(args.outdir, "12_portfolio_weights.png"),
    )

    # ── CSV summary ───────────────────────────────────────────────────────────
    csv_rows = [
        {"metric": f"MC VaR {args.confidence:.0%}",  "value": var},
        {"metric": f"MC CVaR {args.confidence:.0%}", "value": cvar},
        {"metric": "n_sims",   "value": args.n_sims},
        {"metric": "horizon",  "value": args.horizon},
        {"metric": "tickers",  "value": " ".join(summary["tickers"])},
    ]
    pd.DataFrame(csv_rows).to_csv(
        os.path.join(args.outdir, "mc_summary.csv"), index=False
    )

    print(f"Outputs saved to {args.outdir}/")


if __name__ == "__main__":
    main()
