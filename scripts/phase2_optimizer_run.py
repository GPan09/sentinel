"""
Phase 2 Deliverable 3 — Strategy Optimizer & Scorecard Runner
Generates charts 21–24 and scorecard.csv into sentinel/outputs/phase2/.

Usage
-----
    cd ~/Documents/Claude/Projects/Passion\ project\ finance/sentinel
    python3 scripts/phase2_optimizer_run.py
"""

import argparse, pathlib, sys, os, subprocess

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sentinel.backtest.optimizer import (
    sma_grid_search_universe,
    ensemble_signals,
    portfolio_equity,
    monthly_returns_calendar,
    build_scorecard,
)
from sentinel.backtest.optimizer_plots import (
    plot_sma_heatmap,
    plot_ensemble_equity,
    plot_monthly_calendar,
    plot_scorecard_table,
)
from sentinel.backtest.ma_strategy import run_sma_universe
from sentinel.backtest.momentum import (
    tsmom_signals,
    portfolio_returns_from_signals,
    equity_curve_from_returns,
)
from sentinel.backtest.engine import Backtester


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2 D3 — Optimizer & Scorecard")
    p.add_argument("--tickers", nargs="+",
                   default=["AAPL", "GOOGL", "MSFT", "BRK-B", "SPY"])
    p.add_argument("--period", default="2y")
    p.add_argument("--outdir", default="outputs/phase2")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = REPO_ROOT / args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. data ────────────────────────────────────────────────────────────────
    print(f"Downloading prices: {args.tickers}  period={args.period} …")
    raw = yf.download(args.tickers, period=args.period,
                      auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(args.tickers[0])
    log_rets = np.log(raw / raw.shift(1))
    print(f"  {len(raw)} trading days × {raw.shape[1]} tickers")

    # ── 2. chart 21 — SMA parameter grid ──────────────────────────────────────
    print("Computing SMA parameter grid (this takes ~30 seconds) …")
    grid = sma_grid_search_universe(
        raw,
        fast_range=range(5, 45, 5),
        slow_range=range(20, 130, 10),
    )
    best_fast = grid.stack().idxmax()[0]
    best_slow = grid.stack().idxmax()[1]
    print(f"  Best SMA pair: FA{best_fast}/SL{best_slow}  "
          f"Sharpe={grid.loc[best_fast, best_slow]:.2f}")

    fig21 = plot_sma_heatmap(
        grid      = grid,
        save_path = str(out_dir / "21_sma_heatmap.png"),
    )
    plt.close(fig21)

    # ── 3. ensemble signals ────────────────────────────────────────────────────
    print("Building ensemble signals …")
    ens_union, sma_sig, mom_sig = ensemble_signals(
        raw, sma_fast=20, sma_slow=50,
        mom_lookback=252, mom_skip=21, mode="union")
    ens_inter, _, _ = ensemble_signals(
        raw, sma_fast=20, sma_slow=50,
        mom_lookback=252, mom_skip=21, mode="intersection")

    # equity curves
    initial = 10_000.0
    eq_union  = portfolio_equity(raw, ens_union,  initial)
    eq_inter  = portfolio_equity(raw, ens_inter,  initial)
    eq_bh     = equity_curve_from_returns(log_rets.mean(axis=1).fillna(0), initial)

    # SMA-only portfolio (equal-weight across tickers)
    sma_sigs  = run_sma_universe(raw, fast=20, slow=50)
    r_sma_list = []
    for ticker in args.tickers:
        sig, _, _ = sma_sigs[ticker]
        bt = Backtester(raw[ticker], sig).run()
        r_sma_list.append(bt.returns_strat.rename(ticker))
    r_sma_port = pd.concat(r_sma_list, axis=1).mean(axis=1)
    eq_sma = equity_curve_from_returns(r_sma_port, initial)

    # TS-mom portfolio
    ts_sig, _ = tsmom_signals(raw, 252, 21)
    r_ts  = portfolio_returns_from_signals(raw, ts_sig)
    eq_ts = equity_curve_from_returns(r_ts, initial)

    # ── 4. chart 22 — ensemble equity ─────────────────────────────────────────
    print("Plotting chart 22: ensemble equity curves …")
    fig22 = plot_ensemble_equity(
        equities={
            "Ensemble (Union)":        eq_union,
            "Ensemble (Intersection)": eq_inter,
            "SMA 20/50":               eq_sma,
            "TS-Mom":                  eq_ts,
            "EW B&H":                  eq_bh,
        },
        save_path=str(out_dir / "22_ensemble_equity.png"),
    )
    plt.close(fig22)

    # ── 5. chart 23 — monthly returns calendar (ensemble union) ───────────────
    print("Plotting chart 23: monthly returns calendar …")
    r_union = portfolio_returns_from_signals(raw, ens_union)
    cal = monthly_returns_calendar(r_union)
    fig23 = plot_monthly_calendar(
        calendar      = cal,
        strategy_name = "Ensemble (Union)",
        save_path     = str(out_dir / "23_monthly_calendar.png"),
    )
    plt.close(fig23)

    # ── 6. chart 24 — performance scorecard ───────────────────────────────────
    print("Plotting chart 24: performance scorecard …")
    r_union = portfolio_returns_from_signals(raw, ens_union)
    r_inter = portfolio_returns_from_signals(raw, ens_inter)
    r_ew    = log_rets.mean(axis=1).fillna(0)

    scorecard = build_scorecard({
        "SMA 20/50":               r_sma_port,
        "TS-Mom":                  r_ts,
        "Ensemble (Union)":        r_union,
        "Ensemble (Intersection)": r_inter,
        "EW B&H":                  r_ew,
    })
    print("\n" + scorecard.to_string())

    fig24 = plot_scorecard_table(
        scorecard = scorecard,
        save_path = str(out_dir / "24_scorecard.png"),
    )
    plt.close(fig24)

    # ── 7. save scorecard CSV ──────────────────────────────────────────────────
    scorecard.to_csv(out_dir / "scorecard.csv")

    # ── 8. done ───────────────────────────────────────────────────────────────
    new_files = ["21_sma_heatmap.png", "22_ensemble_equity.png",
                 "23_monthly_calendar.png", "24_scorecard.png", "scorecard.csv"]
    print(f"\n✅  Done! New files in {out_dir}:")
    for f in new_files:
        if (out_dir / f).exists():
            print(f"   {f}")

    # ── 9. git commit + push ──────────────────────────────────────────────────
    print("\nCommitting Phase 2 D3 files to GitHub …")
    os.chdir(REPO_ROOT)
    cmds = [
        ["git", "add",
         "sentinel/backtest/optimizer.py",
         "sentinel/backtest/optimizer_plots.py",
         "scripts/phase2_optimizer_run.py"],
        ["git", "commit", "-m",
         "Add Phase2 D3: SMA grid search, ensemble strategy, scorecard (charts 21-24)"],
        ["git", "push", "-u", "origin", "main"],   # -u sets upstream permanently
    ]
    for cmd in cmds:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ⚠️  {' '.join(cmd[:2])}: {result.stderr.strip()}")
        else:
            print(f"  ✓  {' '.join(cmd[:3])}")


if __name__ == "__main__":
    main()
