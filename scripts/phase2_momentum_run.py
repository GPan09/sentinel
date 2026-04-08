"""
Phase 2 Deliverable 2 — Momentum Strategy Runner
Generates charts 17–20 and momentum_summary.csv into sentinel/outputs/phase2/.

Usage
-----
    cd ~/Documents/Claude/Projects/Passion\ project\ finance/sentinel
    python3 scripts/phase2_momentum_run.py

Optional flags
--------------
    --tickers  AAPL MSFT GOOGL BRK-B SPY
    --period   2y
    --lookback 252    (trailing return window in days, default 252 = 1 year)
    --skip     21     (short-term reversal skip, default 21 = 1 month)
    --top-k    2      (number of tickers to hold in CS-mom portfolio)
    --outdir   outputs/phase2
"""

import argparse, pathlib, sys, os, subprocess

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sentinel.backtest.momentum import (
    tsmom_signals,
    csmom_signals,
    portfolio_returns_from_signals,
    equity_curve_from_returns,
)
from sentinel.backtest.momentum_plots import (
    plot_momentum_heatmap,
    plot_csmom_equity,
    plot_tsmom_equities,
    plot_strategy_correlation,
)
# reuse SMA crossover results for the correlation chart
from sentinel.backtest.ma_strategy import run_sma_universe
from sentinel.backtest.engine import Backtester


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2 D2 — Momentum Backtest")
    p.add_argument("--tickers",  nargs="+",
                   default=["AAPL", "GOOGL", "MSFT", "BRK-B", "SPY"])
    p.add_argument("--period",   default="2y")
    p.add_argument("--lookback", type=int, default=252)
    p.add_argument("--skip",     type=int, default=21)
    p.add_argument("--top-k",    type=int, default=2, dest="top_k")
    p.add_argument("--outdir",   default="outputs/phase2")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = REPO_ROOT / args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. download data ───────────────────────────────────────────────────────
    print(f"Downloading prices: {args.tickers}  period={args.period} …")
    raw = yf.download(args.tickers, period=args.period,
                      auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(args.tickers[0])
    print(f"  {len(raw)} trading days × {raw.shape[1]} tickers")

    log_rets = np.log(raw / raw.shift(1))

    # ── 2. time-series momentum signals ───────────────────────────────────────
    print(f"Computing TS-Mom signals (lookback={args.lookback}, skip={args.skip}) …")
    ts_signals, scores = tsmom_signals(raw, args.lookback, args.skip)

    # ── 3. cross-sectional momentum signals ───────────────────────────────────
    print(f"Computing CS-Mom signals (top-{args.top_k}) …")
    cs_signals, _ = csmom_signals(raw, args.lookback, args.skip, args.top_k)

    # ── 4. equity curves ───────────────────────────────────────────────────────
    # TS-mom: per-ticker (equal weight within signal)
    tsmom_equities = {}
    bh_equities    = {}
    for ticker in args.tickers:
        sig1d = ts_signals[[ticker]]
        r_ts  = portfolio_returns_from_signals(raw[[ticker]], sig1d)
        r_bh  = log_rets[ticker].fillna(0)
        tsmom_equities[ticker] = equity_curve_from_returns(r_ts)
        bh_equities[ticker]    = equity_curve_from_returns(r_bh)

    # CS-mom: portfolio of top-k tickers, rebalanced monthly
    r_cs  = portfolio_returns_from_signals(raw, cs_signals)
    r_ew  = log_rets.mean(axis=1).fillna(0)   # equal-weight benchmark
    csmom_equity = equity_curve_from_returns(r_cs)
    ew_equity    = equity_curve_from_returns(r_ew)

    # Print TS-mom stats
    print("\nTime-Series Momentum results:")
    for ticker in args.tickers:
        eq = tsmom_equities[ticker]
        ret = eq.iloc[-1] / eq.iloc[0] - 1
        bh  = bh_equities[ticker].iloc[-1] / bh_equities[ticker].iloc[0] - 1
        print(f"  {ticker:6s}  TS-mom={ret:+.1%}  B&H={bh:+.1%}")

    cs_ret = csmom_equity.iloc[-1] / csmom_equity.iloc[0] - 1
    ew_ret = ew_equity.iloc[-1]    / ew_equity.iloc[0]    - 1
    print(f"\nCross-Sectional Momentum (top-{args.top_k}): {cs_ret:+.1%}  "
          f"vs EW B&H: {ew_ret:+.1%}")

    # ── 5. chart 17 — momentum score heatmap ──────────────────────────────────
    print("\nPlotting chart 17: momentum score heatmap …")
    fig17 = plot_momentum_heatmap(
        scores    = scores,
        save_path = str(out_dir / "17_momentum_heatmap.png"),
    )
    plt.close(fig17)

    # ── 6. chart 18 — CS-mom equity curve ─────────────────────────────────────
    print("Plotting chart 18: CS-mom equity curve …")
    fig18 = plot_csmom_equity(
        csmom_equity = csmom_equity,
        bh_equity    = ew_equity,
        top_k        = args.top_k,
        n_total      = len(args.tickers),
        save_path    = str(out_dir / "18_csmom_equity.png"),
    )
    plt.close(fig18)

    # ── 7. chart 19 — TS-mom equity per ticker ────────────────────────────────
    print("Plotting chart 19: TS-mom equity curves …")
    fig19 = plot_tsmom_equities(
        tsmom_equities = tsmom_equities,
        bh_equities    = bh_equities,
        save_path      = str(out_dir / "19_tsmom_equities.png"),
    )
    plt.close(fig19)

    # ── 8. chart 20 — strategy correlation ────────────────────────────────────
    print("Plotting chart 20: strategy correlation matrix …")

    # Build SMA crossover portfolio return series for comparison
    sma_sigs = run_sma_universe(raw, fast=20, slow=50)
    r_sma_list = []
    for ticker in args.tickers:
        sig, _, _ = sma_sigs[ticker]
        bt = Backtester(raw[ticker], sig).run()
        r_sma_list.append(bt.returns_strat.rename(ticker))
    r_sma_port = pd.concat(r_sma_list, axis=1).mean(axis=1).rename("SMA20/50")

    strategy_returns = {
        "SMA 20/50":  r_sma_port,
        "TS-Mom":     portfolio_returns_from_signals(raw, ts_signals),
        "CS-Mom":     r_cs,
        "EW B&H":     r_ew,
    }
    fig20 = plot_strategy_correlation(
        returns_dict = strategy_returns,
        save_path    = str(out_dir / "20_strategy_correlation.png"),
    )
    plt.close(fig20)

    # ── 9. momentum_summary.csv ───────────────────────────────────────────────
    print("Saving momentum_summary.csv …")
    rows = []
    for ticker in args.tickers:
        eq = tsmom_equities[ticker]
        bh = bh_equities[ticker]
        ts_r = log_rets[ticker].reindex(eq.index)
        ts_strat_r = portfolio_returns_from_signals(
            raw[[ticker]], ts_signals[[ticker]])
        rows.append({
            "ticker":        ticker,
            "tsmom_return":  eq.iloc[-1] / eq.iloc[0] - 1,
            "bh_return":     bh.iloc[-1] / bh.iloc[0] - 1,
            "active_days_pct": ts_signals[ticker].mean(),
        })
    rows.append({
        "ticker":        f"CS-Mom top-{args.top_k}",
        "tsmom_return":  cs_ret,
        "bh_return":     ew_ret,
        "active_days_pct": cs_signals.mean().mean(),
    })
    pd.DataFrame(rows).to_csv(out_dir / "momentum_summary.csv", index=False)

    # ── 10. done ──────────────────────────────────────────────────────────────
    files = sorted(out_dir.glob("17_*.png")) + \
            sorted(out_dir.glob("18_*.png")) + \
            sorted(out_dir.glob("19_*.png")) + \
            sorted(out_dir.glob("20_*.png")) + \
            [out_dir / "momentum_summary.csv"]
    print(f"\n✅  Done! New files in {out_dir}:")
    for f in files:
        if f.exists():
            print(f"   {f.name}")

    # ── 11. git commit ────────────────────────────────────────────────────────
    print("\nCommitting Phase 2 D2 files to GitHub …")
    os.chdir(REPO_ROOT)
    cmds = [
        ["git", "add",
         "sentinel/backtest/momentum.py",
         "sentinel/backtest/momentum_plots.py",
         "scripts/phase2_momentum_run.py"],
        ["git", "commit", "-m",
         "Add Phase2 D2: momentum strategy (TS-mom, CS-mom, charts 17-20)"],
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
