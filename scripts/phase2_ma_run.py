"""
Phase 2 Deliverable 1 — SMA Crossover Backtester Runner
Generates charts 13–16 and backtest_summary.csv into sentinel/outputs/phase2/.

Usage
-----
    cd ~/Documents/Claude/Projects/Passion\ project\ finance/sentinel
    python3 scripts/phase2_ma_run.py

Optional flags
--------------
    --tickers  AAPL MSFT GOOGL        (default: AAPL GOOGL MSFT BRK-B SPY)
    --period   2y                     (yfinance period string, default 2y)
    --fast     20                     (fast SMA window, default 20)
    --slow     50                     (slow SMA window, default 50)
    --signal   AAPL                   (ticker for Chart 13 detail view, default first)
    --outdir   outputs/phase2         (output directory, default outputs/phase2)
"""

import argparse, pathlib, sys, os

# Allow running from repo root or scripts/ directory
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import yfinance as yf

from sentinel.backtest.engine      import Backtester
from sentinel.backtest.ma_strategy import run_sma_universe
from sentinel.backtest.backtest_plots import (
    plot_price_signals,
    plot_equity_curves,
    plot_rolling_sharpe,
    plot_drawdowns,
)


# ── arg parsing ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 2 D1 — SMA Crossover Backtest")
    p.add_argument("--tickers", nargs="+",
                   default=["AAPL", "GOOGL", "MSFT", "BRK-B", "SPY"])
    p.add_argument("--period",  default="2y")
    p.add_argument("--fast",    type=int, default=20)
    p.add_argument("--slow",    type=int, default=50)
    p.add_argument("--signal",  default=None,
                   help="Ticker for Chart 13 detail view")
    p.add_argument("--outdir",  default="outputs/phase2")
    return p.parse_args()


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    out_dir = REPO_ROOT / args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_ticker = args.signal or args.tickers[0]

    # ── 1. download prices ──────────────────────────────────────────────────────
    print(f"Downloading prices: {args.tickers}  period={args.period} …")
    raw = yf.download(args.tickers, period=args.period,
                      auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series):            # single ticker edge case
        raw = raw.to_frame(args.tickers[0])
    print(f"  {len(raw)} trading days × {raw.shape[1]} tickers")

    # ── 2. generate signals ─────────────────────────────────────────────────────
    print(f"Computing SMA{args.fast}/{args.slow} crossover signals …")
    signals = run_sma_universe(raw, fast=args.fast, slow=args.slow)

    # ── 3. backtest each ticker ─────────────────────────────────────────────────
    print("Running backtests …")
    backtests = {}
    for ticker in args.tickers:
        sig, ma_f, ma_s = signals[ticker]
        bt = Backtester(raw[ticker], sig).run()
        backtests[ticker] = bt
        m = bt.metrics()
        print(f"  {ticker:6s}  ret={m['total_return']:+.1%}  "
              f"SR={m['sharpe']:.2f}  MDD={m['max_drawdown']:.1%}  "
              f"WR={m['win_rate']:.1%}  trades={m['n_trades']}")

    # ── 4. chart 13 — price + signals (detail ticker) ──────────────────────────
    print(f"\nPlotting chart 13: {detail_ticker} price + signals …")
    sig13, maf13, mas13 = signals[detail_ticker]
    fig13 = plot_price_signals(
        prices   = raw[detail_ticker],
        ma_fast  = maf13,
        ma_slow  = mas13,
        signal   = sig13,
        ticker   = detail_ticker,
        fast     = args.fast,
        slow     = args.slow,
        save_path= str(out_dir / "13_price_signals.png"),
    )
    import matplotlib.pyplot as plt; plt.close(fig13)

    # ── 5. chart 14 — equity curves ─────────────────────────────────────────────
    print("Plotting chart 14: equity curves …")
    fig14 = plot_equity_curves(
        backtests = backtests,
        save_path = str(out_dir / "14_equity_curves.png"),
    )
    plt.close(fig14)

    # ── 6. chart 15 — rolling sharpe ────────────────────────────────────────────
    print("Plotting chart 15: rolling Sharpe …")
    fig15 = plot_rolling_sharpe(
        backtests = backtests,
        save_path = str(out_dir / "15_rolling_sharpe.png"),
    )
    plt.close(fig15)

    # ── 7. chart 16 — drawdowns ─────────────────────────────────────────────────
    print("Plotting chart 16: drawdowns …")
    fig16 = plot_drawdowns(
        backtests = backtests,
        save_path = str(out_dir / "16_drawdowns.png"),
    )
    plt.close(fig16)

    # ── 8. backtest_summary.csv ──────────────────────────────────────────────────
    print("Saving backtest_summary.csv …")
    rows = []
    for ticker, bt in backtests.items():
        m = bt.metrics()
        rows.append({"ticker": ticker, **m})
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "backtest_summary.csv", index=False)

    # ── 9. done ──────────────────────────────────────────────────────────────────
    files = sorted(out_dir.glob("*.png")) + sorted(out_dir.glob("*.csv"))
    print(f"\n✅  Done! {len(files)} files in {out_dir}")
    for f in files:
        print(f"   {f.name}")

    # ── 10. git commit ───────────────────────────────────────────────────────────
    print("\nCommitting Phase 2 D1 files to GitHub …")
    import subprocess
    os.chdir(REPO_ROOT)
    cmds = [
        ["git", "add",
         "sentinel/backtest/engine.py",
         "sentinel/backtest/ma_strategy.py",
         "sentinel/backtest/backtest_plots.py",
         "scripts/phase2_ma_run.py"],
        ["git", "commit", "-m",
         "Add Phase2 D1: SMA crossover backtester (engine, strategy, plots, runner)"],
        ["git", "push"],
    ]
    for cmd in cmds:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ⚠️  {' '.join(cmd)}\n  {result.stderr.strip()}")
        else:
            print(f"  ✓  {' '.join(cmd[:3])}")


if __name__ == "__main__":
    main()
