"""
Phase 3 Deliverable 2 — Fat-Tail Distribution Runner
Generates charts 29–32 and fattail_summary.csv into sentinel/outputs/phase3/.

Usage
-----
    cd ~/Documents/Claude/Projects/Passion\ project\ finance/sentinel
    python3 scripts/phase3_fattail_run.py
"""

import argparse, pathlib, sys, os, subprocess

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sentinel.crash.fat_tails import fit_universe, metrics_universe
from sentinel.crash.fat_tail_plots import (
    plot_distribution_fits,
    plot_qq,
    plot_tail_dashboard,
    plot_var_underestimation,
)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 3 D2 — Fat-Tail Distributions")
    p.add_argument("--tickers", nargs="+",
                   default=["AAPL", "GOOGL", "MSFT", "BRK-B", "SPY"])
    p.add_argument("--period",  default="3y")
    p.add_argument("--outdir",  default="outputs/phase3")
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
    log_rets = np.log(raw / raw.shift(1)).dropna()
    print(f"  {len(raw)} trading days × {raw.shape[1]} tickers")

    # ── 2. fit distributions ───────────────────────────────────────────────────
    print("Fitting Normal and Student-t distributions …")
    fits = fit_universe(raw)

    print("\nStudent-t degrees of freedom (lower = fatter tails):")
    for ticker, f in fits.items():
        tf = f["Student-t"]
        print(f"  {ticker:6s}  ν={tf['df']:.2f}")

    # ── 3. tail metrics ────────────────────────────────────────────────────────
    print("\nComputing tail risk metrics …")
    metrics = metrics_universe(raw)
    print("\n" + metrics[["excess_kurtosis", "skewness",
                           "tail_ratio_3sigma", "jb_pvalue"]].to_string())

    # ── 4. chart 29 — distribution fits ───────────────────────────────────────
    print("\nPlotting chart 29: distribution fits …")
    fig29 = plot_distribution_fits(log_rets, fits,
                                   save_path=str(out_dir / "29_distribution_fits.png"))
    plt.close(fig29)

    # ── 5. chart 30 — QQ plots ────────────────────────────────────────────────
    print("Plotting chart 30: QQ plots …")
    fig30 = plot_qq(log_rets, fits,
                    save_path=str(out_dir / "30_qq_plots.png"))
    plt.close(fig30)

    # ── 6. chart 31 — tail risk dashboard ─────────────────────────────────────
    print("Plotting chart 31: tail risk dashboard …")
    fig31 = plot_tail_dashboard(metrics,
                                save_path=str(out_dir / "31_tail_dashboard.png"))
    plt.close(fig31)

    # ── 7. chart 32 — VaR underestimation ─────────────────────────────────────
    print("Plotting chart 32: VaR underestimation …")
    fig32 = plot_var_underestimation(metrics,
                                     save_path=str(out_dir / "32_var_underestimation.png"))
    plt.close(fig32)

    # ── 8. save summary ────────────────────────────────────────────────────────
    metrics.to_csv(out_dir / "fattail_summary.csv")
    print("Saved fattail_summary.csv")

    # ── 9. done ────────────────────────────────────────────────────────────────
    new_files = ["29_distribution_fits.png", "30_qq_plots.png",
                 "31_tail_dashboard.png", "32_var_underestimation.png",
                 "fattail_summary.csv"]
    print(f"\n✅  Done! New files in {out_dir}:")
    for f in new_files:
        if (out_dir / f).exists():
            print(f"   {f}")

    # ── 10. git commit + push ─────────────────────────────────────────────────
    print("\nCommitting Phase 3 D2 files to GitHub …")
    os.chdir(REPO_ROOT)
    cmds = [
        ["git", "add",
         "sentinel/crash/fat_tails.py",
         "sentinel/crash/fat_tail_plots.py",
         "scripts/phase3_fattail_run.py"],
        ["git", "commit", "-m",
         "Add Phase3 D2: fat-tail distributions, Student-t fit, QQ plots (charts 29-32)"],
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
