"""
Phase 3 Deliverable 3 — Correlation Spike Detection Runner
Generates charts 33–36 and spike_summary.csv into sentinel/outputs/phase3/.

Usage
-----
    cd ~/Documents/Claude/Projects/Passion\ project\ finance/sentinel
    python3 scripts/phase3_spike_run.py
"""

import argparse, pathlib, sys, os, subprocess

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sentinel.crash.correlation_spike import crash_warning_summary
from sentinel.crash.spike_plots import (
    plot_apc_spikes,
    plot_corr_heatmap_snapshots,
    plot_conditional_var,
    plot_crash_dashboard,
)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 3 D3 — Correlation Spike Detection")
    p.add_argument("--tickers", nargs="+",
                   default=["AAPL", "GOOGL", "MSFT", "BRK-B", "SPY"])
    p.add_argument("--period",  default="3y")
    p.add_argument("--window",  type=int,   default=60)
    p.add_argument("--k",       type=float, default=1.5)
    p.add_argument("--confidence", type=float, default=0.95)
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
    print(f"  {len(raw)} trading days × {raw.shape[1]} tickers")

    # ── 2. run crash warning pipeline ─────────────────────────────────────────
    print(f"\nRunning crash warning pipeline (window={args.window}, k={args.k}) …")
    summary = crash_warning_summary(raw, args.window, args.k, args.confidence)

    print(f"\n  Current APC:    {summary['current_apc']:.3f}")
    print(f"  Current regime: {summary['current_regime']}")
    print(f"  Spike days:     {summary['n_spike_days']}")
    print(f"  Spike episodes: {summary['n_episodes']}")

    if not summary["episodes"].empty:
        print("\n  Spike episodes:")
        print(summary["episodes"].to_string(index=False))

    print("\n  Conditional VaR by regime:")
    print(summary["conditional_var"].to_string())

    # ── 3. chart 33 — APC + spikes ────────────────────────────────────────────
    print("\nPlotting chart 33: APC + spike detection …")
    fig33 = plot_apc_spikes(
        apc      = summary["apc"],
        spikes   = summary["spikes"],
        regime   = summary["regime"],
        episodes = summary["episodes"],
        save_path= str(out_dir / "33_apc_spikes.png"),
    )
    plt.close(fig33)

    # ── 4. chart 34 — correlation snapshots ───────────────────────────────────
    print("Plotting chart 34: correlation heatmap snapshots …")
    fig34 = plot_corr_heatmap_snapshots(
        prices     = raw,
        window     = args.window,
        n_snapshots= 4,
        save_path  = str(out_dir / "34_corr_snapshots.png"),
    )
    plt.close(fig34)

    # ── 5. chart 35 — conditional VaR ─────────────────────────────────────────
    print("Plotting chart 35: conditional VaR by regime …")
    fig35 = plot_conditional_var(
        cond_var_df = summary["conditional_var"],
        confidence  = args.confidence,
        save_path   = str(out_dir / "35_conditional_var.png"),
    )
    plt.close(fig35)

    # ── 6. chart 36 — crash dashboard ─────────────────────────────────────────
    print("Plotting chart 36: composite crash dashboard …")
    fig36 = plot_crash_dashboard(
        prices   = raw,
        apc      = summary["apc"],
        spikes   = summary["spikes"],
        save_path= str(out_dir / "36_crash_dashboard.png"),
    )
    plt.close(fig36)

    # ── 7. spike_summary.csv ──────────────────────────────────────────────────
    cond_var_df = summary["conditional_var"].copy()
    cond_var_df.to_csv(out_dir / "spike_summary.csv")
    print("Saved spike_summary.csv")

    # ── 8. done ────────────────────────────────────────────────────────────────
    new_files = ["33_apc_spikes.png", "34_corr_snapshots.png",
                 "35_conditional_var.png", "36_crash_dashboard.png",
                 "spike_summary.csv"]
    print(f"\n✅  Done! New files in {out_dir}:")
    for f in new_files:
        if (out_dir / f).exists():
            print(f"   {f}")

    # ── 9. git commit + push ──────────────────────────────────────────────────
    print("\nCommitting Phase 3 D3 files to GitHub …")
    os.chdir(REPO_ROOT)
    cmds = [
        ["git", "add",
         "sentinel/crash/correlation_spike.py",
         "sentinel/crash/spike_plots.py",
         "scripts/phase3_spike_run.py"],
        ["git", "commit", "-m",
         "Add Phase3 D3: correlation spike detection, crash dashboard (charts 33-36)"],
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
