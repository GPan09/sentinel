r"""
Phase 4 Deliverable 2 — Multivariate HMM + Model Selection Runner
Generates charts 41–44 and CSV summaries into sentinel/outputs/phase4/.

Pipeline
--------
  1. Download a basket of tickers (default: SPY, AAPL, GOOGL, MSFT, BRK-B)
  2. Build a 3-feature observation matrix per day:
       • SPY log return            (D1's only signal)
       • 21-day rolling realised σ (Phase 3 / D2 addition)
       • 60-day APC across basket  (Phase 3 / D2 addition)
  3. Fit a multivariate Gaussian HMM at K = 2, 3, 4, 5 and pick the
     K that minimises BIC. (AIC is logged for comparison.)
  4. Refit at the chosen K and produce charts 41–44 + summary CSVs.

Usage
-----
    cd ~/Documents/Claude/Projects/Passion\ project\ finance/sentinel
    python3 scripts/phase4_d2_run.py
    python3 scripts/phase4_d2_run.py --benchmark SPY --period 5y
    python3 scripts/phase4_d2_run.py --k_min 2 --k_max 5
    python3 scripts/phase4_d2_run.py --force_k 3        # skip BIC, fit K=3
"""

import argparse, os, pathlib, subprocess, sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sentinel.regime.hmm_detector import (
    build_features,
    compare_k,
    choose_k,
    regime_summary_mv,
)
from sentinel.regime.regime_plots import (
    plot_k_selection,
    plot_mv_posteriors_with_features,
    plot_feature_pairs_by_state,
    plot_feature_boxplots_by_state,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 4 D2 — Multivariate HMM + Model Selection")
    p.add_argument("--tickers", nargs="+",
                   default=["SPY", "AAPL", "GOOGL", "MSFT", "BRK-B"],
                   help="Basket used for APC and benchmark price.")
    p.add_argument("--benchmark", default="SPY",
                   help="Ticker whose log returns and σ feed the HMM.")
    p.add_argument("--period", default="5y")
    p.add_argument("--vol_window", type=int, default=21)
    p.add_argument("--apc_window", type=int, default=60)
    p.add_argument("--k_min", type=int, default=2)
    p.add_argument("--k_max", type=int, default=5)
    p.add_argument("--force_k", type=int, default=None,
                   help="If set, skip BIC and fit this K directly.")
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--n_iter", type=int, default=500)
    p.add_argument("--outdir", default="outputs/phase4")
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
    print(f"  {len(raw)} trading days × {raw.shape[1]} tickers  "
          f"|  {raw.index[0].date()} → {raw.index[-1].date()}")

    # ── 2. features ────────────────────────────────────────────────────────────
    print(f"\nBuilding features (vol_{args.vol_window}d, apc_{args.apc_window}d) …")
    feats = build_features(
        raw, benchmark=args.benchmark,
        vol_window=args.vol_window,
        apc_window=args.apc_window,
    )
    print(f"  features shape: {feats.shape}  columns: {list(feats.columns)}")
    print("  feature head:")
    print(feats.head(3).round(5).to_string())

    # ── 3. model selection ────────────────────────────────────────────────────
    if args.force_k is not None:
        chosen = args.force_k
        comp   = compare_k(
            feats,
            k_values=tuple(range(args.k_min, args.k_max + 1)),
            seed=args.seed, n_iter=args.n_iter,
        )
        print(f"\n--force_k specified → using K = {chosen}")
    else:
        ks = tuple(range(args.k_min, args.k_max + 1))
        print(f"\nFitting HMMs for K ∈ {list(ks)} and comparing …")
        comp = compare_k(feats, k_values=ks,
                         seed=args.seed, n_iter=args.n_iter)
        print(comp.round(2).to_string())
        chosen = choose_k(comp, criterion="bic")
        aic_choice = choose_k(comp, criterion="aic")
        print(f"\n  BIC chooses K = {chosen}")
        print(f"  AIC chooses K = {aic_choice}  "
              f"({'agrees' if aic_choice == chosen else 'disagrees'})")

    # ── 4. fit at chosen K + analytics ─────────────────────────────────────────
    print(f"\nFitting final multivariate HMM with K = {chosen} …")
    summary = regime_summary_mv(
        feats, n_states=chosen,
        seed=args.seed, n_iter=args.n_iter,
    )

    print("\n  State characteristics (return-based):")
    print(summary["characteristics"].round(5).to_string())

    print("\n  Per-feature per-state stats (mean ± std):")
    print(summary["feature_stats"].round(5).to_string())

    print("\n  Transition matrix:")
    print(summary["trans_matrix"].round(4).to_string())

    print("\n  Stationary distribution:")
    print((summary["stationary"] * 100).round(2).to_string())

    print("\n  Expected regime durations (days):")
    print(summary["durations"].round(1).to_string())

    print(f"\n  Log-likelihood: {summary['log_likelihood']:.2f}")
    print(f"  BIC           : {summary['bic']:.2f}")
    print(f"  AIC           : {summary['aic']:.2f}")
    print(f"\n  Regime episodes: {len(summary['segments'])}")
    print(f"  Current state : {summary['current_state']}  "
          f"(posterior: "
          f"{', '.join(f'{l}={p:.0%}' for l, p in summary['current_post'].items())})")

    # ── 5. chart 41 — model selection ─────────────────────────────────────────
    print("\nPlotting chart 41: BIC/AIC vs K …")
    fig41 = plot_k_selection(comp, chosen_k=chosen,
                             save_path=str(out_dir / "41_k_selection.png"))
    plt.close(fig41)

    # ── 6. chart 42 — price + posteriors + features ──────────────────────────
    print("Plotting chart 42: MV posteriors + features …")
    fig42 = plot_mv_posteriors_with_features(
        prices     = raw[args.benchmark],
        posteriors = summary["posteriors"],
        features   = summary["features"],
        ticker     = args.benchmark,
        save_path  = str(out_dir / "42_mv_posteriors.png"),
    )
    plt.close(fig42)

    # ── 7. chart 43 — feature scatter pairs ──────────────────────────────────
    print("Plotting chart 43: feature-pair scatter by regime …")
    fig43 = plot_feature_pairs_by_state(
        features         = summary["features"],
        viterbi_labeled  = summary["viterbi_labeled"],
        labels           = summary["labels"],
        save_path        = str(out_dir / "43_feature_pairs.png"),
    )
    plt.close(fig43)

    # ── 8. chart 44 — feature boxplots ───────────────────────────────────────
    print("Plotting chart 44: per-feature boxplots by regime …")
    fig44 = plot_feature_boxplots_by_state(
        features         = summary["features"],
        viterbi_labeled  = summary["viterbi_labeled"],
        labels           = summary["labels"],
        save_path        = str(out_dir / "44_feature_boxplots.png"),
    )
    plt.close(fig44)

    # ── 9. CSVs ───────────────────────────────────────────────────────────────
    comp.to_csv(out_dir / "k_selection.csv")
    summary["characteristics"].to_csv(out_dir / "regime_summary_mv.csv")
    summary["feature_stats"].to_csv(out_dir / "feature_stats_mv.csv")
    summary["trans_matrix"].to_csv(out_dir / "transition_matrix_mv.csv")
    summary["segments"].to_csv(out_dir / "regime_episodes_mv.csv", index=False)

    # ── 10. done ──────────────────────────────────────────────────────────────
    new_files = ["41_k_selection.png", "42_mv_posteriors.png",
                 "43_feature_pairs.png", "44_feature_boxplots.png",
                 "k_selection.csv", "regime_summary_mv.csv",
                 "feature_stats_mv.csv", "transition_matrix_mv.csv",
                 "regime_episodes_mv.csv"]
    print(f"\n✅  Done! New files in {out_dir}:")
    for f in new_files:
        if (out_dir / f).exists():
            print(f"   {f}")

    # ── 11. git commit + push ─────────────────────────────────────────────────
    print("\nCommitting Phase 4 D2 files to GitHub …")
    os.chdir(REPO_ROOT)
    cmds = [
        ["git", "add",
         "sentinel/regime/hmm_detector.py",
         "sentinel/regime/regime_plots.py",
         "scripts/phase4_d2_run.py"],
        ["git", "commit", "-m",
         "Add Phase4 D2: multivariate HMM + BIC model selection (charts 41-44)"],
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
