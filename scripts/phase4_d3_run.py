r"""
Phase 4 Deliverable 3 — Regime-Aware Strategy + Nowcast Dashboard Runner
Generates charts 45-48 and CSV summaries into sentinel/outputs/phase4/.

Pipeline
--------
  1. Download basket of tickers (default: SPY, AAPL, GOOGL, MSFT, BRK-B)
  2. Build 3-feature observation matrix (from D2)
  3. Fit multivariate Gaussian HMM at K=3 (canonical from D2)
  4. Compute regime-gated signal: hold when P(non-Bear regimes) > threshold
  5. Backtest strategy vs buy-and-hold using the Phase-2 Backtester
  6. Compute rolling 1-year alpha, regime-change signal, nowcast
  7. Emit charts 45-48 and CSVs

Strategy specification
----------------------
At K=3, regimes are {Bear, Sideways, Bull}. The strategy holds SPY whenever
P(Sideways ∪ Bull) > τ (default 0.5), and sits in cash otherwise. The signal
is lagged one day before applying to returns to avoid look-ahead bias.

Usage
-----
    cd ~/Documents/Claude/Projects/Passion\ project\ finance/sentinel
    python3 scripts/phase4_d3_run.py
    python3 scripts/phase4_d3_run.py --benchmark SPY --period 5y
    python3 scripts/phase4_d3_run.py --threshold 0.6
    python3 scripts/phase4_d3_run.py --k_states 5       # use 5-state model

D3.1 — gate-mode comparison
---------------------------
    # Run all three gate modes side-by-side and emit chart 49 + sweep CSV
    python3 scripts/phase4_d3_run.py --sweep_modes

    # Or pick a single non-default gate mode for charts 45-48
    python3 scripts/phase4_d3_run.py --gate_mode viterbi
    python3 scripts/phase4_d3_run.py --gate_mode hysteresis \
            --enter_threshold 0.65 --exit_threshold 0.35 --min_hold_days 7
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
    regime_summary_mv,
)
from sentinel.regime.regime_plots import (
    plot_regime_gated_equity,
    plot_rolling_alpha,
    plot_regime_transitions_nowcast,
    plot_nowcast_dashboard,
    plot_gate_mode_comparison,
)
from sentinel.strategy.regime_strategy import (
    run_strategy,
    rolling_alpha,
    regime_change_score,
    detect_transition_days,
    regime_nowcast,
    nowcast_to_row,
    classify_labels,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 4 D3 — Regime-aware strategy + nowcast")
    p.add_argument("--tickers", nargs="+",
                   default=["SPY", "AAPL", "GOOGL", "MSFT", "BRK-B"])
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--period", default="5y")
    p.add_argument("--vol_window", type=int, default=21)
    p.add_argument("--apc_window", type=int, default=60)
    p.add_argument("--k_states", type=int, default=3,
                   help="HMM states. D2 canonical is 3.")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Posterior mass on bullish regimes required to be invested.")
    p.add_argument("--alpha_window", type=int, default=252,
                   help="Rolling alpha lookback in trading days.")
    p.add_argument("--change_lookback", type=int, default=5,
                   help="Days back for regime-change L1 distance.")
    p.add_argument("--change_threshold", type=float, default=0.5,
                   help="Score threshold to flag a transition day.")
    p.add_argument("--initial", type=float, default=10_000.0)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--n_iter",  type=int, default=500)
    p.add_argument("--outdir",  default="outputs/phase4")
    # ── D3.1 gate-mode args ──
    p.add_argument("--gate_mode",
                   choices=["posterior", "viterbi", "hysteresis"],
                   default="posterior",
                   help="Signal-construction mode for the primary strategy.")
    p.add_argument("--enter_threshold", type=float, default=0.6,
                   help="Hysteresis: enter when P_bullish > this.")
    p.add_argument("--exit_threshold",  type=float, default=0.35,
                   help="Hysteresis: exit when P_bullish < this.")
    p.add_argument("--min_hold_days",   type=int,   default=5,
                   help="Hysteresis: minimum holding period after a flip.")
    p.add_argument("--sweep_modes", action="store_true",
                   help="Run ALL three gate modes and emit chart 49 comparison.")
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
    print(f"\nBuilding features …")
    feats = build_features(raw, benchmark=args.benchmark,
                           vol_window=args.vol_window,
                           apc_window=args.apc_window)
    print(f"  features shape: {feats.shape}")

    # ── 3. fit HMM ────────────────────────────────────────────────────────────
    print(f"\nFitting multivariate HMM (K={args.k_states}) …")
    summary = regime_summary_mv(
        feats, n_states=args.k_states,
        seed=args.seed, n_iter=args.n_iter,
    )
    labels = summary["labels"]
    print(f"  Labels: {labels}")
    print(f"  Current regime: {summary['current_state']}  "
          f"(posterior: "
          f"{', '.join(f'{l}={p:.0%}' for l, p in summary['current_post'].items())})")

    # ── 4. strategy ────────────────────────────────────────────────────────────
    groups = classify_labels(labels)
    bullish = groups["bullish"]
    bearish = groups["bearish"]
    print(f"\nRegime gate: bullish={bullish}  bearish={bearish}")
    if args.gate_mode == "posterior":
        print(f"Policy [posterior]: hold {args.benchmark} when "
              f"P({' ∪ '.join(bullish)}) > {args.threshold:.2f}")
    elif args.gate_mode == "viterbi":
        print(f"Policy [viterbi]: hold {args.benchmark} when "
              f"Viterbi state ∈ {bullish}")
    else:
        print(f"Policy [hysteresis]: enter when P_bullish > "
              f"{args.enter_threshold:.2f}, exit when < "
              f"{args.exit_threshold:.2f}, min hold {args.min_hold_days}d")

    strat = run_strategy(
        prices=raw[args.benchmark],
        posteriors=summary["posteriors"],
        bullish_labels=bullish,
        threshold=args.threshold,
        initial=args.initial,
        gate_mode=args.gate_mode,
        viterbi_labeled=summary["viterbi_labeled"],
        enter_threshold=args.enter_threshold,
        exit_threshold=args.exit_threshold,
        min_hold_days=args.min_hold_days,
    )

    m_s = strat["metrics_strat"]
    m_b = strat["metrics_bh"]
    print("\n  Strategy vs Buy-and-Hold:")
    print(f"    {'Metric':<16}{'Strategy':>14}{'Buy-Hold':>14}")
    for key, label in [("cagr", "CAGR"),
                        ("sharpe", "Sharpe"),
                        ("max_drawdown", "Max Drawdown")]:
        sv, bv = m_s.get(key, 0), m_b.get(key, 0)
        if key == "sharpe":
            print(f"    {label:<16}{sv:>+14.2f}{bv:>+14.2f}")
        else:
            print(f"    {label:<16}{sv*100:>+13.2f}%{bv*100:>+13.2f}%")
    print(f"    {'n_trades':<16}{m_s.get('n_trades', 0):>14}{'—':>14}")
    print(f"    {'win_rate':<16}{m_s.get('win_rate', 0)*100:>+13.1f}%{'—':>14}")

    # ── 5. rolling alpha ───────────────────────────────────────────────────────
    bt = strat["backtester"]
    ralpha = rolling_alpha(bt.returns_strat, bt.returns_bh,
                           window=args.alpha_window)
    ralpha_valid = ralpha.dropna()
    if len(ralpha_valid):
        print(f"\nRolling {args.alpha_window}-day alpha  |  "
              f"min {ralpha_valid.min()*100:+.1f}%  "
              f"mean {ralpha_valid.mean()*100:+.1f}%  "
              f"max {ralpha_valid.max()*100:+.1f}%")

    # ── 6. regime change ──────────────────────────────────────────────────────
    print(f"\nRegime-change signal (L1 over {args.change_lookback}d) …")
    change = regime_change_score(summary["posteriors"],
                                 lookback=args.change_lookback)
    transitions = detect_transition_days(
        summary["posteriors"],
        lookback=args.change_lookback,
        threshold=args.change_threshold,
    )
    print(f"  {len(transitions)} transition days detected "
          f"(threshold={args.change_threshold})")

    # ── 7. nowcast ─────────────────────────────────────────────────────────────
    nc = regime_nowcast(summary, strat, recent_days=60)
    print(f"\n  Nowcast as of {nc['as_of']}:")
    print(f"    current regime        : {nc['current_state']}")
    print(f"    expected days in state: {nc['expected_days_in_regime']:.1f}")
    print(f"    bullish posterior mass: {nc['bullish_prob_today']*100:.1f}%")
    print(f"    position today        : "
          f"{'INVESTED' if nc['invested'] else 'CASH'}")

    # ── 7b. gate-mode sweep (D3.1) ────────────────────────────────────────────
    sweep_runs = None
    if args.sweep_modes:
        print("\nSweeping all three gate modes …")
        sweep_configs = [
            ("posterior",  dict(gate_mode="posterior",  threshold=args.threshold)),
            ("viterbi",    dict(gate_mode="viterbi")),
            ("hysteresis", dict(gate_mode="hysteresis",
                                enter_threshold=args.enter_threshold,
                                exit_threshold=args.exit_threshold,
                                min_hold_days=args.min_hold_days)),
        ]
        sweep_runs = {}
        sweep_alphas = {}
        for mode, cfg in sweep_configs:
            run = run_strategy(
                prices=raw[args.benchmark],
                posteriors=summary["posteriors"],
                bullish_labels=bullish,
                viterbi_labeled=summary["viterbi_labeled"],
                initial=args.initial,
                **cfg,
            )
            sweep_runs[mode]   = run
            bt_m               = run["backtester"]
            sweep_alphas[mode] = rolling_alpha(bt_m.returns_strat,
                                               bt_m.returns_bh,
                                               window=args.alpha_window)

        # Side-by-side comparison print
        print(f"\n  {'Mode':<12}{'Trades':>8}{'%Inv':>8}{'CAGR':>10}"
              f"{'Sharpe':>9}{'MaxDD':>10}")
        bh_m = sweep_runs["posterior"]["metrics_bh"]
        print(f"  {'buy-hold':<12}{'—':>8}{'100.0%':>8}"
              f"{bh_m['cagr']*100:>+9.2f}%{bh_m['sharpe']:>+9.2f}"
              f"{bh_m['max_drawdown']*100:>+9.2f}%")
        for mode, run in sweep_runs.items():
            m   = run["metrics_strat"]
            sig = run["signal"]
            n_t = int(sig.diff().abs().fillna(0).sum())
            print(f"  {mode:<12}{n_t:>8}{sig.mean()*100:>7.1f}%"
                  f"{m['cagr']*100:>+9.2f}%{m['sharpe']:>+9.2f}"
                  f"{m['max_drawdown']*100:>+9.2f}%")

    # ── 8. charts ──────────────────────────────────────────────────────────────
    print("\nPlotting chart 45: regime-gated equity curve …")
    fig45 = plot_regime_gated_equity(
        equity_strat     = bt.equity_strat,
        equity_bh        = bt.equity_bh,
        signal           = strat["signal"],
        viterbi_labeled  = summary["viterbi_labeled"],
        labels           = labels,
        metrics_strat    = m_s,
        metrics_bh       = m_b,
        ticker           = args.benchmark,
        save_path        = str(out_dir / "45_regime_gated_equity.png"),
    )
    plt.close(fig45)

    print("Plotting chart 46: rolling alpha …")
    fig46 = plot_rolling_alpha(
        rolling_alpha    = ralpha,
        viterbi_labeled  = summary["viterbi_labeled"],
        labels           = labels,
        window_days      = args.alpha_window,
        save_path        = str(out_dir / "46_rolling_alpha.png"),
    )
    plt.close(fig46)

    print("Plotting chart 47: regime transitions + recent posterior …")
    fig47 = plot_regime_transitions_nowcast(
        posteriors     = summary["posteriors"],
        change_score   = change,
        transitions    = transitions,
        recent_window  = 90,
        save_path      = str(out_dir / "47_regime_transitions.png"),
    )
    plt.close(fig47)

    print("Plotting chart 48: master nowcast dashboard …")
    fig48 = plot_nowcast_dashboard(
        nowcast       = nc,
        posteriors    = summary["posteriors"],
        equity_strat  = bt.equity_strat,
        equity_bh     = bt.equity_bh,
        summary       = summary,
        ticker        = args.benchmark,
        save_path     = str(out_dir / "48_nowcast_dashboard.png"),
    )
    plt.close(fig48)

    if sweep_runs:
        print("Plotting chart 49: gate-mode comparison …")
        fig49 = plot_gate_mode_comparison(
            runs            = sweep_runs,
            equity_bh       = bt.equity_bh,
            rolling_alphas  = sweep_alphas,
            ticker          = args.benchmark,
            save_path       = str(out_dir / "49_gate_mode_comparison.png"),
        )
        plt.close(fig49)

    # ── 9. CSVs ───────────────────────────────────────────────────────────────
    pd.DataFrame({
        "equity_strat": bt.equity_strat,
        "equity_bh":    bt.equity_bh,
        "signal":       strat["signal"].reindex(bt.equity_strat.index),
    }).to_csv(out_dir / "strategy_equity.csv")
    ralpha.to_csv(out_dir / "rolling_alpha.csv", header=True)
    change.to_csv(out_dir / "regime_change_score.csv", header=True)
    if len(transitions):
        transitions.to_csv(out_dir / "transition_days.csv", index=False)
    nowcast_to_row(nc).to_csv(out_dir / "nowcast.csv", index=False)

    # Strategy metrics comparison CSV
    pd.DataFrame({
        "strategy": pd.Series(m_s),
        "buy_hold": pd.Series(m_b),
    }).to_csv(out_dir / "strategy_metrics.csv")

    # D3.1 — gate-mode sweep CSV
    if sweep_runs:
        sweep_rows = {}
        for mode, run in sweep_runs.items():
            m = run["metrics_strat"]
            sig = run["signal"]
            sweep_rows[mode] = {
                "cagr":          m.get("cagr"),
                "sharpe":        m.get("sharpe"),
                "max_drawdown":  m.get("max_drawdown"),
                "n_trades":      int(sig.diff().abs().fillna(0).sum()),
                "pct_invested":  float(sig.mean()),
            }
        sweep_rows["buy_hold"] = {
            "cagr":         sweep_runs["posterior"]["metrics_bh"].get("cagr"),
            "sharpe":       sweep_runs["posterior"]["metrics_bh"].get("sharpe"),
            "max_drawdown": sweep_runs["posterior"]["metrics_bh"].get("max_drawdown"),
            "n_trades":     None,
            "pct_invested": 1.0,
        }
        pd.DataFrame(sweep_rows).T.to_csv(out_dir / "gate_mode_sweep.csv")

    # ── 10. done ──────────────────────────────────────────────────────────────
    new_files = [
        "45_regime_gated_equity.png", "46_rolling_alpha.png",
        "47_regime_transitions.png",  "48_nowcast_dashboard.png",
        "strategy_equity.csv", "rolling_alpha.csv",
        "regime_change_score.csv", "transition_days.csv",
        "nowcast.csv", "strategy_metrics.csv",
    ]
    if sweep_runs:
        new_files += ["49_gate_mode_comparison.png", "gate_mode_sweep.csv"]
    print(f"\n✅  Done! New files in {out_dir}:")
    for f in new_files:
        if (out_dir / f).exists():
            print(f"   {f}")

    # ── 11. git commit + push ─────────────────────────────────────────────────
    print("\nCommitting Phase 4 D3 files to GitHub …")
    os.chdir(REPO_ROOT)
    commit_msg = (
        "Phase4 D3.1: add viterbi + hysteresis gate modes + chart 49"
        if args.sweep_modes
        else "Phase4 D3 rerun ({} mode)".format(args.gate_mode)
    )
    cmds = [
        ["git", "add",
         "sentinel/strategy/__init__.py",
         "sentinel/strategy/regime_strategy.py",
         "sentinel/regime/regime_plots.py",
         "scripts/phase4_d3_run.py"],
        ["git", "commit", "-m", commit_msg],
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
