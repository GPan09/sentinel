"""
scripts/phase0_run.py — Phase 0 runner.

This is the entry point for Phase 0. Run it from the repo root:

    python scripts/phase0_run.py

Or customize tickers/dates at the command line:

    python scripts/phase0_run.py --tickers AAPL TSLA NVDA --start 2020-01-01 --end 2024-12-31

What it does:
  1. Downloads adjusted closing prices via yfinance
  2. Computes daily log returns
  3. Prints a summary statistics table
  4. Saves four charts to outputs/phase0/
"""

import sys
import os
import argparse

# Allow running from the repo root without installing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sentinel.data.fetcher import (
    fetch_prices,
    compute_log_returns,
    compute_volatility,
    compute_summary_stats,
)
from sentinel.utils.plotting import (
    plot_prices,
    plot_returns,
    plot_return_distribution,
    plot_volatility_bar,
)

try:
    from config import DEFAULT_TICKERS, DEFAULT_START, DEFAULT_END
except ImportError:
    DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "SPY", "BRK-B"]
    DEFAULT_START = "2019-01-01"
    DEFAULT_END   = "2024-12-31"


def parse_args():
    p = argparse.ArgumentParser(description="Sentinel Phase 0 — Data Ingestion & Analysis")
    p.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS,
                   help="Space-separated list of tickers")
    p.add_argument("--start", default=DEFAULT_START, help="Start date YYYY-MM-DD")
    p.add_argument("--end",   default=DEFAULT_END,   help="End date   YYYY-MM-DD")
    p.add_argument("--no-save", action="store_true",
                   help="Don't save charts to disk (just display them)")
    return p.parse_args()


def run(tickers, start, end, save_charts=True):
    out = "outputs/phase0"

    # ── Step 1: Fetch data ────────────────────────────────────────────────────
    prices = fetch_prices(tickers, start=start, end=end)

    # ── Step 2: Compute returns ───────────────────────────────────────────────
    returns = compute_log_returns(prices)

    # ── Step 3: Print summary stats ───────────────────────────────────────────
    stats = compute_summary_stats(returns)

    print("=" * 65)
    print("SENTINEL — Phase 0 Summary Statistics")
    print("=" * 65)
    print()

    # Format for readability
    display = stats.copy()
    display["mean_daily_return"]  = display["mean_daily_return"].map("{:.4f}".format)
    display["mean_annual_return"] = display["mean_annual_return"].map("{:.2%}".format)
    display["daily_volatility"]   = display["daily_volatility"].map("{:.4f}".format)
    display["annual_volatility"]  = display["annual_volatility"].map("{:.2%}".format)
    display["sharpe_approx"]      = display["sharpe_approx"].map("{:.3f}".format)

    print(display.to_string())
    print()

    # Explain what you're reading
    print("Column guide:")
    print("  mean_annual_return  — annualized avg log return (μ × 252)")
    print("  annual_volatility   — annualized std dev (σ_daily × √252)")
    print("  sharpe_approx       — return / vol, assumes risk-free rate = 0")
    print("  skewness            — <0 = left tail (crashes worse than rallies)")
    print("  kurtosis            — >0 = fat tails (more extreme moves than normal)")
    print()

    if not save_charts:
        return stats

    # ── Step 4: Charts ────────────────────────────────────────────────────────
    vol = compute_volatility(returns, annualize=True)

    print("Generating charts...")
    plot_prices(
        prices,
        normalize=True,
        save_path=f"{out}/01_prices_normalized.png",
    )
    plot_returns(
        returns,
        save_path=f"{out}/02_daily_returns.png",
    )
    plot_return_distribution(
        returns,
        save_path=f"{out}/03_return_distributions.png",
    )
    plot_volatility_bar(
        vol,
        save_path=f"{out}/04_annualized_volatility.png",
    )

    print()
    print(f"All outputs saved to: {out}/")
    print()
    print("Phase 0 complete. What to look for in your charts:")
    print("  01_prices_normalized — which assets grew the most? compounding matters.")
    print("  02_daily_returns     — can you spot COVID (March 2020)?")
    print("  03_distributions     — are the tails wider than the orange normal curve?")
    print("  04_volatility        — compare SPY (diversified) vs individual stocks.")

    return stats


if __name__ == "__main__":
    args = parse_args()
    run(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        save_charts=not args.no_save,
    )
