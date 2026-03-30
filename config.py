"""
config.py — Central configuration for Sentinel.

Instead of hardcoding tickers or dates inside every script, we keep them here.
This means you only have to change one file when you want to swap assets or
extend your date range.
"""

# ── Default tickers ────────────────────────────────────────────────────────────
# These are used by any script that doesn't receive explicit tickers.
# Good starting set: a mega-cap tech, a broad ETF, and some sector diversity.
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "SPY", "BRK-B"]

# ── Date range ─────────────────────────────────────────────────────────────────
# Five years of daily data is a solid baseline for most analyses.
DEFAULT_START = "2019-01-01"
DEFAULT_END   = "2024-12-31"

# ── Market calendar ────────────────────────────────────────────────────────────
# There are ~252 trading days in a calendar year (365 days minus weekends,
# minus ~9 US market holidays). This constant shows up everywhere in finance:
# you'll use it to annualize volatility, Sharpe ratios, and more.
TRADING_DAYS_PER_YEAR = 252

# ── Output directories ─────────────────────────────────────────────────────────
OUTPUT_DIR = "outputs"
DATA_DIR   = "data/raw"
