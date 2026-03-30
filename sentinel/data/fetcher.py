"""
sentinel/data/fetcher.py — Data ingestion and return calculations.

Phase 0 deliverable: given a list of tickers, produce daily returns,
annualized volatility, and a summary statistics table.

MATH PRIMER — read this before the code:
─────────────────────────────────────────────────────────────────────────────
Why log returns?

There are two ways to measure a day's return.

  Simple return:  r_t = (P_t - P_{t-1}) / P_{t-1}

  Log return:     r_t = ln(P_t / P_{t-1})

For small moves they're nearly identical — the first-order Taylor expansion
of ln(1 + x) around x=0 is just x, so ln(P_t/P_{t-1}) ≈ (P_t-P_{t-1})/P_{t-1}.
You'll prove this properly when you hit Taylor series in Calc 2.

Log returns have two crucial advantages:

  1. Time-additivity.
     Total return over T days = ln(P_T / P_0)
                              = ln(P_T/P_{T-1}) + ln(P_{T-1}/P_{T-2}) + ... + ln(P_1/P_0)
                              = sum of daily log returns.
     Simple returns don't add — they multiply, which is messier.

  2. Statistical convenience.
     Under reasonable assumptions, log returns are approximately normally
     distributed. This becomes critical in Phase 1 when we model them.

Why annualize volatility?

Volatility (σ) is just the standard deviation of log returns. Computed on
daily data, it's a daily volatility. Comparing daily σ across assets or to
benchmarks is awkward, so we convert to annual units.

The key insight: if daily returns are independent (not correlated with each
other day to day — a rough but useful assumption), then variance adds:

  Var(r_1 + r_2 + ... + r_252) = 252 × Var(r_daily)   [since returns are i.i.d.]

Taking square roots:
  σ_annual = σ_daily × √252

That √252 appears constantly in quant finance. It comes directly from the
square-root-of-time rule for independent random variables.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

# Import from project config — fall back to sensible defaults if running
# the module in isolation.
try:
    from config import DEFAULT_TICKERS, DEFAULT_START, DEFAULT_END, TRADING_DAYS_PER_YEAR, DATA_DIR
except ImportError:
    DEFAULT_TICKERS = ["AAPL", "MSFT", "SPY"]
    DEFAULT_START = "2019-01-01"
    DEFAULT_END   = "2024-12-31"
    TRADING_DAYS_PER_YEAR = 252
    DATA_DIR = "data/raw"


# ── 1. Price fetching ──────────────────────────────────────────────────────────

def fetch_prices(
    tickers: list[str],
    start: str = DEFAULT_START,
    end: str   = DEFAULT_END,
    save: bool = False,
) -> pd.DataFrame:
    """
    Download adjusted closing prices for a list of tickers from Yahoo Finance.

    We use 'adjusted' close prices (the 'Close' column in yfinance's auto-adjust
    mode) which accounts for stock splits and dividend payments. This is crucial:
    raw prices make it look like a stock crashed on ex-dividend day, which would
    corrupt your return calculations.

    Parameters
    ----------
    tickers : list of ticker strings, e.g. ["AAPL", "MSFT", "SPY"]
    start   : start date string "YYYY-MM-DD"
    end     : end date string   "YYYY-MM-DD"
    save    : if True, cache the raw CSV to DATA_DIR (useful for offline work)

    Returns
    -------
    prices : pd.DataFrame, shape (trading_days, n_tickers)
             index = DatetimeIndex of trading days
             columns = ticker symbols
             values  = adjusted closing price in USD
    """
    print(f"Fetching data for: {tickers}")
    print(f"Date range: {start} → {end}\n")

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,   # adjust for splits + dividends automatically
        progress=False,
    )

    # yfinance returns a multi-level column index when you pass multiple tickers.
    # We only want the "Close" level.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # Single ticker — yfinance returns flat columns
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # Drop any days where ALL tickers have NaN (e.g. market holidays that
    # yfinance occasionally includes with missing data).
    prices = prices.dropna(how="all")

    # Forward-fill remaining NaNs (rare, but can happen at IPO boundaries).
    prices = prices.ffill()

    print(f"Loaded {len(prices)} trading days for {list(prices.columns)}\n")

    if save:
        os.makedirs(DATA_DIR, exist_ok=True)
        path = os.path.join(DATA_DIR, "prices.csv")
        prices.to_csv(path)
        print(f"Prices saved to {path}")

    return prices


# ── 2. Log returns ─────────────────────────────────────────────────────────────

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from a price DataFrame.

    Formula: r_t = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})

    Implementation: np.log(prices).diff()
    - np.log() applies the natural log element-wise.
    - .diff() computes the difference between consecutive rows.
    - Combined, this gives ln(P_t) - ln(P_{t-1}) = ln(P_t / P_{t-1}).

    The first row becomes NaN because there's no P_{t-1} on day 0.
    We drop it with .dropna().

    Parameters
    ----------
    prices : pd.DataFrame of adjusted closing prices

    Returns
    -------
    returns : pd.DataFrame of the same shape minus one row
              values are dimensionless log returns (e.g. 0.012 means +1.2%)
    """
    log_returns = np.log(prices).diff().dropna()
    return log_returns


# ── 3. Volatility ──────────────────────────────────────────────────────────────

def compute_volatility(
    returns: pd.DataFrame,
    annualize: bool = True,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """
    Compute the volatility (standard deviation of log returns) for each ticker.

    Daily volatility: σ_daily = std(r_t)
                                uses ddof=1 (Bessel's correction — divides by
                                N-1 instead of N to get an unbiased estimate
                                of population variance from a sample)

    Annualized volatility: σ_annual = σ_daily × √(trading_days)

    The √T scaling comes from the property of variance under independence:
      Var(X_1 + X_2 + ... + X_T) = T × Var(X)  when X_i are i.i.d.
    So Std(sum) = √T × Std(X), and annualizing σ_daily means T = 252.

    Parameters
    ----------
    returns      : pd.DataFrame of log returns
    annualize    : if True, multiply by √252; if False, return daily σ
    trading_days : trading days per year (default 252)

    Returns
    -------
    vol : pd.Series indexed by ticker, values in same units as returns
          (annualized: typically 0.15–0.40 for individual stocks)
    """
    daily_vol = returns.std(ddof=1)

    if annualize:
        return daily_vol * np.sqrt(trading_days)
    return daily_vol


# ── 4. Summary statistics ──────────────────────────────────────────────────────

def compute_summary_stats(
    returns: pd.DataFrame,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """
    Produce a clean summary statistics table for a set of return series.

    Columns returned:
      mean_daily_return  — arithmetic mean of daily log returns
      mean_annual_return — annualized: mean_daily × 252
                           (this is a rough estimate; proper annualization
                            uses exp(μ × 252) - 1, but the linear approximation
                            is close enough for small daily returns)
      daily_volatility   — std dev of daily log returns
      annual_volatility  — daily_vol × √252
      sharpe_approx      — annual_return / annual_vol  (assumes risk-free ≈ 0)
                           A proper Sharpe subtracts the risk-free rate;
                           we'll refine this in Phase 2.
      skewness           — 3rd standardized moment: E[(r-μ)³] / σ³
                           Positive = right tail (big gains more common than big losses)
                           Negative = left tail (bad crashes more extreme than good rallies)
                           Most equity return distributions have NEGATIVE skew.
      kurtosis           — 4th standardized moment (excess): E[(r-μ)⁴]/σ⁴ - 3
                           Normal distribution = 0. Positive = fat tails (extreme
                           moves happen more often than a normal dist would predict).
                           This is called "leptokurtosis" and it's a big deal in Phase 3.

    Parameters
    ----------
    returns      : pd.DataFrame of log returns
    trading_days : trading days per year

    Returns
    -------
    stats : pd.DataFrame, one row per ticker
    """
    daily_vol  = returns.std(ddof=1)
    annual_vol = daily_vol * np.sqrt(trading_days)
    mean_daily = returns.mean()
    mean_annual = mean_daily * trading_days

    stats = pd.DataFrame({
        "mean_daily_return":  mean_daily.round(6),
        "mean_annual_return": mean_annual.round(4),
        "daily_volatility":   daily_vol.round(6),
        "annual_volatility":  annual_vol.round(4),
        "sharpe_approx":      (mean_annual / annual_vol).round(3),
        "skewness":           returns.skew().round(3),
        "kurtosis":           returns.kurt().round(3),   # excess kurtosis (normal = 0)
    })

    return stats


# ── 5. Quick sanity check ──────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run this file directly to do a quick smoke test.
    prices  = fetch_prices(DEFAULT_TICKERS)
    returns = compute_log_returns(prices)
    stats   = compute_summary_stats(returns)

    print("=== Summary Statistics ===")
    print(stats.to_string())
    print()
    print(f"Date range in data: {returns.index[0].date()} → {returns.index[-1].date()}")
    print(f"Total trading days:  {len(returns)}")
