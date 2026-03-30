# Sentinel

**A Quantitative Market Intelligence Engine**

Sentinel connects four mathematical systems into a unified platform for portfolio analysis, strategy backtesting, crash detection, and market regime classification. Built from scratch in Python over 24 weeks.

---

## Architecture

| Layer | Module | Description | Timeline |
|-------|--------|-------------|----------|
| 0 | `sentinel/data` | Data ingestion, log returns, volatility | Weeks 1–2 |
| 1 | `sentinel/risk` | Portfolio VaR (3 methods), correlation matrices, Monte Carlo | Weeks 3–6 |
| 2 | `sentinel/backtest` | Strategy backtester — Sharpe, max drawdown, win rate | Weeks 7–11 |
| 3 | `sentinel/crash` | Crash warning dashboard — GARCH, fat tails, volatility clustering | Weeks 12–16 |
| 4 | `sentinel/regime` | Hidden Markov Model market regime classifier | Weeks 17–20 |
| 5 | Web UI | Streamlit/Flask frontend unifying all layers | Weeks 21–24 |

---

## Phase 0 — Foundation (current)

**Goal:** Pull historical price data, compute returns and volatility, produce diagnostic charts.

**Math covered:** log returns, variance, standard deviation, annualization via √T scaling.

### Quickstart

```bash
# Clone and set up
git clone https://github.com/<your-username>/sentinel.git
cd sentinel
pip install -r requirements.txt

# Run Phase 0 analysis with default tickers (AAPL, MSFT, GOOGL, SPY, BRK-B)
python scripts/phase0_run.py

# Or specify your own
python scripts/phase0_run.py --tickers AAPL TSLA NVDA --start 2020-01-01 --end 2024-12-31
```

### Output

Four charts saved to `outputs/phase0/`:

| File | What it shows |
|------|---------------|
| `01_prices_normalized.png` | All tickers normalized to 100 at start — pure performance comparison |
| `02_daily_returns.png` | Daily log returns — reveals volatility clustering and crash events |
| `03_return_distributions.png` | Histogram vs normal fit — shows fat tails (excess kurtosis) |
| `04_annualized_volatility.png` | Annual σ per ticker — first-pass risk comparison |

### Summary statistics explained

| Column | Formula | Interpretation |
|--------|---------|----------------|
| `mean_daily_return` | mean(r_t) | Avg daily log return |
| `mean_annual_return` | μ_daily × 252 | Annualized expected return |
| `daily_volatility` | std(r_t) | Daily risk |
| `annual_volatility` | σ_daily × √252 | Annual risk |
| `sharpe_approx` | μ_annual / σ_annual | Return per unit of risk (risk-free ≈ 0) |
| `skewness` | E[(r−μ)³]/σ³ | Tail asymmetry — equities usually negative |
| `kurtosis` | E[(r−μ)⁴]/σ⁴ − 3 | Fat-tail excess vs normal distribution |

---

## Repo Structure

```
sentinel/
├── config.py                   # Central configuration (tickers, dates)
├── requirements.txt
├── scripts/
│   └── phase0_run.py           # Phase 0 entry point
├── sentinel/
│   ├── data/
│   │   └── fetcher.py          # Price fetch, log returns, stats
│   ├── risk/                   # Phase 1 (pending)
│   ├── backtest/               # Phase 2 (pending)
│   ├── crash/                  # Phase 3 (pending)
│   ├── regime/                 # Phase 4 (pending)
│   └── utils/
│       └── plotting.py         # Shared charting utilities
├── data/                       # Raw data cache (gitignored)
└── outputs/                    # Generated charts (gitignored)
```

---

## Tech Stack

- **Data:** `yfinance`, `pandas`, `numpy`
- **Math/Stats:** `scipy`, `statsmodels`, `hmmlearn`
- **Visualization:** `matplotlib`, `seaborn`, `plotly`
- **Frontend (Phase 5):** Streamlit or Flask

---

## Mathematical Progression

The math deepens with each phase — designed to be learned incrementally from a Calculus 1 baseline:

- **Phase 0:** Mean, variance, standard deviation, log returns, √T volatility scaling
- **Phase 1:** Linear algebra — covariance/correlation matrices, portfolio variance σ²_p = wᵀΣw
- **Phase 2:** Time series — moving averages, optimization, Sharpe/Sortino ratios
- **Phase 3:** Stochastic processes — GARCH models, Student-t fat tails, extreme value theory
- **Phase 4:** Probabilistic ML — Hidden Markov Models, Expectation-Maximization, Viterbi algorithm

By Phase 4, the math reaches upper-division undergraduate / early graduate level in quantitative finance.

---

*Built by Gavin | 2026*
