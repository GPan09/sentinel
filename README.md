# Sentinel

**A Quantitative Market Intelligence Engine**

Sentinel connects four mathematical systems into a unified platform for portfolio analysis, strategy backtesting, crash detection, and market regime classification. Built from scratch in Python over 24 weeks.

---

## Architecture

| Layer | Module | Description | Timeline |
|-------|--------|-------------|----------|
| 0 | `sentinel/data` | Data ingestion, log returns, volatility | Weeks 1-2 |
| 1 | `sentinel/risk` | Portfolio VaR (3 methods), correlation matrices, Monte Carlo | Weeks 3-6 |
| 2 | `sentinel/backtest` | Strategy backtester - Sharpe, max drawdown, win rate | Weeks 7-11 |
| 3 | `sentinel/crash` | Crash warning dashboard - GARCH, fat tails, volatility clustering | Weeks 12-16 |
| 4 | `sentinel/regime` | Hidden Markov Model market regime classifier | Weeks 17-20 |
| 5 | Web UI | Streamlit/Flask frontend unifying all layers | Weeks 21-24 |

---

## Quickstart

```bash
git clone https://github.com/GPan09/sentinel.git
cd sentinel
pip install -r requirements.txt
python scripts/phase0_run.py
```

Or customize:
```bash
python scripts/phase0_run.py --tickers AAPL TSLA NVDA --start 2020-01-01 --end 2024-12-31
```

---

## Tech Stack

- **Data:** `yfinance`, `pandas`, `numpy`
- **Math/Stats:** `scipy`, `statsmodels`, `hmmlearn`
- **Visualization:** `matplotlib`, `seaborn`, `plotly`
- **Frontend (Phase 5):** Streamlit or Flask

---

*Built by Gavin | 2026*
