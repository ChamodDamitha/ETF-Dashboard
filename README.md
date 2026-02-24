# 📊 ASX ETF Daily Dashboard — Fully Dynamic · GitHub Actions

Emails a live ASX ETF dashboard every weekday at **8:00 AM AEST**.
Covers: **IVV · FANG · VAS · QAU · NDQ · VGS**

> **Zero hardcoded data.** Every number on the dashboard — prices, returns,
> volatility, allocations, and all 10 years of portfolio simulation — is
> computed fresh from Yahoo Finance on the day the workflow runs.

---

## What's dynamic

| Element | How it's computed |
|---------|-------------------|
| Current price, day change | Live from `yfinance` info API |
| 52W high / low, distance from high | Live from `yfinance` info API |
| YTD return | Jan 1 → today from daily close history |
| 1-year return | 365 days ago → today from daily close history |
| Annualised volatility | Std dev of daily returns × √252, past 12 months |
| AUM, MER, dividend yield | Live from `yfinance` info API |
| Annual returns table | Year-by-year from 10 years of daily history |
| 14-month line chart | Monthly closes, indexed to 100 |
| **Portfolio allocations** | **Auto-computed today from live 1Y returns & volatility** |
| 10-year simulation | Compounded using live annual returns + live allocations |
| Verdict ratings | Rules engine on live YTD / 1Y return / distance from high |

---

## Portfolio strategies (all auto-weighted daily)

**Momentum** — Rank ETFs by 1Y return. Best performer gets the highest weight (rank 6 of 6 points down to rank 1). Any ETF with a negative 1Y return is excluded (0% weight). Remaining weights normalised to 100%.

**Risk-Adjusted** — Weight proportional to `1Y return ÷ annualised volatility` (Sharpe proxy). Rewards high-return, low-vol ETFs. Negative ratios = 0% weight.

**Equal Weight** — 1/N across all ETFs. Used as a benchmark to see whether the active strategies add value.

Allocations are printed in the terminal on each run and displayed as a table in the dashboard.

---

## Project structure

```
your-repo/
├── .github/
│   └── workflows/
│       └── daily-dashboard.yml   ← Schedule & GitHub Actions config
├── scripts/
│   └── generate_dashboard.py     ← All logic: fetch → compute → render → email
├── output/                       ← Auto-created at runtime; gitignored
│   ├── latest.html
│   └── etf-dashboard-YYYY-MM-DD.html
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup — 5 Steps

### Step 1 — Create a private GitHub repo

```bash
git init
git remote add origin https://github.com/YOUR_USERNAME/etf-dashboard.git
git add .
git commit -m "init"
git push -u origin main
```

### Step 2 — Get a Gmail App Password

1. [myaccount.google.com](https://myaccount.google.com) → Security → **App Passwords**
2. Create one named "ETF Dashboard"
3. Copy the 16-character password

### Step 3 — Add GitHub Secrets

**Settings → Secrets and variables → Actions → New repository secret**

| Secret | Value |
|--------|-------|
| `EMAIL_SENDER` | Gmail address to send from |
| `EMAIL_APP_PASSWORD` | 16-char App Password |
| `EMAIL_RECIPIENT` | Where to receive the report |

### Step 4 — Test manually

Actions tab → **Daily ASX ETF Dashboard** → **Run workflow** → check your inbox in ~90 seconds.

### Step 5 — It runs itself

Every weekday at 8:00 AM AEST automatically, for free.

---

## Customisation

### Add or swap ETFs
```python
TICKERS = {
    "VHY": {"yahoo": "VHY.AX", "name": "Vanguard High Yield ETF", "color": "#2a6a3a", "cls": "vhy"},
    ...
}
```
The portfolio allocations, charts, and simulation will automatically include the new ticker.

### Change the portfolio weighting logic
Edit `compute_portfolio_allocations()` in `generate_dashboard.py`. The function receives the full `etf_data` dict with all live metrics and returns `{ strategy_name: { ticker: weight } }`.

### Change the starting capital
```python
INITIAL = 50_000.0
```

### Change the lookback window
```python
ten_years_ago = date(today.year - 7, 1, 1)  # 7 years instead of 10
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Prices show N/A | Yahoo Finance rate-limits occasionally. Re-run manually or wait for next day. |
| Allocations sum ≠ 100% | Can happen if all ETFs have negative 1Y returns (weights normalised; rare). |
| Email goes to spam | Add the sender address to your contacts. |
| Wrong time | Cron runs in UTC — use [crontab.guru](https://crontab.guru) to recalculate. |

---

## Cost

Free. ~40 of GitHub's 2,000 free minutes/month.
