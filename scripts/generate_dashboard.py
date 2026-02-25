"""
ASX ETF Dashboard Generator — Fully Dynamic
=============================================
Zero hardcoded data. Everything is computed from live Yahoo Finance data on the day.

DYNAMIC ELEMENTS:
  ✓ Current prices, day change, 52W range
  ✓ YTD return, 1-year return, 10-year annual returns per calendar year
  ✓ AUM, MER, dividend yield, P/E
  ✓ 14-month monthly indexed chart
  ✓ Portfolio allocations — AI-recommended by Claude using live metrics, returns & volatility
  ✓ 10-year portfolio simulation — all returns from Yahoo Finance history
  ✓ Verdict ratings — AI-powered using live metrics, historical data & news context

PORTFOLIO STRATEGIES:
  AI-powered — Claude recommends 3 distinct strategies based on live data
  Fallback   — algorithmic (Momentum, Risk-Adj, Equal) if AI unavailable

Dependencies: yfinance>=0.2.36, python-dateutil>=2.8.2, numpy>=1.24
"""

import json
import os
import smtplib
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from zoneinfo import ZoneInfo

import json as _json_module
import urllib.request
import numpy as np
import yfinance as yf

AEST           = ZoneInfo("Australia/Sydney")
INITIAL        = 10_000.0
RISK_FREE_RATE = 4.0   # % p.a. — RBA cash rate proxy for Sharpe/Sortino
SNAPSHOT_PATH  = Path(__file__).parent.parent / "data" / "last_run.json"

# ── Only static config: ticker identifiers and display metadata ──────────────
# NO prices, NO returns, NO weights — all fetched/computed live
TICKERS = {
    "IVV":  {"yahoo": "IVV.AX",   "name": "iShares S&P 500 ETF",         "color": "#c8440a", "cls": "ivv"},
    "FANG": {"yahoo": "FANG.AX",  "name": "Global X FANG+ ETF",          "color": "#1a3a8a", "cls": "fang"},
    "VAS":  {"yahoo": "VAS.AX",   "name": "Vanguard Aust. Shares ETF",   "color": "#1a7a4a", "cls": "vas"},
    "QAU":  {"yahoo": "QAU.AX",   "name": "BetaShares Gold Bullion ETF", "color": "#b8920a", "cls": "qau"},
    "GOLD": {"yahoo": "GOLD.AX",  "name": "ETFS Physical Gold ETF",      "color": "#a07820", "cls": "gold"},
    "VGS":  {"yahoo": "VGS.AX",   "name": "Vanguard MSCI Intl ETF",      "color": "#0a6a8a", "cls": "vgs"},
}

# ── Candidate pool: popular ASX ETFs screened at runtime for top performers ──
# Each run the top N by 1Y return (not already in TICKERS) are added to the
# live report, so the dashboard always shows what the market is rewarding.
CANDIDATE_POOL = {
    "NDQ":   {"yahoo": "NDQ.AX",   "name": "BetaShares Nasdaq 100 ETF"},
    "SEMI":  {"yahoo": "SEMI.AX",  "name": "BetaShares Global Semiconductors ETF"},
    "HACK":  {"yahoo": "HACK.AX",  "name": "BetaShares Cybersecurity ETF"},
    "RBTZ":  {"yahoo": "RBTZ.AX",  "name": "BetaShares Global Robotics & AI ETF"},
    "ASIA":  {"yahoo": "ASIA.AX",  "name": "BetaShares Asia Technology Tigers ETF"},
    "ETHI":  {"yahoo": "ETHI.AX",  "name": "BetaShares Global Sustainability Leaders ETF"},
    "QUAL":  {"yahoo": "QUAL.AX",  "name": "VanEck MSCI Intl Quality ETF"},
    "A200":  {"yahoo": "A200.AX",  "name": "BetaShares Australia 200 ETF"},
    "IOZ":   {"yahoo": "IOZ.AX",   "name": "iShares Core S&P/ASX 200 ETF"},
    "GDX":   {"yahoo": "GDX.AX",   "name": "VanEck Gold Miners ETF"},
    "DRUG":  {"yahoo": "DRUG.AX",  "name": "BetaShares Healthcare Innovators ETF"},
    "VDHG":  {"yahoo": "VDHG.AX",  "name": "Vanguard Diversified High Growth ETF"},
}

# Extra colors assigned to dynamically discovered ETFs (cycles if needed)
_EXTRA_COLORS = ["#8a2a6a", "#2a6a4a", "#6a3a8a", "#8a5a2a", "#2a4a8a", "#5a8a2a"]


# ── STEP 0: Discover top-performing ETFs from the candidate pool ─────────────

def discover_top_etfs(now, n=3):
    """
    Screens CANDIDATE_POOL for the top N ETFs by 1-year return (excluding any
    already present in TICKERS).  Returns a list of (ticker, meta) tuples ready
    to be merged into TICKERS before the main data fetch.

    Uses a lightweight 1-year history fetch (no .info call) to keep it fast.
    Any candidates that error are silently skipped.
    """
    today        = now.date()
    one_year_ago = today - relativedelta(years=1)
    results      = []

    for ticker, meta in CANDIDATE_POOL.items():
        if ticker in TICKERS:
            continue
        try:
            hist  = yf.Ticker(meta["yahoo"]).history(
                start=str(one_year_ago), end=str(today), auto_adjust=True
            )
            close = hist["Close"] if not hist.empty else None
            if close is None or len(close) < 20:
                continue
            one_yr = (close.iloc[-1] / close.iloc[0] - 1) * 100
            results.append((ticker, meta, one_yr))
        except Exception:
            continue

    results.sort(key=lambda x: -x[2])
    top      = results[:n]
    rejected = results[n:]   # candidates screened but not added this run

    discovered = {}
    for i, (ticker, meta, one_yr) in enumerate(top):
        color = _EXTRA_COLORS[i % len(_EXTRA_COLORS)]
        cls   = ticker.lower()
        discovered[ticker] = {**meta, "color": color, "cls": cls}
        print(f"  [{ticker}] discovered — 1Y {one_yr:+.1f}%  ({meta['name']})")

    # Build screened-pool table data (ticker → {name, one_yr_ret})
    screened_pool = {
        t: {"name": meta["name"], "one_yr_ret": ret}
        for t, meta, ret in rejected
    }

    return discovered, screened_pool


# ── STEP 1: Fetch all live data ───────────────────────────────────────────────

def fetch_all_data(now):
    """
    Fetches everything from Yahoo Finance in a single history pull per ticker.
    Returns a dict keyed by ticker with all computed metrics.
    """
    today          = now.date()
    start_of_year  = date(today.year, 1, 1)
    twenty_years_ago = date(today.year - 20, 1, 1)
    ten_years_ago  = date(today.year - 10, 1, 1)
    five_years_ago = today - relativedelta(years=5)
    one_year_ago   = today - relativedelta(years=1)

    result = {}

    for ticker, meta in TICKERS.items():
        print(f"  [{ticker}] fetching...", end=" ", flush=True)
        try:
            t    = yf.Ticker(meta["yahoo"])
            info = t.info

            # ── Spot price & day change ──────────────────────────────────────
            price      = info.get("currentPrice") or info.get("regularMarketPrice")
            prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
            day_chg    = ((price - prev_close) / prev_close * 100) if price and prev_close else None

            # ── 52W range ────────────────────────────────────────────────────
            w52_high  = info.get("fiftyTwoWeekHigh")
            w52_low   = info.get("fiftyTwoWeekLow")
            from_high = ((price - w52_high) / w52_high * 100) if price and w52_high else None

            # ── Fund metadata ────────────────────────────────────────────────
            aum       = info.get("totalAssets")
            mer       = info.get("annualReportExpenseRatio") or info.get("expenseRatio")
            div_yield = info.get("yield") or info.get("trailingAnnualDividendYield")

            # ── Pull up to 20-year daily history in one request ──────────────
            hist  = t.history(start=str(twenty_years_ago), end=str(today), auto_adjust=True)
            close = hist["Close"] if not hist.empty else None

            # YTD return
            ytd_ret = None
            if close is not None:
                sl = close[close.index.date >= start_of_year]
                if len(sl) > 1:
                    ytd_ret = (sl.iloc[-1] / sl.iloc[0] - 1) * 100

            # 1-year return
            one_yr_ret = None
            if close is not None:
                sl = close[close.index.date >= one_year_ago]
                if len(sl) > 1:
                    one_yr_ret = (sl.iloc[-1] / sl.iloc[0] - 1) * 100

            # 5-year, 10-year, 20-year total returns
            def _period_ret(cl, since):
                if cl is None: return None
                sl = cl[cl.index.date >= since]
                return (sl.iloc[-1] / sl.iloc[0] - 1) * 100 if len(sl) > 1 else None

            five_yr_ret   = _period_ret(close, five_years_ago)
            ten_yr_ret    = _period_ret(close, ten_years_ago)
            twenty_yr_ret = _period_ret(close, twenty_years_ago)

            # 1-year daily volatility (annualised) — used for risk-adjusted weighting
            vol_1y = None
            daily_rets_1y = []
            daily_rets_10y = []
            daily_rets_10y_dates = []
            if close is not None:
                sl = close[close.index.date >= one_year_ago]
                if len(sl) > 20:
                    dr = sl.pct_change().dropna()
                    vol_1y        = float(dr.std() * np.sqrt(252) * 100)
                    daily_rets_1y = dr.tolist()
                sl10 = close[close.index.date >= ten_years_ago]
                if len(sl10) > 200:
                    dr10 = sl10.pct_change().dropna()
                    daily_rets_10y       = dr10.tolist()
                    daily_rets_10y_dates = [d.date() for d in dr10.index]

            # Sharpe & Sortino ratios (1Y, risk-free = RISK_FREE_RATE %)
            sharpe_1y  = None
            sortino_1y = None
            if one_yr_ret is not None and vol_1y and vol_1y > 0:
                sharpe_1y = round((one_yr_ret - RISK_FREE_RATE) / vol_1y, 2)
            if one_yr_ret is not None and daily_rets_1y:
                neg = [r for r in daily_rets_1y if r < 0]
                if len(neg) > 5:
                    dv = float(np.std(neg) * np.sqrt(252) * 100)
                    if dv > 0:
                        sortino_1y = round((one_yr_ret - RISK_FREE_RATE) / dv, 2)

            # Drawdown: max over full history and current from rolling peak
            max_drawdown  = None
            curr_drawdown = None
            if close is not None and len(close) > 1:
                roll_max      = close.cummax()
                dd_series     = (close - roll_max) / roll_max * 100
                max_drawdown  = round(float(dd_series.min()), 1)
                curr_drawdown = round(float(dd_series.iloc[-1]), 1)

            # 50-day & 200-day moving averages (for regime detection)
            ma_50d = ma_200d = is_above_200ma = None
            if close is not None:
                if len(close) >= 200:
                    ma_200d = float(close.rolling(200).mean().iloc[-1])
                if len(close) >= 50:
                    ma_50d  = float(close.rolling(50).mean().iloc[-1])
                if price and ma_200d:
                    is_above_200ma = price > ma_200d

            # Annual returns per full calendar year (start_year → last full year)
            annual_returns = {}
            if close is not None:
                for yr in range(ten_years_ago.year, today.year):
                    sl = close[
                        (close.index.date >= date(yr, 1, 1)) &
                        (close.index.date <= date(yr, 12, 31))
                    ]
                    if len(sl) > 1:
                        annual_returns[str(yr)] = round((sl.iloc[-1] / sl.iloc[0] - 1) * 100, 2)

            # Current year partial (YTD)
            if ytd_ret is not None:
                annual_returns[str(today.year)] = round(ytd_ret, 2)

            # 10-year monthly indexed data for line chart (resampled from daily history)
            monthly_indexed = []
            if close is not None:
                cl10 = close[close.index.date >= ten_years_ago]
                if len(cl10) > 1:
                    mo_close = cl10.resample("ME").last().dropna()
                    if len(mo_close) > 1:
                        base = mo_close.iloc[0]
                        monthly_indexed = [round(v / base * 100, 2) for v in mo_close]

            result[ticker] = {
                **meta,
                "price":           price,
                "day_chg":         day_chg,
                "ytd_ret":         ytd_ret,
                "one_yr_ret":      one_yr_ret,
                "five_yr_ret":     five_yr_ret,
                "ten_yr_ret":      ten_yr_ret,
                "twenty_yr_ret":   twenty_yr_ret,
                "vol_1y":          vol_1y,
                "sharpe_1y":       sharpe_1y,
                "sortino_1y":      sortino_1y,
                "max_drawdown":    max_drawdown,
                "curr_drawdown":   curr_drawdown,
                "ma_50d":          ma_50d,
                "ma_200d":         ma_200d,
                "is_above_200ma":  is_above_200ma,
                "daily_rets_1y":        daily_rets_1y,
                "daily_rets_10y":       daily_rets_10y,
                "daily_rets_10y_dates": daily_rets_10y_dates,
                "w52_high":        w52_high,
                "w52_low":         w52_low,
                "from_high":       from_high,
                "aum":             aum,
                "mer":             mer,
                "div_yield":       div_yield,
                "annual_returns":  annual_returns,
                "monthly_indexed": monthly_indexed,
            }
            parts = [f"${price:.2f}" if price else "no price"]
            if ytd_ret is not None:    parts.append(f"YTD:{ytd_ret:+.1f}%")
            if one_yr_ret is not None: parts.append(f"1Y:{one_yr_ret:+.1f}%")
            if five_yr_ret is not None:  parts.append(f"5Y:{five_yr_ret:+.1f}%")
            if ten_yr_ret is not None:   parts.append(f"10Y:{ten_yr_ret:+.1f}%")
            if vol_1y is not None:     parts.append(f"vol:{vol_1y:.1f}%")
            print("  ".join(parts))

        except Exception as exc:
            print(f"ERROR — {exc}")
            result[ticker] = {
                **meta, "price": None, "error": str(exc),
                "annual_returns": {}, "monthly_indexed": [], "daily_rets_1y": [], "daily_rets_10y": [], "daily_rets_10y_dates": [],
                "one_yr_ret": None, "ytd_ret": None, "vol_1y": None,
                "five_yr_ret": None, "ten_yr_ret": None, "twenty_yr_ret": None,
                "sharpe_1y": None, "sortino_1y": None,
                "max_drawdown": None, "curr_drawdown": None,
                "ma_50d": None, "ma_200d": None, "is_above_200ma": None,
            }

    return result


# ── STEP 2: Auto-compute portfolio allocations from live returns ──────────────

def compute_portfolio_allocations(etf_data):
    """
    Derives three portfolio strategies purely from live 1Y returns & volatility.

    MOMENTUM portfolio:
      Weight proportional to rank of 1Y return.
      Best 1Y performer gets highest weight. Any negative-return ETF gets 0 weight.

    RISK-ADJUSTED portfolio:
      Weight proportional to (1Y return / annualised vol), i.e. a simple Sharpe proxy.
      Negative ratios get 0 weight.

    EQUAL portfolio:
      1/N across all ETFs (benchmark).

    Returns dict: { portfolio_name: { ticker: weight, ... } }
    """
    tickers = list(etf_data.keys())

    one_yr = {t: etf_data[t].get("one_yr_ret") or 0 for t in tickers}
    vols   = {t: etf_data[t].get("vol_1y") or 15.0 for t in tickers}  # default 15% if missing

    # ── Momentum: rank-based weights ─────────────────────────────────────────
    # Sort by 1Y return descending; assign weights 6,5,4,3,2,1 for ranks 1→6
    # Zero-weight any ETF with negative 1Y return
    ranked = sorted(tickers, key=lambda t: one_yr[t], reverse=True)
    n = len(ranked)
    raw_momentum = {}
    for i, t in enumerate(ranked):
        raw_momentum[t] = max(0, (n - i)) if one_yr[t] > 0 else 0
    total_m = sum(raw_momentum.values()) or 1
    momentum_allocs = {t: round(raw_momentum[t] / total_m, 4) for t in tickers}

    # ── Risk-adjusted: Sharpe-proxy weights ──────────────────────────────────
    sharpe = {t: max(0, one_yr[t] / vols[t]) for t in tickers}
    total_s = sum(sharpe.values()) or 1
    risk_adj_allocs = {t: round(sharpe[t] / total_s, 4) for t in tickers}

    # ── Equal weight ─────────────────────────────────────────────────────────
    equal_allocs = {t: round(1 / n, 4) for t in tickers}

    return {
        "Momentum":     {"allocs": momentum_allocs,  "color": "#c8440a", "dashed": False},
        "Risk-Adjusted":{"allocs": risk_adj_allocs,  "color": "#1a7a4a", "dashed": False},
        "Equal Weight": {"allocs": equal_allocs,     "color": "#4a6ad8", "dashed": True},
    }


def allocation_desc(allocs):
    """Format allocation dict as human-readable string, skipping zeros."""
    parts = sorted(
        [(t, w) for t, w in allocs.items() if w > 0.001],
        key=lambda x: -x[1]
    )
    return " · ".join(f"{t} {w*100:.0f}%" for t, w in parts)


# ── STEP 3: 10-year portfolio simulation ─────────────────────────────────────

def build_portfolio_series(etf_data, portfolios, now):
    """
    Simulates $INITIAL invested at start_year Jan 1 through today.
    Each year uses the actual annual return fetched from Yahoo Finance.
    Portfolio allocations are the LIVE auto-computed weights.
    """
    today      = now.date()
    start_year = today.year - 10
    all_years  = [str(y) for y in range(start_year, today.year + 1)]

    result = {}
    for pname, pmeta in portfolios.items():
        value       = INITIAL
        values      = [INITIAL]
        year_labels = [f"Jan {start_year}"]

        for yr in all_years:
            port_ret = sum(
                (etf_data.get(etf, {}).get("annual_returns", {}).get(yr) or 0) / 100 * w
                for etf, w in pmeta["allocs"].items()
            )
            value *= (1 + port_ret)
            values.append(round(value, 2))
            year_labels.append(f"YTD {today.strftime('%b')}" if yr == str(today.year) else yr)

        elapsed = today.year - start_year + today.timetuple().tm_yday / 365
        cagr    = ((value / INITIAL) ** (1 / elapsed) - 1) * 100 if elapsed > 0 else 0

        result[pname] = {
            "values":    values,
            "labels":    year_labels,
            "final":     round(value, 2),
            "total_pct": round((value / INITIAL - 1) * 100, 1),
            "cagr":      round(cagr, 1),
            "color":     pmeta["color"],
            "desc":      allocation_desc(pmeta["allocs"]),
        }

    return result


# ── STEP 3a: Analytics — correlations, regime, DCA, benchmarks ───────────────

def compute_correlations(etf_data):
    """
    Per-year Pearson correlations averaged across up to 10 calendar years.
    For each year present in the data, compute the pairwise correlation of
    daily returns, then return the mean across all qualifying years.
    """
    import pandas as pd

    tickers = list(etf_data.keys())

    # Build a DataFrame of daily returns indexed by date, one column per ticker
    frames = {}
    for t in tickers:
        rets  = etf_data[t].get("daily_rets_10y", [])
        dates = etf_data[t].get("daily_rets_10y_dates", [])
        if len(rets) == len(dates) and len(rets) > 200:
            frames[t] = pd.Series(rets, index=pd.to_datetime(dates))

    if not frames:
        return {}

    df = pd.DataFrame(frames).sort_index()
    valid = list(df.columns)

    years = sorted(df.index.year.unique())

    # Accumulate per-year correlations
    year_corr_sum   = {t1: {t2: 0.0 for t2 in valid} for t1 in valid}
    year_corr_count = {t1: {t2: 0   for t2 in valid} for t1 in valid}

    for yr in years:
        yr_df = df[df.index.year == yr].dropna(how="all")
        if len(yr_df) < 20:
            continue
        # Only include column pairs that both have ≥20 non-NaN obs this year
        for t1 in valid:
            for t2 in valid:
                if t1 == t2:
                    year_corr_sum[t1][t2]   += 1.0
                    year_corr_count[t1][t2] += 1
                    continue
                pair = yr_df[[t1, t2]].dropna()
                if len(pair) < 20:
                    continue
                corr = pair[t1].corr(pair[t2])
                if not np.isnan(corr):
                    year_corr_sum[t1][t2]   += corr
                    year_corr_count[t1][t2] += 1

    matrix = {}
    for t1 in valid:
        matrix[t1] = {}
        for t2 in valid:
            cnt = year_corr_count[t1][t2]
            if cnt > 0:
                matrix[t1][t2] = round(year_corr_sum[t1][t2] / cnt, 2)
    return matrix


def detect_market_regime(etf_data):
    """
    Bull/bear/mixed regime from the proportion of ETFs above their 200-day MA.
    Returns (label, color_hex, description).
    """
    flagged = [(t, d.get("is_above_200ma")) for t, d in etf_data.items()
               if d.get("is_above_200ma") is not None]
    if not flagged:
        return "unknown", "#9a9690", "Insufficient MA data for regime detection."
    above = sum(1 for _, f in flagged if f)
    total = len(flagged)
    pct   = above / total
    if pct >= 0.70:
        return "bull",  "#1a7a4a", f"{above}/{total} ETFs above 200-day MA — broad-based uptrend."
    if pct <= 0.30:
        return "bear",  "#c8440a", f"Only {above}/{total} ETFs above 200-day MA — risk-off conditions."
    return "mixed", "#b8920a", f"{above}/{total} ETFs above 200-day MA — mixed signals, selective exposure."


def build_dca_series(etf_data, portfolios, now):
    """
    Dollar-cost averaging simulation: $1,000 invested at the start of each
    calendar year for 10 years (total $10,000 invested, same as lump sum).
    Returns parallel structure to build_portfolio_series().
    """
    today      = now.date()
    start_year = today.year - 10
    all_years  = [str(y) for y in range(start_year, today.year + 1)]
    annual     = INITIAL / 10  # $1,000/year

    result = {}
    for pname, pmeta in portfolios.items():
        value  = 0.0
        values = [0.0]
        for yr in all_years:
            value += annual
            port_ret = sum(
                (etf_data.get(etf, {}).get("annual_returns", {}).get(yr) or 0) / 100 * w
                for etf, w in pmeta["allocs"].items()
            )
            value *= (1 + port_ret)
            values.append(round(value, 2))

        elapsed   = today.year - start_year + today.timetuple().tm_yday / 365
        cagr      = ((value / INITIAL) ** (1 / elapsed) - 1) * 100 if elapsed > 0 else 0
        total_pct = (value / INITIAL - 1) * 100

        result[pname] = {
            "values":    values,
            "final":     round(value, 2),
            "total_pct": round(total_pct, 1),
            "cagr":      round(cagr, 1),
            "color":     pmeta["color"],
        }
    return result


def fetch_benchmark_series(now):
    """
    Fetches 10-year annual returns for ^AXJO (ASX 200) to use as a reference
    line on the portfolio simulation chart.  Returns a dict shaped like a
    single portfolio_series entry, or None on failure.
    """
    today      = now.date()
    start_year = today.year - 10
    start_date = date(start_year, 1, 1)
    try:
        hist  = yf.Ticker("^AXJO").history(start=str(start_date), end=str(today), auto_adjust=True)
        close = hist["Close"] if not hist.empty else None
        if close is None or len(close) < 50:
            return None

        # Compute annual returns per year
        annual_rets = {}
        for yr in range(start_year, today.year):
            sl = close[(close.index.date >= date(yr, 1, 1)) & (close.index.date <= date(yr, 12, 31))]
            if len(sl) > 1:
                annual_rets[str(yr)] = (sl.iloc[-1] / sl.iloc[0] - 1)
        # YTD
        sl_ytd = close[close.index.date >= date(today.year, 1, 1)]
        if len(sl_ytd) > 1:
            annual_rets[str(today.year)] = (sl_ytd.iloc[-1] / sl_ytd.iloc[0] - 1)

        # Simulate $10,000 lump sum
        value  = INITIAL
        values = [INITIAL]
        for yr in [str(y) for y in range(start_year, today.year + 1)]:
            value *= (1 + annual_rets.get(yr, 0))
            values.append(round(value, 2))

        elapsed   = today.year - start_year + today.timetuple().tm_yday / 365
        cagr      = ((value / INITIAL) ** (1 / elapsed) - 1) * 100 if elapsed > 0 else 0
        total_pct = (value / INITIAL - 1) * 100
        print(f"  ✓ ^AXJO benchmark: ${value:,.0f}  ({total_pct:+.1f}%  CAGR {cagr:.1f}%)")
        return {"values": values, "final": round(value, 2),
                "total_pct": round(total_pct, 1), "cagr": round(cagr, 1)}
    except Exception as e:
        print(f"  ⚠ Benchmark fetch failed: {e}")
        return None


def compute_equal_weight_drift(etf_data):
    """
    Shows how an equal-weight portfolio (1/N at start of year) has drifted
    given YTD returns.  Returns {ticker: drift_pct} where positive = overweight.
    """
    tickers = list(etf_data.keys())
    n       = len(tickers)
    if n == 0:
        return {}
    target = 1.0 / n
    # Growth factor for each ETF since Jan 1
    ytd_factors = {t: 1 + (etf_data[t].get("ytd_ret") or 0) / 100 for t in tickers}
    total       = sum(ytd_factors.values())
    current_wt  = {t: ytd_factors[t] / total for t in tickers}
    return {t: round((current_wt[t] - target) * 100, 1) for t in tickers}


# ── Snapshot persistence (for "What Changed" and verdict history) ─────────────

def load_last_snapshot():
    """Load the previous run's snapshot from data/last_run.json, or {} if absent."""
    try:
        if SNAPSHOT_PATH.exists():
            with open(SNAPSHOT_PATH) as f:
                return _json_module.load(f)
    except Exception:
        pass
    return {}


def save_snapshot(etf_data, ai_content, now):
    """Persist key metrics and verdicts to data/last_run.json for tomorrow's diff."""
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": now.strftime("%Y-%m-%d"),
        "etf_data": {
            t: {k: d.get(k) for k in
                ("price", "one_yr_ret", "ytd_ret", "vol_1y", "sharpe_1y",
                 "sortino_1y", "max_drawdown", "curr_drawdown", "from_high")}
            for t, d in etf_data.items()
        },
        "verdicts": {t: v.get("label") for t, v in
                     (ai_content or {}).get("verdicts", {}).items()},
    }
    with open(SNAPSHOT_PATH, "w") as f:
        _json_module.dump(payload, f, indent=2)


def generate_whatchanged_summary(etf_data, snapshot, ai_content, now):
    """
    Calls Claude with a diff of today vs yesterday's metrics and asks for a
    concise narrative of what materially changed.  Returns an HTML string.
    Falls back to a plain text diff if the API call fails.
    """
    if not snapshot:
        return ""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return ""

    prev_date  = snapshot.get("date", "previous run")
    prev_etfs  = snapshot.get("etf_data", {})
    prev_verd  = snapshot.get("verdicts", {})
    curr_verd  = {t: v.get("label") for t, v in
                  (ai_content or {}).get("verdicts", {}).items()}

    diff_lines = [f"Previous run: {prev_date}  →  Today: {now.strftime('%d %b %Y')}\n"]
    for t, d in etf_data.items():
        p = prev_etfs.get(t, {})
        parts = []
        for key, label in [("price", "price"), ("one_yr_ret", "1Y"), ("ytd_ret", "YTD"),
                            ("sharpe_1y", "Sharpe"), ("curr_drawdown", "drawdown")]:
            cur = d.get(key)
            prv = p.get(key)
            if cur is not None and prv is not None:
                delta = cur - prv
                if abs(delta) > 0.1:
                    parts.append(f"{label}: {prv:+.2f}→{cur:+.2f} (Δ{delta:+.2f})")
        vd_change = ""
        if prev_verd.get(t) and curr_verd.get(t) and prev_verd[t] != curr_verd[t]:
            vd_change = f"  verdict: {prev_verd[t]}→{curr_verd[t]}"
        if parts or vd_change:
            diff_lines.append(f"{t}: {', '.join(parts)}{vd_change}")

    if len(diff_lines) <= 1:
        return ""
    diff_text = "\n".join(diff_lines)

    prompt = f"""You are writing a concise daily briefing for an Australian ETF investor.

Here are the changes from the previous trading day:
{diff_text}

Write a single short paragraph (3-5 sentences, max 80 words) summarising what materially changed.
Highlight any verdict upgrades/downgrades, meaningful price moves, and trend shifts.
Be specific: mention ticker names and numbers. Do not use markdown. Plain prose only."""

    try:
        payload = _json_module.dumps({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 200,
            "messages": [{"role": "user", "content": prompt}]
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=payload,
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw  = _json_module.loads(resp.read().decode())
            text = raw["content"][0]["text"].strip()
            print(f"  ✓ What-changed summary generated ({len(text)} chars)")
            return text
    except Exception as e:
        print(f"  ⚠ What-changed summary failed: {e}")
        return ""


# ── STEP 3b: AI-generated portfolio strategies via Anthropic API ──────────────

def generate_ai_portfolios(etf_data, now):
    """
    Calls the Anthropic API to generate 3 distinct AI-recommended portfolio
    strategies based on today's live ETF metrics, returns & volatility.

    Returns a dict in the same format as compute_portfolio_allocations(), with
    an added 'rationale' field per portfolio.
    Falls back to None if the API is unavailable (caller then uses
    compute_portfolio_allocations() as fallback).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None

    tickers = list(etf_data.keys())

    summary_lines = [f"Date: {now.strftime('%d %B %Y')} (AEST)\n"]
    for ticker, d in etf_data.items():
        price    = f"${d['price']:.2f}"       if d.get('price')        else 'N/A'
        ytd      = f"{d['ytd_ret']:+.1f}%"    if d.get('ytd_ret') is not None    else 'N/A'
        one_yr   = f"{d['one_yr_ret']:+.1f}%" if d.get('one_yr_ret') is not None else 'N/A'
        vol      = f"{d['vol_1y']:.1f}%"      if d.get('vol_1y') is not None     else 'N/A'
        frm_high = f"{d['from_high']:+.1f}%"  if d.get('from_high') is not None  else 'N/A'
        ann      = d.get('annual_returns', {})
        recent   = {yr: v for yr, v in ann.items() if int(yr) >= (now.year - 5)}
        ann_str  = ", ".join(f"{yr}:{v:+.1f}%" for yr, v in sorted(recent.items()))
        summary_lines.append(
            f"{ticker} ({d['name']}): price={price}, YTD={ytd}, 1Y={one_yr}, "
            f"vol={vol}, from52wHigh={frm_high}, annualReturns=[{ann_str}]"
        )
    data_summary = "\n".join(summary_lines)
    tickers_list = ", ".join(tickers)

    prompt = f"""You are a portfolio manager constructing 3 distinct ETF portfolio strategies for an Australian retail investor.

Available ETFs: {tickers_list}

Today's live data:
{data_summary}

Create exactly 3 portfolios with genuinely different investment philosophies (e.g. aggressive growth, balanced, defensive/income).
For each portfolio:
1. Choose a short descriptive name (3-4 words max, e.g. "Growth Tilt", "Defensive Core", "Balanced Blend")
2. Allocate weights across ALL six ETFs (must sum to exactly 100, whole numbers only, minimum 0)
3. Write a single concise sentence (max 20 words) explaining the strategy rationale based on the current data

Return ONLY a JSON object with this exact structure:
{{
  "portfolios": [
    {{
      "name": "Portfolio Name",
      "rationale": "One sentence rationale grounded in today's data.",
      "allocations": {{"IVV": 30, "FANG": 20, "VAS": 20, "QAU": 10, "GOLD": 10, "VGS": 10}}
    }},
    {{...}},
    {{...}}
  ]
}}

Rules:
- Each portfolio's allocations must sum to exactly 100
- All weights must be non-negative integers
- The 3 portfolios must be meaningfully distinct from each other
- Base the allocations on the actual performance data provided
- Return ONLY the JSON, no markdown fences, no commentary"""

    try:
        payload = _json_module.dumps({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 800,
            "messages": [{"role": "user", "content": prompt}]
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw  = _json_module.loads(resp.read().decode("utf-8"))
            text = raw["content"][0]["text"].strip()
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:])
            if text.endswith("```"):
                text = "\n".join(text.split("\n")[:-1])
            result = _json_module.loads(text)

        portfolios_ai = result.get("portfolios", [])
        if len(portfolios_ai) != 3:
            raise ValueError(f"Expected 3 portfolios, got {len(portfolios_ai)}")

        colors = ["#c8440a", "#1a7a4a", "#4a6ad8"]
        dashed = [False, False, True]
        portfolios_dict = {}
        for i, p in enumerate(portfolios_ai):
            raw_allocs = p.get("allocations", {})
            allocs = {t: raw_allocs.get(t, 0) / 100.0 for t in tickers}
            total  = sum(allocs.values()) or 1
            allocs = {t: round(v / total, 4) for t, v in allocs.items()}
            portfolios_dict[p["name"]] = {
                "allocs":    allocs,
                "color":     colors[i],
                "dashed":    dashed[i],
                "rationale": p.get("rationale", ""),
            }

        print(f"  ✓ AI portfolios: {', '.join(portfolios_dict.keys())}")
        return portfolios_dict

    except Exception as e:
        print(f"  ⚠ AI portfolio generation failed: {e} — using computed fallback")
        return None


# ── STEP 4: AI-generated news & insights via Anthropic API ───────────────────

def fetch_ai_news(etf_data, now, regime=None):
    """
    Calls the Anthropic API with today's live ETF metrics and asks Claude to:
    1. Identify 6 current macro/geopolitical news themes affecting these ETFs
    2. Write a short insight paragraph for each ETF
    Returns: { "news": [...6 cards...], "insights": {ticker: text} }
    Falls back to placeholder text if API key missing or call fails.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("  ⚠ ANTHROPIC_API_KEY not set — skipping AI news section")
        return _fallback_news(etf_data)

    # Build a concise data summary to send to Claude (includes historical returns)
    summary_lines = [f"Date: {now.strftime('%d %B %Y')} (AEST)\n"]
    if regime:
        rlabel, _, rdesc = regime
        summary_lines.append(f"Market regime: {rlabel.upper()} — {rdesc}\n")
    for ticker, d in etf_data.items():
        price    = f"${d['price']:.2f}"       if d.get('price')    else 'N/A'
        day      = f"{d['day_chg']:+.2f}%"    if d.get('day_chg')  else 'N/A'
        ytd      = f"{d['ytd_ret']:+.1f}%"    if d.get('ytd_ret')  else 'N/A'
        one_yr   = f"{d['one_yr_ret']:+.1f}%" if d.get('one_yr_ret') else 'N/A'
        frm_high = f"{d['from_high']:+.1f}%"  if d.get('from_high') else 'N/A'
        vol      = f"{d['vol_1y']:.1f}%"      if d.get('vol_1y')   else 'N/A'
        sharpe   = f"{d['sharpe_1y']:.2f}"    if d.get('sharpe_1y') is not None else 'N/A'
        drawdn   = f"{d['curr_drawdown']:+.1f}%" if d.get('curr_drawdown') is not None else 'N/A'
        ma_sig   = ("above" if d.get("is_above_200ma") else "below") if d.get("is_above_200ma") is not None else "N/A"
        # Include last 5 years of annual returns for historical context
        ann = d.get('annual_returns', {})
        recent_ann = {yr: v for yr, v in ann.items() if int(yr) >= (now.year - 5)}
        ann_str = ", ".join(f"{yr}:{v:+.1f}%" for yr, v in sorted(recent_ann.items()))
        summary_lines.append(
            f"{ticker} ({d['name']}): "
            f"price={price}, day={day}, YTD={ytd}, 1Y={one_yr}, "
            f"from52wHigh={frm_high}, vol={vol}, Sharpe={sharpe}, "
            f"drawdown={drawdn}, 200dMA={ma_sig}, annualReturns=[{ann_str}]"
        )
    data_summary = "\n".join(summary_lines)

    prompt = f"""You are a senior market analyst writing a daily ASX ETF briefing for an Australian investor.

Here is today's live ETF data including multi-year historical returns from Yahoo Finance:
{data_summary}

Your task: Generate a JSON object with exactly this structure:
{{
  "news": [
    {{
      "severity": "red"|"amber"|"green",
      "tag": "short category label (e.g. Trade Policy, Geopolitics, Monetary Policy)",
      "headline": "concise news headline (max 12 words)",
      "body": "2-3 sentence explanation of the news event and why it matters for markets (max 60 words)",
      "impact": "1-2 sentences on which specific ETFs are affected and how (bullish/bearish)"
    }},
    ... (exactly 6 news items total)
  ],
  "insights": {{
    {', '.join(f'"{t}": "2-3 sentence insight for {t} (max 70 words)"' for t in etf_data)}
  }},
  "verdicts": {{
    {', '.join(f'"{t}": {{"label": "Strong Buy|Buy|Accumulate|Hold|Watch", "cls": "buy|accum|hold|watch", "note": "max 25 words"}}' for t in etf_data)}
  }}
}}

Rules:
- News items must reflect REAL current macro themes as of {now.strftime('%B %Y')} (tariffs, central bank policy, geopolitics, tech regulation, commodities, FX)
- Severity: red = high negative risk, amber = mixed/uncertain, green = positive catalyst
- Insights must reference the actual YTD and 1Y return numbers provided
- 2 red items, 2 amber items, 2 green items
- Verdicts MUST be informed by: (a) multi-year historical return pattern, (b) current YTD/1Y momentum, (c) distance from 52-week high, (d) annualised volatility, (e) the macro news themes identified above
- cls values: "buy" for Strong Buy or Buy, "accum" for Accumulate, "hold" for Hold, "watch" for Watch
- Return ONLY the JSON object, no preamble, no markdown fences"""

    try:
        payload = _json_module.dumps({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 2000,
            "messages": [{"role": "user", "content": prompt}]
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = _json_module.loads(resp.read().decode("utf-8"))
            text = raw["content"][0]["text"].strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:])
            if text.endswith("```"):
                text = "\n".join(text.split("\n")[:-1])
            result = _json_module.loads(text)
            n_news     = len(result.get('news', []))
            n_insights = len(result.get('insights', {}))
            n_verdicts = len(result.get('verdicts', {}))
            print(f"  ✓ AI content generated ({n_news} news, {n_insights} insights, {n_verdicts} verdicts)")
            return result
    except Exception as e:
        print(f"  ⚠ AI news fetch failed: {e} — using fallback")
        return _fallback_news(etf_data)


def _fallback_news(etf_data):
    """Minimal fallback if Anthropic API unavailable — uses rule-based verdicts."""
    news = [
        {"severity": "amber", "tag": "Markets", "headline": "Live market data loaded — AI news unavailable",
         "body": "Set the ANTHROPIC_API_KEY secret in GitHub to enable AI-generated daily news and ETF insights.",
         "impact": "Add ANTHROPIC_API_KEY to your GitHub repo secrets to enable this section."},
    ]
    insights = {t: f"{t}: Add ANTHROPIC_API_KEY to GitHub secrets to enable AI-generated insights for each ETF." for t in etf_data}
    # Generate rule-based verdicts as fallback
    verdicts = {}
    for ticker, d in etf_data.items():
        label, cls, note = _compute_verdict_rules(d.get("ytd_ret"), d.get("one_yr_ret"), d.get("from_high"))
        verdicts[ticker] = {"label": label, "cls": cls, "note": note}
    return {"news": news, "insights": insights, "verdicts": verdicts}


# ── Verdict engine ────────────────────────────────────────────────────────────

def _compute_verdict_rules(ytd, one_yr, from_high):
    """Fallback rules-based verdict when AI is unavailable."""
    ytd = ytd or 0; one_yr = one_yr or 0; from_high = from_high or 0
    if ytd > 3 and from_high > -2:
        return "Strong Buy", "buy",   f"+{ytd:.1f}% YTD near highs. Core holding."
    if one_yr > 10 and from_high > -15:
        return "Buy",        "buy",   f"{one_yr:.1f}% 1Y return. Good entry zone."
    if one_yr > 5 and -20 < from_high <= -5:
        return "Accumulate", "accum", f"Pulling {from_high:.1f}% from highs. Accumulate."
    if -10 <= ytd <= 3:
        return "Hold",       "hold",  f"YTD {ytd:+.1f}%. Hold and monitor."
    return "Watch",          "watch", f"Down {ytd:.1f}% YTD. High risk."


def _get_verdict(ticker, d, ai_verdicts):
    """
    Returns (label, cls, note) for a ticker.
    Uses AI-generated verdict if available; falls back to rules-based logic.
    """
    _cls_map = {
        "Strong Buy": "buy", "Buy": "buy",
        "Accumulate": "accum", "Hold": "hold", "Watch": "watch",
    }
    if ai_verdicts:
        v = ai_verdicts.get(ticker)
        if v and v.get("label"):
            label = v["label"]
            cls   = v.get("cls") or _cls_map.get(label, "hold")
            note  = v.get("note", "")
            return label, cls, note
    return _compute_verdict_rules(d.get("ytd_ret"), d.get("one_yr_ret"), d.get("from_high"))


# ── Formatters ───────────────────────────────────────────────────────────────

def fp(v):    return f"${v:.2f}" if v is not None else "N/A"
def fpct(v):  return f"{v:+.2f}%" if v is not None else "N/A"
def faum(v):  return "N/A" if v is None else (f"${v/1e9:.1f}B" if v >= 1e9 else f"${v/1e6:.0f}M")
def fmer(v):  return f"{v*100:.2f}%" if v is not None else "N/A"
def fyld(v):  return f"{v*100:.2f}%" if v is not None else "N/A"
def ccls(v):  return "up" if (v or 0) > 0 else ("dn" if (v or 0) < 0 else "neu")
def arrow(v): return "▲" if (v or 0) > 0 else "▼"


# ── HTML generation ──────────────────────────────────────────────────────────

def generate_html(etf_data, portfolios, portfolio_series, now, ai_content=None,
                  correlations=None, regime=None, dca_series=None,
                  what_changed=None, snapshot=None, benchmark_series=None,
                  screened_pool=None, eq_drift=None):
    tickers   = list(etf_data.keys())
    today     = now.date()
    today_str = now.strftime("%d %B %Y, %I:%M %p AEST")
    start_yr  = today.year - 10

    # Extract AI verdicts early so they're available for cards and verdict rows
    ai_verdicts = (ai_content or {}).get("verdicts", {})

    # Build month labels for the 10-year monthly chart (dynamic based on actual data length)
    n_months_data = max(
        (len(etf_data[t].get("monthly_indexed", [])) for t in tickers),
        default=120,
    )
    n_months_data = max(n_months_data, 2)
    month_labels = [
        (datetime(now.year, now.month, 1) - relativedelta(months=i)).strftime("%b %y")
        for i in range(n_months_data - 1, -1, -1)
    ]

    # Annual returns table (last 6 years)
    table_years = [str(y) for y in range(today.year, today.year - 6, -1)]
    table_rows  = ""
    for yr in table_years:
        label = f"{yr} YTD" if yr == str(today.year) else yr
        cells = "".join(
            f"<td class='{ccls(v)}'>{fpct(v)}</td>"
            if (v := etf_data[t].get("annual_returns", {}).get(yr)) is not None
            else "<td class='neu'>N/A</td>"
            for t in tickers
        )
        table_rows += f"<tr><td>{label}</td>{cells}</tr>"

    # Previous verdicts for trend badges
    prev_verdicts = (snapshot or {}).get("verdicts", {})
    _vrank = {"Strong Buy": 5, "Buy": 4, "Accumulate": 3, "Hold": 2, "Watch": 1}

    # Price cards
    cards_html = ""
    for ticker in tickers:
        d         = etf_data[ticker]
        ytd       = d.get("ytd_ret");  one_yr = d.get("one_yr_ret");  fh = d.get("from_high")
        five_yr   = d.get("five_yr_ret")
        ten_yr    = d.get("ten_yr_ret")
        twenty_yr = d.get("twenty_yr_ret")
        dchg      = d.get("day_chg")
        sharpe    = d.get("sharpe_1y")
        sortino   = d.get("sortino_1y")
        max_dd    = d.get("max_drawdown")
        curr_dd   = d.get("curr_drawdown")
        ma_above  = d.get("is_above_200ma")
        day_s     = f"{arrow(dchg)} {abs(dchg):.2f}% today" if dchg is not None else "— today"
        _, v_cls, v_note = _get_verdict(ticker, d, ai_verdicts)
        # MA signal badge
        ma_badge = ""
        if ma_above is not None:
            ma_col  = "#1a7a4a" if ma_above else "#c8440a"
            ma_text = "▲ 200d MA" if ma_above else "▼ 200d MA"
            ma_badge = f'<div style="font-size:7.5px;color:{ma_col};margin-top:2px">{ma_text}</div>'
        # 1Y rank badge
        all_1y    = [(t, etf_data[t].get("one_yr_ret") or 0) for t in tickers]
        rank      = sorted(all_1y, key=lambda x: -x[1]).index((ticker, one_yr or 0)) + 1
        rank_html = f'<div class="rank">#{rank} 1Y</div>'
        cards_html += f"""
    <div class="pc {d['cls']}">
      <div style="display:flex;justify-content:space-between;align-items:flex-start">
        <div><div class="pc-ticker">{ticker}</div><div class="pc-name">{d['name']}</div>{ma_badge}</div>
        {rank_html}
      </div>
      <div class="pc-price">{fp(d.get('price'))}</div>
      <div class="pc-chg {ccls(dchg)}">{day_s}</div>
      <div class="div"></div>
      <div class="pr"><span class="pl">52W HIGH</span><span class="pv">{fp(d.get('w52_high'))}</span></div>
      <div class="pr"><span class="pl">52W LOW</span><span class="pv">{fp(d.get('w52_low'))}</span></div>
      <div class="pr"><span class="pl">FROM HIGH</span><span class="pv {ccls(fh)}">{fpct(fh)}</span></div>
      <div class="pr"><span class="pl">CURR DRAWDOWN</span><span class="pv {ccls(curr_dd)}">{f"{curr_dd:+.1f}%" if curr_dd is not None else "N/A"}</span></div>
      <div class="pr"><span class="pl">MAX DRAWDOWN</span><span class="pv dn">{f"{max_dd:.1f}%" if max_dd is not None else "N/A"}</span></div>
      <div class="pr"><span class="pl">1Y RETURN</span><span class="pv {ccls(one_yr)}">{fpct(one_yr)}</span></div>
      <div class="pr"><span class="pl">5Y RETURN</span><span class="pv {ccls(five_yr)}">{fpct(five_yr)}</span></div>
      <div class="pr"><span class="pl">10Y RETURN</span><span class="pv {ccls(ten_yr)}">{fpct(ten_yr)}</span></div>
      <div class="pr"><span class="pl">20Y RETURN</span><span class="pv {ccls(twenty_yr)}">{fpct(twenty_yr)}</span></div>
      <div class="pr"><span class="pl">YTD</span><span class="pv {ccls(ytd)}">{fpct(ytd)}</span></div>
      <div class="pr"><span class="pl">SHARPE (1Y)</span><span class="pv {ccls(sharpe)}">{f"{sharpe:.2f}" if sharpe is not None else "N/A"}</span></div>
      <div class="pr"><span class="pl">SORTINO (1Y)</span><span class="pv {ccls(sortino)}">{f"{sortino:.2f}" if sortino is not None else "N/A"}</span></div>
      <div class="pr"><span class="pl">VOL (1Y)</span><span class="pv">{f"{d['vol_1y']:.1f}%" if d.get('vol_1y') else 'N/A'}</span></div>
      <div class="pr"><span class="pl">MER</span><span class="pv">{fmer(d.get('mer'))}</span></div>
      <div class="pr"><span class="pl">YIELD</span><span class="pv">{fyld(d.get('div_yield'))}</span></div>
      <div class="pr"><span class="pl">AUM</span><span class="pv">{faum(d.get('aum'))}</span></div>
    </div>"""

    # Verdict row
    verdicts_html = ""
    verdict_source = "AI-Powered" if ai_verdicts else "Rules-Based"
    regime_badge  = ""
    if regime:
        rlabel, rcolor, rdesc = regime
        regime_badge = f' · <span style="color:{rcolor};font-weight:500">{rlabel.upper()} REGIME</span> — {rdesc}'
    for ticker in tickers:
        d = etf_data[ticker]
        v_label, v_cls, v_note = _get_verdict(ticker, d, ai_verdicts)
        # Trend badge vs previous run
        trend_html = ""
        prev_lbl   = prev_verdicts.get(ticker)
        if prev_lbl and prev_lbl != v_label:
            cur_r  = _vrank.get(v_label, 0)
            prev_r = _vrank.get(prev_lbl, 0)
            if cur_r > prev_r:
                trend_html = f'<div style="font-size:7.5px;color:#1a7a4a;margin-top:3px">↑ from {prev_lbl}</div>'
            else:
                trend_html = f'<div style="font-size:7.5px;color:#c8440a;margin-top:3px">↓ from {prev_lbl}</div>'
        verdicts_html += f"""
    <div class="vc">
      <div class="vt" style="color:{d['color']}">{ticker}</div>
      <div class="vb {v_cls}">{v_label}</div>
      {trend_html}
      <div class="vn">{v_note}</div>
    </div>"""

    # Portfolio allocation table
    alloc_rows = ""
    for ticker in tickers:
        d   = etf_data[ticker]
        tds = "".join(
            f"<td>{portfolios[p]['allocs'].get(ticker, 0)*100:.0f}%</td>"
            for p in portfolios
        )
        alloc_rows += f"<tr><td><span class='dot' style='background:{d['color']}'></span>{ticker} — {d['name']}</td>{tds}</tr>"

    # Monthly ETF chart
    monthly_ds = []
    for ticker in tickers:
        d    = etf_data[ticker]
        vals = d.get("monthly_indexed", [])
        while len(vals) < n_months_data: vals = [100.0] + vals
        vals = vals[-n_months_data:]
        monthly_ds.append({
            "label": ticker, "data": vals,
            "borderColor": d["color"],
            "borderWidth": 2.5, "pointRadius": 0, "pointHoverRadius": 4,
            "tension": 0.4, "fill": False,
        })

    # Portfolio chart datasets + stat cards
    port_ds         = []
    port_stats_html = ""
    port_labels_clean = [l.replace("\n", " ") for l in list(portfolio_series.values())[0]["labels"]]

    for pname, ps in portfolio_series.items():
        port_ds.append({
            "label": pname, "data": ps["values"],
            "borderColor": ps["color"],
            "backgroundColor": ps["color"] + "12",
            "borderWidth": 2.5,
            "pointRadius": 3, "pointHoverRadius": 6,
            "pointBackgroundColor": ps["color"],
            "tension": 0.35, "fill": True,
            "borderDash": [5, 3] if portfolios[pname]["dashed"] else [],
        })
        sign    = "+" if ps["total_pct"] > 0 else ""
        ret_col = "#4aaa74" if ps["total_pct"] > 0 else "#e87050"
        port_stats_html += f"""
        <div style="background:var(--bg2);border-radius:3px;padding:14px 16px;border-left:2px solid {ps['color']}">
          <div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;color:var(--ink3);margin-bottom:4px">{pname}</div>
          <div style="font-family:'Fraunces',serif;font-size:22px;font-weight:900;color:var(--ink);letter-spacing:-1px">${ps['final']:,.0f}</div>
          <div style="font-size:10px;color:{ret_col};margin:3px 0">{sign}{ps['total_pct']}% total · {ps['cagr']}% CAGR p.a.</div>
          <div style="font-size:8px;color:var(--ink2);line-height:1.5">{ps['desc']}</div>
        </div>"""

    # DCA overlay lines on portfolio chart
    if dca_series:
        for pname, ds in dca_series.items():
            port_ds.append({
                "label": f"{pname} (DCA)", "data": ds["values"],
                "borderColor": ds["color"],
                "borderWidth": 1.5,
                "pointRadius": 0, "pointHoverRadius": 4,
                "tension": 0.35, "fill": False,
                "borderDash": [4, 4],
            })

    # ASX 200 benchmark line
    if benchmark_series:
        port_ds.append({
            "label": "ASX 200 (benchmark)", "data": benchmark_series["values"],
            "borderColor": "#9a9690",
            "borderWidth": 1.5,
            "pointRadius": 0, "pointHoverRadius": 4,
            "tension": 0.35, "fill": False,
            "borderDash": [6, 3],
        })

    etf_legend  = "".join(f'<div class="li"><div class="ld" style="background:{etf_data[t]["color"]}"></div>{t}</div>' for t in tickers)
    port_legend = "".join(f'<div class="li"><div class="ld" style="background:{portfolio_series[p]["color"]}"></div><span style="color:var(--ink)">{p}</span></div>' for p in portfolios)
    if benchmark_series:
        port_legend += '<div class="li"><div class="ld" style="background:#9a9690;border-style:dashed"></div><span style="color:var(--ink2)">ASX 200</span></div>'

    # ── AI content: news, insights ───────────────────────────────────────────
    news_items  = (ai_content or {}).get("news", [])
    insights    = (ai_content or {}).get("insights", {})

    news_cards_html = ""
    for item in news_items:
        sev   = item.get("severity", "amber")
        tag   = item.get("tag", "")
        head  = item.get("headline", "")
        body  = item.get("body", "")
        impact= item.get("impact", "")
        icon  = {"red": "🔴", "amber": "🟡", "green": "🟢"}.get(sev, "⚪")
        news_cards_html += f"""
      <div class="news-block {sev}">
        <div class="news-tag">{icon} {tag}</div>
        <div class="news-head">{head}</div>
        <div class="news-body">{body}</div>
        <div class="news-impact">⚡ {impact}</div>
      </div>"""

    if not news_cards_html:
        news_cards_html = '<div style="color:#5a5650;font-size:10px;padding:12px 0">No news available. Set ANTHROPIC_API_KEY to enable AI-generated news.</div>'

    insights_html = ""
    for ticker in tickers:
        d    = etf_data[ticker]
        text = insights.get(ticker, "")
        if text:
            insights_html += f"""
      <div class="ins {d['cls']}">
        <div class="it">{ticker} — {d['name']}</div>
        <div class="ix">{text}</div>
      </div>"""

    # ── Correlation heatmap ───────────────────────────────────────────────────
    corr_html = ""
    if correlations:
        valid_t = [t for t in tickers if t in correlations]
        header  = "<tr><th></th>" + "".join(
            f'<th style="color:{etf_data[t]["color"]}">{t}</th>' for t in valid_t
        ) + "</tr>"
        rows = ""
        for t1 in valid_t:
            cells = f'<td style="font-weight:500;color:{etf_data[t1]["color"]}">{t1}</td>'
            for t2 in valid_t:
                v = correlations.get(t1, {}).get(t2)
                if v is None:
                    cells += "<td>—</td>"
                else:
                    if t1 == t2:
                        bg = "rgba(100,100,100,0.15)"
                    elif v > 0:
                        bg = f"rgba(26,122,74,{min(v*0.6,0.55):.2f})"
                    else:
                        bg = f"rgba(200,68,10,{min(abs(v)*0.6,0.55):.2f})"
                    cells += f'<td style="background:{bg};text-align:center">{v:.2f}</td>'
            rows += f"<tr>{cells}</tr>"
        corr_html = f"""
  <div class="tbl-card" style="margin-bottom:20px">
    <div class="sec-label">10-Year Correlation Matrix — Per-Year Pearson Averaged Across Up To 10 Calendar Years</div>
    <div style="overflow-x:auto"><table>{header}<tbody>{rows}</tbody></table></div>
    <div style="font-size:8.5px;color:var(--ink3);margin-top:10px;font-style:italic">
      Each cell = average of up to 10 annual Pearson correlations · Green = positive (move together) · Red = negative (inverse) · Darker = stronger signal
    </div>
  </div>"""

    # ── Portfolio drift from equal weight ─────────────────────────────────────
    drift_rows = ""
    if eq_drift:
        for t in tickers:
            d_val = eq_drift.get(t, 0)
            col   = "#1a7a4a" if d_val > 0.5 else ("#c8440a" if d_val < -0.5 else "#9a9690")
            sign  = "+" if d_val >= 0 else ""
            drift_rows += f'<span style="margin-right:16px;font-size:9px">' \
                          f'<span style="color:{etf_data[t]["color"]}">{t}</span> ' \
                          f'<span style="color:{col}">{sign}{d_val:.1f}pp</span></span>'

    drift_html = ""
    if drift_rows:
        drift_html = f"""
    <div style="background:var(--bg2);border-radius:3px;padding:11px 14px;margin-top:10px;margin-bottom:0">
      <div style="font-size:8px;letter-spacing:2px;text-transform:uppercase;color:var(--ink3);margin-bottom:6px">
        YTD Drift From Equal Weight (1/N target)
      </div>
      <div>{drift_rows}</div>
    </div>"""

    # ── DCA stat cards ────────────────────────────────────────────────────────
    dca_stats_html = ""
    if dca_series:
        for pname, ds in dca_series.items():
            sign    = "+" if ds["total_pct"] > 0 else ""
            ret_col = "#4aaa74" if ds["total_pct"] > 0 else "#e87050"
            dca_stats_html += f"""
        <div style="background:var(--bg2);border-radius:3px;padding:14px 16px;border-left:2px dashed {ds['color']}">
          <div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;color:var(--ink3);margin-bottom:4px">{pname} — DCA</div>
          <div style="font-family:'Fraunces',serif;font-size:22px;font-weight:900;color:var(--ink);letter-spacing:-1px">${ds['final']:,.0f}</div>
          <div style="font-size:10px;color:{ret_col};margin:3px 0">{sign}{ds['total_pct']}% on $10k invested · {ds['cagr']}% CAGR p.a.</div>
          <div style="font-size:8px;color:var(--ink2)">$1,000/year over 10 years vs $10,000 lump sum</div>
        </div>"""

    # ── Screened pool table ───────────────────────────────────────────────────
    screened_html = ""
    if screened_pool:
        pool_rows = "".join(
            f"<tr><td>{t}</td><td style='color:#9a9690'>{meta['name']}</td>"
            f"<td class='{ccls(meta['one_yr_ret'])}'>{meta['one_yr_ret']:+.1f}%</td>"
            f"<td style='color:#5a5650;font-size:8.5px'>Not selected this run</td></tr>"
            for t, meta in sorted(screened_pool.items(),
                                  key=lambda x: -x[1]["one_yr_ret"])
        )
        screened_html = f"""
  <div class="tbl-card" style="margin-top:20px">
    <div class="sec-label">Candidate Pool — Screened But Not Added This Run</div>
    <div style="overflow-x:auto">
    <table>
      <thead><tr><th>Ticker</th><th>Name</th><th>1Y Return</th><th>Status</th></tr></thead>
      <tbody>{pool_rows}</tbody>
    </table>
    </div>
    <div style="font-size:8.5px;color:var(--ink3);margin-top:8px;font-style:italic">
      Top 3 by 1Y return are added to the live report each run. Pool is re-screened daily.
    </div>
  </div>"""

    # ── What Changed banner ───────────────────────────────────────────────────
    whatchanged_html = ""
    if what_changed:
        prev_date = (snapshot or {}).get("date", "")
        prev_lbl  = f" vs {prev_date}" if prev_date else ""
        whatchanged_html = f"""
  <div style="background:#f0ece8;border-left:3px solid #c8440a;border-radius:3px;
              padding:13px 16px;margin-bottom:18px;font-size:10px;line-height:1.7;color:#3a3630">
    <div style="font-size:8px;letter-spacing:2px;text-transform:uppercase;color:#9a9690;margin-bottom:5px">
      What Changed{prev_lbl}
    </div>
    {what_changed}
  </div>"""

    # ── Dynamic CSS strings — built from all active tickers (core + discovered) ─
    css_vars      = ";".join(f"--{etf_data[t]['cls']}:{etf_data[t]['color']}" for t in tickers)
    pc_before_css = "".join(f".pc.{etf_data[t]['cls']}::before{{background:var(--{etf_data[t]['cls']})}}" for t in tickers)
    pc_ticker_css = "".join(f".pc.{etf_data[t]['cls']} .pc-ticker{{color:var(--{etf_data[t]['cls']})}}" for t in tickers)
    ins_css       = "".join(f".ins.{etf_data[t]['cls']}{{border-color:var(--{etf_data[t]['cls']})}}" for t in tickers)
    grid_cols     = f"repeat({len(tickers)},1fr)"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ASX ETF Dashboard — {now.strftime('%d %b %Y')}</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Fraunces:ital,wght@0,300;0,600;0,900;1,300;1,600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{{--bg:#f4f1e8;--bg2:#ebe8de;--card:#fff;--ink:#1c1916;--ink2:#5a5650;--ink3:#9a9690;--border:#dedad0;
  {css_vars};}}
*{{margin:0;padding:0;box-sizing:border-box;}}body{{background:var(--bg);color:var(--ink);font-family:'DM Mono',monospace;}}
header{{padding:34px 48px 22px;border-bottom:1px solid var(--border);display:flex;align-items:flex-end;justify-content:space-between;}}
.h-title{{font-family:'Fraunces',serif;font-size:38px;font-weight:900;letter-spacing:-2px;line-height:1;}}
.h-title em{{font-style:italic;color:var(--ivv);}}
.h-sub{{font-size:9px;color:var(--ink3);letter-spacing:2px;margin-top:5px;text-transform:uppercase;}}
.h-right{{text-align:right;font-size:10px;color:var(--ink3);}}
.h-right strong{{display:block;font-family:'Fraunces',serif;font-size:14px;color:var(--ink2);margin-bottom:3px;font-weight:600;}}
.main{{padding:26px 48px 56px;max-width:1440px;margin:0 auto;}}
.sec-label{{font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--ink3);margin-bottom:14px;display:flex;align-items:center;gap:10px;}}
.sec-label::after{{content:'';flex:1;height:1px;background:var(--border);}}
.cards-grid{{display:grid;grid-template-columns:{grid_cols};gap:11px;margin-bottom:22px;}}
.pc{{background:var(--card);border:1px solid var(--border);border-radius:4px;padding:14px 13px;position:relative;overflow:hidden;}}
.pc::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;}}
{pc_before_css}
.pc-ticker{{font-size:10px;letter-spacing:2px;font-weight:500;}}
{pc_ticker_css}
.pc-name{{font-size:8.5px;color:var(--ink3);margin-bottom:8px;line-height:1.4;min-height:20px;}}
.pc-price{{font-family:'Fraunces',serif;font-size:23px;font-weight:900;letter-spacing:-1px;}}
.pc-chg{{font-size:11px;font-weight:500;margin:3px 0 10px;}}
.up{{color:#1a7a4a}}.dn{{color:#c8440a}}.neu{{color:var(--ink3)}}
.div{{height:1px;background:var(--border);margin-bottom:9px;}}
.pr{{display:flex;justify-content:space-between;margin-bottom:4px;}}
.pl{{font-size:8.5px;color:var(--ink3);letter-spacing:.5px;}}.pv{{font-size:9.5px;font-weight:500;}}
.rank{{font-size:8px;background:var(--bg2);border-radius:2px;padding:2px 5px;color:var(--ink3);white-space:nowrap;}}
.chart-card,.tbl-card{{background:var(--card);border:1px solid var(--border);border-radius:4px;padding:22px;margin-bottom:20px;}}
.chart-hdr{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:16px;flex-wrap:wrap;gap:10px;}}
.chart-t{{font-family:'Fraunces',serif;font-size:14px;font-weight:700;}}
.chart-s{{font-size:8.5px;color:var(--ink3);letter-spacing:.5px;margin-top:3px;}}
.legend{{display:flex;flex-wrap:wrap;gap:10px;}}.li{{display:flex;align-items:center;gap:5px;font-size:9.5px;font-weight:500;}}
.ld{{width:20px;height:2.5px;border-radius:2px;flex-shrink:0;}}
table{{width:100%;border-collapse:collapse;}}
th{{font-size:8.5px;letter-spacing:1px;color:var(--ink3);text-transform:uppercase;padding:0 0 8px;border-bottom:1px solid var(--border);font-weight:500;text-align:left;}}
th:not(:first-child){{text-align:right;}}
td{{padding:7px 0;font-size:9.5px;border-bottom:1px solid var(--bg2);color:var(--ink2);}}
td:first-child{{color:var(--ink);font-weight:500;}}td:not(:first-child){{text-align:right;}}
tr:last-child td{{border-bottom:none;}}
.dot{{display:inline-block;width:6px;height:6px;border-radius:50%;margin-right:5px;vertical-align:middle;}}
.verdict-row{{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:20px;}}
.vc{{background:var(--card);border:1px solid var(--border);border-radius:4px;padding:13px 11px;text-align:center;}}
.vt{{font-size:9.5px;font-weight:500;letter-spacing:2px;margin-bottom:5px;}}
.vb{{display:inline-block;font-size:7.5px;letter-spacing:1.5px;text-transform:uppercase;padding:3px 7px;border-radius:2px;font-weight:500;margin-bottom:6px;}}
.vb.buy{{background:rgba(26,122,74,.1);color:#1a7a4a}}.vb.hold{{background:rgba(184,146,10,.1);color:var(--qau)}}
.vb.watch{{background:rgba(200,68,10,.1);color:var(--ivv)}}.vb.accum{{background:rgba(10,106,138,.1);color:var(--vgs)}}
.vn{{font-size:8.5px;color:var(--ink3);line-height:1.5;}}
.dark{{background:var(--bg2);border-radius:4px;padding:28px 32px;margin-bottom:20px;}}
.dark .sec-label{{color:var(--ink3);}}.dark .sec-label::after{{background:var(--border);}}
.port-chart-card{{background:var(--card);border:1px solid var(--border);border-radius:4px;padding:22px;margin-top:20px;}}
.port-stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:11px;margin-top:14px;}}
.alloc-note{{font-size:8.5px;color:var(--ink3);margin-top:10px;font-style:italic;}}
.news-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:13px;margin-bottom:24px;}}
.news-block{{background:var(--card);border-radius:3px;padding:15px 17px;border-left:3px solid var(--border);}}
.news-block.red{{border-color:#c8440a;}}.news-block.amber{{border-color:#b8920a;}}.news-block.green{{border-color:#1a7a4a;}}
.news-tag{{font-size:8px;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;font-weight:500;}}
.news-block.red .news-tag{{color:#c8440a;}}.news-block.amber .news-tag{{color:#b8920a;}}.news-block.green .news-tag{{color:#4aaa74;}}
.news-head{{font-size:12px;color:var(--ink);font-weight:500;margin-bottom:6px;line-height:1.4;}}
.news-body{{font-size:10px;color:var(--ink2);line-height:1.65;}}
.news-impact{{font-size:9px;margin-top:8px;padding-top:8px;border-top:1px solid var(--border);letter-spacing:.5px;}}
.news-block.red .news-impact{{color:#e87050;}}.news-block.amber .news-impact{{color:#d8a830;}}.news-block.green .news-impact{{color:#4aaa74;}}
.ig{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin-bottom:22px;}}
.ins{{border-left:2px solid var(--border);padding-left:13px;}}
{ins_css}
.it{{font-size:9px;letter-spacing:2px;text-transform:uppercase;color:var(--ink3);margin-bottom:6px;}}
.ix{{font-size:11px;color:var(--ink2);line-height:1.75;}}
footer{{padding:14px 48px;border-top:1px solid var(--border);display:flex;justify-content:space-between;font-size:8.5px;color:var(--ink3);}}
@media(max-width:1024px){{
  .cards-grid{{grid-template-columns:repeat(3,1fr);}}
  .verdict-row{{grid-template-columns:repeat(3,1fr);}}
  .news-grid{{grid-template-columns:repeat(2,1fr);}}
  .ig{{grid-template-columns:repeat(2,1fr);}}
}}
@media(max-width:768px){{
  header{{padding:20px 22px 14px;}}
  .main{{padding:16px 20px 36px;}}
  footer{{padding:12px 20px;flex-direction:column;gap:5px;}}
  .cards-grid{{grid-template-columns:repeat(2,1fr);gap:9px;}}
  .verdict-row{{grid-template-columns:repeat(2,1fr);}}
  .news-grid{{grid-template-columns:1fr;}}
  .ig{{grid-template-columns:1fr;}}
  .port-stats{{grid-template-columns:repeat(2,1fr);}}
  .chart-card,.tbl-card{{padding:16px 14px;}}
  .dark{{padding:20px 18px;}}
  .port-chart-card{{padding:16px 14px;}}
}}
@media(max-width:480px){{
  header{{padding:14px;flex-direction:column;align-items:flex-start;gap:10px;}}
  .h-title{{font-size:26px;letter-spacing:-1px;}}
  .h-right{{text-align:left;}}
  .main{{padding:12px 12px 28px;}}
  footer{{padding:10px 12px;flex-direction:column;gap:5px;}}
  .cards-grid{{grid-template-columns:repeat(2,1fr);gap:8px;}}
  .verdict-row{{grid-template-columns:repeat(2,1fr);gap:8px;}}
  .news-grid{{grid-template-columns:1fr;}}
  .ig{{grid-template-columns:1fr;}}
  .port-stats{{grid-template-columns:1fr;}}
  .chart-card,.tbl-card{{padding:12px 10px;}}
  .dark{{padding:16px 12px;}}
  .port-chart-card{{padding:12px 10px;}}
  .chart-hdr{{flex-direction:column;gap:10px;}}
  .pc{{padding:11px 10px;}}
  .pc-price{{font-size:18px;}}
  .legend{{gap:7px;}}
}}
</style>
</head>
<body>
<header>
  <div>
    <div class="h-title">ASX <em>ETF</em> Pulse</div>
    <div class="h-sub">IVV · FANG · VAS · QAU · GOLD · VGS &nbsp;|&nbsp; AUD &nbsp;|&nbsp; Fully Dynamic</div>
  </div>
  <div class="h-right">
    <strong>{today_str}</strong>
    All data live from Yahoo Finance · Allocations auto-computed today
  </div>
</header>
<div class="main">
  {whatchanged_html}
  <div class="sec-label">Live Prices — {now.strftime('%d %b %Y')} · Ranked by 1Y Return</div>
  <div class="cards-grid">{cards_html}</div>

  <div class="chart-card">
    <div class="chart-hdr">
      <div><div class="chart-t">10-Year Monthly Performance — Indexed to 100</div>
      <div class="chart-s">Monthly closing prices · 10-year lookback · Fetched live from Yahoo Finance · AUD</div></div>
      <div class="legend">{etf_legend}</div>
    </div>
    <canvas id="etfChart" style="max-height:270px"></canvas>
  </div>

  <div class="tbl-card">
    <div class="sec-label">Annual Returns — Computed Live from Yahoo Finance History</div>
    <div style="overflow-x:auto">
    <table>
      <thead><tr><th>Year</th>{''.join(f'<th><span class="dot" style="background:{etf_data[t]["color"]}"></span>{t}</th>' for t in tickers)}</tr></thead>
      <tbody>{table_rows}</tbody>
    </table>
    </div>
  </div>

  {corr_html}

  <div class="sec-label">Verdict — {verdict_source} · Live Yahoo Finance Data + News Context{regime_badge}</div>
  <div class="verdict-row">{verdicts_html}</div>

  <div class="dark">
    <div class="sec-label">Global Events Impacting Your Portfolio — {now.strftime('%d %b %Y')}</div>
    <div class="news-grid">{news_cards_html}</div>

    {'<div class="sec-label">ETF Insights — In Context of Global Events</div><div class="ig">' + insights_html + '</div>' if insights_html else ''}

    <div class="sec-label">AI-Powered Portfolio Strategies — Allocations Recommended by Claude Based on Today's Live Data</div>

    <div class="tbl-card" style="padding:18px 20px;margin-bottom:0">
      <div style="overflow-x:auto">
      <table>
        <thead>
          <tr>
            <th style="color:#6a6660">ETF</th>
            {''.join(f'<th style="color:{portfolio_series[p]["color"]}">{p}</th>' for p in portfolios)}
          </tr>
        </thead>
        <tbody>{alloc_rows}</tbody>
      </table>
      </div>
      <div class="alloc-note">
        {'  ·  '.join(f'<span style="color:{portfolio_series[p]["color"]}">{p}</span>: {portfolios[p].get("rationale", allocation_desc(portfolios[p]["allocs"]))}' for p in portfolios)}
      </div>
      {drift_html}
    </div>

    <div class="port-chart-card">
      <div class="chart-hdr">
        <div><div class="chart-t">10-Year Portfolio Simulation — $10,000 Lump Sum &amp; DCA · Jan {start_yr}</div>
        <div class="chart-s">Annual returns from Yahoo Finance · Solid = lump sum · Dashed = DCA $1k/yr · Grey = ASX 200 benchmark · AUD</div></div>
        <div class="legend">{port_legend}</div>
      </div>
      <canvas id="portChart" style="max-height:310px"></canvas>
      <div class="port-stats">{port_stats_html}</div>
      {f'<div style="margin-top:12px"><div style="font-size:8px;letter-spacing:2px;text-transform:uppercase;color:var(--ink3);margin-bottom:8px">DCA Equivalent ($1,000/year — same total invested)</div><div class="port-stats">{dca_stats_html}</div></div>' if dca_stats_html else ''}
    </div>
  </div>

  {screened_html}

</div>
<footer>
  <span>Generated {today_str} · Zero hardcoded values · Prices, returns, volatility all live from Yahoo Finance · Portfolio strategies powered by Claude AI</span>
  <span>⚠ Not financial advice. Past performance ≠ future results.</span>
</footer>
<script>
Chart.defaults.font.family="'DM Mono',monospace";Chart.defaults.font.size=10;
new Chart(document.getElementById('etfChart').getContext('2d'),{{
  type:'line',data:{{labels:{json.dumps(month_labels)},datasets:{json.dumps(monthly_ds)}}},
  options:{{responsive:true,interaction:{{mode:'index',intersect:false}},
    plugins:{{legend:{{display:false}},tooltip:{{backgroundColor:'#1c1916',titleColor:'#9a9690',bodyColor:'#f0ece4',borderColor:'#3a3630',borderWidth:1,padding:11,
      callbacks:{{label:c=>` ${{c.dataset.label}}: ${{c.parsed.y.toFixed(1)}}`}}}}}},
    scales:{{x:{{grid:{{color:'rgba(0,0,0,0.04)'}},ticks:{{color:'#9a9690'}}}},
      y:{{grid:{{color:'rgba(0,0,0,0.04)'}},ticks:{{color:'#9a9690',callback:v=>v.toFixed(0)}},
        title:{{display:true,text:'Indexed (base=100)',color:'#9a9690',font:{{size:9}}}}}}}}}}
}});
new Chart(document.getElementById('portChart').getContext('2d'),{{
  type:'line',data:{{labels:{json.dumps(port_labels_clean)},datasets:{json.dumps(port_ds)}}},
  options:{{responsive:true,interaction:{{mode:'index',intersect:false}},
    plugins:{{legend:{{display:false}},tooltip:{{backgroundColor:'#1c1916',titleColor:'#9a9690',bodyColor:'#f0ece4',borderColor:'#3a3630',borderWidth:1,padding:13,
      callbacks:{{label:c=>{{const v=c.parsed.y,g=((v/{INITIAL}-1)*100).toFixed(1),s=g>=0?'+':'';return`  ${{c.dataset.label}}: $${{Math.round(v).toLocaleString()}} (${{s}}${{g}}%)`;}}}}}}}},
    scales:{{x:{{grid:{{color:'rgba(0,0,0,0.04)'}},ticks:{{color:'#9a9690',maxRotation:0}}}},
      y:{{grid:{{color:'rgba(0,0,0,0.04)'}},ticks:{{color:'#9a9690',callback:v=>'$'+(v/1000).toFixed(0)+'k'}},
        title:{{display:true,text:'Portfolio Value (AUD)',color:'#9a9690',font:{{size:9}}}}}}}}}}
}});
</script>
</body>
</html>"""


# ── Email ────────────────────────────────────────────────────────────────────

def send_email(recipient, now, dashboard_url):
    """
    Sends a notification email with a clickable link to the hosted dashboard.
    No attachment — the link opens directly in the browser.
    """
    sender   = os.environ["EMAIL_SENDER"]
    password = os.environ["EMAIL_APP_PASSWORD"]

    date_str = now.strftime("%d %b %Y")
    time_str = now.strftime("%I:%M %p AEST")

    # Plain-text fallback
    plain = (
        f"ASX ETF Dashboard — {date_str}\n\n"
        f"Your daily ETF Dashboard is ready.\n\n"
        f"View it here: {dashboard_url}\n\n"
        f"Generated {date_str} at {time_str}.\n"
        f"Not financial advice. Past performance does not guarantee future results."
    )

    # HTML notification email — no attachment, just the link
    html_email = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ASX ETF Dashboard — {date_str}</title>
</head>
<body style="margin:0;padding:0;background:#f4f1e8;font-family:'Courier New',Courier,monospace;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#f4f1e8;padding:40px 20px;">
    <tr><td align="center">
      <table width="520" cellpadding="0" cellspacing="0" style="background:#ffffff;border:1px solid #dedad0;border-radius:4px;overflow:hidden;">
        <!-- Header bar -->
        <tr><td style="background:#1c1916;padding:22px 32px;">
          <div style="font-size:22px;font-weight:900;color:#f4f1e8;letter-spacing:-1px;">ASX <span style="color:#c8440a;font-style:italic;">ETF</span> Pulse</div>
          <div style="font-size:10px;color:#6a6660;letter-spacing:2px;margin-top:4px;text-transform:uppercase;">Daily Dashboard · {date_str}</div>
        </td></tr>
        <!-- Body -->
        <tr><td style="padding:32px 32px 24px;">
          <p style="margin:0 0 16px;font-size:13px;color:#5a5650;line-height:1.7;">
            Your daily ASX ETF Dashboard is ready — AI-powered verdicts, live Yahoo Finance data, macro news, and portfolio simulations.
          </p>
          <p style="margin:0 0 28px;font-size:11px;color:#9a9690;line-height:1.6;">
            Generated {date_str} at {time_str} · IVV · FANG · VAS · QAU · GOLD · VGS
          </p>
          <!-- CTA Button -->
          <table cellpadding="0" cellspacing="0" width="100%">
            <tr><td align="center">
              <a href="{dashboard_url}"
                 style="display:inline-block;background:#1c1916;color:#f4f1e8;text-decoration:none;
                        padding:14px 40px;border-radius:3px;font-size:11px;font-weight:500;
                        letter-spacing:2px;text-transform:uppercase;">
                View Dashboard &rarr;
              </a>
            </td></tr>
          </table>
          <p style="margin:20px 0 0;font-size:10px;color:#bab6ae;text-align:center;">
            Or copy this link: <a href="{dashboard_url}" style="color:#c8440a;word-break:break-all;">{dashboard_url}</a>
          </p>
        </td></tr>
        <!-- Footer -->
        <tr><td style="border-top:1px solid #ebe8de;padding:14px 32px;">
          <p style="margin:0;font-size:9px;color:#bab6ae;text-align:center;">
            Not financial advice &nbsp;·&nbsp; Past performance does not guarantee future results
          </p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""

    msg = MIMEMultipart("alternative")
    msg["From"]    = sender
    msg["To"]      = recipient
    msg["Subject"] = f"ASX ETF Dashboard — {date_str}"
    msg.attach(MIMEText(plain,      "plain"))
    msg.attach(MIMEText(html_email, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
        srv.login(sender, password)
        srv.sendmail(sender, recipient, msg.as_string())
    print(f"  ✓ Sent to {recipient} (link: {dashboard_url})")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    now = datetime.now(AEST)
    print(f"\n🚀 ASX ETF Dashboard — {now.strftime('%d %b %Y %H:%M AEST')}")
    print("  Fully dynamic. Prices, returns, volatility & allocations all computed today.\n")

    print("📂 Loading previous snapshot...")
    snapshot = load_last_snapshot()
    if snapshot.get("date"):
        print(f"  Previous run: {snapshot['date']}")
    else:
        print("  No previous snapshot — this appears to be the first run.")

    print("\n🔍 Screening candidate pool for top performers...")
    discovered, screened_pool = discover_top_etfs(now, n=3)
    TICKERS.update(discovered)
    if discovered:
        print(f"  Added {len(discovered)} ETF(s): {', '.join(discovered)}")
    if screened_pool:
        print(f"  Screened (not added): {', '.join(screened_pool)}")

    print("\n📡 Fetching live data from Yahoo Finance...")
    etf_data = fetch_all_data(now)

    print("\n📊 Computing analytics (correlations, regime, drift)...")
    correlations = compute_correlations(etf_data)
    regime       = detect_market_regime(etf_data)
    eq_drift     = compute_equal_weight_drift(etf_data)
    rlabel, rcolor, rdesc = regime
    print(f"  Regime: {rlabel.upper()} — {rdesc}")

    print("\n🤖 Generating AI-powered portfolio strategies (Anthropic API)...")
    portfolios = generate_ai_portfolios(etf_data, now)
    if portfolios is None:
        print("  Falling back to algorithmic portfolios...")
        portfolios = compute_portfolio_allocations(etf_data)
    for pname, pmeta in portfolios.items():
        print(f"  {pname:15s} → {allocation_desc(pmeta['allocs'])}")

    print("\n📐 Running simulations (lump sum, DCA, benchmark)...")
    portfolio_series = build_portfolio_series(etf_data, portfolios, now)
    for pname, ps in portfolio_series.items():
        print(f"  {pname:15s} → ${ps['final']:>10,.0f}  ({ps['total_pct']:+.1f}%  CAGR {ps['cagr']:.1f}%)")
    dca_series       = build_dca_series(etf_data, portfolios, now)
    benchmark_series = fetch_benchmark_series(now)

    print("\n📰 Fetching AI-generated news, insights & what-changed (Anthropic API)...")
    ai_content   = fetch_ai_news(etf_data, now, regime=regime)
    what_changed = generate_whatchanged_summary(etf_data, snapshot, ai_content, now)

    print("\n💾 Saving snapshot...")
    save_snapshot(etf_data, ai_content, now)

    print("\n🎨 Generating HTML...")
    html = generate_html(
        etf_data, portfolios, portfolio_series, now, ai_content,
        correlations=correlations, regime=regime, dca_series=dca_series,
        what_changed=what_changed, snapshot=snapshot,
        benchmark_series=benchmark_series, screened_pool=screened_pool,
        eq_drift=eq_drift,
    )

    out = Path(__file__).parent.parent / "output"
    out.mkdir(exist_ok=True)
    fname = out / f"etf-dashboard-{now.strftime('%Y-%m-%d')}.html"
    fname.write_text(html, encoding="utf-8")
    (out / "latest.html").write_text(html, encoding="utf-8")
    print(f"  ✓ Saved → {fname.name}")

    print("\n📧 Sending email...")
    recipient     = os.environ.get("EMAIL_RECIPIENT") or os.environ["EMAIL_SENDER"]
    dashboard_url = os.environ.get("DASHBOARD_URL", "")
    if not dashboard_url:
        print("  ⚠ DASHBOARD_URL not set — link in email will be empty")
    send_email(recipient, now, dashboard_url)
    print("\n✅ Done.\n")


if __name__ == "__main__":
    main()
