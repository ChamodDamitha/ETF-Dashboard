"""
ASX ETF Dashboard Generator — Fully Dynamic
=============================================
Zero hardcoded data. Everything is computed from live Yahoo Finance data on the day.

DYNAMIC ELEMENTS:
  ✓ Current prices, day change, 52W range
  ✓ YTD return, 1-year return, 10-year annual returns per calendar year
  ✓ AUM, MER, dividend yield, P/E
  ✓ 14-month monthly indexed chart
  ✓ Portfolio allocations — auto-weighted daily by 1Y performance rank
  ✓ 10-year portfolio simulation — all returns from Yahoo Finance history
  ✓ Verdict ratings — AI-powered using live metrics, historical data & news context

PORTFOLIO WEIGHTING STRATEGIES (all auto-computed):
  Momentum   — weighted by rank of 1Y return (best performers get most weight)
  Risk-Adj   — weighted by 1Y return / volatility (Sharpe-style)
  Equal      — equal weight across all ETFs

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

AEST = ZoneInfo("Australia/Sydney")
INITIAL = 10_000.0

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


# ── STEP 1: Fetch all live data ───────────────────────────────────────────────

def fetch_all_data(now):
    """
    Fetches everything from Yahoo Finance in a single history pull per ticker.
    Returns a dict keyed by ticker with all computed metrics.
    """
    today         = now.date()
    start_of_year = date(today.year, 1, 1)
    ten_years_ago = date(today.year - 10, 1, 1)
    one_year_ago  = today - relativedelta(years=1)
    fourteen_mo   = today - relativedelta(months=14)

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

            # ── Pull full 10-year daily history in one request ───────────────
            hist  = t.history(start=str(ten_years_ago), end=str(today), auto_adjust=True)
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

            # 1-year daily volatility (annualised) — used for risk-adjusted weighting
            vol_1y = None
            if close is not None:
                sl = close[close.index.date >= one_year_ago]
                if len(sl) > 20:
                    daily_rets = sl.pct_change().dropna()
                    vol_1y = float(daily_rets.std() * np.sqrt(252) * 100)  # % annualised

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

            # 14-month monthly indexed data for line chart
            monthly_indexed = []
            hist_mo = t.history(
                start=str(fourteen_mo), end=str(today),
                interval="1mo", auto_adjust=True,
            )
            if not hist_mo.empty and len(hist_mo) > 1:
                base = hist_mo["Close"].iloc[0]
                monthly_indexed = [round(v / base * 100, 2) for v in hist_mo["Close"]]

            result[ticker] = {
                **meta,
                "price":           price,
                "day_chg":         day_chg,
                "ytd_ret":         ytd_ret,
                "one_yr_ret":      one_yr_ret,
                "vol_1y":          vol_1y,
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
            if ytd_ret is not None:  parts.append(f"YTD:{ytd_ret:+.1f}%")
            if one_yr_ret is not None: parts.append(f"1Y:{one_yr_ret:+.1f}%")
            if vol_1y is not None:  parts.append(f"vol:{vol_1y:.1f}%")
            print("  ".join(parts))

        except Exception as exc:
            print(f"ERROR — {exc}")
            result[ticker] = {
                **meta, "price": None, "error": str(exc),
                "annual_returns": {}, "monthly_indexed": [],
                "one_yr_ret": None, "ytd_ret": None, "vol_1y": None,
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
    tickers = list(TICKERS.keys())

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


# ── STEP 4: AI-generated news & insights via Anthropic API ───────────────────

def fetch_ai_news(etf_data, now):
    """
    Calls the Anthropic API with today's live ETF metrics and asks Claude to:
    1. Identify 6 current macro/geopolitical news themes affecting these ETFs
    2. Write a short insight paragraph for each ETF
    Returns: { "news": [...6 cards...], "insights": {ticker: text} }
    Falls back to placeholder text if API key missing or call fails.
    """
    api_key = os.environ["ANTHROPIC_API_KEY"]
    if not api_key:
        print("  ⚠ ANTHROPIC_API_KEY not set — skipping AI news section")
        return _fallback_news(etf_data)

    # Build a concise data summary to send to Claude (includes historical returns)
    summary_lines = [f"Date: {now.strftime('%d %B %Y')} (AEST)\n"]
    for ticker, d in etf_data.items():
        price    = f"${d['price']:.2f}"       if d.get('price')    else 'N/A'
        day      = f"{d['day_chg']:+.2f}%"    if d.get('day_chg')  else 'N/A'
        ytd      = f"{d['ytd_ret']:+.1f}%"    if d.get('ytd_ret')  else 'N/A'
        one_yr   = f"{d['one_yr_ret']:+.1f}%" if d.get('one_yr_ret') else 'N/A'
        frm_high = f"{d['from_high']:+.1f}%"  if d.get('from_high') else 'N/A'
        vol      = f"{d['vol_1y']:.1f}%"      if d.get('vol_1y')   else 'N/A'
        # Include last 5 years of annual returns for historical context
        ann = d.get('annual_returns', {})
        recent_ann = {yr: v for yr, v in ann.items() if int(yr) >= (now.year - 5)}
        ann_str = ", ".join(f"{yr}:{v:+.1f}%" for yr, v in sorted(recent_ann.items()))
        summary_lines.append(
            f"{ticker} ({d['name']}): "
            f"price={price}, day={day}, YTD={ytd}, 1Y={one_yr}, "
            f"from52wHigh={frm_high}, vol(ann)={vol}, "
            f"annualReturns=[{ann_str}]"
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
    "IVV": "2-3 sentence insight connecting today's price action and macro context to IVV specifically (max 70 words)",
    "FANG": "...",
    "VAS": "...",
    "QAU": "...",
    "GOLD": "...",
    "VGS": "..."
  }},
  "verdicts": {{
    "IVV": {{
      "label": "Strong Buy"|"Buy"|"Accumulate"|"Hold"|"Watch",
      "cls": "buy"|"buy"|"accum"|"hold"|"watch",
      "note": "1-2 sentence reasoning grounded in the historical returns, current momentum, volatility, distance from 52W high, and the macro news context above (max 25 words)"
    }},
    "FANG": {{}},
    "VAS": {{}},
    "QAU": {{}},
    "GOLD": {{}},
    "VGS": {{}}
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
        with urllib.request.urlopen(req, timeout=30) as resp:
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

def generate_html(etf_data, portfolios, portfolio_series, now, ai_content=None):
    tickers   = list(TICKERS.keys())
    today     = now.date()
    today_str = now.strftime("%d %B %Y, %I:%M %p AEST")
    start_yr  = today.year - 10

    # Extract AI verdicts early so they're available for cards and verdict rows
    ai_verdicts = (ai_content or {}).get("verdicts", {})

    # Build month labels for the 14-month chart
    month_labels = [
        (datetime(now.year, now.month, 1) - relativedelta(months=i)).strftime("%b %y")
        for i in range(13, -1, -1)
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

    # Price cards
    cards_html = ""
    for ticker in tickers:
        d      = etf_data[ticker]
        ytd    = d.get("ytd_ret");  one_yr = d.get("one_yr_ret");  fh = d.get("from_high")
        dchg   = d.get("day_chg")
        day_s  = f"{arrow(dchg)} {abs(dchg):.2f}% today" if dchg is not None else "— today"
        _, v_cls, v_note = _get_verdict(ticker, d, ai_verdicts)
        # Show 1Y rank badge
        all_1y    = [(t, etf_data[t].get("one_yr_ret") or 0) for t in tickers]
        rank      = sorted(all_1y, key=lambda x: -x[1]).index((ticker, one_yr or 0)) + 1
        rank_html = f'<div class="rank">#{rank} 1Y</div>'
        cards_html += f"""
    <div class="pc {d['cls']}">
      <div style="display:flex;justify-content:space-between;align-items:flex-start">
        <div><div class="pc-ticker">{ticker}</div><div class="pc-name">{d['name']}</div></div>
        {rank_html}
      </div>
      <div class="pc-price">{fp(d.get('price'))}</div>
      <div class="pc-chg {ccls(dchg)}">{day_s}</div>
      <div class="div"></div>
      <div class="pr"><span class="pl">52W HIGH</span><span class="pv">{fp(d.get('w52_high'))}</span></div>
      <div class="pr"><span class="pl">52W LOW</span><span class="pv">{fp(d.get('w52_low'))}</span></div>
      <div class="pr"><span class="pl">FROM HIGH</span><span class="pv {ccls(fh)}">{fpct(fh)}</span></div>
      <div class="pr"><span class="pl">1Y RETURN</span><span class="pv {ccls(one_yr)}">{fpct(one_yr)}</span></div>
      <div class="pr"><span class="pl">YTD</span><span class="pv {ccls(ytd)}">{fpct(ytd)}</span></div>
      <div class="pr"><span class="pl">VOL (1Y)</span><span class="pv">{f"{d['vol_1y']:.1f}%" if d.get('vol_1y') else 'N/A'}</span></div>
      <div class="pr"><span class="pl">MER</span><span class="pv">{fmer(d.get('mer'))}</span></div>
      <div class="pr"><span class="pl">YIELD</span><span class="pv">{fyld(d.get('div_yield'))}</span></div>
      <div class="pr"><span class="pl">AUM</span><span class="pv">{faum(d.get('aum'))}</span></div>
    </div>"""

    # Verdict row
    verdicts_html = ""
    verdict_source = "AI-Powered" if ai_verdicts else "Rules-Based"
    for ticker in tickers:
        d = etf_data[ticker]
        v_label, v_cls, v_note = _get_verdict(ticker, d, ai_verdicts)
        verdicts_html += f"""
    <div class="vc">
      <div class="vt" style="color:{d['color']}">{ticker}</div>
      <div class="vb {v_cls}">{v_label}</div>
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
        while len(vals) < 14: vals = [100.0] + vals
        vals = vals[-14:]
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
        <div style="background:#1a1810;border-radius:3px;padding:14px 16px;border-left:2px solid {ps['color']}">
          <div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;color:#6a6660;margin-bottom:4px">{pname}</div>
          <div style="font-family:'Fraunces',serif;font-size:22px;font-weight:900;color:#f0ece4;letter-spacing:-1px">${ps['final']:,.0f}</div>
          <div style="font-size:10px;color:{ret_col};margin:3px 0">{sign}{ps['total_pct']}% total · {ps['cagr']}% CAGR p.a.</div>
          <div style="font-size:8px;color:#5a5650;line-height:1.5">{ps['desc']}</div>
        </div>"""

    etf_legend  = "".join(f'<div class="li"><div class="ld" style="background:{TICKERS[t]["color"]}"></div>{t}</div>' for t in tickers)
    port_legend = "".join(f'<div class="li"><div class="ld" style="background:{portfolio_series[p]["color"]}"></div><span style="color:#c8c4bc">{p}</span></div>' for p in portfolios)

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

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ASX ETF Dashboard — {now.strftime('%d %b %Y')}</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Fraunces:ital,wght@0,300;0,600;0,900;1,300;1,600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{{--bg:#f4f1e8;--bg2:#ebe8de;--card:#fff;--ink:#1c1916;--ink2:#5a5650;--ink3:#9a9690;--border:#dedad0;
  --ivv:#c8440a;--fang:#1a3a8a;--vas:#1a7a4a;--qau:#b8920a;--gold:#a07820;--vgs:#0a6a8a;}}
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
.cards-grid{{display:grid;grid-template-columns:repeat(6,1fr);gap:11px;margin-bottom:22px;}}
.pc{{background:var(--card);border:1px solid var(--border);border-radius:4px;padding:14px 13px;position:relative;overflow:hidden;}}
.pc::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;}}
.pc.ivv::before{{background:var(--ivv)}}.pc.fang::before{{background:var(--fang)}}.pc.vas::before{{background:var(--vas)}}
.pc.qau::before{{background:var(--qau)}}.pc.gold::before{{background:var(--gold)}}.pc.vgs::before{{background:var(--vgs)}}
.pc-ticker{{font-size:10px;letter-spacing:2px;font-weight:500;}}
.pc.ivv .pc-ticker{{color:var(--ivv)}}.pc.fang .pc-ticker{{color:var(--fang)}}.pc.vas .pc-ticker{{color:var(--vas)}}
.pc.qau .pc-ticker{{color:var(--qau)}}.pc.gold .pc-ticker{{color:var(--gold)}}.pc.vgs .pc-ticker{{color:var(--vgs)}}
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
.dark{{background:var(--ink);border-radius:4px;padding:28px 32px;margin-bottom:20px;}}
.dark .sec-label{{color:#5a5650;}}.dark .sec-label::after{{background:#2a2820;}}
.port-chart-card{{background:#161410;border:1px solid #2a2820;border-radius:4px;padding:22px;margin-top:20px;}}
.port-stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:11px;margin-top:14px;}}
.alloc-note{{font-size:8.5px;color:#6a6660;margin-top:10px;font-style:italic;}}
.news-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:13px;margin-bottom:24px;}}
.news-block{{background:#161410;border-radius:3px;padding:15px 17px;border-left:3px solid #3a3830;}}
.news-block.red{{border-color:#c8440a;}}.news-block.amber{{border-color:#b8920a;}}.news-block.green{{border-color:#1a7a4a;}}
.news-tag{{font-size:8px;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;font-weight:500;}}
.news-block.red .news-tag{{color:#c8440a;}}.news-block.amber .news-tag{{color:#b8920a;}}.news-block.green .news-tag{{color:#4aaa74;}}
.news-head{{font-size:12px;color:#e8e4dc;font-weight:500;margin-bottom:6px;line-height:1.4;}}
.news-body{{font-size:10px;color:#7a7670;line-height:1.65;}}
.news-impact{{font-size:9px;margin-top:8px;padding-top:8px;border-top:1px solid #2a2820;letter-spacing:.5px;}}
.news-block.red .news-impact{{color:#e87050;}}.news-block.amber .news-impact{{color:#d8a830;}}.news-block.green .news-impact{{color:#4aaa74;}}
.ig{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin-bottom:22px;}}
.ins{{border-left:2px solid #3a3830;padding-left:13px;}}
.ins.ivv{{border-color:var(--ivv)}}.ins.fang{{border-color:var(--fang)}}.ins.vas{{border-color:var(--vas)}}
.ins.qau{{border-color:var(--qau)}}.ins.gold{{border-color:var(--gold)}}.ins.vgs{{border-color:var(--vgs)}}
.it{{font-size:9px;letter-spacing:2px;text-transform:uppercase;color:#6a6660;margin-bottom:6px;}}
.ix{{font-size:11px;color:#bfbab2;line-height:1.75;}}
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

  <div class="sec-label">Live Prices — {now.strftime('%d %b %Y')} · Ranked by 1Y Return</div>
  <div class="cards-grid">{cards_html}</div>

  <div class="chart-card">
    <div class="chart-hdr">
      <div><div class="chart-t">14-Month Performance — Indexed to 100</div>
      <div class="chart-s">Monthly closing prices · Fetched live from Yahoo Finance · AUD</div></div>
      <div class="legend">{etf_legend}</div>
    </div>
    <canvas id="etfChart" style="max-height:270px"></canvas>
  </div>

  <div class="tbl-card">
    <div class="sec-label">Annual Returns — Computed Live from Yahoo Finance History</div>
    <div style="overflow-x:auto">
    <table>
      <thead><tr><th>Year</th>{''.join(f'<th><span class="dot" style="background:{TICKERS[t]["color"]}"></span>{t}</th>' for t in tickers)}</tr></thead>
      <tbody>{table_rows}</tbody>
    </table>
    </div>
  </div>

  <div class="sec-label">Verdict — {verdict_source} · Live Yahoo Finance Data + News Context</div>
  <div class="verdict-row">{verdicts_html}</div>

  <div class="dark">
    <div class="sec-label">Global Events Impacting Your Portfolio — {now.strftime('%d %b %Y')}</div>
    <div class="news-grid">{news_cards_html}</div>

    {'<div class="sec-label">ETF Insights — In Context of Global Events</div><div class="ig">' + insights_html + '</div>' if insights_html else ''}

    <div class="sec-label">Auto-Computed Portfolio Allocations — Based on Today's Live 1Y Returns &amp; Volatility</div>

    <div class="tbl-card" style="background:#1a1810;border-color:#2a2820;padding:18px 20px;margin-bottom:0">
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
        Momentum: rank-weighted by 1Y return (top performers get most weight, negative-return ETFs excluded) ·
        Risk-Adjusted: weighted by 1Y return ÷ annualised volatility (Sharpe proxy) ·
        Equal Weight: 1/N benchmark · Recomputed every run from live data.
      </div>
    </div>

    <div class="port-chart-card">
      <div class="chart-hdr">
        <div><div class="chart-t" style="color:#f0ece4">10-Year Portfolio Simulation — $10,000 Invested Jan {start_yr}</div>
        <div class="chart-s" style="color:#6a6660">Annual returns from Yahoo Finance · Allocations are today's auto-computed weights · AUD</div></div>
        <div class="legend">{port_legend}</div>
      </div>
      <canvas id="portChart" style="max-height:310px"></canvas>
      <div class="port-stats">{port_stats_html}</div>
    </div>
  </div>

</div>
<footer>
  <span>Generated {today_str} · Zero hardcoded values · Prices, returns, volatility, allocations all live from Yahoo Finance</span>
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
    plugins:{{legend:{{display:false}},tooltip:{{backgroundColor:'#0e0c0a',titleColor:'#8a8680',bodyColor:'#c8c4bc',borderColor:'#2a2820',borderWidth:1,padding:13,
      callbacks:{{label:c=>{{const v=c.parsed.y,g=((v/{INITIAL}-1)*100).toFixed(1),s=g>=0?'+':'';return`  ${{c.dataset.label}}: $${{Math.round(v).toLocaleString()}} (${{s}}${{g}}%)`;}}}}}}}},
    scales:{{x:{{grid:{{color:'rgba(255,255,255,0.03)'}},ticks:{{color:'#6a6660',maxRotation:0}}}},
      y:{{grid:{{color:'rgba(255,255,255,0.03)'}},ticks:{{color:'#6a6660',callback:v=>'$'+(v/1000).toFixed(0)+'k'}},
        title:{{display:true,text:'Portfolio Value (AUD)',color:'#6a6660',font:{{size:9}}}}}}}}}}
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

    print("📡 Fetching live data from Yahoo Finance...")
    etf_data = fetch_all_data(now)

    print("\n⚖️  Auto-computing portfolio allocations from live 1Y returns & volatility...")
    portfolios = compute_portfolio_allocations(etf_data)
    for pname, pmeta in portfolios.items():
        print(f"  {pname:15s} → {allocation_desc(pmeta['allocs'])}")

    print("\n📐 Running 10-year simulation on live returns...")
    portfolio_series = build_portfolio_series(etf_data, portfolios, now)
    for pname, ps in portfolio_series.items():
        print(f"  {pname:15s} → ${ps['final']:>10,.0f}  ({ps['total_pct']:+.1f}%  CAGR {ps['cagr']:.1f}%)")

    print("\n📰 Fetching AI-generated news & insights (Anthropic API)...")
    ai_content = fetch_ai_news(etf_data, now)

    print("\n🎨 Generating HTML...")
    html = generate_html(etf_data, portfolios, portfolio_series, now, ai_content)

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
