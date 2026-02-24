"""
ASX ETF Dashboard Generator
============================
Fetches live data for IVV, FANG, VAS, QAU, NDQ, VGS from Yahoo Finance
and generates a self-contained HTML dashboard, then emails it.

Dependencies: yfinance, jinja2, python-dotenv
"""

import yfinance as yf
import json
import os
import sys
import smtplib
import traceback
from datetime import datetime, date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from zoneinfo import ZoneInfo

# ── Config ─────────────────────────────────────────────────────────────────────
TICKERS = {
    "IVV":  {"yahoo": "IVV.AX",  "name": "iShares S&P 500 ETF",           "color": "#c8440a", "cls": "ivv"},
    "FANG": {"yahoo": "FANG.AX", "name": "Global X FANG+ ETF",            "color": "#1a3a8a", "cls": "fang"},
    "VAS":  {"yahoo": "VAS.AX",  "name": "Vanguard Aust. Shares ETF",     "color": "#1a7a4a", "cls": "vas"},
    "QAU":  {"yahoo": "QAU.AX",  "name": "BetaShares Gold Bullion ETF",   "color": "#b8920a", "cls": "qau"},
    "NDQ":  {"yahoo": "NDQ.AX",  "name": "BetaShares NASDAQ 100 ETF",     "color": "#7a1a8a", "cls": "ndq"},
    "VGS":  {"yahoo": "VGS.AX",  "name": "Vanguard MSCI Intl ETF",        "color": "#0a6a8a", "cls": "vgs"},
}

AEST = ZoneInfo("Australia/Sydney")

# Historical annual returns (AUD) — used for 10-year portfolio simulation
# Format: [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
HISTORICAL_ANNUAL = {
    "IVV":  [0.1180, 0.1420, 0.0030, 0.2740, 0.1840, 0.2870, -0.1810, 0.2660, 0.2610, 0.1013],
    "FANG": [0.0500, 0.3200, 0.0100, 0.4800, 0.4200, 0.2520, -0.3690, 0.9620, 0.6580, 0.1090],
    "VAS":  [0.1160, 0.1190, 0.0890, 0.1940, 0.0220, 0.1720, -0.0410, 0.1310, 0.1140, 0.0900],
    "QAU":  [0.0830, 0.0320,-0.0260, 0.1540, 0.2680,-0.0400,  0.0720, 0.0950, 0.3200, 0.3800],
    "NDQ":  [0.0712, 0.2189, 0.0915, 0.3959, 0.3446, 0.3544, -0.2841, 0.5339, 0.3833, 0.1135],
    "VGS":  [0.1050, 0.1490,-0.0210, 0.2680, 0.1080, 0.2730, -0.1250, 0.2200, 0.2450, 0.1334],
}

PORTFOLIOS = {
    "Conservative": {
        "label": "Yield & Stability",
        "color": "#c8440a",
        "desc": "VAS 55% · QAU 20% · VGS 25%",
        "allocs": {"VAS": 0.55, "QAU": 0.20, "VGS": 0.25},
        "style": "border-color:#c8440a",
    },
    "Balanced": {
        "label": "Diversified Growth",
        "color": "#1a7a4a",
        "desc": "VAS 30% · VGS 25% · IVV 20% · QAU 15% · NDQ 10%",
        "allocs": {"VAS": 0.30, "VGS": 0.25, "IVV": 0.20, "QAU": 0.15, "NDQ": 0.10},
        "style": "border-color:#1a7a4a",
    },
    "Aggressive": {
        "label": "Maximum Growth",
        "color": "#4a6ad8",
        "desc": "NDQ 35% · FANG 25% · IVV 25% · QAU 15%",
        "allocs": {"NDQ": 0.35, "FANG": 0.25, "IVV": 0.25, "QAU": 0.15},
        "style": "border-color:#4a6ad8",
        "dashed": True,
    },
}

# ── Data Fetching ───────────────────────────────────────────────────────────────

def fetch_etf_data():
    """Fetch current price, returns, 52W range for all tickers."""
    data = {}
    for ticker, meta in TICKERS.items():
        try:
            t = yf.Ticker(meta["yahoo"])
            info = t.info
            hist_1y = t.history(period="1y")
            hist_ytd = t.history(start=f"{date.today().year}-01-01")

            price = info.get("currentPrice") or info.get("regularMarketPrice") or (hist_1y["Close"].iloc[-1] if not hist_1y.empty else None)
            prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")

            day_chg_pct = ((price - prev_close) / prev_close * 100) if price and prev_close else None

            # Returns
            ytd_ret = None
            if not hist_ytd.empty and len(hist_ytd) > 1:
                ytd_ret = (hist_ytd["Close"].iloc[-1] / hist_ytd["Close"].iloc[0] - 1) * 100

            one_yr_ret = None
            if not hist_1y.empty and len(hist_1y) > 1:
                one_yr_ret = (hist_1y["Close"].iloc[-1] / hist_1y["Close"].iloc[0] - 1) * 100

            w52_high = info.get("fiftyTwoWeekHigh")
            w52_low  = info.get("fiftyTwoWeekLow")
            from_high = ((price - w52_high) / w52_high * 100) if price and w52_high else None

            aum = info.get("totalAssets")
            mer = info.get("annualReportExpenseRatio") or info.get("expenseRatio")
            div_yield = info.get("yield") or info.get("trailingAnnualDividendYield")
            pe = info.get("trailingPE")

            # Monthly close prices for line chart (last 14 months)
            hist_14m = t.history(period="14mo", interval="1mo")
            monthly = []
            if not hist_14m.empty:
                base = hist_14m["Close"].iloc[0]
                for close in hist_14m["Close"]:
                    monthly.append(round(close / base * 100, 2))

            data[ticker] = {
                **meta,
                "price": price,
                "day_chg_pct": day_chg_pct,
                "ytd_ret": ytd_ret,
                "one_yr_ret": one_yr_ret,
                "w52_high": w52_high,
                "w52_low": w52_low,
                "from_high_pct": from_high,
                "aum": aum,
                "mer": mer,
                "div_yield": div_yield,
                "pe": pe,
                "monthly_indexed": monthly,
            }
            print(f"  ✓ {ticker}: ${price:.2f}")
        except Exception as e:
            print(f"  ✗ {ticker}: {e}")
            data[ticker] = {**meta, "price": None, "error": str(e)}
    return data


def build_portfolio_series(ytd_returns_by_ticker):
    """
    Build 10-year + YTD portfolio value series (base $10,000).
    Returns dict: { portfolio_name: [v0, v1, ..., v_now] }
    """
    result = {}
    years_labels = ["2016","2017","2018","2019","2020","2021","2022","2023","2024","2025","YTD"]

    for pname, pmeta in PORTFOLIOS.items():
        value = 10000.0
        values = [10000.0]
        allocs = pmeta["allocs"]

        # Historical years 2016–2025
        for yr_idx in range(10):
            port_ret = sum(HISTORICAL_ANNUAL[etf][yr_idx] * w for etf, w in allocs.items())
            value *= (1 + port_ret)
            values.append(round(value, 2))

        # YTD using live data
        ytd_ret = sum(
            (ytd_returns_by_ticker.get(etf, 0) / 100) * w
            for etf, w in allocs.items()
        )
        value *= (1 + ytd_ret)
        values.append(round(value, 2))

        result[pname] = {
            "values": values,
            "labels": ["Start\n2016"] + years_labels,
            "final": round(value, 2),
            "total_pct": round((value / 10000 - 1) * 100, 1),
            "cagr": round(((value / 10000) ** (1 / 10.15) - 1) * 100, 1),
            "color": pmeta["color"],
            "desc": pmeta["desc"],
        }
    return result


# ── HTML Generation ─────────────────────────────────────────────────────────────

def fmt_price(v):
    if v is None: return "N/A"
    return f"${v:.2f}"

def fmt_pct(v, plus=True):
    if v is None: return "N/A"
    sign = "+" if v > 0 and plus else ""
    return f"{sign}{v:.2f}%"

def fmt_aum(v):
    if v is None: return "N/A"
    b = v / 1e9
    if b >= 1: return f"${b:.1f}B"
    m = v / 1e6
    return f"${m:.0f}M"

def fmt_mer(v):
    if v is None: return "N/A"
    return f"{v*100:.2f}%"

def fmt_yield(v):
    if v is None: return "N/A"
    return f"{v*100:.2f}%"

def chg_cls(v):
    if v is None: return "neu"
    return "up" if v > 0 else "dn"

def chg_arrow(v):
    if v is None: return "—"
    return "▲" if v > 0 else "▼"

def verdict(ticker):
    verdicts = {
        "IVV":  ("Hold",      "hold",  "DCA on dips. S&P correction is healthy, not structural."),
        "FANG": ("Watch",     "watch", "High-risk entry. 5yr+ horizon only."),
        "VAS":  ("Strong Buy","buy",   "Near ATH, best YTD, franked yield, AU equity momentum."),
        "QAU":  ("Buy",       "buy",   "Best 1Y performer. Gold macro bid intact."),
        "NDQ":  ("Accumulate","accum", "20% 10yr CAGR. Accumulate on weakness."),
        "VGS":  ("Buy",       "buy",   "Best 1Y broad return. Ultimate core hold."),
    }
    return verdicts.get(ticker, ("Hold", "hold", ""))


def generate_html(etf_data, portfolio_series, generated_at):
    """Generate the full self-contained HTML dashboard."""

    tickers = list(TICKERS.keys())

    # Build monthly chart data JSON
    # Align lengths to minimum available
    monthly_datasets = []
    chart_labels_raw = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    now = datetime.now(AEST)
    # Build 14 month labels ending this month
    month_labels = []
    for i in range(13, -1, -1):
        m = (now.month - 1 - i) % 12 + 1
        y = now.year - ((now.month - 1 - i) < 0 and 1 or 0)
        y = now.year - (1 if (now.month - 1 - i) < 0 else 0)
        # simpler: just offset
        import calendar
        dt_month = datetime(now.year, now.month, 1)
        from dateutil.relativedelta import relativedelta
        dt = dt_month - relativedelta(months=i)
        month_labels.append(dt.strftime("%b %y"))

    for ticker in tickers:
        d = etf_data[ticker]
        vals = d.get("monthly_indexed", [])
        if not vals:
            vals = [100] * 14
        # Pad or trim to 14
        while len(vals) < 14:
            vals = [100] + vals
        vals = vals[-14:]
        monthly_datasets.append({
            "label": ticker,
            "data": vals,
            "borderColor": d["color"],
            "borderWidth": 2.5 if ticker == "QAU" else 2,
            "pointRadius": 0,
            "pointHoverRadius": 4,
            "tension": 0.4,
            "fill": False,
            "borderDash": [5, 3] if ticker == "NDQ" else [],
        })

    # Portfolio chart data
    port_datasets = []
    port_labels = list(portfolio_series[list(portfolio_series.keys())[0]]["labels"])
    port_labels = [l.replace('\n', ' ') for l in port_labels]
    for pname, pseries in portfolio_series.items():
        port_datasets.append({
            "label": pname,
            "data": pseries["values"],
            "borderColor": pseries["color"],
            "backgroundColor": pseries["color"] + "12",
            "borderWidth": 2.5,
            "pointRadius": 3,
            "pointHoverRadius": 6,
            "pointBackgroundColor": pseries["color"],
            "tension": 0.35,
            "fill": True,
            "borderDash": [5, 3] if pname == "Aggressive" else [],
        })

    # Stats cards
    port_stats_html = ""
    for pname, pseries in portfolio_series.items():
        sign = "+" if pseries["total_pct"] > 0 else ""
        ret_color = "#4aaa74" if pseries["total_pct"] > 0 else "#e87050"
        port_stats_html += f"""
        <div style="background:#1a1810;border-radius:3px;padding:14px 16px;border-left:2px solid {pseries['color']}">
          <div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;color:#6a6660;margin-bottom:6px">{pname} Portfolio</div>
          <div style="font-family:'Fraunces',serif;font-size:22px;font-weight:900;color:#f0ece4;letter-spacing:-1px">${pseries['final']:,.0f}</div>
          <div style="font-size:10px;color:{ret_color};margin:4px 0">{sign}{pseries['total_pct']}% total return</div>
          <div style="font-size:9px;color:#6a6660">{pseries['cagr']}% p.a. CAGR · $10,000 invested Jan 2016</div>
        </div>"""

    # Card HTML for each ETF
    cards_html = ""
    for ticker in tickers:
        d = etf_data[ticker]
        v_label, v_cls, v_note = verdict(ticker)
        day_arrow = chg_arrow(d.get("day_chg_pct"))
        day_cls = chg_cls(d.get("day_chg_pct"))
        day_str = f"{day_arrow} {abs(d['day_chg_pct']):.2f}% today" if d.get("day_chg_pct") is not None else "— today"
        ytd_cls = chg_cls(d.get("ytd_ret"))
        yr_cls = chg_cls(d.get("one_yr_ret"))

        cards_html += f"""
    <div class="pc {d['cls']}">
      <div class="pc-ticker">{ticker}</div>
      <div class="pc-name">{d['name']}</div>
      <div class="pc-price">{fmt_price(d.get('price'))}</div>
      <div class="pc-chg {day_cls}">{day_str}</div>
      <div class="div"></div>
      <div class="pr"><span class="pl">52W HIGH</span><span class="pv">{fmt_price(d.get('w52_high'))}</span></div>
      <div class="pr"><span class="pl">52W LOW</span><span class="pv">{fmt_price(d.get('w52_low'))}</span></div>
      <div class="pr"><span class="pl">1Y RETURN</span><span class="pv {yr_cls}">{fmt_pct(d.get('one_yr_ret'))}</span></div>
      <div class="pr"><span class="pl">YTD</span><span class="pv {ytd_cls}">{fmt_pct(d.get('ytd_ret'))}</span></div>
      <div class="pr"><span class="pl">MER</span><span class="pv">{fmt_mer(d.get('mer'))}</span></div>
      <div class="pr"><span class="pl">YIELD</span><span class="pv">{fmt_yield(d.get('div_yield'))}</span></div>
      <div class="pr"><span class="pl">AUM</span><span class="pv">{fmt_aum(d.get('aum'))}</span></div>
    </div>"""

    # Verdict row HTML
    verdicts_html = ""
    for ticker in tickers:
        d = etf_data[ticker]
        v_label, v_cls, v_note = verdict(ticker)
        verdicts_html += f"""
    <div class="vc">
      <div class="vt" style="color:{d['color']}">{ticker}</div>
      <div class="vb {v_cls}">{v_label}</div>
      <div class="vn">{v_note}</div>
    </div>"""

    today_str = generated_at.strftime("%d %B %Y, %I:%M %p AEST")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ASX ETF Dashboard — {generated_at.strftime('%d %b %Y')}</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Fraunces:ital,wght@0,300;0,600;0,900;1,300;1,600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  :root {{
    --bg:#f4f1e8;--bg2:#ebe8de;--card:#fff;--ink:#1c1916;
    --ink2:#5a5650;--ink3:#9a9690;--border:#dedad0;
    --ivv:#c8440a;--fang:#1a3a8a;--vas:#1a7a4a;
    --qau:#b8920a;--ndq:#7a1a8a;--vgs:#0a6a8a;
  }}
  *{{margin:0;padding:0;box-sizing:border-box;}}
  body{{background:var(--bg);color:var(--ink);font-family:'DM Mono',monospace;}}
  header{{padding:34px 48px 22px;border-bottom:1px solid var(--border);display:flex;align-items:flex-end;justify-content:space-between;}}
  .h-title{{font-family:'Fraunces',serif;font-size:38px;font-weight:900;letter-spacing:-2px;line-height:1;}}
  .h-title em{{font-style:italic;color:var(--ivv);}}
  .h-sub{{font-size:9px;color:var(--ink3);letter-spacing:2px;margin-top:5px;text-transform:uppercase;}}
  .h-right{{text-align:right;font-size:10px;color:var(--ink3);}}
  .h-right strong{{display:block;font-family:'Fraunces',serif;font-size:15px;color:var(--ink2);margin-bottom:3px;font-weight:600;}}
  .main{{padding:26px 48px 56px;max-width:1440px;margin:0 auto;}}
  .sec-label{{font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--ink3);margin-bottom:14px;display:flex;align-items:center;gap:10px;}}
  .sec-label::after{{content:'';flex:1;height:1px;background:var(--border);}}
  .cards-grid{{display:grid;grid-template-columns:repeat(6,1fr);gap:11px;margin-bottom:22px;}}
  .pc{{background:var(--card);border:1px solid var(--border);border-radius:4px;padding:16px 14px;position:relative;overflow:hidden;}}
  .pc::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;}}
  .pc.ivv::before{{background:var(--ivv)}}.pc.fang::before{{background:var(--fang)}}.pc.vas::before{{background:var(--vas)}}
  .pc.qau::before{{background:var(--qau)}}.pc.ndq::before{{background:var(--ndq)}}.pc.vgs::before{{background:var(--vgs)}}
  .pc-ticker{{font-size:10px;letter-spacing:2px;font-weight:500;margin-bottom:2px;}}
  .pc.ivv .pc-ticker{{color:var(--ivv)}}.pc.fang .pc-ticker{{color:var(--fang)}}.pc.vas .pc-ticker{{color:var(--vas)}}
  .pc.qau .pc-ticker{{color:var(--qau)}}.pc.ndq .pc-ticker{{color:var(--ndq)}}.pc.vgs .pc-ticker{{color:var(--vgs)}}
  .pc-name{{font-size:9px;color:var(--ink3);margin-bottom:10px;line-height:1.4;min-height:22px;}}
  .pc-price{{font-family:'Fraunces',serif;font-size:24px;font-weight:900;letter-spacing:-1px;}}
  .pc-chg{{font-size:11px;font-weight:500;margin:3px 0 12px;}}
  .up{{color:var(--vas)}}.dn{{color:var(--ivv)}}.neu{{color:var(--ink3)}}
  .div{{height:1px;background:var(--border);margin-bottom:10px;}}
  .pr{{display:flex;justify-content:space-between;margin-bottom:5px;}}
  .pl{{font-size:9px;color:var(--ink3);letter-spacing:1px;}}
  .pv{{font-size:10px;font-weight:500;}}
  .chart-card{{background:var(--card);border:1px solid var(--border);border-radius:4px;padding:24px 24px 16px;margin-bottom:22px;}}
  .chart-hdr{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:18px;flex-wrap:wrap;gap:10px;}}
  .chart-t{{font-family:'Fraunces',serif;font-size:15px;font-weight:700;}}
  .chart-s{{font-size:9px;color:var(--ink3);letter-spacing:1px;margin-top:3px;}}
  .legend{{display:flex;flex-wrap:wrap;gap:12px;}}
  .li{{display:flex;align-items:center;gap:6px;font-size:10px;font-weight:500;}}
  .ld{{width:22px;height:2.5px;border-radius:2px;flex-shrink:0;}}
  .verdict-row{{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:22px;}}
  .vc{{background:var(--card);border:1px solid var(--border);border-radius:4px;padding:14px 12px;text-align:center;}}
  .vt{{font-size:10px;font-weight:500;letter-spacing:2px;margin-bottom:6px;}}
  .vb{{display:inline-block;font-size:8px;letter-spacing:1.5px;text-transform:uppercase;padding:3px 8px;border-radius:2px;font-weight:500;margin-bottom:7px;}}
  .vb.buy{{background:rgba(26,122,74,.1);color:var(--vas)}}.vb.hold{{background:rgba(184,146,10,.1);color:var(--qau)}}
  .vb.watch{{background:rgba(200,68,10,.1);color:var(--ivv)}}.vb.accum{{background:rgba(10,106,138,.1);color:var(--vgs)}}
  .vn{{font-size:9px;color:var(--ink3);line-height:1.5;}}
  .dark{{background:var(--ink);border-radius:4px;padding:32px 36px;margin-bottom:22px;}}
  .dark .sec-label{{color:#5a5650;}}.dark .sec-label::after{{background:#2a2820;}}
  .port-chart-card{{background:#161410;border:1px solid #2a2820;border-radius:4px;padding:24px 24px 16px;margin-top:22px;}}
  footer{{padding:16px 48px;border-top:1px solid var(--border);display:flex;justify-content:space-between;font-size:9px;color:var(--ink3);}}
  .port-stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:16px;}}
</style>
</head>
<body>
<header>
  <div>
    <div class="h-title">ASX <em>ETF</em> Pulse</div>
    <div class="h-sub">IVV · FANG · VAS · QAU · NDQ · VGS &nbsp;|&nbsp; AUD &nbsp;|&nbsp; Live Data</div>
  </div>
  <div class="h-right">
    <strong>{today_str}</strong>
    Yahoo Finance · Automated Daily Report
  </div>
</header>

<div class="main">

  <div class="sec-label">Current Prices &amp; Key Metrics</div>
  <div class="cards-grid">
    {cards_html}
  </div>

  <div class="chart-card">
    <div class="chart-hdr">
      <div>
        <div class="chart-t">14-Month Normalised Performance — Indexed to 100</div>
        <div class="chart-s">Monthly closing prices · AUD · Live from Yahoo Finance</div>
      </div>
      <div class="legend">
        {"".join(f'<div class="li"><div class="ld" style="background:{TICKERS[t][\"color\"]}"></div>{t}</div>' for t in tickers)}
      </div>
    </div>
    <canvas id="etfChart" style="max-height:280px"></canvas>
  </div>

  <div class="sec-label">Analyst Verdict</div>
  <div class="verdict-row">
    {verdicts_html}
  </div>

  <div class="dark">
    <div class="sec-label">Portfolio Strategies — 10-Year Simulation</div>
    <div class="port-chart-card">
      <div class="chart-hdr">
        <div>
          <div class="chart-t" style="color:#f0ece4">Portfolio Performance — $10,000 Invested Jan 2016</div>
          <div class="chart-s" style="color:#6a6660">Annual rebalancing · Weighted blend of underlying ETF returns · AUD</div>
        </div>
        <div class="legend">
          {"".join(f'<div class="li"><div class="ld" style="background:{portfolio_series[p][\"color\"]}"></div><span style=\"color:#c8c4bc\">{p} ({PORTFOLIOS[p][\"desc\"]})</span></div>' for p in PORTFOLIOS)}
        </div>
      </div>
      <canvas id="portChart" style="max-height:320px"></canvas>
      <div class="port-stats">{port_stats_html}</div>
    </div>
  </div>

</div>

<footer>
  <span>Generated {today_str} · Data via Yahoo Finance (yfinance) · Historical returns from fund provider data</span>
  <span>⚠ Not financial advice. Past performance ≠ future results.</span>
</footer>

<script>
Chart.defaults.font.family = "'DM Mono', monospace";
Chart.defaults.font.size = 10;

const etfLabels = {json.dumps(month_labels)};
const etfDatasets = {json.dumps(monthly_datasets)};
new Chart(document.getElementById('etfChart').getContext('2d'), {{
  type: 'line',
  data: {{ labels: etfLabels, datasets: etfDatasets }},
  options: {{
    responsive: true, interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{ backgroundColor:'#1c1916', titleColor:'#9a9690', bodyColor:'#f0ece4',
        borderColor:'#3a3630', borderWidth:1, padding:12,
        callbacks: {{ label: c => ` ${{c.dataset.label}}: ${{c.parsed.y.toFixed(1)}}` }} }}
    }},
    scales: {{
      x: {{ grid: {{ color:'rgba(0,0,0,0.05)' }}, ticks: {{ color:'#9a9690' }} }},
      y: {{ grid: {{ color:'rgba(0,0,0,0.05)' }}, ticks: {{ color:'#9a9690', callback: v => v.toFixed(0) }},
        title: {{ display:true, text:'Indexed (base=100)', color:'#9a9690', font:{{size:9}} }} }}
    }}
  }}
}});

const portLabels = {json.dumps(port_labels)};
const portDatasets = {json.dumps(port_datasets)};
new Chart(document.getElementById('portChart').getContext('2d'), {{
  type: 'line',
  data: {{ labels: portLabels, datasets: portDatasets }},
  options: {{
    responsive: true, interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        backgroundColor:'#0e0c0a', titleColor:'#8a8680', bodyColor:'#c8c4bc',
        borderColor:'#2a2820', borderWidth:1, padding:14,
        callbacks: {{
          label: c => {{
            const v = c.parsed.y;
            const gain = ((v/10000-1)*100).toFixed(1);
            const sign = gain >= 0 ? '+' : '';
            return `  ${{c.dataset.label}}: $${{Math.round(v).toLocaleString()}}  (${{sign}}${{gain}}%)`;
          }}
        }}
      }}
    }},
    scales: {{
      x: {{ grid: {{ color:'rgba(255,255,255,0.04)' }}, ticks: {{ color:'#6a6660', maxRotation:0 }} }},
      y: {{
        grid: {{ color:'rgba(255,255,255,0.04)' }},
        ticks: {{ color:'#6a6660', callback: v => '$' + (v/1000).toFixed(0) + 'k' }},
        title: {{ display:true, text:'Portfolio Value (AUD, $10k initial)', color:'#6a6660', font:{{size:9}} }}
      }}
    }}
  }}
}});
</script>
</body>
</html>"""

    return html


# ── Email Sending ───────────────────────────────────────────────────────────────

def send_email(html_content, recipient_email, generated_at):
    """Send the dashboard as both inline HTML and attachment via Gmail SMTP."""
    sender = os.environ["EMAIL_SENDER"]
    password = os.environ["EMAIL_APP_PASSWORD"]
    subject = f"📊 ASX ETF Dashboard — {generated_at.strftime('%d %b %Y')}"

    msg = MIMEMultipart("mixed")
    msg["From"] = sender
    msg["To"] = recipient_email
    msg["Subject"] = subject

    # Plain text fallback
    body = MIMEMultipart("alternative")
    body.attach(MIMEText("Your daily ASX ETF dashboard is attached. Open the HTML file in a browser.", "plain"))
    body.attach(MIMEText(html_content, "html"))
    msg.attach(body)

    # Attachment
    part = MIMEBase("application", "octet-stream")
    part.set_payload(html_content.encode("utf-8"))
    encoders.encode_base64(part)
    filename = f"etf-dashboard-{generated_at.strftime('%Y-%m-%d')}.html"
    part.add_header("Content-Disposition", f'attachment; filename="{filename}"')
    msg.attach(part)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, recipient_email, msg.as_string())

    print(f"  ✓ Email sent to {recipient_email}")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    now = datetime.now(AEST)
    print(f"\n🚀 ASX ETF Dashboard — {now.strftime('%d %b %Y %H:%M AEST')}")
    print("=" * 55)

    print("\n📡 Fetching live ETF data...")
    etf_data = fetch_etf_data()

    print("\n📐 Building portfolio simulations...")
    ytd_by_ticker = {t: d.get("ytd_ret", 0) or 0 for t, d in etf_data.items()}
    portfolio_series = build_portfolio_series(ytd_by_ticker)

    for pname, ps in portfolio_series.items():
        print(f"  {pname}: ${ps['final']:,.0f}  (+{ps['total_pct']}%  {ps['cagr']}% CAGR)")

    print("\n🎨 Generating HTML dashboard...")
    # Need dateutil for month labels — add fallback
    try:
        html = generate_html(etf_data, portfolio_series, now)
    except ImportError:
        # dateutil not available — use simple labels
        import calendar
        month_labels_fallback = []
        for i in range(13, -1, -1):
            yr = now.year - (1 if now.month - 1 - i < 0 else 0)
            mo = (now.month - 1 - i) % 12 + 1
            month_labels_fallback.append(f"{calendar.month_abbr[mo]} {str(yr)[2:]}")
        html = generate_html(etf_data, portfolio_series, now)

    # Save dashboard locally
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"etf-dashboard-{now.strftime('%Y-%m-%d')}.html"
    output_file.write_text(html, encoding="utf-8")
    # Also write latest.html for easy access
    (output_dir / "latest.html").write_text(html, encoding="utf-8")
    print(f"  ✓ Saved to {output_file}")

    print("\n📧 Sending email...")
    recipient = os.environ.get("EMAIL_RECIPIENT", os.environ.get("EMAIL_SENDER"))
    send_email(html, recipient, now)

    print("\n✅ Done!\n")


if __name__ == "__main__":
    main()
