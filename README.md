# 📊 ASX ETF Daily Dashboard — GitHub Actions Automation

Automatically generates and emails a live ASX ETF dashboard every weekday at **8:00 AM AEST**.

Covers: **IVV · FANG · VAS · QAU · NDQ · VGS**

---

## What it does

Every weekday morning the workflow:
1. Fetches **live prices** for all 6 ETFs from Yahoo Finance (free, no API key needed)
2. Generates a **self-contained HTML dashboard** with charts, performance data, and 10-year portfolio simulation
3. **Emails it to you** as both inline HTML and a downloadable attachment
4. Saves it as a **GitHub Actions artifact** (viewable in the Actions tab for 30 days)

---

## Setup — 5 Steps

### Step 1 — Fork or create the repo

Create a new **private** GitHub repository and push these files to it:

```
your-repo/
├── .github/
│   └── workflows/
│       └── daily-dashboard.yml
├── scripts/
│   └── generate_dashboard.py
├── requirements.txt
└── README.md
```

### Step 2 — Enable a Gmail App Password

The script sends email via Gmail SMTP. You need an **App Password** (not your regular Gmail password).

1. Go to your Google Account → **Security**
2. Enable **2-Step Verification** if not already on
3. Go to **Security → App Passwords**
4. Create a new app password — name it "ETF Dashboard"
5. Copy the 16-character password (e.g. `abcd efgh ijkl mnop`)

> ⚠️ Use a dedicated Gmail account for sending, not your personal one.

### Step 3 — Add GitHub Secrets

In your GitHub repo, go to **Settings → Secrets and variables → Actions → New repository secret** and add:

| Secret Name         | Value                                      |
|---------------------|--------------------------------------------|
| `EMAIL_SENDER`      | Your Gmail address (e.g. `you@gmail.com`)  |
| `EMAIL_APP_PASSWORD`| The 16-char App Password from Step 2       |
| `EMAIL_RECIPIENT`   | Where to send the dashboard (can be same)  |

### Step 4 — Push and test

1. Push all files to your repo
2. Go to **Actions** tab in GitHub
3. Click **"Daily ASX ETF Dashboard"** workflow
4. Click **"Run workflow"** → **"Run workflow"** (manual trigger)
5. Watch it run — check your inbox in ~60 seconds ✉️

### Step 5 — Confirm the schedule

The workflow runs automatically at these times:

| AEST (Sydney)      | UTC                  |
|--------------------|----------------------|
| Mon–Fri 8:00 AM    | Sun–Thu 10:00 PM     |

To change the time, edit the `cron` line in `.github/workflows/daily-dashboard.yml`:
```yaml
- cron: "0 22 * * 0-4"   # 10pm UTC Sun-Thu = 8am AEST Mon-Fri
```

Use [crontab.guru](https://crontab.guru) to calculate cron times.

---

## Customisation

### Change ETFs
Edit `TICKERS` in `scripts/generate_dashboard.py`. Any Yahoo Finance ticker works.
```python
TICKERS = {
    "IVV": {"yahoo": "IVV.AX", "name": "...", "color": "#c8440a", "cls": "ivv"},
    ...
}
```

### Change portfolio allocations
Edit `PORTFOLIOS` in the same file:
```python
PORTFOLIOS = {
    "Conservative": {
        "allocs": {"VAS": 0.55, "QAU": 0.20, "VGS": 0.25},
        ...
    },
}
```

### Change email send time
Modify the cron schedule in `.github/workflows/daily-dashboard.yml`.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Email not arriving | Check spam folder. Verify App Password is correct in Secrets. |
| Workflow fails | Check the Actions log. yfinance sometimes has brief outages — re-run manually. |
| Prices show N/A | Yahoo Finance rate-limits occasionally. The script will retry on next run. |
| Wrong timezone | The cron runs in UTC. Adjust the cron expression for your timezone. |

---

## Cost

**Completely free.** GitHub Actions provides 2,000 minutes/month on free accounts. This workflow uses ~2 minutes per run × 5 days/week × 4 weeks = ~40 minutes/month.

---

## Files

```
.github/workflows/daily-dashboard.yml   — Schedule & Actions config
scripts/generate_dashboard.py           — Main Python script
requirements.txt                        — Python dependencies
output/                                 — Generated dashboards (auto-created)
```
