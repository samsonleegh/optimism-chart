# SG Optimism Charts

Optimism-channel charts and BUY / HOLD / SELL signals for SG (SGX) stocks.

## What it does
For each stock, on a **log-price** chart over 10 years it fits a straight
trend channel:

| Line | Meaning |
|------|---------|
| **100%** | best-fit line through the **peaks** (expensive) |
| **0%**   | best-fit line through the **troughs** (cheap) |
| **50%**  | midway — fair value |
| **75% / 25%** | midway of the upper / lower halves |

The latest price's position in the channel = **optimism %**.
`≤25%` → **BUY**, `≥75%` → **SELL**, else **HOLD** (thresholds in `optimism.py`).

Channel lines are found by *iterative envelope regression* (fit OLS → keep points
above/below the line → refit) so peaks/troughs are not hand-picked.

> Axis note: "log-log" was requested, but the trend model is exponential growth,
> which is a straight line on a **log-price axis vs linear time** (semi-log).
> A log time axis would only straighten power-law growth. Change in `make_chart`.

Two market tabs: **🇸🇬 Singapore** and **🇺🇸 United States**.

## Run locally (live server)
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python optimism.py     # one-off: writes ocbc_optimism.png
.venv/bin/python app.py          # dashboard at http://127.0.0.1:5000
```
Split-cadence background worker (in `app.py`):
- **Channels** (10y fit) re-computed **daily** — slow, barely moves intraday.
- **Prices + optimism % + charts** refreshed every **15 min** — a cheap quote per stock.

Tune `PRICE_REFRESH_MINUTES` / `CHANNEL_REFRESH_HOURS` at the top of `app.py`.

## Deploy free on GitHub Pages (static)
Pages can't run Flask, so a scheduled Action regenerates a **static snapshot**
(`build_static.py` → `site/`) and publishes it. Steps:
1. Push this repo to GitHub.
2. Repo **Settings → Pages → Source: GitHub Actions**.
3. `.github/workflows/pages.yml` then builds on every push and every ~30 min.

Build the static site locally too: `.venv/bin/python build_static.py` → open `site/index.html`.

> Scheduled Actions use a cron but are best-effort and often delayed under GitHub
> load — treat "every 30 min" as approximate, not a guaranteed live feed. For a
> truly live 15-min dashboard, run `app.py` on an always-on host (Render/Railway/Fly).

## Files
- `optimism.py`     — engine: fit channel (`fit_channel`), evaluate price (`evaluate`), chart PNG (`make_chart`).
- `tickers.py`      — `MARKETS` = SG + US universes. Add `(ticker, name)` pairs to widen.
- `app.py`          — live Flask dashboard, tabs, split-cadence scheduler.
- `build_static.py` — static-site generator for GitHub Pages.
- `.github/workflows/pages.yml` — scheduled build + deploy to Pages.

## Caveats
- Data is Yahoo Finance (`.SI`), end-of-period and delayed — not a live feed.
- Prices are **nominal** (not dividend-adjusted), so a 10y channel reflects the
  price chart you'd recognise, not total return.
- This is a valuation heuristic, **not investment advice**.
