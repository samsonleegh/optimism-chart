"""
Optimism Chart dashboard for SG stocks.

A background worker recomputes the optimism channel for every stock in the
universe every REFRESH_MINUTES (default 15) and caches the result + chart PNG.
The web UI shows a sortable table of BUY / HOLD / SELL calls and a chart page
per stock.

Run:  .venv/bin/python app.py     then open http://127.0.0.1:5000
"""

from __future__ import annotations

import threading
import time
import traceback
from datetime import datetime, timezone

from flask import Flask, Response, render_template_string, abort, request

import optimism
from tickers import MARKETS, ALL_STOCKS

PRICE_REFRESH_MINUTES = 15     # latest price + optimism % + chart
CHANNEL_REFRESH_HOURS = 24     # re-fit the 10y log-linear channel (slow-moving)

app = Flask(__name__)

# ---- shared cache (written by worker thread, read by request handlers) ----
# ticker -> {"name", "channel": Channel|None, "result", "png": bytes|None, "error"}
_lock = threading.Lock()
_cache: dict[str, dict] = {}
_last_channel: datetime | None = None
_last_price: datetime | None = None
_running = False


def _refresh_channels() -> None:
    """Slow path: download 10y history and re-fit every channel. Runs daily."""
    global _last_channel
    for market, ticker, name in ALL_STOCKS:
        try:
            channel = optimism.fit_channel(ticker, name=name)
            with _lock:
                e = _cache.setdefault(ticker, {"name": name, "market": market,
                                               "result": None, "png": None})
                e["channel"] = channel
                e["error"] = None
        except Exception as exc:
            with _lock:
                e = _cache.setdefault(ticker, {"name": name, "market": market,
                                               "result": None, "png": None})
                e["channel"] = None
                e["error"] = str(exc)
            traceback.print_exc()
        time.sleep(0.4)  # be gentle with the data source
    _last_channel = datetime.now(timezone.utc)
    print(f"[{_last_channel:%H:%M:%S} UTC] re-fit {len(ALL_STOCKS)} channels")


def _refresh_prices() -> None:
    """Fast path: pull the latest quote, re-evaluate + re-render. Runs every 15 min."""
    global _last_price
    for market, ticker, name in ALL_STOCKS:
        with _lock:
            e = _cache.get(ticker)
        if not e or not e.get("channel"):
            continue  # channel not fitted yet (or errored)
        try:
            price, date = optimism.latest_quote(ticker)
            result = optimism.evaluate(e["channel"], price, date)
            png = optimism.make_chart(result, e["channel"])
            with _lock:
                e["result"], e["png"], e["error"] = result, png, None
        except Exception as exc:
            with _lock:
                e["error"] = str(exc)
            traceback.print_exc()
        time.sleep(0.25)
    _last_price = datetime.now(timezone.utc)
    print(f"[{_last_price:%H:%M:%S} UTC] refreshed prices")


def _worker() -> None:
    cycles_per_channel = max(1, int(CHANNEL_REFRESH_HOURS * 60 / PRICE_REFRESH_MINUTES))
    while True:
        try:
            _refresh_channels()              # daily slow path
        except Exception:
            traceback.print_exc()
        for _ in range(cycles_per_channel):  # 15-min fast path until next channel refit
            try:
                _refresh_prices()
            except Exception:
                traceback.print_exc()
            time.sleep(PRICE_REFRESH_MINUTES * 60)


def _start_worker_once() -> None:
    global _running
    with _lock:
        if _running:
            return
        _running = True
    threading.Thread(target=_worker, daemon=True).start()


# ----------------------------- views --------------------------------------
DASH_HTML = """
<!doctype html><html><head>
<meta charset="utf-8"><title>SG Optimism Charts</title>
<meta http-equiv="refresh" content="60">
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#1f2933;background:#f7f9fb}
 h1{margin:0 0 4px} .sub{color:#667;margin-bottom:18px;font-size:13px}
 table{border-collapse:collapse;width:100%;background:#fff;box-shadow:0 1px 3px #0001;border-radius:8px;overflow:hidden}
 th,td{padding:9px 12px;text-align:left;border-bottom:1px solid #eef1f4;font-size:14px}
 th{background:#2c3e50;color:#fff;font-weight:600}
 tr:hover{background:#f0f6ff}
 a{color:#2c3e50;text-decoration:none;font-weight:600}
 .tabs{margin:14px 0 0} .tabs a{display:inline-block;padding:8px 18px;border-radius:8px 8px 0 0;
   background:#dfe6ee;color:#445;margin-right:4px;font-size:14px}
 .tabs a.active{background:#2c3e50;color:#fff}
 .rec{font-weight:700;padding:2px 10px;border-radius:12px;color:#fff;font-size:12px}
 .BUY{background:#27ae60}.SELL{background:#c0392b}.HOLD{background:#f39c12}
 .bar{height:9px;border-radius:5px;background:linear-gradient(90deg,#27ae60,#f1c40f,#c0392b);position:relative;width:140px}
 .bar i{position:absolute;top:-3px;width:3px;height:15px;background:#1f2933}
 .err{color:#c0392b;font-size:12px}
</style></head><body>
<h1>Optimism Charts — {{market_label}}</h1>
<div class="sub">
 0% = trough line (cheap) · 100% = peak line (expensive). BUY ≤ {{buy}}% · SELL ≥ {{sell}}%.
 Prices every {{refresh}} min · channels re-fit daily · last price: {{last_price}} · last channels: {{last_channel}}
</div>
<div class="tabs">
 {% for code, label in markets %}
 <a href="/?market={{code}}" class="{{'active' if code==market else ''}}">{{label}}</a>
 {% endfor %}
</div>
<table>
<tr><th>Stock</th><th>Last</th><th>Optimism</th><th></th><th>Fair (50%)</th><th>Call</th></tr>
{% for row in rows %}
<tr>
 <td><a href="/chart/{{row.ticker}}">{{row.name}}</a> <span style="color:#99a">{{row.ticker}}</span></td>
 {% if row.error %}
   <td colspan="5" class="err">error: {{row.error}}</td>
 {% else %}
   <td>{{'%.2f'|format(row.last)}}</td>
   <td>{{'%.0f'|format(row.optimism)}}%</td>
   <td><div class="bar"><i style="left:{{row.clamped}}%"></i></div></td>
   <td>{{'%.2f'|format(row.fair)}}</td>
   <td><span class="rec {{row.rec}}">{{row.rec}}</span></td>
 {% endif %}
</tr>
{% endfor %}
</table>
{% if not rows %}<p>Computing first batch… refresh in a few seconds.</p>{% endif %}
</body></html>
"""


@app.route("/")
def dashboard():
    market = request.args.get("market", next(iter(MARKETS)))
    if market not in MARKETS:
        market = next(iter(MARKETS))
    rows = []
    with _lock:
        items = list(_cache.items())
    for ticker, entry in items:
        if entry.get("market") != market:
            continue
        if entry["error"] or entry["result"] is None:
            rows.append(dict(ticker=ticker, name=entry["name"],
                             error=entry["error"] or "pending"))
            continue
        r = entry["result"]
        rows.append(dict(
            ticker=ticker, name=r.name, last=r.last_price, optimism=r.optimism,
            clamped=max(0, min(100, r.optimism)), fair=r.line_price_50,
            rec=r.recommendation, error=None,
        ))
    order = {"SELL": 0, "BUY": 1, "HOLD": 2, None: 3}
    rows.sort(key=lambda x: (order.get(x.get("rec"), 3), -(x.get("optimism") or 0)))
    lp = _last_price.strftime("%Y-%m-%d %H:%M UTC") if _last_price else "pending…"
    lc = _last_channel.strftime("%Y-%m-%d %H:%M UTC") if _last_channel else "pending…"
    return render_template_string(DASH_HTML, rows=rows, last_price=lp, last_channel=lc,
                                  refresh=PRICE_REFRESH_MINUTES,
                                  market=market, market_label=MARKETS[market][0],
                                  markets=[(c, lbl) for c, (lbl, _) in MARKETS.items()],
                                  buy=int(optimism.BUY_BELOW),
                                  sell=int(optimism.SELL_ABOVE))


CHART_HTML = """
<!doctype html><html><head><meta charset="utf-8"><title>{{name}}</title>
<meta http-equiv="refresh" content="60">
<style>body{font-family:-apple-system,sans-serif;margin:24px;text-align:center}
a{color:#2c3e50}</style></head><body>
<p><a href="/?market={{market}}">&larr; back to {{market}} stocks</a></p>
<img src="/chart/{{ticker}}.png" style="max-width:100%;box-shadow:0 1px 4px #0002;border-radius:8px">
</body></html>
"""


@app.route("/chart/<ticker>")
def chart_page(ticker):
    with _lock:
        entry = _cache.get(ticker)
    if not entry:
        abort(404)
    return render_template_string(CHART_HTML, ticker=ticker, name=entry["name"],
                                  market=entry.get("market", ""))


@app.route("/chart/<ticker>.png")
def chart_png(ticker):
    with _lock:
        entry = _cache.get(ticker)
    if not entry or not entry.get("png"):
        abort(404)
    return Response(entry["png"], mimetype="image/png")


_start_worker_once()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
