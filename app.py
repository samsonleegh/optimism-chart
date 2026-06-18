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
import smartmoney
from tickers import MARKETS, ALL_STOCKS

PRICE_REFRESH_MINUTES = 15     # latest price + optimism % + chart
CHANNEL_REFRESH_HOURS = 24     # re-fit the 10y log-linear channel (slow-moving)

app = Flask(__name__)

# ---- shared cache (written by worker thread, read by request handlers) ----
# ticker -> {"name", "channel": Channel|None, "result", "png": bytes|None,
#            "smart": SmartMoneyResult|None, "error"}
_lock = threading.Lock()
_cache: dict[str, dict] = {}
_market_risk: smartmoney.MarketRisk | None = None
_last_channel: datetime | None = None
_last_price: datetime | None = None
_last_smart: datetime | None = None
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


def _refresh_smartmoney() -> None:
    """Daily: pull 1y OHLCV and compute the smart-money scoreboard per stock."""
    global _last_smart, _market_risk
    for market, ticker, name in ALL_STOCKS:
        try:
            res = smartmoney.analyze(ticker, name=name)
            with _lock:
                e = _cache.setdefault(ticker, {"name": name, "market": market,
                                               "channel": None, "result": None,
                                               "png": None, "error": None})
                e["smart"], e["smart_error"] = res, None
        except Exception as exc:
            with _lock:
                e = _cache.setdefault(ticker, {"name": name, "market": market,
                                               "channel": None, "result": None,
                                               "png": None, "error": None})
                e["smart"], e["smart_error"] = None, str(exc)
            traceback.print_exc()
        time.sleep(0.3)  # be gentle with the data source
    try:
        risk = smartmoney.market_risk()
        with _lock:
            _market_risk = risk
    except Exception:
        traceback.print_exc()
    _last_smart = datetime.now(timezone.utc)
    print(f"[{_last_smart:%H:%M:%S} UTC] computed smart-money scores")


def _worker() -> None:
    cycles_per_channel = max(1, int(CHANNEL_REFRESH_HOURS * 60 / PRICE_REFRESH_MINUTES))
    while True:
        try:
            _refresh_channels()              # daily slow path
        except Exception:
            traceback.print_exc()
        try:
            _refresh_smartmoney()            # daily smart-money scoreboard
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
 <a href="/smartmoney?market={{market}}" style="background:#5b2c83">🧠 Smart Money</a>
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
        if entry.get("error") or entry.get("result") is None:
            rows.append(dict(ticker=ticker, name=entry["name"],
                             error=entry.get("error") or "pending"))
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


SMART_HTML = """
<!doctype html><html><head>
<meta charset="utf-8"><title>Smart Money Scoreboard</title>
<meta http-equiv="refresh" content="120">
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#1f2933;background:#f7f9fb}
 h1{margin:0 0 4px} .sub{color:#667;margin-bottom:14px;font-size:13px}
 table{border-collapse:collapse;width:100%;background:#fff;box-shadow:0 1px 3px #0001;border-radius:8px;overflow:hidden}
 th,td{padding:8px 10px;text-align:right;border-bottom:1px solid #eef1f4;font-size:13px}
 th:first-child,td:first-child{text-align:left}
 th{background:#5b2c83;color:#fff;font-weight:600;cursor:default}
 tr:hover{background:#f5f0fb}
 a{color:#5b2c83;text-decoration:none;font-weight:600}
 .tabs{margin:14px 0 0} .tabs a{display:inline-block;padding:8px 18px;border-radius:8px 8px 0 0;
   background:#dfe6ee;color:#445;margin-right:4px;font-size:14px}
 .tabs a.active{background:#5b2c83;color:#fff}
 .rec{font-weight:700;padding:2px 9px;border-radius:12px;color:#fff;font-size:12px}
 .BUY{background:#27ae60}.SELL{background:#c0392b}.HOLD{background:#f39c12}
 .score{font-weight:700} .pos{color:#27ae60}.neg{color:#c0392b}
 .hc{background:#5b2c83;color:#fff;border-radius:6px;padding:1px 6px;font-size:11px;margin-left:4px}
 .meter{display:inline-block;padding:6px 14px;border-radius:8px;color:#fff;font-weight:700;font-size:14px}
 .GREEN{background:#27ae60}.AMBER{background:#f39c12}.RED{background:#c0392b}
 .risk{background:#fff;border-radius:8px;box-shadow:0 1px 3px #0001;padding:12px 16px;margin:6px 0 16px}
 .err{color:#c0392b;font-size:12px} .note{color:#667;font-size:12px}
 .flow{height:8px;border-radius:4px;background:#e74c3c;position:relative;width:90px;display:inline-block;vertical-align:middle}
 .flow i{position:absolute;left:0;top:0;height:8px;border-radius:4px;background:#27ae60}
</style></head><body>
<h1>🧠 Smart Money Scoreboard — {{market_label}} <span style="font-size:13px;color:#999">(experimental)</span></h1>
<div class="sub">
 Volume split into proxy <b>ask (buy-up)</b> vs <b>bid (sell-down)</b> from daily OHLCV →
 Smart Money &amp; Accumulation scores (0–100). BUY ≥ {{buy}} · SELL ≤ {{sell}} · ⭐ = high-conviction (≥{{hc}}).
 Heuristic, daily data, <b>not advice</b>. Last computed: {{last_smart}}.
</div>
<div class="risk">
 <span class="meter {{risk.rating}}">Market Risk: {{risk.rating}}</span>
 &nbsp; STI {{risk.sti_trend}}
 {% if risk.vix is not none %}· VIX {{'%.0f'|format(risk.vix)}}{% endif %}
 {% if risk.sp_change is not none %}· S&P {{'%+.1f'|format(risk.sp_change)}}%{% endif %}
 {% if risk.notes %}<div class="note">{{ '; '.join(risk.notes) }}</div>{% endif %}
</div>
<div class="tabs">
 {% for code, label in markets %}
 <a href="/smartmoney?market={{code}}" class="{{'active' if code==market else ''}}">{{label}}</a>
 {% endfor %}
 <a href="/?market={{market}}" style="background:#dfe6ee;color:#445">← Optimism</a>
</div>
<table>
<tr><th>#</th><th>Stock</th><th>Last</th><th>Chg%</th><th>RSI</th><th>MACD</th>
 <th>RelVol</th><th>Ask vol%</th><th></th><th>CMF</th><th>Trend</th>
 <th>SM Score</th><th>Accum</th><th>Conf</th><th>Call</th></tr>
{% for r in rows %}
<tr>
 <td>{{loop.index}}</td>
 <td><a href="/smartmoney/{{r.ticker}}">{{r.name}}</a>{% if r.high_conviction %}<span class="hc">⭐</span>{% endif %}</td>
 {% if r.error %}
   <td colspan="13" class="err">error: {{r.error}}</td>
 {% else %}
   <td>{{'%.3g'|format(r.last_price)}}</td>
   <td class="{{'pos' if r.change_pct>=0 else 'neg'}}">{{'%+.1f'|format(r.change_pct)}}</td>
   <td>{{'%.0f'|format(r.rsi)}}</td>
   <td>{{r.macd_cross}}</td>
   <td>{{'%.2f'|format(r.rel_volume)}}×</td>
   <td>{{'%.0f'|format(r.buy_ratio)}}%</td>
   <td><span class="flow"><i style="width:{{r.buy_ratio}}%"></i></span></td>
   <td class="{{'pos' if r.cmf>=0 else 'neg'}}">{{'%+.2f'|format(r.cmf)}}</td>
   <td>{{r.trend}}</td>
   <td class="score">{{'%.0f'|format(r.smart_money_score)}}</td>
   <td>{{'%.0f'|format(r.accumulation_score)}}</td>
   <td>{{'%.0f'|format(r.confidence)}}</td>
   <td><span class="rec {{r.recommendation}}">{{r.recommendation}}</span></td>
 {% endif %}
</tr>
{% endfor %}
</table>
{% if not rows %}<p>Computing first batch… refresh in a minute.</p>{% endif %}
</body></html>
"""


@app.route("/smartmoney")
def smartmoney_board():
    market = request.args.get("market", next(iter(MARKETS)))
    if market not in MARKETS:
        market = next(iter(MARKETS))
    rows = []
    with _lock:
        items = list(_cache.items())
        risk = _market_risk
    for ticker, entry in items:
        if entry.get("market") != market:
            continue
        res, err = entry.get("smart"), entry.get("smart_error")
        if res is None:
            rows.append(dict(ticker=ticker, name=entry["name"],
                             error=err or "pending", high_conviction=False))
        else:
            d = res.as_dict()
            d["error"] = None
            rows.append(d)
    rows.sort(key=lambda r: (r.get("error") is not None, -(r.get("smart_money_score") or 0)))
    risk = risk or smartmoney.MarketRisk(rating="AMBER", notes=["computing…"])
    ls = _last_smart.strftime("%Y-%m-%d %H:%M UTC") if _last_smart else "pending…"
    return render_template_string(
        SMART_HTML, rows=rows, risk=risk, last_smart=ls,
        market=market, market_label=MARKETS[market][0],
        markets=[(c, lbl) for c, (lbl, _) in MARKETS.items()],
        buy=int(smartmoney.BUY_ABOVE), sell=int(smartmoney.SELL_BELOW),
        hc=int(smartmoney.HIGH_CONVICTION))


SMART_DETAIL_HTML = """
<!doctype html><html><head><meta charset="utf-8"><title>{{r.name}} — Smart Money</title>
<meta http-equiv="refresh" content="120">
<style>
 body{font-family:-apple-system,sans-serif;margin:24px;color:#1f2933;background:#f7f9fb;max-width:760px}
 a{color:#5b2c83} h1{margin:0 0 2px}
 .rec{font-weight:700;padding:3px 12px;border-radius:12px;color:#fff}
 .BUY{background:#27ae60}.SELL{background:#c0392b}.HOLD{background:#f39c12}
 .card{background:#fff;border-radius:8px;box-shadow:0 1px 3px #0001;padding:16px 20px;margin:14px 0}
 table{border-collapse:collapse;width:100%} td{padding:6px 8px;border-bottom:1px solid #eef1f4;font-size:14px}
 td:last-child{text-align:right;font-weight:600}
 .big{font-size:34px;font-weight:800;color:#5b2c83}
</style></head><body>
<p><a href="/smartmoney?market={{market}}">← back to Smart Money ({{market}})</a> ·
   <a href="/chart/{{r.ticker}}">optimism chart →</a></p>
<h1>{{r.name}} <span style="color:#99a;font-size:16px">{{r.ticker}}</span></h1>
<p>Last {{'%.4g'|format(r.last_price)}} ({{'%+.1f'|format(r.change_pct)}}%) ·
   <span class="rec {{r.recommendation}}">{{r.recommendation}}</span>
   {% if r.high_conviction %}⭐ high-conviction{% endif %}</p>
<div class="card">
 <div class="big">{{'%.0f'|format(r.smart_money_score)}}<span style="font-size:16px;color:#999">/100 Smart Money</span></div>
 <table>
  <tr><td>Accumulation score</td><td>{{'%.0f'|format(r.accumulation_score)}}/100</td></tr>
  <tr><td>Proxy ask (buy-up) volume, last {{flow_window}}d</td><td>{{'%.0f'|format(r.buy_ratio)}}%</td></tr>
  <tr><td>Chaikin Money Flow (20d)</td><td>{{'%+.3f'|format(r.cmf)}}</td></tr>
  <tr><td>Relative volume (vs 20d avg)</td><td>{{'%.2f'|format(r.rel_volume)}}×</td></tr>
  <tr><td>RSI (14)</td><td>{{'%.0f'|format(r.rsi)}}</td></tr>
  <tr><td>MACD</td><td>{{r.macd_cross}} ({{'%+.3f'|format(r.macd_hist)}})</td></tr>
  <tr><td>Trend (EMA20 {{'%.3g'|format(r.ema20)}} / EMA50 {{'%.3g'|format(r.ema50)}})</td><td>{{r.trend}}</td></tr>
  <tr><td>Candlestick</td><td>{{r.candlestick}}</td></tr>
 </table>
</div>
<div class="card">
 <h3 style="margin:0 0 6px">Next-day trading plan <span style="color:#999;font-size:12px">(ATR heuristic)</span></h3>
 <table>
  <tr><td>Support / Resistance (20d)</td><td>{{'%.4g'|format(r.support)}} / {{'%.4g'|format(r.resistance)}}</td></tr>
  <tr><td>Entry zone</td><td>{{'%.4g'|format(r.entry_low)}} – {{'%.4g'|format(r.entry_high)}}</td></tr>
  <tr><td>Breakout buy above</td><td>{{'%.4g'|format(r.breakout)}}</td></tr>
  <tr><td>Stop-loss</td><td>{{'%.4g'|format(r.stop_loss)}}</td></tr>
  <tr><td>Target 1 / Target 2</td><td>{{'%.4g'|format(r.target1)}} / {{'%.4g'|format(r.target2)}}</td></tr>
  <tr><td>Risk / reward (to T1)</td><td>{{'%.2f'|format(r.risk_reward)}}</td></tr>
 </table>
</div>
<p style="color:#888;font-size:12px">Heuristic proxy from daily OHLCV (Yahoo, delayed). Not investment advice.</p>
</body></html>
"""


@app.route("/smartmoney/<ticker>")
def smartmoney_detail(ticker):
    with _lock:
        entry = _cache.get(ticker)
    if not entry:
        abort(404)
    res = entry.get("smart")
    if res is None:
        abort(404)
    return render_template_string(SMART_DETAIL_HTML, r=res,
                                  market=entry.get("market", ""),
                                  flow_window=smartmoney.FLOW_WINDOW)


_start_worker_once()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
