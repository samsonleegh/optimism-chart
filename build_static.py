"""
Static-site generator for GitHub Pages.

Computes the optimism channel for every stock once and writes a fully static site
to ./site  (HTML + chart PNGs) — no server, no background thread. A GitHub Actions
cron re-runs this every ~15-30 min and republishes to Pages.

Run locally:  .venv/bin/python build_static.py   then open site/index.html
"""

from __future__ import annotations

import os
import time
import html
import traceback
from datetime import datetime, timezone

import optimism
import smartmoney
from tickers import MARKETS

OUT = "site"
CHARTS = os.path.join(OUT, "charts")
SMART = os.path.join(OUT, "smart")

CSS = """
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#1f2933;background:#f7f9fb}
 h1{margin:0 0 4px} .sub{color:#667;margin-bottom:8px;font-size:13px}
 table{border-collapse:collapse;width:100%;background:#fff;box-shadow:0 1px 3px #0001;border-radius:8px;overflow:hidden}
 th,td{padding:9px 12px;text-align:left;border-bottom:1px solid #eef1f4;font-size:14px}
 th{background:#2c3e50;color:#fff;font-weight:600}
 tr:hover{background:#f0f6ff}
 a{color:#2c3e50;text-decoration:none;font-weight:600}
 .tabs{margin:14px 0 0}
 .tabs a{display:inline-block;padding:8px 18px;border-radius:8px 8px 0 0;background:#dfe6ee;color:#445;margin-right:4px;font-size:14px}
 .tabs a.active{background:#2c3e50;color:#fff}
 .rec{font-weight:700;padding:2px 10px;border-radius:12px;color:#fff;font-size:12px}
 .BUY{background:#27ae60}.SELL{background:#c0392b}.HOLD{background:#f39c12}
 .bar{height:9px;border-radius:5px;background:linear-gradient(90deg,#27ae60,#f1c40f,#c0392b);position:relative;width:140px}
 .bar i{position:absolute;top:-3px;width:3px;height:15px;background:#1f2933}
 .err{color:#c0392b;font-size:12px} img{max-width:100%;box-shadow:0 1px 4px #0002;border-radius:8px}
 .tabs a.smartlink{background:#5b2c83;color:#fff}
 .tabs.smart a.active{background:#5b2c83}
 table.smart th,table.smart td{text-align:right}
 table.smart th:first-child,table.smart td:first-child{text-align:left}
 table.smart th{background:#5b2c83}
 .score{font-weight:700} .pos{color:#27ae60}.neg{color:#c0392b}
 .hc{background:#5b2c83;color:#fff;border-radius:6px;padding:1px 6px;font-size:11px;margin-left:4px}
 .meter{display:inline-block;padding:6px 14px;border-radius:8px;color:#fff;font-weight:700;font-size:14px}
 .GREEN{background:#27ae60}.AMBER{background:#f39c12}.RED{background:#c0392b}
 .risk{background:#fff;border-radius:8px;box-shadow:0 1px 3px #0001;padding:12px 16px;margin:6px 0 16px}
 .note{color:#667;font-size:12px}
 .flow{height:8px;border-radius:4px;background:#e74c3c;position:relative;width:90px;display:inline-block;vertical-align:middle}
 .flow i{position:absolute;left:0;top:0;height:8px;border-radius:4px;background:#27ae60}
 .card{background:#fff;border-radius:8px;box-shadow:0 1px 3px #0001;padding:16px 20px;margin:14px 0;max-width:760px}
 .card table{box-shadow:none} .card td{text-align:left} .card td:last-child{text-align:right;font-weight:600}
 .big{font-size:32px;font-weight:800;color:#5b2c83}
"""

REC_ORDER = {"SELL": 0, "BUY": 1, "HOLD": 2}


def opt_file(code: str) -> str:
    return "index.html" if code == FIRST else f"{code.lower()}.html"


def smart_file(code: str) -> str:
    return "smart.html" if code == FIRST else f"smart-{code.lower()}.html"


def page(title: str, body: str) -> str:
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")


def tabs(active: str) -> str:
    links = "".join(
        f"<a href='{opt_file(code)}' "
        f"class='{'active' if code == active else ''}'>{html.escape(label)}</a>"
        for code, (label, _) in MARKETS.items())
    links += f"<a href='{smart_file(active)}' class='smartlink'>🧠 Smart Money</a>"
    return f"<div class='tabs'>{links}</div>"


def smart_tabs(active: str) -> str:
    links = "".join(
        f"<a href='{smart_file(code)}' "
        f"class='{'active' if code == active else ''}'>{html.escape(label)}</a>"
        for code, (label, _) in MARKETS.items())
    links += f"<a href='{opt_file(active)}' style='background:#dfe6ee;color:#445'>← Optimism</a>"
    return f"<div class='tabs smart'>{links}</div>"


def chart_page(result, code) -> str:
    # this page lives in site/charts/, so back-link goes up one level and the
    # image sits alongside it in the same directory.
    back = "index.html" if code == FIRST else f"{code.lower()}.html"
    body = (f"<p><a href='../{back}'>&larr; back</a></p>"
            f"<img src='{html.escape(result.ticker)}.png'>")
    return page(f"{result.name} ({result.ticker})", body)


def market_page(code, label, results, stamp) -> str:
    rows = ""
    for r in sorted(results, key=lambda x: (REC_ORDER.get(x.recommendation, 3), -x.optimism)):
        clamped = max(0, min(100, r.optimism))
        rows += (
            f"<tr><td><a href='charts/{html.escape(r.ticker)}.html'>{html.escape(r.name)}</a> "
            f"<span style='color:#99a'>{html.escape(r.ticker)}</span></td>"
            f"<td>{r.last_price:.2f}</td><td>{r.optimism:.0f}%</td>"
            f"<td><div class='bar'><i style='left:{clamped}%'></i></div></td>"
            f"<td>{r.line_price_50:.2f}</td>"
            f"<td><span class='rec {r.recommendation}'>{r.recommendation}</span></td></tr>")
    body = (
        f"<h1>Optimism Charts — {html.escape(label)}</h1>"
        f"<div class='sub'>0% = trough (cheap) · 100% = peak (expensive). "
        f"BUY &le; {int(optimism.BUY_BELOW)}% · SELL &ge; {int(optimism.SELL_ABOVE)}% · "
        f"built {stamp}</div>"
        f"{tabs(code)}"
        f"<table><tr><th>Stock</th><th>Last</th><th>Optimism</th><th></th>"
        f"<th>Fair (50%)</th><th>Call</th></tr>{rows}</table>")
    return page(f"Optimism — {label}", body)


def risk_banner(risk) -> str:
    vix = f" · VIX {risk.vix:.0f}" if risk.vix is not None else ""
    sp = f" · S&P {risk.sp_change:+.1f}%" if risk.sp_change is not None else ""
    notes = (f"<div class='note'>{html.escape('; '.join(risk.notes))}</div>"
             if risk.notes else "")
    return (f"<div class='risk'><span class='meter {risk.rating}'>Market Risk: {risk.rating}</span> "
            f"&nbsp; STI {html.escape(risk.sti_trend)}{vix}{sp}{notes}</div>")


def smart_detail_page(r, code, chart_html) -> str:
    back = smart_file(code)
    body = (
        f"<p><a href='../{back}'>&larr; back to Smart Money</a> · "
        f"<a href='../charts/{html.escape(r.ticker)}.html'>optimism chart &rarr;</a></p>"
        f"<h1>{html.escape(r.name)} <span style='color:#99a;font-size:16px'>{html.escape(r.ticker)}</span></h1>"
        f"<p>Last {r.last_price:.4g} ({r.change_pct:+.1f}%) · "
        f"<span class='rec {r.recommendation}'>{r.recommendation}</span> "
        f"{'⭐ high-conviction' if r.high_conviction else ''}</p>"
        f"<div style='background:#fff;border-radius:8px;box-shadow:0 1px 4px #0002;padding:6px'>{chart_html}</div>"
        f"<div class='card'><div class='big'>{r.smart_money_score:.0f}"
        f"<span style='font-size:16px;color:#999'>/100 Smart Money</span></div><table>"
        f"<tr><td>Accumulation score</td><td>{r.accumulation_score:.0f}/100</td></tr>"
        f"<tr><td>Proxy ask (buy-up) volume, last {smartmoney.FLOW_WINDOW}d</td><td>{r.buy_ratio:.0f}%</td></tr>"
        f"<tr><td>Chaikin Money Flow (20d)</td><td>{r.cmf:+.3f}</td></tr>"
        f"<tr><td>Relative volume (vs 20d avg)</td><td>{r.rel_volume:.2f}×</td></tr>"
        f"<tr><td>RSI (14)</td><td>{r.rsi:.0f}</td></tr>"
        f"<tr><td>MACD</td><td>{r.macd_cross} ({r.macd_hist:+.3f})</td></tr>"
        f"<tr><td>Trend (EMA20 {r.ema20:.3g} / EMA50 {r.ema50:.3g})</td><td>{r.trend}</td></tr>"
        f"<tr><td>Candlestick</td><td>{html.escape(r.candlestick)}</td></tr></table></div>"
        f"<div class='card'><h3 style='margin:0 0 6px'>Next-day trading plan "
        f"<span style='color:#999;font-size:12px'>(ATR heuristic)</span></h3><table>"
        f"<tr><td>Support / Resistance (20d)</td><td>{r.support:.4g} / {r.resistance:.4g}</td></tr>"
        f"<tr><td>Entry zone</td><td>{r.entry_low:.4g} – {r.entry_high:.4g}</td></tr>"
        f"<tr><td>Breakout buy above</td><td>{r.breakout:.4g}</td></tr>"
        f"<tr><td>Stop-loss</td><td>{r.stop_loss:.4g}</td></tr>"
        f"<tr><td>Target 1 / Target 2</td><td>{r.target1:.4g} / {r.target2:.4g}</td></tr>"
        f"<tr><td>Risk / reward (to T1)</td><td>{r.risk_reward:.2f}</td></tr></table></div>"
        f"<p style='color:#888;font-size:12px'>Heuristic proxy from daily OHLCV "
        f"(Yahoo, delayed). Not investment advice.</p>")
    return page(f"{r.name} — Smart Money", body)


def smart_page(code, label, results, risk, stamp) -> str:
    rows = ""
    for i, r in enumerate(sorted(results, key=lambda x: -x.smart_money_score), 1):
        hc = "<span class='hc'>⭐</span>" if r.high_conviction else ""
        cc = "pos" if r.change_pct >= 0 else "neg"
        cmfc = "pos" if r.cmf >= 0 else "neg"
        rows += (
            f"<tr><td>{i}</td>"
            f"<td><a href='smart/{html.escape(r.ticker)}.html'>{html.escape(r.name)}</a>{hc}</td>"
            f"<td>{r.last_price:.3g}</td>"
            f"<td class='{cc}'>{r.change_pct:+.1f}</td>"
            f"<td>{r.rsi:.0f}</td><td>{r.macd_cross}</td>"
            f"<td>{r.rel_volume:.2f}×</td><td>{r.buy_ratio:.0f}%</td>"
            f"<td><span class='flow'><i style='width:{r.buy_ratio:.0f}%'></i></span></td>"
            f"<td class='{cmfc}'>{r.cmf:+.2f}</td><td>{r.trend}</td>"
            f"<td class='score'>{r.smart_money_score:.0f}</td>"
            f"<td>{r.accumulation_score:.0f}</td><td>{r.confidence:.0f}</td>"
            f"<td><span class='rec {r.recommendation}'>{r.recommendation}</span></td></tr>")
    body = (
        f"<h1>🧠 Smart Money Scoreboard — {html.escape(label)} "
        f"<span style='font-size:13px;color:#999'>(experimental)</span></h1>"
        f"<div class='sub'>Volume split into proxy <b>ask (buy-up)</b> vs <b>bid (sell-down)</b> "
        f"from daily OHLCV → Smart Money &amp; Accumulation scores (0–100). "
        f"BUY ≥ {int(smartmoney.BUY_ABOVE)} · SELL ≤ {int(smartmoney.SELL_BELOW)} · "
        f"⭐ = high-conviction (≥{int(smartmoney.HIGH_CONVICTION)}). "
        f"Heuristic, daily data, <b>not advice</b> · built {stamp}</div>"
        f"{risk_banner(risk)}{smart_tabs(code)}"
        f"<table class='smart'><tr><th>#</th><th>Stock</th><th>Last</th><th>Chg%</th>"
        f"<th>RSI</th><th>MACD</th><th>RelVol</th><th>Ask vol%</th><th></th><th>CMF</th>"
        f"<th>Trend</th><th>SM Score</th><th>Accum</th><th>Conf</th><th>Call</th></tr>"
        f"{rows}</table>")
    return page(f"Smart Money — {label}", body)


def build() -> None:
    os.makedirs(CHARTS, exist_ok=True)
    os.makedirs(SMART, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    try:
        risk = smartmoney.market_risk()
    except Exception:
        traceback.print_exc()
        risk = smartmoney.MarketRisk(rating="AMBER", notes=["risk meter unavailable"])

    for code, (label, stocks) in MARKETS.items():
        results, smart_results = [], []
        for ticker, name in stocks:
            try:
                result, channel = optimism.compute(ticker, name=name)
                png = optimism.make_chart(result, channel)
                with open(os.path.join(CHARTS, f"{ticker}.png"), "wb") as f:
                    f.write(png)
                with open(os.path.join(CHARTS, f"{ticker}.html"), "w") as f:
                    f.write(chart_page(result, code))
                results.append(result)
                print(f"  {code} {ticker:10s} {result.optimism:5.0f}%  {result.recommendation}")
            except Exception as exc:
                print(f"  {code} {ticker:10s} ERROR {exc}")
                traceback.print_exc()
            # Smart Money scoreboard (separate daily-OHLCV fetch)
            try:
                sres, sdf = smartmoney.compute(ticker, name=name)
                chart_html = smartmoney.make_chart(sres, sdf)
                with open(os.path.join(SMART, f"{ticker}.html"), "w") as f:
                    f.write(smart_detail_page(sres, code, chart_html))
                smart_results.append(sres)
                print(f"  {code} {ticker:10s} SM {sres.smart_money_score:5.0f}  {sres.recommendation}")
            except Exception as exc:
                print(f"  {code} {ticker:10s} SM ERROR {exc}")
            time.sleep(0.3)
        fname = opt_file(code)
        with open(os.path.join(OUT, fname), "w") as f:
            f.write(market_page(code, label, results, stamp))
        with open(os.path.join(OUT, smart_file(code)), "w") as f:
            f.write(smart_page(code, label, smart_results, risk, stamp))
        print(f"wrote {OUT}/{fname} ({len(results)} opt, {len(smart_results)} smart)")


FIRST = next(iter(MARKETS))

if __name__ == "__main__":
    build()
    print("done -> open site/index.html")
