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
from tickers import MARKETS

OUT = "site"
CHARTS = os.path.join(OUT, "charts")

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
"""

REC_ORDER = {"SELL": 0, "BUY": 1, "HOLD": 2}


def page(title: str, body: str) -> str:
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")


def tabs(active: str) -> str:
    links = "".join(
        f"<a href='{('index' if code == FIRST else code.lower())}.html' "
        f"class='{'active' if code == active else ''}'>{html.escape(label)}</a>"
        for code, (label, _) in MARKETS.items())
    return f"<div class='tabs'>{links}</div>"


def chart_page(result, code) -> str:
    body = (f"<p><a href='{('index' if code == FIRST else code.lower())}.html'>&larr; back</a></p>"
            f"<img src='charts/{html.escape(result.ticker)}.png'>")
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


def build() -> None:
    os.makedirs(CHARTS, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    for code, (label, stocks) in MARKETS.items():
        results = []
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
            time.sleep(0.3)
        fname = "index.html" if code == FIRST else f"{code.lower()}.html"
        with open(os.path.join(OUT, fname), "w") as f:
            f.write(market_page(code, label, results, stamp))
        print(f"wrote {OUT}/{fname} ({len(results)} stocks)")


FIRST = next(iter(MARKETS))

if __name__ == "__main__":
    build()
    print("done -> open site/index.html")
