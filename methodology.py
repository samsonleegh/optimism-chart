"""
Shared methodology content for the dashboard (live app + static site).

`body_html()` returns a self-contained HTML fragment (its own <style> + content)
that both app.py and build_static.py wrap in their page shell. It reads the real
constants/weights from optimism.py and smartmoney.py, so the explanation can never
drift from what the code actually computes.
"""

from __future__ import annotations

import optimism
import smartmoney as sm

# component key -> (display label, plain-English formula)
_SMART_DESC = {
    "cmf": ("Chaikin Money Flow (20d)",
            "Money-flow volume ÷ volume over 20 days, then mapped (CMF+0.2)/0.4×100 → 0–100. "
            "The smoothed, volume-weighted money-flow signal."),
    "entry": ("Entry quality",
              "0.6×MACD-timing + 0.4×volume. MACD-timing rewards a <b>bullish MACD crossover "
              "near or below the zero line</b> (early in a move, not extended); volume rewards "
              "above-average participation."),
    "trend": ("Trend",
              "Counts how many are true: price&gt;EMA20, price&gt;EMA50, EMA20&gt;EMA50 → 0 / 33 / 67 / 100."),
    "momentum": ("Momentum",
                 "0.6×RSI-score + 0.4×MACD-score. RSI(14) rescaled 30–70 → 0–100; "
                 "MACD = 100/50/0 for bull/flat/bear."),
    "obv": ("On-Balance Volume trend",
            "20-day change in OBV, normalised by average volume, through tanh → 0–100."),
    "ad": ("Accumulation / Distribution trend",
           "20-day change in the A/D line, normalised, through tanh → 0–100."),
}


def _weight_rows(weights: dict) -> str:
    rows = ""
    for k, w in sorted(weights.items(), key=lambda kv: -kv[1]):
        label, desc = _SMART_DESC[k]
        rows += (f"<tr><td><b>{label}</b></td><td class='w'>{w*100:.0f}%</td>"
                 f"<td>{desc}</td></tr>")
    rows += (f"<tr class='tot'><td>Total</td><td class='w'>"
             f"{sum(weights.values())*100:.0f}%</td><td></td></tr>")
    return rows


def body_html() -> str:
    o_buy, o_sell = int(optimism.BUY_BELOW), int(optimism.SELL_ABOVE)
    anchors, sep_m = optimism.N_ANCHORS, optimism.PEAK_MIN_SEP_DAYS / 30.4
    s_buy, s_sell, s_hc = int(sm.BUY_ABOVE), int(sm.SELL_BELOW), int(sm.HIGH_CONVICTION)
    fw, cw = sm.FLOW_WINDOW, sm.CMF_WINDOW
    rg_spread, rg_look = sm.REGIME_SPREAD_PCT, sm.REGIME_LOOKBACK

    _order = ["cmf", "entry", "trend", "momentum", "obv", "ad"]
    _lab = {"cmf": "CMF · money flow", "entry": "Entry quality", "trend": "Trend",
            "momentum": "Momentum", "obv": "OBV", "ad": "A/D"}
    regime_rows = ""
    for _k in _order:
        regime_rows += (
            f"<tr><td><b>{_lab[_k]}</b></td>"
            f"<td class='w'>{round(sm.REGIME_WEIGHTS['trending'].get(_k, 0) * 100)}%</td>"
            f"<td class='w'>{round(sm.REGIME_WEIGHTS['ranging'].get(_k, 0) * 100)}%</td>"
            f"<td class='w'>{round(sm.REGIME_WEIGHTS['transitional'].get(_k, 0) * 100)}%</td></tr>")

    return f"""
<style>
 .method{{max-width:900px}}
 .method h2{{margin:26px 0 6px;border-bottom:2px solid #e6e9ee;padding-bottom:4px}}
 .method h3{{margin:18px 0 4px;color:#33415c}}
 .method p,.method li{{line-height:1.5}}
 .method code{{background:#eef1f4;padding:1px 5px;border-radius:4px;font-size:13px}}
 .method table{{border-collapse:collapse;width:100%;margin:10px 0;background:#fff;
   box-shadow:0 1px 3px #0001;border-radius:8px;overflow:hidden}}
 .method th,.method td{{padding:7px 10px;text-align:left;border-bottom:1px solid #eef1f4;font-size:13px;vertical-align:top}}
 .method th{{background:#33415c;color:#fff}}
 .method td.w{{text-align:right;font-weight:700;white-space:nowrap}}
 .method tr.tot td{{font-weight:700;background:#f5f7fa}}
 .method .box{{background:#fff;border-left:4px solid #5b2c83;border-radius:6px;
   padding:10px 14px;margin:12px 0;box-shadow:0 1px 3px #0001}}
 .method .warn{{border-left-color:#e67e22;background:#fff9f0}}
 .method .f{{background:#eef1f4;border-radius:6px;padding:8px 12px;font-family:ui-monospace,Menlo,monospace;font-size:13px;overflow-x:auto}}
</style>
<div class="method">
<h1>📖 Methodology</h1>
<p>How every number on this dashboard is computed, step by step. Everything is built
from <b>daily/weekly OHLCV</b> (open/high/low/close/volume) — there is no paid feed.</p>

<div class="box warn"><b>Not investment advice.</b> These are heuristics on delayed,
free data (Yahoo Finance). The Optimism chart follows a published parallel-channel
valuation method; the Smart Money score is our own experimental construction.</div>

<h2>1 · Data source</h2>
<ul>
<li><b>Optimism (10-year)</b>: 10 years of <b>weekly</b> closing prices.</li>
<li><b>Optimism (1-year)</b> &amp; <b>Smart Money</b>: 1 year of <b>daily</b> OHLCV.</li>
<li>Source is <b>Yahoo Finance</b> via <code>yfinance</code> — end-of-day, ~15-min delayed,
    <b>nominal</b> (not dividend-adjusted). It gives only total daily volume, <b>not</b> real bid/ask volume.</li>
<li>The static site rebuilds roughly every 30 min; each rebuild re-fetches fresh data.</li>
</ul>

<h2>2 · Optimism chart (parallel channel)</h2>
<p>Work in <b>log-price</b> space (<code>y = ln(price)</code>) vs linear time, so steady
compounding is a straight line. The channel is drawn as
<b>five parallel lines</b> (one shared slope):</p>
<ol>
<li><b>P50 (fair value)</b> — the <b>least-squares regression line</b> through log-price.
    Drawn first; it balances the area above and below. Its slope sets the whole channel.</li>
<li><b>P100 (expensive)</b> / <b>P0 (cheap)</b> — parallel to P50, lifted/dropped to rest on the
    <b>peaks / troughs</b>. Anchored on ≥ {anchors} peaks/troughs that are ≥ ~{sep_m:.0f} months apart
    (≈6 weeks on the 1-year chart), so a single spike doesn't set the channel.</li>
<li><b>P75 / P25</b> — midway P50↔P100 and P0↔P50.</li>
</ol>
<p><b>Optimism %</b> = where today's price sits between P0 (0%) and P100 (100%), linear in log-price:</p>
<div class="f">optimism% = 100 × (ln price − ln P0) / (ln P100 − ln P0)</div>
<p>Signal: <b>≤ {o_buy}% → BUY</b> (cheap) · <b>≥ {o_sell}% → SELL</b> (expensive) · else HOLD.
Each stock shows a <b>10-year</b> (long-term) and a <b>1-year</b> (short-term) channel.</p>

<h2>3 · Smart Money score (0–100)</h2>
<p>An <b>experimental</b> weighted blend of six components, each scored 0–100. Because the
weights sum to 100%, the total is bounded 0–100. The <b>weights adapt to each stock's market
regime</b> (see §3a); the table below is the neutral (<i>Mixed</i>) profile.</p>
<table><tr><th>Component</th><th>Weight (Mixed)</th><th>How it's computed</th></tr>
{_weight_rows(sm.SMART_WEIGHTS)}</table>
<p>Signal: <b>≥ {s_buy} → BUY</b> · <b>≤ {s_sell} → SELL</b> · else HOLD · <b>⭐ high-conviction ≥ {s_hc}</b>.</p>

<div class="box"><b>On the proxy "Ask volume %".</b> Real executed bid/ask volume isn't in the
free feed, so we estimate buying pressure from where the close sits in the day's range
(using the typical price (H+L+C)/3), summed over {fw} days. It is shown for information
but is <b>no longer part of the score</b> — the smoothed CMF ({cw}-day) carries the money-flow signal instead.</div>

<h3>Sub-score formulas</h3>
<ul>
<li><b>CMF</b>: <code>(CMF + 0.2) / 0.4 × 100</code>, clipped 0–100.</li>
<li><b>Trend</b>: 1 point each for price&gt;EMA20, price&gt;EMA50, EMA20&gt;EMA50 → ×100/3.</li>
<li><b>Momentum</b>: <code>0.6·RSI-score + 0.4·MACD-score</code>; RSI-score = <code>(RSI−30)/40×100</code>; MACD-score = 100/50/0 (bull/flat/bear).</li>
<li><b>Entry quality</b>: <code>0.6·MACD-timing + 0.4·volume</code>. MACD-timing = <code>100 − 30·max(z,0)</code> for a bull cross, where <code>z</code> is the MACD line's distance from zero in std units — so a cross <b>near/below zero</b> scores ~100, an extended one scores lower.</li>
<li><b>OBV / A/D trend</b>: 20-day change, normalised by average volume, through <code>50 + 50·tanh(·)</code>.</li>
</ul>

<h3 id="regime">3a · Regime-adaptive weights</h3>
<p>Different market conditions reward different signals, so each stock's <b>default</b> weights
switch with its <b>regime</b>, detected from its own price action (no new indicator):</p>
<ul>
<li><b>EMA separation</b>: |EMA20 − EMA50| as a % of price — a wide gap ({rg_spread}%+) means a trend.</li>
<li><b>Directional efficiency</b> (a simplified Kaufman ratio over {rg_look} days):
    <code>|net move| / total path</code> — ~1 = straight-line trend, ~0 = choppy.</li>
</ul>
<p>Two hits → <b>📈 Trend</b>, zero → <b>🔁 Range</b>, one → <b>⚖️ Mixed</b> (the neutral blend, so the
score doesn't jump as a stock crosses the boundary). Weights per regime:</p>
<table><tr><th>Component</th><th>📈 Trend</th><th>🔁 Range</th><th>⚖️ Mixed</th></tr>
{regime_rows}</table>
<p><b>Trend</b> leans on trend / entry / OBV; <b>Range</b> leans on momentum (RSI) &amp; money-flow
(CMF / A-D). On the scoreboard you can also flip on <b>Manual override</b> to apply one weighting
to every stock. <span style="color:#a55">Caveat: this is a regime layer on top of an already-composite
score — sanity-check it against known trending / ranging periods before trusting it live.</span></p>

<h3>Accumulation score</h3>
<p>A volume-only view (is smart money quietly accumulating?): {" + ".join(f"{int(w*100)}%·{_SMART_DESC[k][0].split(' (')[0]}" for k,w in sm.ACCUM_WEIGHTS.items())}.</p>

<h3>Confidence</h3>
<div class="f">confidence = Smart Money score × (0.85 + 0.15 × min(relVolume, 2))</div>
<p>The score nudged ±15% by volume: a signal on heavy volume is more trustworthy than one on a quiet day. Capped at 100.</p>

<h2>4 · Next-day trading plan (ATR heuristic)</h2>
<ul>
<li><b>Support / Resistance</b> = lowest low / highest high of the last 20 days.</li>
<li><b>Entry zone</b> = last − 0.5·ATR(14) … last. <b>Breakout</b> = resistance × 1.005.</li>
<li><b>Stop</b> = min(20-day support, last − 1.5·ATR). <b>Targets</b> = last + 1.5× / 2.5× the risk (risk = last − stop).</li>
</ul>
<div class="box warn">Targets are risk-multiples, so risk/reward to Target 1 is always 1.5 by construction — a mechanical rule, not a read of real chart resistance.</div>

<h2>5 · Market Risk Meter</h2>
<p>🟢 / 🟡 / 🔴 banner from: STI vs its 50-day EMA, the VIX level, and overnight S&amp;P 500 —
a quick "be aggressive / selective / defensive" read. Best-effort; degrades gracefully if a feed is missing.</p>

<h2>6 · Caveats</h2>
<ul>
<li>Free, delayed, end-of-day data — <b>not</b> a live or tick feed.</li>
<li>The ask/bid split is a <b>proxy</b>, not real order flow (which needs a broker/paid feed).</li>
<li>The Smart Money weights &amp; thresholds are judgment calls, easily tuned.</li>
<li><b>Not investment advice.</b></li>
</ul>
</div>
"""
