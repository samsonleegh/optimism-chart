"""
Forward-test track record (paper trading the high-confidence calls).

A static site forgets everything between builds, so to see whether a BUY call
actually worked out we keep a small JSON ledger committed in the repo. Each time
a stock first becomes a *high-confidence* call we open a "pick" stamped with the
entry date, entry price and the full signal snapshot at that moment. On every
later build we re-read the live price and measure how the pick is doing, and once
the evaluation horizon passes we *settle* it with a WIN / MISS verdict and a short
diagnosis of what the original analysis looked like when it missed.

Two signal sources feed the forward test (each pick is tagged):
  * "smart"     — Smart Money ⭐ high-conviction BUY (score >= HIGH_CONVICTION)
  * "optimism"  — Optimism-channel deep BUY (optimism % <= OPT_BUY_BELOW)

Everything here is deliberately network-free: the caller passes in the OHLCV
frames it already downloaded, so this module is fully unit-testable offline and
adds no extra Yahoo calls. Heuristic, delayed daily data — NOT investment advice.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import pandas as pd

# ---- forward-test parameters ----------------------------------------------
HORIZON_TD = 20            # trading days after entry at which a pick is settled (~1 month)
CHECKPOINTS_TD = (5, 10, 20)   # trading-day marks surfaced on open positions
LEDGER_PATH = "track_record.json"
LEDGER_VERSION = 1

# entry gates (kept in step with the two engines' own thresholds)
SMART_HIGH_CONVICTION = 85.0   # smartmoney.HIGH_CONVICTION
OPT_BUY_BELOW = 25.0           # optimism.BUY_BELOW
# An Optimism deep-BUY is only tracked when Smart Money *confirms* it — otherwise
# we'd open a basket of "cheap but still falling" names, which isn't high conviction.
CONFIRM_SMART_MIN = 65.0       # smartmoney.BUY_ABOVE: min Smart Money score to confirm
CONFIRM_BAD_TREND = "down"     # ...and the stock must not be in a clear downtrend

# market code -> benchmark ticker (for the "vs market" alpha read)
BENCHMARKS = {"SG": "^STI", "US": "^GSPC"}

SOURCE_LABEL = {"smart": "🧠 Smart Money ⭐", "optimism": "📉 Optimism deep-BUY"}


# ============================================================================
# ledger persistence
# ============================================================================
def load_ledger(path: str = LEDGER_PATH) -> dict:
    """Load the ledger, or return a fresh empty one if the file is missing/corrupt."""
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("picks"), list):
                data.setdefault("version", LEDGER_VERSION)
                return data
        except (ValueError, OSError):
            pass
    return {"version": LEDGER_VERSION, "picks": []}


def save_ledger(ledger: dict, path: str = LEDGER_PATH) -> None:
    """Write the ledger with stable key ordering so diffs stay small."""
    with open(path, "w") as f:
        json.dump(ledger, f, indent=2, sort_keys=True)
        f.write("\n")


# ============================================================================
# small date / price helpers (work on a daily OHLCV frame)
# ============================================================================
def _as_date(ts) -> str:
    return pd.Timestamp(ts).date().isoformat()


def _price_on_or_before(close: pd.Series, date_iso: str) -> float | None:
    """Closing price on `date_iso`, or the last trading day before it."""
    if close is None or len(close) == 0:
        return None
    cutoff = pd.Timestamp(date_iso)
    sub = close[close.index <= cutoff]
    if len(sub) == 0:
        return None
    return float(sub.iloc[-1])


def _bars_after(df: pd.DataFrame, date_iso: str) -> pd.DataFrame:
    """Rows strictly after the entry date (the forward window)."""
    if df is None or len(df) == 0:
        return df
    return df[df.index > pd.Timestamp(date_iso)]


# ============================================================================
# opening new picks
# ============================================================================
def _is_index(ticker: str) -> bool:
    """Indices/benchmarks (e.g. ^STI) are tracked as context, not tradeable calls."""
    return ticker.startswith("^") or ticker in BENCHMARKS.values()


def _has_open(ledger: dict, ticker: str, source: str) -> bool:
    return any(p["ticker"] == ticker and p["source"] == source and p["status"] == "open"
               for p in ledger["picks"])


def _pick_id(ticker: str, source: str, entry_date: str) -> str:
    return f"{source}:{ticker}:{entry_date}"


def open_picks(ledger: dict, code: str,
               smart_results: list, opt_results: list,
               sdf_map: dict, bench_close: pd.Series | None,
               risk_rating: str) -> list[dict]:
    """Open a pick for every NEW high-confidence call in this market.

    A pick is opened only if there is no already-open pick for the same
    (ticker, source); once a pick settles, a fresh signal can open a new one.
    Returns the list of picks opened in this call.
    """
    opened: list[dict] = []
    smart_by_ticker = {r.ticker: r for r in smart_results}

    def _entry(ticker):
        sdf = sdf_map.get(ticker)
        if sdf is None or len(sdf) == 0:
            return None, None, None
        entry_date = _as_date(sdf.index[-1])
        entry_price = float(sdf["Close"].iloc[-1])
        bench_entry = _price_on_or_before(bench_close, entry_date) if bench_close is not None else None
        return entry_date, entry_price, bench_entry

    # --- source 1: Smart Money high-conviction BUY ---
    for r in smart_results:
        if r.recommendation == "BUY" and getattr(r, "high_conviction", False):
            if _is_index(r.ticker) or _has_open(ledger, r.ticker, "smart"):
                continue
            entry_date, entry_price, bench_entry = _entry(r.ticker)
            if entry_date is None:
                continue
            pick = _new_pick(code, r.ticker, r.name, "smart", entry_date, entry_price,
                             bench_entry, _smart_snapshot(r, risk_rating))
            ledger["picks"].append(pick)
            opened.append(pick)

    # --- source 2: Optimism deep-BUY, CONFIRMED by Smart Money ---
    for r in opt_results:
        if not (r.recommendation == "BUY" and r.optimism <= OPT_BUY_BELOW):
            continue
        if _is_index(r.ticker) or _has_open(ledger, r.ticker, "optimism"):
            continue
        # confirmation gate: cheap on the channel is not enough — Smart Money must
        # agree (score >= CONFIRM_SMART_MIN) and the stock must not be falling.
        sr = smart_by_ticker.get(r.ticker)
        if sr is None or sr.smart_money_score < CONFIRM_SMART_MIN or sr.trend == CONFIRM_BAD_TREND:
            continue
        entry_date, entry_price, bench_entry = _entry(r.ticker)
        if entry_date is None:
            continue
        snap = {"optimism_pct": float(r.optimism),
                "fair_price_50": float(getattr(r, "line_price_50", 0.0)),
                "market_risk": risk_rating,
                "smart_money_score": float(sr.smart_money_score),
                "regime": sr.regime, "rsi": float(sr.rsi),
                "trend": sr.trend, "macd_cross": sr.macd_cross}
        pick = _new_pick(code, r.ticker, r.name, "optimism", entry_date, entry_price,
                         bench_entry, snap)
        ledger["picks"].append(pick)
        opened.append(pick)

    return opened


def _smart_snapshot(r, risk_rating: str) -> dict:
    return {"smart_money_score": float(r.smart_money_score),
            "accumulation_score": float(r.accumulation_score),
            "confidence": float(r.confidence),
            "entry_score": float(r.entry_score),
            "regime": r.regime, "rsi": float(r.rsi), "trend": r.trend,
            "macd_cross": r.macd_cross, "cmf": float(r.cmf),
            "rel_volume": float(r.rel_volume), "buy_ratio": float(r.buy_ratio),
            "target1": float(r.target1), "target2": float(r.target2),
            "stop_loss": float(r.stop_loss), "market_risk": risk_rating}


def _new_pick(code, ticker, name, source, entry_date, entry_price,
              bench_entry, snapshot) -> dict:
    return {"id": _pick_id(ticker, source, entry_date),
            "market": code, "ticker": ticker, "name": name, "source": source,
            "entry_date": entry_date, "entry_price": round(entry_price, 4),
            "benchmark": BENCHMARKS.get(code), "benchmark_entry": bench_entry,
            "entry_signals": snapshot, "status": "open", "result": None}


# ============================================================================
# live metrics + settlement
# ============================================================================
def _metrics(pick: dict, sdf: pd.DataFrame | None, bench_close: pd.Series | None) -> dict:
    """Compute the live forward-test metrics for one pick from current data."""
    m = {"current_price": None, "ret_pct": None, "bench_ret_pct": None,
         "alpha_pct": None, "days_held": None, "hit_t1": False, "hit_stop": False,
         "max_gain_pct": None, "max_drawdown_pct": None, "current_date": None}
    if sdf is None or len(sdf) == 0:
        return m
    entry_price = pick["entry_price"]
    fwd = _bars_after(sdf, pick["entry_date"])
    m["days_held"] = int(len(fwd))
    cur = float(sdf["Close"].iloc[-1])
    m["current_price"] = round(cur, 4)
    m["current_date"] = _as_date(sdf.index[-1])
    if entry_price:
        m["ret_pct"] = round((cur / entry_price - 1.0) * 100.0, 2)
    # path extremes over the forward window (for target/stop + MFE/MAE)
    if len(fwd):
        hi = float(fwd["High"].max())
        lo = float(fwd["Low"].min())
        if entry_price:
            m["max_gain_pct"] = round((hi / entry_price - 1.0) * 100.0, 2)
            m["max_drawdown_pct"] = round((lo / entry_price - 1.0) * 100.0, 2)
        sig = pick["entry_signals"]
        t1, stop = sig.get("target1"), sig.get("stop_loss")
        if t1:
            m["hit_t1"] = bool(hi >= t1)
        if stop:
            m["hit_stop"] = bool(lo <= stop)
    # benchmark return over the same window
    be = pick.get("benchmark_entry")
    if be and bench_close is not None and len(bench_close):
        cb = float(bench_close.iloc[-1])
        m["bench_ret_pct"] = round((cb / be - 1.0) * 100.0, 2)
        if m["ret_pct"] is not None:
            m["alpha_pct"] = round(m["ret_pct"] - m["bench_ret_pct"], 2)
    return m


def _verdict(m: dict) -> str:
    """WIN if the pick made money AND beat its market; else MISS.

    (If we have no benchmark, fall back to absolute return.)
    """
    ret = m.get("ret_pct")
    if ret is None:
        return "MISS"
    alpha = m.get("alpha_pct")
    if alpha is None:
        return "WIN" if ret > 0 else "MISS"
    return "WIN" if (ret > 0 and alpha > 0) else "MISS"


def diagnose(pick: dict, m: dict, verdict: str) -> list[str]:
    """Rules-based 'what did the entry analysis look like' notes.

    For a MISS these read as *what may have been wrong* with the original call;
    for a WIN they read as what worked. Deterministic, first-pass commentary —
    the richer written post-mortem is left to a Claude review of the ledger.
    """
    sig = pick["entry_signals"]
    notes: list[str] = []
    ret = m.get("ret_pct")
    alpha = m.get("alpha_pct")

    if m.get("hit_stop"):
        notes.append("Price hit the planned stop-loss during the window.")
    if m.get("hit_t1"):
        notes.append("Price reached Target&nbsp;1 at some point (max gain "
                     f"{m.get('max_gain_pct')}%).")

    if verdict == "MISS":
        if alpha is not None and ret is not None and ret > 0 >= alpha:
            notes.append("Rose, but lagged the market — a broad-market tide, "
                         "not a stock-specific edge.")
        if alpha is not None and m.get("bench_ret_pct") is not None and m["bench_ret_pct"] < -2:
            notes.append(f"The whole market fell ({m['bench_ret_pct']}%) over the window — "
                         "a risk-off tape the entry didn't account for.")
        if sig.get("market_risk") in ("AMBER", "RED"):
            notes.append(f"Market-risk meter was {sig['market_risk']} at entry — "
                         "conviction signal fired into a shaky tape.")
        if sig.get("regime") == "ranging":
            notes.append("Entered in a <b>ranging</b> regime, where breakouts often fail "
                         "and the trend-following read is weaker.")
        if sig.get("rsi") is not None and sig["rsi"] >= 65:
            notes.append(f"RSI was already hot ({sig['rsi']:.0f}) at entry — the move may "
                         "have been chased near a short-term top.")
        if pick["source"] == "optimism" and sig.get("smart_money_score") is not None \
                and sig["smart_money_score"] < 50:
            notes.append("Cheap on the 10y channel, but Smart Money flow was weak "
                         f"({sig['smart_money_score']:.0f}/100) — value without a catalyst.")
        if not notes:
            notes.append("Signals looked clean at entry; underperformance here is the "
                         "base-rate noise a heuristic will always have.")
    else:  # WIN
        if sig.get("regime") == "trending":
            notes.append("Caught a name already in a clean <b>trend</b>.")
        if sig.get("market_risk") == "GREEN":
            notes.append("Entered with a GREEN market-risk backdrop.")
        if not notes:
            notes.append("Call played out as intended.")
    return notes


def settle_due(ledger: dict, code: str, sdf_map: dict,
               bench_close: pd.Series | None) -> list[dict]:
    """Close every open pick in this market that has reached the horizon.

    The final metrics + verdict + diagnosis are frozen into the pick so they
    never drift afterwards. Returns the picks settled in this call.
    """
    settled: list[dict] = []
    for p in ledger["picks"]:
        if p["market"] != code or p["status"] != "open":
            continue
        sdf = sdf_map.get(p["ticker"])
        m = _metrics(p, sdf, bench_close)
        if m["days_held"] is not None and m["days_held"] >= HORIZON_TD:
            verdict = _verdict(m)
            p["status"] = "closed"
            p["result"] = {"verdict": verdict, "metrics": m,
                           "diagnosis": diagnose(p, m, verdict),
                           "settled_date": m.get("current_date")}
            settled.append(p)
    return settled


# ============================================================================
# view model for rendering (pure data; build_static turns it into HTML)
# ============================================================================
def build_view(ledger: dict, sdf_map_by_market: dict,
               bench_close_by_market: dict) -> dict:
    """Assemble everything the Forward Test page needs, across all markets.

    `sdf_map_by_market`   : {code: {ticker: ohlcv_df}}
    `bench_close_by_market`: {code: benchmark_close_series}
    """
    open_rows, closed_rows = [], []
    for p in ledger["picks"]:
        code = p["market"]
        sdf = sdf_map_by_market.get(code, {}).get(p["ticker"])
        bench = bench_close_by_market.get(code)
        if p["status"] == "closed" and p.get("result"):
            row = {**p, "live": p["result"]["metrics"]}
            closed_rows.append(row)
        else:
            row = {**p, "live": _metrics(p, sdf, bench)}
            open_rows.append(row)

    # newest first for open, most-recently-settled first for closed
    open_rows.sort(key=lambda r: r["entry_date"], reverse=True)
    closed_rows.sort(key=lambda r: (r["result"] or {}).get("settled_date") or "", reverse=True)

    return {"open": open_rows, "closed": closed_rows, "stats": _stats(closed_rows)}


def _stats(closed_rows: list[dict]) -> dict:
    """Aggregate hit-rate / average return stats over settled picks."""
    def agg(rows):
        n = len(rows)
        if not n:
            return {"n": 0, "wins": 0, "win_rate": None, "avg_ret": None, "avg_alpha": None}
        wins = sum(1 for r in rows if (r["result"] or {}).get("verdict") == "WIN")
        rets = [r["live"].get("ret_pct") for r in rows if r["live"].get("ret_pct") is not None]
        alphas = [r["live"].get("alpha_pct") for r in rows if r["live"].get("alpha_pct") is not None]
        return {"n": n, "wins": wins,
                "win_rate": round(wins / n * 100.0, 0),
                "avg_ret": round(sum(rets) / len(rets), 2) if rets else None,
                "avg_alpha": round(sum(alphas) / len(alphas), 2) if alphas else None}

    out = {"all": agg(closed_rows)}
    for src in ("smart", "optimism"):
        out[src] = agg([r for r in closed_rows if r["source"] == src])
    return out
