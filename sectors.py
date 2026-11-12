"""
Sector rotation — "which sector is the flavour of the moment?"

The optimism and Smart Money tabs are bottom-up (one stock at a time). This is the
top-down layer: tag every name with a sector, roll the daily data up per sector, and
rank sectors hottest -> coldest so you can see where the money is rotating before
picking a stock.

Network-free: it takes the Smart Money results (already computed, which carry last
price, trend, EMA50, Smart-Money score, relative volume) plus each stock's daily
OHLCV frame (for multi-day returns) and a ticker->sector map. No extra Yahoo calls.

"Heat" is a momentum + money blend (0-100):
    0.28 · 1-month return   + 0.22 · 1-week return      (price momentum, 50%)
  + 0.20 · breadth                                       (how many names trend up)
  + 0.18 · avg Smart-Money score + 0.12 · avg rel-volume (money / participation, 30%)
Heuristic on delayed daily data — NOT investment advice.
"""

from __future__ import annotations

import pandas as pd

RET_WINDOWS = {"1w": 5, "1m": 20}          # trading-day lookbacks
MIN_NAMES_FOR_FLAVOUR = 2                   # a 1-stock "sector" isn't a rotation call

# heat-score weights (sum to 1.0)
WEIGHTS = {"ret1m": 0.28, "ret1w": 0.22, "breadth": 0.20, "smart": 0.18, "relvol": 0.12}


def _ret_pct(close: pd.Series | None, bars: int) -> float | None:
    """Percent change over `bars` trading days (falls back to the longest available)."""
    if close is None or len(close) < 2:
        return None
    bars = min(bars, len(close) - 1)
    then = float(close.iloc[-1 - bars])
    now = float(close.iloc[-1])
    return (now / then - 1.0) * 100.0 if then else None


def _lin(v: float | None, lo: float, hi: float) -> float:
    """Map v in [lo, hi] to 0..100 (clipped); None -> neutral 50."""
    if v is None:
        return 50.0
    return max(0.0, min(100.0, (v - lo) / (hi - lo) * 100.0))


def _avg(items, key):
    vals = [it[key] for it in items if it.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def analyze_sectors(smart_results: list, sdf_map: dict, sector_map: dict) -> list[dict]:
    """Return per-sector aggregates ranked by heat (hottest first).

    Each sector dict: sector, n, ret1w, ret1m, breadth (%), avg_sm, avg_relvol,
    heat (0-100), and `stocks` (per-name rows sorted by 1-month return).
    """
    buckets: dict[str, list] = {}
    for r in smart_results:
        sec = sector_map.get(r.ticker)
        if not sec:                       # unmapped ticker or the index -> skip
            continue
        sdf = sdf_map.get(r.ticker)
        close = sdf["Close"] if sdf is not None and "Close" in sdf else None
        ema50 = getattr(r, "ema50", None)
        up = bool(ema50 and r.last_price > ema50)
        buckets.setdefault(sec, []).append({
            "ticker": r.ticker, "name": r.name, "last": r.last_price,
            "chg": r.change_pct, "ret1w": _ret_pct(close, RET_WINDOWS["1w"]),
            "ret1m": _ret_pct(close, RET_WINDOWS["1m"]), "up": up,
            "sm": float(r.smart_money_score), "relvol": float(r.rel_volume),
            "rec": r.recommendation, "trend": r.trend, "rsi": float(r.rsi),
        })

    sectors: list[dict] = []
    for sec, items in buckets.items():
        n = len(items)
        ret1w, ret1m = _avg(items, "ret1w"), _avg(items, "ret1m")
        sm, relvol = _avg(items, "sm"), _avg(items, "relvol")
        breadth = sum(1 for it in items if it["up"]) / n * 100.0
        heat = (WEIGHTS["ret1m"] * _lin(ret1m, -15, 15)
                + WEIGHTS["ret1w"] * _lin(ret1w, -8, 8)
                + WEIGHTS["breadth"] * breadth
                + WEIGHTS["smart"] * (sm if sm is not None else 50.0)
                + WEIGHTS["relvol"] * _lin(relvol, 0.7, 1.6))
        items.sort(key=lambda it: (it["ret1m"] is not None, it["ret1m"] or -1e9), reverse=True)
        sectors.append({
            "sector": sec, "n": n, "ret1w": ret1w, "ret1m": ret1m,
            "breadth": round(breadth, 0), "avg_sm": sm, "avg_relvol": relvol,
            "heat": round(heat, 1), "stocks": items,
        })

    sectors.sort(key=lambda s: s["heat"], reverse=True)
    return sectors


def flavour(sectors: list[dict]) -> dict | None:
    """The hottest sector that has at least MIN_NAMES_FOR_FLAVOUR names."""
    for s in sectors:
        if s["n"] >= MIN_NAMES_FOR_FLAVOUR:
            return s
    return sectors[0] if sectors else None
