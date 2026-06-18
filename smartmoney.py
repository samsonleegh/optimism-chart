"""
Smart Money scoreboard engine (experimental).

Idea
----
"Successful ask volume vs bid volume" = the volume that transacted at the ask
(aggressive *buying*, price pushed up) versus at the bid (aggressive *selling*,
price pushed down). That is true tick / order-flow data and is NOT in the free
Yahoo Finance feed. So we APPROXIMATE it from each daily OHLCV bar using where
the close sits inside the day's range (a standard money-flow proxy):

    buy_fraction  = (Close - Low) / (High - Low)      # closed near the high -> bought up
    buy_volume    = Volume * buy_fraction             # proxy "successful ask volume"
    sell_volume   = Volume * (1 - buy_fraction)       # proxy "successful bid volume"

Rolled up over a recent window this gives a buy/sell pressure %, which (together
with classic accumulation indicators) feeds a Smart Money Score (0-100) and an
Accumulation Score (0-100). These are heuristics for trying the idea out, NOT a
real order-book feed and NOT investment advice.

Everything here is computed from daily OHLCV, so it works with yfinance today.
Things ChatGPT's full spec wanted that the free feed can't give (real reported
block trades, the closing-auction tape) are intentionally omitted, not faked.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd
import yfinance as yf


# ---- recommendation thresholds on the Smart Money Score (0-100) -----------
BUY_ABOVE = 65.0     # strong net accumulation -> BUY
SELL_BELOW = 40.0    # net distribution        -> SELL
FLOW_WINDOW = 10     # trading days for the buy/sell (ask/bid) volume split
CMF_WINDOW = 20      # Chaikin Money Flow window
HIGH_CONVICTION = 85.0  # score at/above which a name is a "high-conviction" idea


# --------------------------- indicator helpers -----------------------------
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(100.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _clip01(v: float) -> float:
    return float(max(0.0, min(100.0, v)))


# --------------------------- the ask/bid split -----------------------------
def buy_sell_volume(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Proxy split of each bar's volume into ask-side (buy) and bid-side (sell)."""
    rng = (df["High"] - df["Low"]).replace(0.0, np.nan)
    buy_frac = ((df["Close"] - df["Low"]) / rng).fillna(0.5).clip(0.0, 1.0)
    buy_vol = df["Volume"] * buy_frac
    sell_vol = df["Volume"] * (1.0 - buy_frac)
    return buy_vol, sell_vol


def _candlestick(df: pd.DataFrame) -> str:
    """Name a simple pattern off the last one/two bars (best-effort)."""
    if len(df) < 2:
        return "—"
    o, h, l, c = (float(df["Open"].iloc[-1]), float(df["High"].iloc[-1]),
                  float(df["Low"].iloc[-1]), float(df["Close"].iloc[-1]))
    po, pc = float(df["Open"].iloc[-2]), float(df["Close"].iloc[-2])
    rng = h - l or 1e-9
    body = abs(c - o)
    upper = h - max(c, o)
    lower = min(c, o) - l
    if body <= 0.1 * rng:
        return "Doji"
    if lower >= 2 * body and upper <= body:
        return "Hammer" if c >= o else "Hanging Man"
    if upper >= 2 * body and lower <= body:
        return "Shooting Star"
    if c > o and pc < po and c >= po and o <= pc:
        return "Bullish Engulfing"
    if c < o and pc > po and o >= pc and c <= po:
        return "Bearish Engulfing"
    return "Bullish Marubozu" if c > o and body > 0.8 * rng else (
        "Bearish Marubozu" if c < o and body > 0.8 * rng else "—")


@dataclass
class SmartMoneyResult:
    ticker: str
    name: str
    end: str
    last_price: float
    change_pct: float
    rsi: float
    macd_hist: float
    macd_cross: str            # "bull" / "bear" / "flat"
    rel_volume: float          # today's volume / 20d average
    buy_ratio: float           # % successful ask (buy) volume over FLOW_WINDOW
    cmf: float                 # Chaikin Money Flow (CMF_WINDOW)
    ema20: float
    ema50: float
    trend: str                 # "up" / "down" / "mixed"
    candlestick: str
    support: float
    resistance: float
    smart_money_score: float   # 0-100
    accumulation_score: float  # 0-100
    confidence: float          # 0-100 (used for ranking)
    recommendation: str        # BUY / HOLD / SELL
    high_conviction: bool
    # next-day trading plan (heuristic, ATR-based)
    entry_low: float
    entry_high: float
    breakout: float
    stop_loss: float
    target1: float
    target2: float
    risk_reward: float
    error: str | None = None

    def as_dict(self):
        return asdict(self)


def fetch_ohlcv(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Daily OHLCV frame with single-level columns Open/High/Low/Close/Volume."""
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):      # flatten ('Close','AAPL') -> 'Close'
        df.columns = df.columns.get_level_values(0)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    return df[cols].dropna()


def analyze(ticker: str, name: str | None = None,
            df: pd.DataFrame | None = None) -> SmartMoneyResult:
    """Compute the smart-money scoreboard row for one stock from daily OHLCV."""
    name = name or ticker
    if df is None:
        df = fetch_ohlcv(ticker)
    if len(df) < 60:
        raise ValueError(f"Not enough daily data for {ticker} ({len(df)} bars)")

    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
    last = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    change_pct = (last / prev - 1.0) * 100.0 if prev else 0.0

    e20 = float(ema(close, 20).iloc[-1])
    e50 = float(ema(close, 50).iloc[-1])
    rsi_v = float(rsi(close).iloc[-1])
    m_line, m_sig, m_hist = macd(close)
    macd_hist = float(m_hist.iloc[-1])
    macd_cross = ("bull" if m_line.iloc[-1] > m_sig.iloc[-1] and macd_hist > 0
                  else "bear" if m_line.iloc[-1] < m_sig.iloc[-1] and macd_hist < 0
                  else "flat")

    avg_vol20 = float(vol.tail(20).mean()) or 1.0
    rel_volume = float(vol.iloc[-1]) / avg_vol20

    # ---- the ask/bid volume proxy + accumulation indicators ----
    buy_vol, sell_vol = buy_sell_volume(df)
    win = min(FLOW_WINDOW, len(df))
    tot = float((buy_vol + sell_vol).tail(win).sum()) or 1.0
    buy_ratio = float(buy_vol.tail(win).sum()) / tot * 100.0

    mfm = ((close - low) - (high - close)) / (high - low).replace(0.0, np.nan)
    mfv = (mfm.fillna(0.0) * vol)
    cmf = float(mfv.tail(CMF_WINDOW).sum() / vol.tail(CMF_WINDOW).sum())
    ad_line = mfv.cumsum()
    obv = (np.sign(close.diff().fillna(0.0)) * vol).cumsum()

    # ---- component sub-scores (each 0-100) ----
    flow_score = _clip01(buy_ratio)                                   # ask/bid pressure
    cmf_score = _clip01((cmf + 0.2) / 0.4 * 100.0)                    # -0.2..+0.2 -> 0..100
    span = min(20, len(df) - 1)
    obv_norm = (float(obv.iloc[-1]) - float(obv.iloc[-1 - span])) / (avg_vol20 * span)
    obv_score = _clip01(50.0 + 50.0 * np.tanh(obv_norm))
    ad_norm = (float(ad_line.iloc[-1]) - float(ad_line.iloc[-1 - span])) / (avg_vol20 * span)
    ad_score = _clip01(50.0 + 50.0 * np.tanh(ad_norm))

    trend_pts = (last > e20) + (last > e50) + (e20 > e50)
    trend_score = trend_pts / 3.0 * 100.0
    trend = "up" if trend_pts == 3 else "down" if trend_pts == 0 else "mixed"

    rsi_score = _clip01((rsi_v - 30.0) / 40.0 * 100.0)
    macd_score = 100.0 if macd_cross == "bull" else 0.0 if macd_cross == "bear" else 50.0
    momentum_score = 0.6 * rsi_score + 0.4 * macd_score

    smart = (0.30 * flow_score + 0.18 * cmf_score + 0.12 * obv_score +
             0.10 * ad_score + 0.15 * trend_score + 0.15 * momentum_score)
    accumulation = 0.40 * flow_score + 0.25 * cmf_score + 0.20 * obv_score + 0.15 * ad_score

    # relative-volume confirmation: conviction grows when the move has volume
    confidence = _clip01(smart * (0.85 + 0.15 * min(rel_volume, 2.0)))

    rec = ("BUY" if smart >= BUY_ABOVE else
           "SELL" if smart <= SELL_BELOW else "HOLD")

    # ---- next-day trading plan (ATR-based, heuristic) ----
    a = float(atr(high, low, close).iloc[-1])
    support = float(low.tail(20).min())
    resistance = float(high.tail(20).max())
    stop_loss = min(support, last - 1.5 * a)
    risk = max(last - stop_loss, 1e-9)
    target1 = last + 1.5 * risk
    target2 = last + 2.5 * risk

    return SmartMoneyResult(
        ticker=ticker, name=name, end=str(close.index[-1].date()),
        last_price=round(last, 4), change_pct=round(change_pct, 2),
        rsi=round(rsi_v, 1), macd_hist=round(macd_hist, 4), macd_cross=macd_cross,
        rel_volume=round(rel_volume, 2), buy_ratio=round(buy_ratio, 1), cmf=round(cmf, 3),
        ema20=round(e20, 4), ema50=round(e50, 4), trend=trend,
        candlestick=_candlestick(df), support=round(support, 4), resistance=round(resistance, 4),
        smart_money_score=round(smart, 1), accumulation_score=round(accumulation, 1),
        confidence=round(confidence, 1), recommendation=rec,
        high_conviction=bool(smart >= HIGH_CONVICTION),
        entry_low=round(last - 0.5 * a, 4), entry_high=round(last, 4),
        breakout=round(resistance * 1.005, 4), stop_loss=round(stop_loss, 4),
        target1=round(target1, 4), target2=round(target2, 4),
        risk_reward=round((target1 - last) / risk, 2),
    )


# ------------------------- market risk meter -------------------------------
@dataclass
class MarketRisk:
    rating: str                 # GREEN / AMBER / RED
    sti_trend: str = "n/a"
    vix: float | None = None
    sp_change: float | None = None
    notes: list = field(default_factory=list)


def _last_change_pct(ticker: str) -> float | None:
    try:
        df = yf.download(ticker, period="5d", interval="1d", auto_adjust=False, progress=False)
        c = df["Close"]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:, 0]
        c = c.dropna()
        return float(c.iloc[-1] / c.iloc[-2] - 1.0) * 100.0
    except Exception:
        return None


def market_risk() -> MarketRisk:
    """Best-effort Green/Amber/Red read on the overall market (degrades gracefully)."""
    risk = 0
    notes: list[str] = []

    # STI trend vs its own 50-day EMA
    sti_trend = "n/a"
    try:
        sti = yf.download("^STI", period="6mo", interval="1d", auto_adjust=False, progress=False)
        c = sti["Close"]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:, 0]
        c = c.dropna()
        above = float(c.iloc[-1]) >= float(ema(c, 50).iloc[-1])
        sti_trend = "above 50EMA" if above else "below 50EMA"
        if not above:
            risk += 1
            notes.append("STI below its 50-day EMA")
    except Exception:
        notes.append("STI trend unavailable")

    # VIX level
    vix = None
    try:
        v = yf.download("^VIX", period="5d", interval="1d", auto_adjust=False, progress=False)
        cv = v["Close"]
        if isinstance(cv, pd.DataFrame):
            cv = cv.iloc[:, 0]
        vix = float(cv.dropna().iloc[-1])
        if vix > 24:
            risk += 2
            notes.append(f"VIX elevated ({vix:.0f})")
        elif vix > 18:
            risk += 1
            notes.append(f"VIX picking up ({vix:.0f})")
    except Exception:
        notes.append("VIX unavailable")

    # US overnight (S&P 500)
    sp = _last_change_pct("^GSPC")
    if sp is not None and sp < -1.0:
        risk += 1
        notes.append(f"S&P 500 fell {sp:.1f}% overnight")

    rating = "GREEN" if risk <= 1 else "AMBER" if risk <= 3 else "RED"
    return MarketRisk(rating=rating, sti_trend=sti_trend, vix=vix, sp_change=sp, notes=notes)


if __name__ == "__main__":
    import json
    res = analyze("O39.SI", name="OCBC")
    print(json.dumps(res.as_dict(), indent=2))
