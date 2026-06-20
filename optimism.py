"""
Optimism Chart engine.

Methodology
-----------
Work in log-price space:  y = ln(price),  x = time in years from the series start.
A stock that compounds at a steady rate is a straight line in this space, so the
trend channel lines are straight lines (they render straight on a log-price axis).

  * 100% optimism line = straight line fit through the PEAKS  (upper envelope)
  *   0% optimism line = straight line fit through the TROUGHS (lower envelope)
  *  50% optimism line = midway between the 0% and 100% lines  (in log space)
  *  75% optimism line = midway between the 50% and 100% lines
  *  25% optimism line = midway between the 0% and 50% lines

The envelope lines are found by *iterative envelope regression*: fit OLS, keep only
the points above (resp. below) the line, refit, repeat. This converges to a straight
line resting on the peaks (resp. troughs) without hand-picking points.

Optimism of the latest price = where it sits inside the channel, 0% at the trough
line, 100% at the peak line. Cheap (low optimism) -> BUY, expensive (high) -> SELL.

Note on "log-log": time is plotted linearly (semi-log). Exponential growth is a
straight line on a log-PRICE axis vs linear time; a log time axis would only
straighten power-law growth, which is not the optimism-chart model.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import yfinance as yf

import plotly.graph_objects as go


# ---- thresholds for the recommendation -----------------------------------
BUY_BELOW = 25.0    # optimism <= 25%  -> undervalued -> BUY
SELL_ABOVE = 75.0   # optimism >= 75%  -> overvalued  -> SELL


@dataclass
class ChannelLine:
    slope: float       # in log-price per year
    intercept: float   # log-price at x = 0

    def at(self, x: np.ndarray | float) -> np.ndarray | float:
        return self.slope * np.asarray(x, dtype=float) + self.intercept

    def price_at(self, x):
        return np.exp(self.at(x))


@dataclass
class OptimismResult:
    ticker: str
    name: str
    start: str
    end: str
    last_price: float
    optimism: float            # 0..100 (can exceed if price breaks the channel)
    recommendation: str        # BUY / HOLD / SELL
    line_price_0: float        # trough-line price at the latest date
    line_price_25: float
    line_price_50: float
    line_price_75: float
    line_price_100: float      # peak-line price at the latest date

    def as_dict(self):
        return asdict(self)


def fit_envelope(x: np.ndarray, y: np.ndarray, upper: bool, iters: int = 12) -> ChannelLine:
    """Iterative envelope regression -> straight line resting on peaks/troughs."""
    slope, intercept = np.polyfit(x, y, 1)
    for _ in range(iters):
        resid = y - (slope * x + intercept)
        mask = resid >= 0 if upper else resid <= 0
        if mask.sum() < 2:
            break
        new_slope, new_intercept = np.polyfit(x[mask], y[mask], 1)
        if np.isclose(new_slope, slope) and np.isclose(new_intercept, intercept):
            slope, intercept = new_slope, new_intercept
            break
        slope, intercept = new_slope, new_intercept
    return ChannelLine(float(slope), float(intercept))


def fetch_prices(ticker: str, period: str = "10y", interval: str = "1wk") -> pd.Series:
    """Return a clean Series of nominal close prices indexed by date."""
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):       # multiindex columns
        close = close.iloc[:, 0]
    return close.dropna()


@dataclass
class Channel:
    """The fitted log-linear channel for one stock (the expensive, slow-moving part)."""
    ticker: str
    name: str
    start: pd.Timestamp          # x = 0 reference date
    lines: dict                  # pct -> ChannelLine
    prices: pd.Series            # historical close series (for plotting)
    x: np.ndarray                # years-from-start for each price point


def fit_channel(ticker: str, name: str | None = None,
                period: str = "10y", interval: str = "1wk",
                prices: pd.Series | None = None) -> Channel:
    """Fit the optimism channel. This is the part that needs full 10y history."""
    if prices is None:
        prices = fetch_prices(ticker, period, interval)
    if len(prices) < 20:
        raise ValueError(f"Not enough data for {ticker} ({len(prices)} points)")

    dates = prices.index
    x = (dates - dates[0]).days.to_numpy(dtype=float) / 365.25
    y = np.log(prices.to_numpy(dtype=float))

    line100 = fit_envelope(x, y, upper=True)
    line0 = fit_envelope(x, y, upper=False)
    line50 = ChannelLine((line0.slope + line100.slope) / 2,
                         (line0.intercept + line100.intercept) / 2)
    line75 = ChannelLine((line50.slope + line100.slope) / 2,
                         (line50.intercept + line100.intercept) / 2)
    line25 = ChannelLine((line0.slope + line50.slope) / 2,
                         (line0.intercept + line50.intercept) / 2)

    return Channel(
        ticker=ticker, name=name or ticker, start=dates[0],
        lines={0: line0, 25: line25, 50: line50, 75: line75, 100: line100},
        prices=prices, x=x,
    )


def evaluate(channel: Channel, last_price: float,
             last_date: pd.Timestamp | None = None) -> OptimismResult:
    """Cheap step: place a (possibly fresh intraday) price inside the channel."""
    if last_date is None:
        last_date = channel.prices.index[-1]
    x_now = (last_date - channel.start).days / 365.25
    y_now = float(np.log(last_price))

    line0, line100 = channel.lines[0], channel.lines[100]
    y0, y100 = float(line0.at(x_now)), float(line100.at(x_now))
    span = y100 - y0
    optimism = 100.0 * (y_now - y0) / span if span > 1e-9 else 50.0

    if optimism <= BUY_BELOW:
        rec = "BUY"
    elif optimism >= SELL_ABOVE:
        rec = "SELL"
    else:
        rec = "HOLD"

    return OptimismResult(
        ticker=channel.ticker, name=channel.name,
        start=str(channel.start.date()), end=str(pd.Timestamp(last_date).date()),
        last_price=round(float(last_price), 4),
        optimism=round(float(optimism), 1), recommendation=rec,
        line_price_0=round(float(line0.price_at(x_now)), 4),
        line_price_25=round(float(channel.lines[25].price_at(x_now)), 4),
        line_price_50=round(float(channel.lines[50].price_at(x_now)), 4),
        line_price_75=round(float(channel.lines[75].price_at(x_now)), 4),
        line_price_100=round(float(line100.price_at(x_now)), 4),
    )


def latest_quote(ticker: str) -> tuple[float, pd.Timestamp]:
    """Cheap fetch of just the most recent close (for the 15-min price refresh)."""
    df = yf.download(ticker, period="5d", interval="1d",
                     auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No recent quote for {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    return float(close.iloc[-1]), close.index[-1]


def compute(ticker: str, name: str | None = None,
            period: str = "10y", interval: str = "1wk",
            prices: pd.Series | None = None) -> tuple[OptimismResult, Channel]:
    """Convenience: fit channel and evaluate at the latest historical close."""
    channel = fit_channel(ticker, name, period, interval, prices)
    result = evaluate(channel, float(channel.prices.iloc[-1]), channel.prices.index[-1])
    return result, channel


_LINE_STYLE = {
    100: ("#c0392b", "100% Optimism (peaks)", "solid", 2.0),
    75:  ("#e67e22", "75% Optimism", "dash", 1.2),
    50:  ("#7f8c8d", "50% Optimism (fair value)", "solid", 1.4),
    25:  ("#27ae60", "25% Optimism", "dash", 1.2),
    0:   ("#1e8449", "0% Optimism (troughs)", "solid", 2.0),
}


def make_chart(result: OptimismResult, channel: Channel) -> str:
    """Interactive Plotly optimism chart, returned as an embeddable HTML fragment.

    Close price and the five channel lines on a log-price axis. Hover for a
    unified readout, drag to zoom, click legend entries to toggle lines.
    """
    prices = channel.prices
    x = channel.x
    dates = prices.index
    lines = channel.lines
    rec_color = {"BUY": "#27ae60", "SELL": "#c0392b", "HOLD": "#f39c12"}[result.recommendation]

    fig = go.Figure()
    for pct in (100, 75, 50, 25, 0):
        color, lbl, dash, w = _LINE_STYLE[pct]
        fig.add_trace(go.Scatter(
            x=dates, y=lines[pct].price_at(x), name=lbl,
            line=dict(color=color, width=w, dash=dash),
            hovertemplate="%{y:.4g}<extra>" + lbl + "</extra>"))
    fig.add_trace(go.Scatter(
        x=dates, y=prices.to_numpy(dtype=float), name="Close price",
        line=dict(color="#2c3e50", width=1.4),
        hovertemplate="%{y:.4g}<extra>Close</extra>"))
    fig.add_trace(go.Scatter(
        x=[pd.Timestamp(result.end)], y=[result.last_price], mode="markers",
        name=f"Now {result.last_price:g} ({result.optimism:.0f}%)",
        marker=dict(size=12, color=rec_color, line=dict(color="white", width=1.5)),
        hovertemplate="%{y:.4g}<extra>Now</extra>"))

    fig.update_yaxes(type="log", title="Price (log scale)")
    fig.update_xaxes(title="Year")
    fig.update_layout(
        hovermode="x unified", template="plotly_white", height=560,
        margin=dict(l=55, r=20, t=60, b=40),
        legend=dict(orientation="h", y=1.04, x=0, font=dict(size=10)),
        title=dict(text=(f"{result.name} ({result.ticker})  —  Optimism "
                         f"{result.optimism:.0f}%  →  {result.recommendation}  ·  log price, 10y"),
                   font=dict(size=14, color=rec_color)))

    return fig.to_html(full_html=False, include_plotlyjs="cdn",
                       config=dict(displaylogo=False, responsive=True, scrollZoom=True),
                       default_height="560px")


if __name__ == "__main__":
    res, channel = compute("O39.SI", name="OCBC")
    with open("ocbc_optimism.html", "w") as f:
        f.write("<!doctype html><meta charset='utf-8'>" + make_chart(res, channel))
    import json
    print(json.dumps(res.as_dict(), indent=2))
    print("saved ocbc_optimism.html")
