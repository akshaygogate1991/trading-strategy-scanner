"""Check tipped stocks against our own rules - facts only, not a buy / sell call.

For each NSE symbol:
  momentum rule   in the 200 most traded? above its 200-day average? 12-1 month return and its
                  rank among the eligible stocks (the paper portfolio buys the top 20)
  trend / risk    1, 3, 6 month returns, distance from the 52-week high, yearly volatility,
                  worst fall in the last 3 years, median daily traded value
  the tip's claim "big returns in 1-1.5 months": over the stock's history, how often a
                  30-trading-day hold gained 15 %+, and how often it lost 15 %+

Universe ranks come from swing_data/prices_all.pkl (the momentum data); each stock's own
prices are fresh from Yahoo.

Run: python stock_check.py HFCL RBZJEWEL NCC SHANKESH NELCAST   -> logs/stock_check.txt
"""
from __future__ import annotations

import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
UNIVERSE, N = 200, 20


def universe_ranks():
    """12-1 month return ranks among the 200 most traded stocks above their 200-day average, as of the last day."""
    with open(HERE / "swing_data" / "prices_all.pkl", "rb") as f:
        prices = pickle.load(f)
    prices.pop("^NSEI", None)
    close = pd.DataFrame({t: d["Close"] for t, d in prices.items()}).sort_index().ffill(limit=5)
    value = pd.DataFrame({t: d["Close"] * d["Volume"] for t, d in prices.items()}).reindex(close.index)
    day = close.index[-1]
    liq = value.rolling(126, min_periods=100).median().loc[day].dropna()
    top = liq.nlargest(UNIVERSE).index
    sma200 = close.rolling(200, min_periods=200).mean().loc[day]
    r = (close.shift(21) / close.shift(252) - 1).loc[day]
    ok = [t for t in top if close.loc[day, t] > sma200.get(t, np.inf)]
    ranked = r[ok].dropna().sort_values(ascending=False)
    return day, liq.rank(ascending=False), ranked, len(liq)


def fetch(sym: str) -> pd.DataFrame:
    import yfinance as yf
    d = yf.download(sym + ".NS", period="max", interval="1d", auto_adjust=True, progress=False)
    d.columns = d.columns.get_level_values(0)
    return d.dropna(subset=["Close"])


def check(sym, day, liq_rank, ranked, n_stocks) -> dict:
    d = fetch(sym)
    c = d["Close"]
    last = c.iloc[-1]
    ret = lambda n: (last / c.iloc[-1 - n] - 1) * 100 if len(c) > n else float("nan")
    sma200 = c.rolling(200).mean().iloc[-1] if len(c) >= 200 else float("nan")
    r12_1 = (c.iloc[-22] / c.iloc[-253] - 1) * 100 if len(c) > 253 else float("nan")
    tv = (d["Close"] * d["Volume"]).tail(126).median() / 1e7
    three = c.tail(756)
    dd3 = ((three / three.cummax()) - 1).min() * 100
    fwd = (c.shift(-30) / c - 1).dropna()
    t = sym + ".NS"
    rank = ranked.index.get_loc(t) + 1 if t in ranked.index else None
    return {"stock": sym, "price": round(last, 2), "since": c.index[0].date(), "days": len(c),
            "1m %": ret(21), "3m %": ret(63), "6m %": ret(126), "12-1m %": r12_1,
            "above 200d": (last > sma200) if not math.isnan(sma200) else "too new",
            "off 52w high %": (last / c.tail(252).max() - 1) * 100,
            "vol %/yr": c.pct_change().tail(252).std() * math.sqrt(252) * 100,
            "worst fall 3y %": dd3, "traded Rs cr/day": tv,
            "liquidity rank": f"{int(liq_rank[t])} of {n_stocks}" if t in liq_rank.index else "not in data",
            "momentum rank": f"{rank} of {len(ranked)}" if rank else "not eligible",
            "30d +15% %": (fwd >= 0.15).mean() * 100 if len(fwd) > 100 else float("nan"),
            "30d -15% %": (fwd <= -0.15).mean() * 100 if len(fwd) > 100 else float("nan"),
            "30d median %": fwd.median() * 100 if len(fwd) > 100 else float("nan")}


def main() -> None:
    syms = [s.upper() for s in sys.argv[1:]] or ["HFCL", "RBZJEWEL", "NCC", "SHANKESH", "NELCAST"]
    day, liq_rank, ranked, n = universe_ranks()
    rows = [check(s, day, liq_rank, ranked, n) for s in syms]
    t = pd.DataFrame(rows).set_index("stock").T
    lines = [f"Momentum ranks as of {day.date()} (top {UNIVERSE} most traded, above 200-day average: "
             f"{len(ranked)} eligible; the paper portfolio holds the top {N}). Stock prices: latest Yahoo close.",
             t.to_string(),
             "",
             "30d +15% / -15% = share of all past 30-trading-day holds that gained 15 %+ / lost 15 %+.",
             "Facts from past prices only - not a recommendation."]
    text = "\n".join(lines)
    print(text)
    (HERE / "logs").mkdir(exist_ok=True)
    (HERE / "logs" / "stock_check.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
