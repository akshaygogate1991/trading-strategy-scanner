"""Do these stocks really move in cycles (up, back down, up again)?

Any stock price wiggles, even a purely random one, so "it goes up and comes
back" is only useful if the swings come back MORE than chance. Three checks per
stock on daily prices (Yahoo, up to 7 years):

  swings      completed moves of 20%+ in either direction: how many per year,
              their typical size and length (what the eye sees on a chart)
  VR(10)      variance ratio of 10-day vs 1-day returns. Below 1 = moves tend to
              reverse (cyclic / mean-reverting), above 1 = moves tend to continue
              (trending). z beyond -2 / +2 means it is unlikely to be chance.
  band trade  buy at the next open after a close below the 20-day average minus
              2 standard deviations; sell at the next open after a close back
              above the 20-day average, or after 30 trading days. Costs: 0.2%
              per side plus slippage of 0.3% per side (0.6% under Rs 50 a share,
              where one tick is a big step). Compared with entering on random days
              and holding just as long (1,000 tries): 'beats random' is the
              share of those tries the real entries beat.

Run:  python cycle_check.py
      python cycle_check.py RCF.NS HINDCOPPER.NS
"""
from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd
import yfinance as yf

WATCHLIST = ["20MICRONS.NS", "ADSL.NS", "PVVINFRA.BO", "RCF.NS", "RPOWER.NS", "GNRL.NS",
             "RELINFRA.NS", "HINDCOPPER.NS", "CUBEXTUB.NS", "SAMMAANCAP.NS", "AEQUS.NS",
             "GRMOVER.NS", "SNOWMAN.NS", "GUJENERGY.NS"]
SWING = 0.20
COST = 0.002


def swings(close: pd.Series, pct: float = SWING) -> list[tuple[str, float, int]]:
    """Zig-zag: completed legs of at least pct, as (direction, size, trading days)."""
    legs, piv_i, piv_p, direction = [], 0, close.iloc[0], 0
    ext_i, ext_p = 0, close.iloc[0]
    for i, p in enumerate(close.values):
        if direction >= 0 and p > ext_p or direction <= 0 and p < ext_p:
            if direction == 0 or (direction > 0 and p > ext_p) or (direction < 0 and p < ext_p):
                ext_i, ext_p = i, p
        if direction == 0:
            if p >= piv_p * (1 + pct):
                direction, ext_i, ext_p = 1, i, p
            elif p <= piv_p * (1 - pct):
                direction, ext_i, ext_p = -1, i, p
            continue
        if direction > 0 and p <= ext_p * (1 - pct):
            legs.append(("up", ext_p / piv_p - 1, ext_i - piv_i))
            piv_i, piv_p, direction, ext_i, ext_p = ext_i, ext_p, -1, i, p
        elif direction < 0 and p >= ext_p * (1 + pct):
            legs.append(("down", ext_p / piv_p - 1, ext_i - piv_i))
            piv_i, piv_p, direction, ext_i, ext_p = ext_i, ext_p, 1, i, p
    return legs


def variance_ratio(close: pd.Series, q: int = 10) -> tuple[float, float]:
    """Lo-MacKinlay variance ratio with the heteroskedasticity-robust z."""
    r = np.log(close).diff().dropna().values
    n, mu = len(r), r.mean()
    var1 = ((r - mu) ** 2).sum() / (n - 1)
    rq = np.convolve(r, np.ones(q), "valid")
    m = q * (n - q + 1) * (1 - q / n)
    varq = ((rq - q * mu) ** 2).sum() / m
    vr = varq / var1
    d2 = (r - mu) ** 2
    theta = sum((2 * (q - j) / q) ** 2 * (d2[j:] * d2[:-j]).sum() / (d2.sum() ** 2) for j in range(1, q))
    return vr, (vr - 1) / math.sqrt(theta)


def band_trades(df: pd.DataFrame, side_cost: float, rng) -> dict:
    c, o = df["Close"], df["Open"]
    mid = c.rolling(20).mean()
    low = mid - 2 * c.rolling(20).std()
    trades, i, n = [], 20, len(df)
    while i < n - 1:
        if c.iloc[i] < low.iloc[i]:
            entry = i + 1
            j = entry
            while j < n - 1 and j - entry < 30 and not c.iloc[j] > mid.iloc[j]:
                j += 1
            exit_ = min(j + 1, n - 1)
            trades.append((entry, exit_, o.iloc[exit_] / o.iloc[entry] - 1 - 2 * side_cost))
            i = exit_
        else:
            i += 1
    if not trades:
        return {"trades": 0}
    rets = np.array([t[2] for t in trades])
    holds = [t[1] - t[0] for t in trades]
    # random entries with the same holding periods
    beat = []
    for _ in range(1000):
        rr = []
        for h in holds:
            s = rng.integers(20, n - 1 - h)
            rr.append(o.iloc[s + h] / o.iloc[s] - 1 - 2 * side_cost)
        beat.append(np.mean(rr))
    half = len(rets) // 2
    return {"trades": len(rets), "win %": (rets > 0).mean() * 100, "avg %": rets.mean() * 100,
            "total %": (np.prod(1 + rets) - 1) * 100, "avg days": np.mean(holds),
            "beats random %": (np.array(beat) < rets.mean()).mean() * 100,
            "avg % 1st half": rets[:half].mean() * 100 if half else float("nan"),
            "avg % 2nd half": rets[half:].mean() * 100 if half else float("nan")}


def main() -> None:
    tickers = sys.argv[1:] or WATCHLIST
    raw = yf.download(tickers, period="7y", auto_adjust=True, group_by="ticker", progress=False)
    rng = np.random.default_rng(3)
    rows = []
    for t in tickers:
        df = raw[t].dropna(subset=["Close"]) if t in raw.columns.get_level_values(0) else pd.DataFrame()
        if len(df) < 120:
            rows.append({"stock": t, "note": f"only {len(df)} days of data"})
            continue
        c = df["Close"]
        years = len(df) / 248
        legs = swings(c)
        ups = [l for l in legs if l[0] == "up"]
        downs = [l for l in legs if l[0] == "down"]
        vr, z = variance_ratio(c)
        price = float(c.iloc[-1])
        side = COST + (0.006 if price < 50 else 0.003)
        b = band_trades(df, side, rng)
        rows.append({
            "stock": t.split(".")[0], "years": round(years, 1), "price": round(price, 2),
            "buy&hold %/yr": round(((c.iloc[-1] / c.iloc[0]) ** (1 / years) - 1) * 100, 1),
            "max fall %": round((c / c.cummax() - 1).min() * 100, 0),
            "20% swings/yr": round(len(legs) / years, 1),
            "typ. up %": round(np.median([l[1] for l in ups]) * 100, 0) if ups else None,
            "typ. down %": round(np.median([l[1] for l in downs]) * 100, 0) if downs else None,
            "days/leg": round(np.median([l[2] for l in legs]), 0) if legs else None,
            "VR(10)": round(vr, 2), "z": round(z, 2),
            "band trades": b["trades"], "win %": round(b.get("win %", float("nan")), 0),
            "avg trade %": round(b.get("avg %", float("nan")), 2),
            "beats random %": round(b.get("beats random %", float("nan")), 0),
            "1st half %": round(b.get("avg % 1st half", float("nan")), 2),
            "2nd half %": round(b.get("avg % 2nd half", float("nan")), 2),
        })
    out = pd.DataFrame(rows).set_index("stock")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(out.to_string())
    out.to_csv("logs/cycle_check.csv")


if __name__ == "__main__":
    main()
