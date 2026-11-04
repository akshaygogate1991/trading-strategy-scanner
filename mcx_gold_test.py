"""Daily trend strategies for MCX gold futures, tested on gold in rupees (COMEX x USD/INR).

MCX gold follows gold in rupees plus India's import duty, so this proxy misses duty-change
jumps (e.g. the Jul 2024 Budget cut, about -6 % in a day on MCX). Futures let you go short and
use leverage; results here are on 1x notional - leverage multiplies the return and the falls.

Rules (signal at a close, position from the next day, 0.05 % per side for fees + slippage):
  hold long                      always long
  200-day avg long/flat | long/short
  50/200 cross long/short
  20-day breakout long/short     go long on a 20-day high close, short on a 20-day low close
  55/20 breakout long/flat       turtle-style: enter on a 55-day high, exit on a 20-day low
  12-month return long/short     month-end signal
Same checks as before: before / from 2015, t of the monthly difference vs holding long.

Run: python mcx_gold_test.py   -> logs/mcx_gold.txt
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from gold_test import gold

HERE = Path(__file__).parent
COST = 0.0005
SPLIT = pd.Timestamp("2015-01-01")


def breakout(px, n_in, n_out, short):
    hi_in, lo_in = px.rolling(n_in).max(), px.rolling(n_in).min()
    lo_out, hi_out = px.rolling(n_out).min(), px.rolling(n_out).max()
    pos, out = 0.0, []
    for p, a, b, c, d in zip(px, hi_in, lo_in, lo_out, hi_out):
        if pos <= 0 and p >= a:
            pos = 1.0
        elif pos > 0 and p <= c:
            pos = -1.0 if short and p <= b else 0.0
        elif pos == 0 and short and p <= b:
            pos = -1.0
        elif pos < 0 and p >= d:
            pos = 1.0 if p >= a else 0.0
        out.append(pos)
    return pd.Series(out, index=px.index)


def stats(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    cagr = lambda x: ((1 + x).prod() ** (252 / len(x)) - 1) * 100
    return {"CAGR %": cagr(r), "MaxDD %": (eq / eq.cummax() - 1).min() * 100,
            "Sharpe": r.mean() / r.std() * math.sqrt(252),
            "CAGR to 2014": cagr(r[r.index < SPLIT]), "CAGR 2015+": cagr(r[r.index >= SPLIT]),
            "worst yr %": ((1 + r).groupby(r.index.year).prod() - 1).min() * 100}


def main() -> None:
    px = gold()["inr"]
    ret = px.pct_change().fillna(0)
    sma50, sma200 = px.rolling(50).mean(), px.rolling(200).mean()
    me = px.groupby(px.index.to_period("M")).tail(1)
    m12 = np.sign(me / me.shift(12) - 1).reindex(px.index).ffill()
    sig = {"hold long": pd.Series(1.0, index=px.index),
           "200-day long/flat": (px > sma200).astype(float),
           "200-day long/short": np.sign(px - sma200),
           "50/200 cross long/short": np.sign(sma50 - sma200),
           "20-day breakout long/short": breakout(px, 20, 20, True),
           "55/20 breakout long/flat": breakout(px, 55, 20, False),
           "12-month return long/short": m12}
    start = px.index[260]
    rows, base = [], None
    for name, s in sig.items():
        pos = s.fillna(0).shift(1).fillna(0)
        r = (pos * ret - pos.diff().abs().fillna(0) * COST)[start:]
        m = (1 + r).groupby(r.index.to_period("M")).prod() - 1
        base = m if base is None else base
        d = m - base
        rows.append({"rule": name, **stats(r), "% days short": (pos[start:] < 0).mean() * 100,
                     "trades/yr": pos[start:].diff().abs().gt(0).sum() / (len(r) / 252),
                     "t vs hold": d.mean() / (d.std() / math.sqrt(len(d))) if d.std() > 0 else float("nan")})
    t = pd.DataFrame(rows).set_index("rule").round(2)
    text = "\n".join([f"Gold in rupees, daily, {start.date()} to {px.index[-1].date()} (1x notional, "
                      f"{COST * 100:.2f} % per side)", t.to_string(), "",
                      "Facts from past prices only; MCX duty jumps not included. t = monthly difference vs hold long."])
    print(text)
    (HERE / "logs" / "mcx_gold.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
