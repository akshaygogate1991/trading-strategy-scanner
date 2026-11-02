"""9 EMA + Fibonacci pivot breakout, INTRADAY (Instagram reel, dropoutengineers).

THE RULES AS GIVEN
  * EMA 9 and Pivot Points Standard, type Fibonacci (daily on an intraday chart)
  * BUY when a big green candle breaks the pivot line and the EMA together
  * SELL when a big red candle breaks them downwards
  * Stop-loss below the pivot line and the EMA (above them for a sell)
  * Reward:risk 1:2 or 1:3

HOW IT IS READ (PROXY where the reel is vague)
  * Daily Fibonacci levels from the previous day: P, R1-R3, S1-S3.
  * "Breaks both": the candle opens below and closes above the EMA AND at least
    one pivot level (mirror for red).
  * "Big": body >= 1.5x the average body of the last 10 candles (PROXY).
    Also tested without the size condition.
  * Entry at the next candle's open, stop at the lower of the broken level and
    the EMA (the higher of the two for a sell).
  * 5 and 15 minute candles; entries until 15:00, flat at 15:20.
  * Fills and exits on 1-minute bars; stop assumed first if both are hit in
    the same minute.  Costs 0.015 % per side (index futures).
  * Control: random candles, random side, stop distance copied from a random
    real trade, same 1:R target.

Run:  python pivot_ema_intraday_test.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import bb_alert_test as B

HERE = os.path.dirname(os.path.abspath(__file__))
COST = 0.00015
RNG = np.random.default_rng(5)
RANDOM_RUNS = 50


def setup(m1, minutes):
    c = m1.resample(f"{minutes}min", origin="start_day", offset="15min",
                    label="left", closed="left").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    daily = m1.groupby(m1.index.normalize()).agg(H=("High", "max"), L=("Low", "min"), C=("Close", "last")).shift(1)
    p = (daily.H + daily.L + daily.C) / 3
    rng = daily.H - daily.L
    lv = pd.DataFrame({"P": p, "R1": p + .382 * rng, "R2": p + .618 * rng, "R3": p + rng,
                       "S1": p - .382 * rng, "S2": p - .618 * rng, "S3": p - rng})
    lv = lv.reindex(c.index.normalize()).to_numpy()
    c["ema"] = c["Close"].ewm(span=9, adjust=False).mean()
    c["body"] = (c["Close"] - c["Open"]).abs()
    c["avg_body"] = c["body"].shift(1).rolling(10).mean()
    return c, lv


def signals(c, lv, big):
    o, cl, e = c["Open"].to_numpy(), c["Close"].to_numpy(), c["ema"].to_numpy()
    size_ok = (c["body"] >= 1.5 * c["avg_body"]).to_numpy() if big else np.ones(len(c), bool)
    side = np.zeros(len(c), int)
    stop = np.full(len(c), np.nan)
    for i in range(len(c)):
        L = lv[i]
        if not size_ok[i] or not np.isfinite(L).all():
            continue
        if cl[i] > o[i] and o[i] < e[i] < cl[i]:
            hit = L[(L > o[i]) & (L < cl[i])]
            if len(hit):
                side[i], stop[i] = 1, min(hit.min(), e[i])
        elif cl[i] < o[i] and cl[i] < e[i] < o[i]:
            hit = L[(L < o[i]) & (L > cl[i])]
            if len(hit):
                side[i], stop[i] = -1, max(hit.max(), e[i])
    return side, stop


def run(M, c, side, stop, rr, risk_pct=None):
    """risk_pct given -> stop placed at that % from entry (random control)."""
    rows, busy = [], -1
    for i in np.flatnonzero(side):
        j = i + 1
        if j >= len(M.c_day) or M.c_day[j] != M.c_day[i] or M.c_hhmm[j] > B.LAST_ENTRY:
            continue
        k = M.c_start[j]
        if k <= busy:
            continue
        s = side[i]
        entry = M.o[k]
        sl = entry * (1 - s * risk_pct[i]) if risk_pct is not None else stop[i]
        risk = s * (entry - sl)
        if risk <= 0:
            continue
        tgt = entry + s * rr * risk
        last, px, q = M.last[k], None, k
        while q <= last:
            if s == 1:
                if M.l[q] <= sl:
                    px = min(sl, M.o[q]) if q > k else sl
                elif M.h[q] >= tgt:
                    px = tgt
            else:
                if M.h[q] >= sl:
                    px = max(sl, M.o[q]) if q > k else sl
                elif M.l[q] <= tgt:
                    px = tgt
            if px is not None:
                break
            q += 1
        if px is None:
            q, px = last, M.cl[last]
        busy = q
        rows.append(dict(day=M.t[k], R=s * (px - entry) / risk,
                         net=s * (px - entry) / entry - 2 * COST, risk_pct=risk / entry))
    return pd.DataFrame(rows)


def line(name, r, rand=None):
    if r.empty:
        return f"{name:<40} no trades"
    net = r.net * 100
    daily = net.groupby(pd.to_datetime(r.day).dt.normalize()).sum()
    t = daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily)))
    h = len(r) // 2
    s = (f"{name:<40} n={len(r):>5}  win={(r.R > 0).mean() * 100:4.1f}%  avgR={r.R.mean():+.2f}  "
         f"net/trade={net.mean():+.3f}%  total={net.sum():+7.1f}%  t={t:+.2f}  "
         f"halves {net.iloc[:h].sum():+.1f}/{net.iloc[h:].sum():+.1f}  stop={r.risk_pct.mean() * 100:.2f}%")
    if rand is not None:
        s += f"\n{'':<40} random entries same stop size/target: avg total {rand.mean():+.1f}%  share >= real {(rand >= net.sum()).mean() * 100:.0f}%"
    return s


def main():
    lines = []
    out = lambda s="": (print(s, flush=True), lines.append(s))
    pooled = {}
    for sym in ("NIFTY", "BANKNIFTY", "SENSEX"):
        m1 = B.clean(B.load_index(sym))
        out(f"\n=== {sym}  {m1.index[0].date()} .. {m1.index[-1].date()}")
        for minutes in (5, 15):
            c, lv = setup(m1, minutes)
            c2 = c.copy()
            c2["up"] = c2["lo"] = np.nan
            M = B.Minutes(m1, c2)
            for big in (True, False):
                side, stop = signals(c, lv, big)
                for rr in (2.0, 3.0):
                    r = run(M, c, side, stop, rr)
                    rand = None
                    if big and not r.empty:
                        tots = []
                        n = int((side != 0).sum())
                        for _ in range(RANDOM_RUNS):
                            rs = np.zeros(len(c), int)
                            pos = RNG.choice(len(c) - 1, size=n, replace=False)
                            rs[pos] = RNG.choice([-1, 1], size=n)
                            rp = np.zeros(len(c))
                            rp[pos] = RNG.choice(r.risk_pct.to_numpy(), size=n)
                            rr_ = run(M, c, rs, None, rr, rp)
                            tots.append(rr_.net.sum() * 100 if len(rr_) else 0)
                        rand = np.array(tots)
                    name = f"{minutes}m {'big candle' if big else 'any size '} 1:{rr:.0f}"
                    out(line(name, r, rand))
                    pooled.setdefault(name, []).append(r)
    out("\n=== POOLED (3 indices)")
    for name, rs in pooled.items():
        out(line(name, pd.concat(rs, ignore_index=True)))
    with open(os.path.join(HERE, "logs", "pivot_ema_intraday.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
