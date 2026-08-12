"""9 EMA + Fibonacci Pivot breakout — long only, tested properly.

THE RULES AS GIVEN
  * 9-period EMA
  * Pivot Points Standard, Fibonacci type
  * A big GREEN candle closes above BOTH the 9 EMA and the pivot line above it
  * Stop-loss at the pivot line below (the "cross line down")
  * Reward:risk 1:2

FIBONACCI PIVOT FORMULA (from the previous period's High/Low/Close)
  P  = (H + L + C) / 3
  R1 = P + 0.382 x (H - L)      S1 = P - 0.382 x (H - L)
  R2 = P + 0.618 x (H - L)      S2 = P - 0.618 x (H - L)
  R3 = P + 1.000 x (H - L)      S3 = P - 1.000 x (H - L)

Pivots are computed from the PREVIOUS week/month and only applied afterwards,
so no future information leaks in. Both weekly and monthly pivots are tested
because the right choice for daily bars is genuinely ambiguous.

WHAT IS TESTED
  1  the strategy alone
  2  the strategy AND the live app's EMA+Elliott signal agreeing
  3  a random-entry control, run many times, as the significance benchmark

Anything I had to invent (e.g. what counts as a "big" candle) is marked PROXY.

Run:  python pivot_strategy_test.py
      python pivot_strategy_test.py --draws 100 --pivot M
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import exit_matrix as E      # pricing, option sim, cached Elliott mask

EMA_LEN = 9
RR = 2.0
BIG_BODY_MULT = 1.5          # PROXY: "big" = body >= 1.5x the recent average body
LOOKBACK_BODY = 10           # PROXY: how far back "recent" reaches
FIB = (0.382, 0.618, 1.000)


def fib_pivots(df: pd.DataFrame, rule: str = "W") -> pd.DataFrame:
    """Fibonacci pivots from the PREVIOUS period, aligned onto daily bars.

    The shift(1) is what keeps this honest: this week's levels come from last
    week's range, which is all a trader could have known.
    """
    # pandas 3 renamed the month-end alias from 'M' to 'ME'
    freq = {"M": "ME", "W": "W"}.get(rule, rule)
    agg = df.resample(freq).agg({"High": "max", "Low": "min", "Close": "last"}).dropna()
    prev = agg.shift(1)
    p = (prev["High"] + prev["Low"] + prev["Close"]) / 3.0
    rng = prev["High"] - prev["Low"]
    out = pd.DataFrame(index=agg.index)
    out["P"] = p
    for i, f in enumerate(FIB, start=1):
        out[f"R{i}"] = p + f * rng
        out[f"S{i}"] = p - f * rng
    return out.reindex(df.index, method="ffill")


def levels_at(piv_row) -> list[float]:
    vals = [v for v in piv_row.values if np.isfinite(v)]
    return sorted(vals)


def entries_pivot(ticker: str, df: pd.DataFrame, rule: str = "W") -> list[int]:
    """A big green candle closing above BOTH the 9 EMA and the pivot above it."""
    d = df.copy()
    d["ema"] = d["Close"].ewm(span=EMA_LEN, adjust=False).mean()
    body = (d["Close"] - d["Open"]).abs()
    d["avg_body"] = body.rolling(LOOKBACK_BODY).mean()
    piv = fib_pivots(df, rule)

    o, c = d["Open"].values, d["Close"].values
    ema, ab = d["ema"].values, d["avg_body"].values
    out: list[int] = []
    for i in range(max(LOOKBACK_BODY, EMA_LEN) + 2, len(d)):
        if not (c[i] > o[i]):                               # must be green
            continue
        if not np.isfinite(ab[i]) or ab[i] <= 0:
            continue
        if (c[i] - o[i]) < BIG_BODY_MULT * ab[i]:           # must be BIG
            continue
        if not (c[i] > ema[i] and c[i - 1] <= ema[i - 1]):  # fresh break of the EMA
            continue
        lv = levels_at(piv.iloc[i])
        if not lv:
            continue
        # the pivot line it broke: highest level below today's close that
        # yesterday's close was still under
        broken = [x for x in lv if c[i] > x >= c[i - 1]]
        if not broken:
            continue
        out.append(i)
    return out


def stop_for(df: pd.DataFrame, i: int, entry: float, rule: str = "W") -> float | None:
    """The 'cross line down' — nearest Fibonacci pivot below entry."""
    piv = fib_pivots(df, rule)
    below = [x for x in levels_at(piv.iloc[i]) if x < entry]
    return max(below) if below else None


def underlying_result(df: pd.DataFrame, entries: list[int], rule: str) -> list[float]:
    """R-multiples on the STOCK, using the 1:2 rule exactly as specified."""
    high, low, close = df["High"].values, df["Low"].values, df["Close"].values
    piv = fib_pivots(df, rule)
    out: list[float] = []
    n = len(df)
    for i in entries:
        entry = float(close[i])
        below = [x for x in levels_at(piv.iloc[i]) if x < entry]
        if not below:
            continue
        stop = max(below)
        risk = entry - stop
        if risk <= 0:
            continue
        target = entry + RR * risk
        r = 0.0
        for k in range(i + 1, min(i + 120, n)):
            if low[k] <= stop:                 # pessimistic: stop first
                r = -1.0
                break
            if high[k] >= target:
                r = RR
                break
        out.append(r)
    return out


def stats(v):
    if not v:
        return 0.0, 0.0, 0, 0.0
    a = np.array(v, dtype=float)
    sd = a.std(ddof=1) if len(a) > 1 else 0.0
    se = sd / np.sqrt(len(a)) if sd else 0.0
    return (float(a.mean()), float(a.mean() / se) if se else 0.0,
            len(a), 100.0 * float((a > 0).mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--pivot", choices=["W", "M", "both"], default="both")
    args = ap.parse_args()

    E._load_mask_cache()
    frames = {}
    for t in E.B.ALL_TICKERS:
        df = E.B.load(t, 4.0)
        if df is not None and len(df) > 250:
            frames[t] = df
    print(f"9 EMA + Fibonacci pivot breakout — long only, {len(frames)} instruments\n")

    rules = ["W", "M"] if args.pivot == "both" else [args.pivot]
    for rule in rules:
        rname = "WEEKLY" if rule == "W" else "MONTHLY"
        print("=" * 78)
        print(f"  {rname} Fibonacci pivots")
        print("=" * 78)

        ent = {t: entries_pivot(t, df, rule) for t, df in frames.items()}
        total = sum(len(v) for v in ent.values())
        if total == 0:
            print("  No signals at all — the entry conditions never coincided.\n")
            continue

        # --- 1. on the underlying, using the 1:2 rule as specified ---
        r_all = []
        for t, df in frames.items():
            r_all += underlying_result(df, ent[t], rule)
        m, tt, n, win = stats(r_all)
        print(f"\n  A. STOCK, your exact 1:2 rule")
        print(f"     trades {n} | win {win:.1f}% | expectancy {m:+.3f}R | t={tt:+.2f}")
        print(f"     (breakeven needs a {100 / (1 + RR):.0f}% win rate)")

        # --- 2. as an option trade, best-known exit geometry ---
        opt = []
        for t, df in frames.items():
            opt += E.simulate(df, ent[t], -0.60, None)
        m2, t2, n2, w2 = stats(opt)
        print(f"\n  B. OPTION (ATM monthly call, -60% stop, no target)")
        print(f"     trades {n2} | win {w2:.1f}% | avg {100 * m2:+.2f}% | t={t2:+.2f}")

        # --- 3. combined with the live app's signal ---
        comb = {}
        for t, df in frames.items():
            mask = E.ema_elliott_mask(t, df)
            comb[t] = [i for i in ent[t] if i < len(mask) and mask[i]]
        c_opt = []
        for t, df in frames.items():
            c_opt += E.simulate(df, comb[t], -0.60, None)
        m3, t3, n3, w3 = stats(c_opt)
        print(f"\n  C. COMBINED with EMA+Elliott")
        print(f"     trades {n3} | win {w3:.1f}% | avg {100 * m3:+.2f}% | t={t3:+.2f}"
              + ("   <-- too few to judge" if n3 < 100 else ""))

        # --- 4. permutation control ---
        print(f"\n  D. Is B better than random? ({args.draws} random runs)")
        rng = np.random.default_rng(7)
        dist = np.empty(args.draws)
        for d_i in range(args.draws):
            vals = []
            for t, df in frames.items():
                k = len(ent[t])
                if k <= 0:
                    continue
                idx = sorted(int(x) for x in rng.integers(121, len(df) - 35, size=k))
                vals += E.simulate(df, idx, -0.60, None)
            dist[d_i] = 100 * float(np.mean(vals)) if vals else 0.0
        sig = 100 * m2
        pct = 100.0 * float((dist < sig).mean())
        lo, hi = np.percentile(dist, [5, 95])
        print(f"     signal {sig:+.2f}%  vs random avg {dist.mean():+.2f}% "
              f"[5-95%: {lo:+.1f}, {hi:+.1f}]")
        print(f"     percentile {pct:.0f}%  ->  "
              + ("BEATS random" if pct > 95 else
                 "WORSE than random" if pct < 5 else "no evidence of edge"))
        print()

    print("Reading this: section A is your rule exactly as described, judged on the")
    print("stock. Section B converts it to an option. Section D is the only one that")
    print("decides anything — a positive average that sits inside the random range")
    print("is a rising market, not a strategy.")


if __name__ == "__main__":
    main()
