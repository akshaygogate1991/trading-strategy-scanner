"""Volatility rank — buy options when they are CHEAP, not when a pattern appears.

A completely different mechanism from everything tested so far. The six previous
strategies all tried to predict DIRECTION from the chart, and all landed inside
the random range. This one ignores direction and asks whether the option is
cheap relative to its own history.

THE IDEA
Volatility mean-reverts. When a stock has been unusually quiet, options priced
off that quiet are cheap; when volatility returns to normal, those options gain
value even before the stock moves anywhere. That is a pricing edge rather than a
forecasting edge - a different thing entirely.

HONEST LIMITATION
We have no historical IMPLIED volatility, so realised (20-day) volatility is
used as the proxy, ranked against its own trailing year. Real IV differs: it
carries an expectations premium and collapses after events. So treat this as a
test of the MECHANISM, not a precise simulation of what you would have paid.

WHAT IS TESTED
  A  CALL  bought at LOW vol rank      (cheap options, plus direction risk)
  B  CALL  bought at HIGH vol rank     (expensive options - should be worse)
  C  STRADDLE at LOW vol rank          (call+put: PURE volatility, no direction)
  D  STRADDLE at HIGH vol rank
  E  random entry control, run many times

If the idea works, A beats B and C beats D - and C is the cleaner evidence,
because a straddle removes direction from the question entirely.

Run:  python vol_rank_test.py
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import pandas as pd

import exit_matrix as E

VOL_WIN = 20            # days used to measure realised volatility
RANK_WIN = 252          # rank it against the trailing year
LOW_RANK, HIGH_RANK = 25.0, 75.0

# VARIANCE RISK PREMIUM — the correction that makes this test honest.
#
# You do not buy options at realised volatility. You buy at IMPLIED volatility,
# which is systematically HIGHER: sellers demand payment for carrying the risk.
# Across equity markets the gap typically runs 10-30% of the volatility level.
#
# Without this, the first version of this test paid random straddles +31% per
# month - a money printer that does not exist. That absurd control number was
# the giveaway that the model was handing out options at a price nobody sells
# at, and that the "edge" was just mean reversion I had selected for.
#
# Entry is charged at RV x (1 + VRP); exits are marked at RV. That is the real
# asymmetry an option buyer faces.
VRP = 0.20
DAYS_TO_EXPIRY = 30
MAX_HOLD = 30
STOP_PCT = -0.60
COST_PCT = 0.03


def bs_put(S, K, T, iv, r=E.RISK_FREE):
    if T <= 0 or iv <= 0:
        return max(0.0, K - S)
    d1 = (math.log(S / K) + (r + iv * iv / 2) * T) / (iv * math.sqrt(T))
    d2 = d1 - iv * math.sqrt(T)
    N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    return K * math.exp(-r * T) * N(-d2) - S * N(-d1)


def vol_and_rank(df: pd.DataFrame):
    ret = np.log(df["Close"] / df["Close"].shift(1))
    vol = (ret.rolling(VOL_WIN).std() * math.sqrt(252)).clip(0.10, 1.20)
    # percentile of today's vol within its own trailing year
    rank = vol.rolling(RANK_WIN).apply(
        lambda w: 100.0 * (w[-1] - np.min(w)) / (np.max(w) - np.min(w) + 1e-9),
        raw=True)
    return vol.values, rank.values


def entries_by_rank(df: pd.DataFrame, low: bool) -> list[int]:
    """Fire when vol rank crosses INTO the cheap (or expensive) zone."""
    _vol, rank = vol_and_rank(df)
    out: list[int] = []
    for i in range(RANK_WIN + 5, len(df) - 2):
        if not np.isfinite(rank[i]) or not np.isfinite(rank[i - 1]):
            continue
        if low and rank[i] < LOW_RANK <= rank[i - 1]:
            out.append(i)
        elif (not low) and rank[i] > HIGH_RANK >= rank[i - 1]:
            out.append(i)
    return out


def sim_straddle(df: pd.DataFrame, entries: list[int]) -> list[float]:
    """Buy an ATM call AND put. Profits from movement or rising volatility,
    in either direction - so the result is not contaminated by trend."""
    vol, _rank = vol_and_rank(df)
    close, high, low = df["Close"].values, df["High"].values, df["Low"].values
    n = len(df)
    out: list[float] = []
    for i in entries:
        if i >= n - 2 or not np.isfinite(vol[i]):
            continue
        S0, v0 = float(close[i]), float(vol[i])
        K = E.app.nearest_strike(S0, "X.NS")
        T0 = DAYS_TO_EXPIRY / 365.0
        # PAY the implied-vol premium on the way in, mark to realised on the way
        # out. This is the asymmetry every option buyer actually faces.
        v_buy = v0 * (1 + VRP)
        p0 = E.bs_call(S0, K, T0, v_buy) + bs_put(S0, K, T0, v_buy)
        if p0 <= 0.01:
            continue
        pnl = -1.0
        for k in range(i + 1, min(i + 1 + MAX_HOLD, n)):
            left = DAYS_TO_EXPIRY - (k - i)
            if left <= 1:
                pnl = (abs(float(close[k]) - K)) / p0 - 1
                break
            T = left / 365.0
            v = float(vol[k]) if np.isfinite(vol[k]) else v0
            val = E.bs_call(float(close[k]), K, T, v) + bs_put(float(close[k]), K, T, v)
            if val / p0 - 1 <= STOP_PCT:
                pnl = STOP_PCT
                break
        else:
            k = min(i + MAX_HOLD, n - 1)
            left = max(DAYS_TO_EXPIRY - (k - i), 1)
            v = float(vol[k]) if np.isfinite(vol[k]) else v0
            val = (E.bs_call(float(close[k]), K, left / 365.0, v)
                   + bs_put(float(close[k]), K, left / 365.0, v))
            pnl = val / p0 - 1
        out.append(float(pnl))
    return out


def stats(v):
    if not v:
        return 0.0, 0.0, 0, 0.0
    a = np.array(v, dtype=float)
    sd = a.std(ddof=1) if len(a) > 1 else 0.0
    se = sd / math.sqrt(len(a)) if sd else 0.0
    return (100 * float(a.mean()), float(a.mean() / se) if se else 0.0,
            len(a), 100.0 * float((a > 0).mean()))


def show(label, vals):
    m, t, n, w = stats(vals)
    flag = "  <-- significant" if abs(t) >= 2 else ""
    print(f"  {label:38s} n={n:5d}  win {w:5.1f}%  avg {m:+7.2f}%  "
          f"net {m - COST_PCT * 100:+7.2f}%  t={t:+5.2f}{flag}")
    return m, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=100)
    args = ap.parse_args()

    frames = {}
    for t in E.B.ALL_TICKERS:
        df = E.B.load(t, 4.0)
        if df is not None and len(df) > RANK_WIN + 60:
            frames[t] = df
    print(f"Volatility-rank test — {len(frames)} instruments, "
          f"realised vol ranked over {RANK_WIN} days")
    print(f"Entry priced at realised vol x {1 + VRP:.2f} (variance risk premium), "
          f"exits marked at realised vol.")
    print("SANITY CHECK: if the RANDOM straddle row is strongly positive, the "
          "model is\nstill mispricing options and NO result here can be trusted.\n")

    low = {t: entries_by_rank(df, True) for t, df in frames.items()}
    high = {t: entries_by_rank(df, False) for t, df in frames.items()}
    print(f"  cheap-vol signals: {sum(len(v) for v in low.values())}   "
          f"expensive-vol signals: {sum(len(v) for v in high.values())}\n")

    print("A/B — CALL only (still carries direction risk)")
    call_low, call_high = [], []
    for t, df in frames.items():
        call_low += E.simulate(df, low[t], STOP_PCT, None)
        call_high += E.simulate(df, high[t], STOP_PCT, None)
    m_cl, _ = show("A  CALL at LOW vol rank (cheap)", call_low)
    m_ch, _ = show("B  CALL at HIGH vol rank (expensive)", call_high)

    print("\nC/D — STRADDLE (call+put: pure volatility, direction removed)")
    str_low, str_high = [], []
    for t, df in frames.items():
        str_low += sim_straddle(df, low[t])
        str_high += sim_straddle(df, high[t])
    m_sl, n_sl = show("C  STRADDLE at LOW vol rank (cheap)", str_low)
    m_sh, _ = show("D  STRADDLE at HIGH vol rank (expensive)", str_high)

    print(f"\nE — permutation control ({args.draws} random-entry runs, straddles)")
    rng = np.random.default_rng(11)
    dist = np.empty(args.draws)
    for d in range(args.draws):
        vals = []
        for t, df in frames.items():
            k = len(low[t])
            if k <= 0:
                continue
            idx = sorted(int(x) for x in rng.integers(RANK_WIN + 5, len(df) - 35, size=k))
            vals += sim_straddle(df, idx)
        dist[d] = 100 * float(np.mean(vals)) if vals else 0.0
    pct = 100.0 * float((dist < m_sl).mean())
    lo, hi = np.percentile(dist, [5, 95])
    print(f"  cheap-vol straddle {m_sl:+.2f}%  vs random {dist.mean():+.2f}% "
          f"[5-95%: {lo:+.1f}, {hi:+.1f}]  ->  percentile {pct:.0f}%")

    print("\n" + "=" * 78)
    edge_dir = m_cl - m_ch
    edge_vol = m_sl - m_sh
    print(f"  Cheap vs expensive, CALLS    : {edge_dir:+.2f}%")
    print(f"  Cheap vs expensive, STRADDLES: {edge_vol:+.2f}%   <-- the clean test")
    print(f"  Cheap straddle vs random     : percentile {pct:.0f}%")
    print()
    if pct > 95 and edge_vol > 0:
        print("  Buying volatility when it is cheap DOES beat random, and beats")
        print("  buying it expensive. This is a pricing edge, not a forecast —")
        print("  the first thing in this project to survive a proper control.")
    elif edge_vol > 0 and pct > 80:
        print("  Points the right way (cheap beats expensive) but does not clear")
        print("  the 95% bar. Suggestive, not proven. Would need longer history")
        print("  or real implied-volatility data to settle.")
    else:
        print("  No edge here either. Volatility rank built from REALISED vol does")
        print("  not predict option returns on this data. Worth noting the proxy is")
        print("  the weak link: real IV rank may behave differently.")
    print("=" * 78)


if __name__ == "__main__":
    main()
