"""RSI DIVERGENCE on Nifty futures — the real thing, not RSI levels.

WHAT DIVERGENCE IS (and how it differs from what we already tested)
Earlier tests asked "is RSI above 80?" — a threshold on one series. Divergence
compares the SHAPE of two series:

  BEARISH:  price makes a HIGHER high, but RSI makes a LOWER high
            -> the new high is not backed by momentum -> SELL
  BULLISH:  price makes a LOWER low,  but RSI makes a HIGHER low
            -> selling is losing force                -> BUY

PARAMETERS — WHERE THESE NUMBERS COME FROM
You asked me to choose. Rather than invent them, these are the defaults from
TradingView's own built-in Divergence Indicator, which is what nearly every
published RSI-divergence script uses:

  pivot lookback left  = 5 bars
  pivot lookback right = 5 bars        (so a pivot is an 11-bar extreme)
  minimum gap between the two pivots   =  5 bars
  maximum gap between the two pivots   = 60 bars

Using the standard values matters: if I picked numbers myself and the result
came out positive, you could not tell whether the strategy worked or whether I
had tuned my way to a nice answer. Change them with --pivot and --range if you
want, but the defaults are the honest starting point.

NO LOOKAHEAD
A pivot needs 5 bars to its RIGHT before it can be identified, so a divergence
found at bar i is only ACTED ON at bar i+5. Skipping that is the single easiest
way to invent an edge that does not exist.

THREE VARIANTS TESTED
  1  divergence on 1-minute only
  2  divergence on 5-minute only
  3  both timeframes agreeing (your original idea)

Run:  python rsi_divergence_test.py
      python rsi_divergence_test.py --target 0.5 --stop 0.25
      python rsi_divergence_test.py --no-cost
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd

import smartapi_data as sd

MONTHS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nifty_months")
RSI_LEN = 14
PIVOT_L = PIVOT_R = 5          # TradingView default
RANGE_MIN, RANGE_MAX = 5, 60   # TradingView default
MAX_HOLD_MIN = 180
LOT_FALLBACK = 65
AGREE_WINDOW_5M = 12           # 5-min bars (= 1 hour) for "both agree"

BROKERAGE_PER_LEG = 20.0
STT_SELL = 0.0005        # 0.05% on the SELL side (verified Oct 2026, was 0.02%)
EXCH_TXN = 0.0000183     # NSE futures transaction charge, each side
STAMP_BUY = 0.00002
SEBI_FEE = 0.000001
GST_RATE = 0.18
SLIPPAGE_PTS = 1.0
CHARGE_COSTS = True


def load_all():
    if not os.path.isdir(MONTHS):
        print("No data. Run fetch_month.py first.")
        return None
    one, five = [], []
    for f in sorted(os.listdir(MONTHS)):
        if not f.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(MONTHS, f), index_col=0, parse_dates=True)
        (one if "ONE_MINUTE" in f else five).append(df)
    if not one or not five:
        return None
    m1, m5 = pd.concat(one), pd.concat(five)
    for d in (m1, m5):
        d.index = pd.DatetimeIndex(d.index).tz_localize(None)
    return (m1[~m1.index.duplicated(keep="last")].sort_index(),
            m5[~m5.index.duplicated(keep="last")].sort_index())


def rsi(close: pd.Series, n: int = RSI_LEN) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def pivots(series: np.ndarray, left: int, right: int):
    """Indices where `series` is a local high / low over left+right bars.

    Pivots are found on the RSI (the oscillator), which is how TradingView's
    divergence indicator does it; price is then read at those same bars.
    """
    highs, lows = [], []
    n = len(series)
    for i in range(left, n - right):
        w = series[i - left: i + right + 1]
        if not np.isfinite(w).all():
            continue
        if series[i] == w.max() and series[i] > series[i - left:i].max():
            highs.append(i)
        elif series[i] == w.min() and series[i] < series[i - left:i].min():
            lows.append(i)
    return highs, lows


def divergences(df: pd.DataFrame, rsi_vals: np.ndarray, left: int, right: int,
                rmin: int, rmax: int):
    """Returns (bearish_confirm_idx, bullish_confirm_idx).

    Each index is the bar at which the divergence becomes KNOWN, i.e. `right`
    bars after the pivot itself — not the pivot bar.
    """
    high, low = df["High"].values, df["Low"].values
    p_hi, p_lo = pivots(rsi_vals, left, right)
    bear, bull = [], []

    for a, b in zip(p_hi, p_hi[1:]):
        gap = b - a
        if not (rmin <= gap <= rmax):
            continue
        # price higher high, RSI lower high
        if high[b] > high[a] and rsi_vals[b] < rsi_vals[a]:
            bear.append(b + right)

    for a, b in zip(p_lo, p_lo[1:]):
        gap = b - a
        if not (rmin <= gap <= rmax):
            continue
        # price lower low, RSI higher low
        if low[b] < low[a] and rsi_vals[b] > rsi_vals[a]:
            bull.append(b + right)

    n = len(df)
    return ([i for i in bear if i < n - 2], [i for i in bull if i < n - 2])


def cost_points(price: float, lot: int) -> float:
    if not CHARGE_COSTS:
        return 0.0
    notional = price * lot
    brokerage = 2 * BROKERAGE_PER_LEG
    exch = EXCH_TXN * notional * 2
    sebi = SEBI_FEE * notional * 2
    gst = GST_RATE * (brokerage + exch + sebi)
    rupees = (brokerage + STT_SELL * notional + exch
              + STAMP_BUY * notional + sebi + gst)
    return rupees / lot + SLIPPAGE_PTS


def simulate(df, entries, side, stop_pct, target_pct, lot):
    close, high, low = df["Close"].values, df["High"].values, df["Low"].values
    day, ym = df["day"].values, df["ym"].values
    n = len(df)
    out = []
    for i in entries:
        entry = float(close[i])
        if side > 0:
            stop, target = entry * (1 - stop_pct), entry * (1 + target_pct)
        else:
            stop, target = entry * (1 + stop_pct), entry * (1 - target_pct)
        res, pts, k = "eod", 0.0, i
        for k in range(i + 1, min(i + 1 + MAX_HOLD_MIN, n)):
            if day[k] != day[i]:
                k -= 1
                break
            hs = (low[k] <= stop) if side > 0 else (high[k] >= stop)
            ht = (high[k] >= target) if side > 0 else (low[k] <= target)
            if hs:                      # pessimistic: stop wins a tied bar
                pts, res = side * (stop - entry), "stop"
                break
            if ht:
                pts, res = side * (target - entry), "target"
                break
        if res == "eod":
            pts = side * (float(close[k]) - entry)
        c = cost_points(entry, lot)
        out.append({"ym": ym[i], "pts": pts, "net": pts - c,
                    "rs": (pts - c) * lot, "res": res})
    return out


def stats(rows):
    if not rows:
        return dict(n=0, win=0.0, net=0.0, t=0.0, rs=0.0)
    a = np.array([r["net"] for r in rows])
    sd_ = a.std(ddof=1) if len(a) > 1 else 0.0
    se = sd_ / math.sqrt(len(a)) if sd_ else 0.0
    return dict(n=len(a),
                win=100 * float((np.array([r["pts"] for r in rows]) > 0).mean()),
                net=float(a.mean()), t=float(a.mean() / se) if se else 0.0,
                rs=float(np.sum([r["rs"] for r in rows])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.25)
    ap.add_argument("--stop", type=float, default=0.50)
    ap.add_argument("--pivot", type=int, default=PIVOT_L)
    ap.add_argument("--range", dest="rng_max", type=int, default=RANGE_MAX)
    ap.add_argument("--rsi-len", type=int, default=RSI_LEN)
    ap.add_argument("--no-cost", action="store_true")
    ap.add_argument("--lot", type=int, default=None)
    ap.add_argument("--draws", type=int, default=50)
    args = ap.parse_args()
    globals()["CHARGE_COSTS"] = not args.no_cost
    TGT, STP = args.target / 100.0, args.stop / 100.0

    loaded = load_all()
    if loaded is None:
        return
    m1, m5 = loaded
    try:
        LOT = args.lot or sd.get_lot_size("^NSEI") or LOT_FALLBACK
    except Exception:
        LOT = args.lot or LOT_FALLBACK

    m1 = m1.copy()
    m1["day"] = m1.index.normalize()
    m1["ym"] = m1.index.to_period("M").astype(str)
    r1 = rsi(m1["Close"], args.rsi_len).values
    r5 = rsi(m5["Close"], args.rsi_len).values

    print(f"RSI DIVERGENCE — Nifty futures, {len(m1):,} one-minute bars")
    print(f"  pivots {args.pivot} bars each side | gap {RANGE_MIN}-{args.rng_max} "
          f"bars | RSI({args.rsi_len})  [TradingView defaults]")
    print(f"  target +{args.target:.2f}% / stop -{args.stop:.2f}% | lot {LOT}"
          + ("  | costs IGNORED" if args.no_cost else ""))

    # 1-minute divergences
    bear1, bull1 = divergences(m1, r1, args.pivot, args.pivot,
                               RANGE_MIN, args.rng_max)
    # 5-minute divergences, mapped onto the 1-minute clock at confirmation time
    m5c = m5.copy()
    bear5, bull5 = divergences(m5c, r5, args.pivot, args.pivot,
                               RANGE_MIN, args.rng_max)
    t_bear5 = [m5c.index[i] for i in bear5 if i < len(m5c)]
    t_bull5 = [m5c.index[i] for i in bull5 if i < len(m5c)]

    def to_1m(times):
        out = []
        for t in times:
            pos = m1.index.searchsorted(t)
            if 0 <= pos < len(m1) - 2:
                out.append(int(pos))
        return sorted(set(out))

    bear5_1m, bull5_1m = to_1m(t_bear5), to_1m(t_bull5)

    # "both agree": a 1-min divergence with a 5-min one of the same type
    # within the last hour
    win_min = AGREE_WINDOW_5M * 5

    def agree(one_list, five_list):
        five = np.array(five_list)
        out = []
        for i in one_list:
            if five.size and ((five <= i) & (five >= i - win_min)).any():
                out.append(i)
        return out

    bear_both, bull_both = agree(bear1, bear5_1m), agree(bull1, bull5_1m)

    variants = [
        ("1  1-minute divergence only", bear1, bull1),
        ("2  5-minute divergence only", bear5_1m, bull5_1m),
        ("3  BOTH timeframes agree", bear_both, bull_both),
    ]

    print(f"\n{'variant':32s}{'sell':>7s}{'buy':>7s}{'trades':>8s}"
          f"{'win%':>8s}{'net pts':>10s}{'t':>7s}{'Rs':>12s}")
    print("-" * 91)
    results = {}
    for name, bears, bulls in variants:
        rows = (simulate(m1, bears, -1, STP, TGT, LOT)
                + simulate(m1, bulls, +1, STP, TGT, LOT))
        s = stats(rows)
        results[name] = (rows, s)
        print(f"{name:32s}{len(bears):>7d}{len(bulls):>7d}{s['n']:>8d}"
              f"{s['win']:>8.1f}{s['net']:>+10.1f}{s['t']:>+7.2f}{s['rs']:>+12,.0f}")

    need = 100 * STP / (STP + TGT)
    print(f"\n  Break-even win rate at this reward:risk = {need:.1f}%")

    # month-by-month for the best variant by rupees
    best_name = max(results, key=lambda k: results[k][1]["rs"])
    rows, s = results[best_name]
    if rows:
        print(f"\n  Month by month — {best_name.strip()}")
        print(f"  {'month':10s}{'trades':>8s}{'win%':>8s}{'net pts':>10s}{'Rs':>12s}")
        print("  " + "-" * 48)
        months = sorted({r["ym"] for r in rows})
        pos = 0
        for ym in months:
            sub = [r for r in rows if r["ym"] == ym]
            st = stats(sub)
            pos += st["rs"] > 0
            print(f"  {ym:10s}{st['n']:>8d}{st['win']:>8.1f}"
                  f"{st['net']:>+10.1f}{st['rs']:>+12,.0f}")
        print(f"\n  {pos} of {len(months)} months profitable")

        # permutation control
        rng = np.random.default_rng(23)
        dist = []
        for _ in range(args.draws):
            idx = sorted(int(x) for x in
                         rng.integers(100, len(m1) - 200, size=max(s["n"], 2)))
            h = len(idx) // 2
            rr = (simulate(m1, idx[:h], -1, STP, TGT, LOT)
                  + simulate(m1, idx[h:], +1, STP, TGT, LOT))
            st = stats(rr)
            if st["n"]:
                dist.append(st["net"])
        if dist:
            d = np.array(dist)
            pct = 100.0 * float((d < s["net"]).mean())
            lo_, hi_ = np.percentile(d, [5, 95])
            print(f"\n  Control ({args.draws} random runs): signal {s['net']:+.1f} "
                  f"vs random {d.mean():+.1f} [5-95%: {lo_:+.1f}, {hi_:+.1f}] "
                  f"-> percentile {pct:.0f}%")
            print("  " + ("BEATS random" if pct > 95 else
                          "WORSE than random" if pct < 5 else
                          "no evidence of edge"))

    print("\n  Note: the best of three variants is shown above. Picking the best")
    print("  of several AFTER seeing results flatters it — so the control, not")
    print("  the rupee total, is what decides whether this is real.")


if __name__ == "__main__":
    main()
