"""Backtest the dual-RSI strategy on the month-by-month Nifty futures data.

Reads every monthly file that fetch_month.py has saved, stitches them into one
continuous series, and runs the same test as before — but on data you have
verified month by month rather than one long pull with silent gaps.

It also reports results PER MONTH. That matters: a strategy that makes money in
three months and loses it in nine is not the same as one that works steadily,
even when the totals match. Monthly figures expose that; a single total hides it.

Run:  python rsi_month_test.py
      python rsi_month_test.py --reverse        (>80 sell, <20 buy)
      python rsi_month_test.py --no-cost
      python rsi_month_test.py --target 0.5 --stop 0.25
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
MAX_HOLD_MIN = 180
LOT_FALLBACK = 65
MARGIN_PCT = 0.15

BROKERAGE_PER_LEG = 20.0
STT_SELL = 0.0005        # 0.05% on the SELL side (verified Oct 2026, was 0.02%)
EXCH_TXN = 0.0000183     # NSE futures transaction charge, each side
STAMP_BUY = 0.00002
SEBI_FEE = 0.000001
GST_RATE = 0.18
SLIPPAGE_PTS = 1.0
CHARGE_COSTS = True


def load_all() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if not os.path.isdir(MONTHS):
        print("No data yet. Start with:  python fetch_month.py --month 2026-09")
        return None
    one, five = [], []
    for f in sorted(os.listdir(MONTHS)):
        if not f.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(MONTHS, f), index_col=0, parse_dates=True)
        (one if "ONE_MINUTE" in f else five).append(df)
    if not one or not five:
        print("Need both 1-minute and 5-minute files. Run fetch_month.py first.")
        return None
    m1 = pd.concat(one)
    m5 = pd.concat(five)
    for d in (m1, m5):
        d.index = pd.DatetimeIndex(d.index).tz_localize(None)
    m1 = m1[~m1.index.duplicated(keep="last")].sort_index()
    m5 = m5[~m5.index.duplicated(keep="last")].sort_index()
    return m1, m5


def rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


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


def prepare(m1: pd.DataFrame, m5: pd.DataFrame, rsi_len: int) -> pd.DataFrame:
    d = m1.copy()
    d["rsi1"] = rsi(d["Close"], rsi_len)
    # shift(1): a 5-minute RSI is only known once that candle has closed
    d["rsi5"] = rsi(m5["Close"], rsi_len).shift(1).reindex(d.index, method="ffill")
    d["day"] = d.index.normalize()
    d["ym"] = d.index.to_period("M").astype(str)
    return d.dropna(subset=["rsi1", "rsi5"])


def signals(df, ob, os_):
    r1, r5 = df["rsi1"].values, df["rsi5"].values
    hi, lo = [], []
    for i in range(1, len(df) - 2):
        a = r1[i] > ob and r5[i] > ob
        ap = r1[i - 1] > ob and r5[i - 1] > ob
        b = r1[i] < os_ and r5[i] < os_
        bp = r1[i - 1] < os_ and r5[i - 1] < os_
        if a and not ap:
            hi.append(i)
        elif b and not bp:
            lo.append(i)
    return hi, lo


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
            if hs:                       # pessimistic: stop wins a tied bar
                pts, res = side * (stop - entry), "stop"
                break
            if ht:
                pts, res = side * (target - entry), "target"
                break
        if res == "eod":
            pts = side * (float(close[k]) - entry)
        c = cost_points(entry, lot)
        out.append({"i": i, "ym": ym[i], "pts": pts, "net": pts - c,
                    "rs": (pts - c) * lot, "res": res, "entry": entry})
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
    ap.add_argument("--ob", type=float, default=80.0)
    ap.add_argument("--os", dest="os_", type=float, default=20.0)
    ap.add_argument("--rsi-len", type=int, default=RSI_LEN)
    ap.add_argument("--reverse", action="store_true")
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

    df = prepare(m1, m5, args.rsi_len)
    months = sorted(df["ym"].unique())
    print(f"NIFTY FUTURES — monthly data, {len(months)} months "
          f"({months[0]} to {months[-1]})")
    print(f"  {len(df):,} one-minute bars | lot {LOT} | "
          f"target +{args.target:.2f}% / stop -{args.stop:.2f}%")
    print(f"  direction: {'REVERSAL (>80 sell, <20 buy)' if args.reverse else 'MOMENTUM (>80 buy, <20 sell)'}"
          + ("  | costs IGNORED" if args.no_cost else ""))

    hi, lo = signals(df, args.ob, args.os_)
    a_side, b_side = (-1, +1) if args.reverse else (+1, -1)
    rows = (simulate(df, hi, a_side, STP, TGT, LOT)
            + simulate(df, lo, b_side, STP, TGT, LOT))
    if not rows:
        print("\n  No trades generated.")
        return
    s = stats(rows)
    need = 100 * STP / (STP + TGT)
    print(f"\n  OVERALL: {s['n']} trades | win {s['win']:.1f}% (need {need:.1f}%) "
          f"| net {s['net']:+.1f} pts | t={s['t']:+.2f} | Rs {s['rs']:+,.0f}")

    # ---------------- month by month ----------------
    print(f"\n  {'month':10s}{'trades':>8s}{'win%':>8s}{'net pts':>10s}{'Rs':>12s}")
    print("  " + "-" * 48)
    wins = 0
    for ym in months:
        sub = [r for r in rows if r["ym"] == ym]
        if not sub:
            print(f"  {ym:10s}{0:>8d}{'-':>8s}{'-':>10s}{'-':>12s}")
            continue
        st = stats(sub)
        wins += st["rs"] > 0
        print(f"  {ym:10s}{st['n']:>8d}{st['win']:>8.1f}"
              f"{st['net']:>+10.1f}{st['rs']:>+12,.0f}")
    print(f"\n  {wins} of {len(months)} months profitable")
    print("  A strategy that works steadily looks different from one carried by")
    print("  a single month, even when the totals match. This is where you see it.")

    # ---------------- control ----------------
    rng = np.random.default_rng(17)
    dist = []
    for _ in range(args.draws):
        idx = sorted(int(x) for x in rng.integers(400, len(df) - 200, size=len(rows)))
        h = len(idx) // 2
        rr = (simulate(df, idx[:h], a_side, STP, TGT, LOT)
              + simulate(df, idx[h:], b_side, STP, TGT, LOT))
        st = stats(rr)
        if st["n"]:
            dist.append(st["net"])
    if dist:
        d = np.array(dist)
        pct = 100.0 * float((d < s["net"]).mean())
        lo_, hi_ = np.percentile(d, [5, 95])
        print(f"\n  Control ({args.draws} random-entry runs): signal "
              f"{s['net']:+.1f} vs random {d.mean():+.1f} "
              f"[5-95%: {lo_:+.1f}, {hi_:+.1f}] -> percentile {pct:.0f}%")
        print("  " + ("BEATS random" if pct > 95 else
                      "WORSE than random" if pct < 5 else
                      "no evidence of edge"))


if __name__ == "__main__":
    main()
