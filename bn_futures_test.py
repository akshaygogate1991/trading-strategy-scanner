"""VWAP + MA on REAL Bank Nifty (or Nifty) futures candles.

THE INSIGHT THAT MAKES THIS POSSIBLE
Angel One deletes expired contracts, so there is no multi-year futures
history. But a LIVE contract carries its own history back to the day it was
listed. BANKNIFTY27OCT26FUT was introduced around late July 2026, so it holds
roughly three months of one-minute candles WITH REAL TRADED VOLUME — which is
exactly what VWAP needs and what the index could never give us.

THE HONEST LIMITATION
For most of its life a contract is a FAR-month contract, where volume is thin.
Volume only concentrates in it during its final month as near-month. So this
script reports volume per month and lets you restrict the test to the liquid
stretch. Do not skip that table: a VWAP computed from 40 contracts an hour is
not the same measurement as one computed from 4,000.

Run:  python bn_futures_test.py --fetch --name BANKNIFTY
      python bn_futures_test.py --name BANKNIFTY --grid
      python bn_futures_test.py --name BANKNIFTY --min-vol 500
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import time

import numpy as np
import pandas as pd

import smartapi_data as sd
import vwap_ma_test as v

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bn_fut_data")


def live_contracts(name: str):
    """Live FUTIDX contracts for `name`, nearest expiry first."""
    master = sd._instrument_master()
    out = []
    today = dt.date.today()
    for r in master:
        if (r.get("exch_seg") == "NFO"
                and r.get("name", "").upper() == name.upper()
                and r.get("instrumenttype") == "FUTIDX"):
            d = sd._parse_expiry(r.get("expiry", ""))
            if d and d > today:
                out.append((d, r["symbol"], r["token"],
                            int(float(r.get("lotsize") or 0))))
    out.sort()
    return out


def fetch_contract(token: str, symbol: str, start: dt.date, end: dt.date):
    client = sd._session()
    frames, cursor, gap, miss = [], start, 0.45, 0
    while cursor <= end:
        if cursor.weekday() < 5:
            got = False
            for attempt in range(4):
                try:
                    resp = client.getCandleData({
                        "exchange": "NFO", "symboltoken": token,
                        "interval": "ONE_MINUTE",
                        "fromdate": cursor.strftime("%Y-%m-%d 09:15"),
                        "todate": cursor.strftime("%Y-%m-%d 15:30"),
                    })
                    rows = resp.get("data") if isinstance(resp, dict) else None
                    if rows:
                        f = pd.DataFrame(rows, columns=["ts", "Open", "High",
                                                        "Low", "Close", "Volume"])
                        f["ts"] = pd.to_datetime(f["ts"])
                        frames.append(f.set_index("ts").astype(float))
                        got = True
                    break
                except Exception as exc:
                    if "access rate" in str(exc).lower() and attempt < 3:
                        gap = min(gap + 0.05, 0.9)
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    break
            if not got:
                miss += 1
            time.sleep(gap)
        cursor += dt.timedelta(days=1)
    if not frames:
        return None, miss
    df = pd.concat(frames)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    return df[~df.index.duplicated(keep="last")].sort_index(), miss


def do_fetch(args):
    os.makedirs(CACHE, exist_ok=True)
    cons = live_contracts(args.name)
    if not cons:
        print(f"No live {args.name} futures found.")
        return
    print(f"{len(cons)} live {args.name} contracts. Fetching each contract's "
          f"own history\n(back to {args.days} days before its expiry, or its "
          f"listing date — whichever is later).\n")
    today = dt.date.today()
    for exp, sym, token, lot in cons:
        path = os.path.join(CACHE, f"{sym}.csv")
        if os.path.exists(path):
            print(f"  {sym:24s} cached")
            continue
        start = exp - dt.timedelta(days=args.days)
        end = min(today, exp)
        print(f"  {sym:24s} {start} -> {end} (lot {lot})", flush=True)
        df, miss = fetch_contract(token, sym, start, end)
        if df is None or not len(df):
            print("      nothing returned (contract may not have existed yet)")
            continue
        df.to_csv(path)
        print(f"      {len(df):,} candles, {miss} days with no data, "
              f"first {df.index[0].date()}")


def load(args) -> pd.DataFrame | None:
    if not os.path.isdir(CACHE):
        return None
    best, best_n = None, 0
    for f in sorted(os.listdir(CACHE)):
        if not f.endswith(".csv") or args.name.upper() not in f.upper():
            continue
        df = pd.read_csv(os.path.join(CACHE, f), index_col=0, parse_dates=True)
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        if len(df) > best_n:          # the contract with the longest history
            best, best_n, name = df, len(df), f
    if best is None:
        return None
    print(f"  using the single longest contract: {name} "
          f"({best_n:,} candles)")
    return best


def liquidity_table(df: pd.DataFrame, min_vol: float) -> pd.DataFrame:
    """Volume per month — the table that decides whether VWAP is meaningful."""
    d = df.copy()
    d["ym"] = d.index.to_period("M").astype(str)
    print(f"\n  LIQUIDITY BY MONTH (VWAP is only as good as the volume)")
    print(f"  {'month':10s}{'candles':>10s}{'% w/ vol':>10s}"
          f"{'median vol':>13s}{'usable?':>10s}")
    print("  " + "-" * 53)
    keep = []
    for ym, g in d.groupby("ym"):
        cov = 100 * float((g["Volume"] > 0).mean())
        med = float(g["Volume"].median())
        ok = med >= min_vol and cov > 90
        keep.append((ym, ok))
        print(f"  {ym:10s}{len(g):>10,}{cov:>10.0f}{med:>13,.0f}"
              f"{'yes' if ok else 'THIN':>10s}")
    good = {ym for ym, ok in keep if ok}
    if not good:
        print(f"\n  No month clears the median-volume bar of {min_vol:,.0f}.")
        return d.iloc[0:0]
    print(f"\n  Keeping {len(good)} of {len(keep)} months "
          f"(median volume >= {min_vol:,.0f} and >90% of candles traded).")
    return d[d["ym"].isin(good)].drop(columns=["ym"])


def control(d, lot, atr_mult, n_trades, draws, target_atr, rng):
    """Random entries with identical stop/target geometry, in points."""
    close, high, low = d["Close"].values, d["High"].values, d["Low"].values
    atr, day = d["atr"].values, d["day"].values
    n = len(d)
    dist = []
    for _ in range(draws):
        nets = []
        for _k in range(n_trades):
            i = int(rng.integers(5, max(n - 200, 6)))
            side = 1 if rng.random() < 0.5 else -1
            e = float(close[i])
            stp = e - side * atr_mult * float(atr[i])
            tgt = (e + side * target_atr * float(atr[i])) if target_atr else None
            pts = 0.0
            for j in range(i + 1, min(i + 200, n)):
                if day[j] != day[i]:
                    pts = side * (float(close[j - 1]) - e)
                    break
                if (side > 0 and low[j] <= stp) or (side < 0 and high[j] >= stp):
                    pts = side * (stp - e)
                    break
                if tgt is not None and ((side > 0 and high[j] >= tgt)
                                        or (side < 0 and low[j] <= tgt)):
                    pts = side * (tgt - e)
                    break
            else:
                pts = side * (float(close[min(i + 200, n - 1)]) - e)
            nets.append(pts - v.round_trip_points(e, lot))
        if nets:
            dist.append(float(np.mean(nets)))
    return dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="BANKNIFTY")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--days", type=int, default=130,
                    help="how far back before expiry to try fetching")
    ap.add_argument("--ma", type=int, default=14)
    ap.add_argument("--atr", type=int, default=14)
    ap.add_argument("--atr-mult", type=float, default=1.5)
    ap.add_argument("--target-atr", type=float, default=0.0)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--slip", type=float, default=3.0,
                    help="slippage in index points per round trip")
    ap.add_argument("--min-vol", type=float, default=200.0,
                    help="median candle volume required to trust VWAP")
    ap.add_argument("--lot", type=int, default=None)
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--no-cost", action="store_true")
    args = ap.parse_args()

    v.SLIPPAGE_PTS = args.slip
    v.CHARGE_COSTS = not args.no_cost

    if args.fetch:
        do_fetch(args)
        print()

    raw = load(args)
    if raw is None:
        print(f"No data. Run:  python bn_futures_test.py --fetch "
              f"--name {args.name}")
        return

    cons = {s: l for _e, s, _t, l in live_contracts(args.name)}
    LOT = args.lot or (list(cons.values())[0] if cons else 30)

    print(f"\n{args.name} FUTURES — VWAP + MA mean reversion (stop and reverse)")
    print(f"  {len(raw):,} one-minute candles, "
          f"{raw.index[0].date()} to {raw.index[-1].date()}")
    print(f"  lot {LOT} | MA {args.ma} | ATR {args.atr} x {args.atr_mult} | "
          f"slippage {args.slip} pts"
          + ("  | costs IGNORED" if args.no_cost else ""))

    raw = liquidity_table(raw, args.min_vol)
    if not len(raw):
        print("\n  Nothing liquid enough to test. Lower --min-vol to see it "
              "anyway,\n  but know that the VWAP would then be noise.")
        return

    d = v.add_indicators(raw, args.ma, args.atr)
    px = float(d["Close"].mean())
    cost = v.round_trip_points(px, LOT)
    atr_pts = float(d["atr"].mean())
    print(f"\n  Measured 1-minute ATR: {atr_pts:.1f} points "
          f"({100 * atr_pts / px:.4f}% of price)")
    print(f"  Round-trip cost: {cost:.1f} points "
          f"({100 * cost / px:.4f}%) = {cost / atr_pts:.2f} ATRs")
    print("  ^ THIS is the number that matters. Everything below is whether")
    print("    the signal clears it.")

    rng = np.random.default_rng(71)
    ladder = ([0.0] if not args.grid
              else [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 0.0])
    if args.target_atr and not args.grid:
        ladder = [args.target_atr]

    print(f"\n  {'target':>10s}{'trades':>8s}{'win%':>7s}{'gross pts':>11s}"
          f"{'net pts':>10s}{'Rs total':>12s}{'vs random':>11s}")
    print("  " + "-" * 69)
    for t in ladder:
        trades = v.run(d, LOT, args.atr_mult, t, 0.0)
        if not trades:
            continue
        s = v.stats(trades)
        dist = control(d, LOT, args.atr_mult, s["n"], args.draws, t, rng)
        pctl = (100.0 * float((np.array(dist) < s["net"]).mean())
                if dist else float("nan"))
        label = f"{t:.1f}x ATR" if t else "none"
        print(f"  {label:>10s}{s['n']:>8d}{s['win']:>7.1f}{s['gross']:>+11.1f}"
              f"{s['net']:>+10.1f}{s['rs']:>+12,.0f}{pctl:>10.0f}%")

    trades = v.run(d, LOT, args.atr_mult, args.target_atr, 0.0)
    print("\n  How trades ended:")
    for res in ("stop", "target", "reverse", "eod"):
        sub = [r["pts"] for r in trades if r["res"] == res]
        if sub:
            print(f"    {res:9s} {len(sub):5d} "
                  f"({100 * len(sub) / len(trades):3.0f}%)  "
                  f"avg {np.mean(sub):+8.1f} pts")

    print(f"\n  Gross must exceed {cost:.1f} points before anything reaches "
          f"your account.")
    print("  'vs random' above 95% would be a real edge; below 5% is worse "
          "than random.")
    print(f"\n  CAUTION: this is a SHORT sample. Treat it as a look, not a "
          f"verdict —\n  12 months of equity data told us more than 3 months "
          f"of anything can.")


if __name__ == "__main__":
    main()
