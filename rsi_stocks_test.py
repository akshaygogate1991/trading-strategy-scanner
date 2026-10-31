"""Dual-RSI reversal strategy across Nifty 50 stock futures.

THE STRATEGY (the REVERSAL reading, which was the profitable direction on Nifty)
  * RSI(14) on 1-minute AND 5-minute candles
  * Both above 80  -> SELL  (overbought, expect a pull-back)
  * Both below 20  -> BUY   (oversold, expect a bounce)
  * Target +0.25%, stop -0.50%, squared off same day

WHY STOCKS ARE A HARSHER TEST THAN NIFTY
Nifty futures are extremely liquid - one or two ticks wide. Stock futures are
not. A 0.25% target on a stock whose bid-ask is already 0.05% wide gives away a
fifth of the move before anything happens. Slippage is therefore modelled much
higher here (0.02% per side vs Nifty's ~0.004%), and that single difference
decides most of these results.

DATA COST - READ THIS FIRST
One year of 1-minute data is ~188 requests per stock; two years is ~376. Across
50 stocks that is 9,000-19,000 requests, or roughly 1-2 hours at Angel One's
rate limit. Every stock is cached separately as it completes, so you can stop
this at any point with Ctrl+C and re-run later to continue where it left off.

Run:  python rsi_stocks_test.py                      (1 year, all stocks)
      python rsi_stocks_test.py --days 730           (2 years)
      python rsi_stocks_test.py --stocks RELIANCE.NS TCS.NS
      python rsi_stocks_test.py --no-cost            (gross result)
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

# Liquid Nifty-50 F&O names. Override with --stocks.
NIFTY50 = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS",
    "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "ITC.NS",
    "LT.NS", "HINDUNILVR.NS", "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS",
    "ULTRACEMCO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "NTPC.NS",
    "POWERGRID.NS", "ONGC.NS", "COALINDIA.NS", "TATASTEEL.NS", "JSWSTEEL.NS",
    "HINDALCO.NS", "M&M.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",
    "DRREDDY.NS", "CIPLA.NS", "ADANIENT.NS", "ADANIPORTS.NS", "GRASIM.NS",
    "NESTLEIND.NS", "BRITANNIA.NS", "EICHERMOT.NS", "HEROMOTOCO.NS",
    "BPCL.NS", "TATACONSUM.NS", "APOLLOHOSP.NS", "DIVISLAB.NS",
    "INDUSINDBK.NS", "SBILIFE.NS", "HDFCLIFE.NS", "SHRIRAMFIN.NS",
    "BAJAJ-AUTO.NS", "TRENT.NS", "BEL.NS", "VEDL.NS",
]

RSI_LEN = 14
OB, OS = 80.0, 20.0
TARGET, STOP = 0.0025, 0.0050
MAX_HOLD_MIN = 180

BROKERAGE_PER_LEG = 20.0
STT_SELL = 0.0005        # 0.05% on the SELL side (verified Oct 2026, was 0.02%)
EXCH_TXN = 0.0000183     # NSE futures transaction charge, each side
STAMP_BUY = 0.00002
SEBI_FEE = 0.000001
GST_RATE = 0.18
SLIP_PCT_STOCK = 0.0002     # 0.02% per side — stock futures, NOT index
MARGIN_PCT = 0.20           # stock futures margin is higher than index

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rsi_stock_data")
CHARGE_COSTS = True


def cost_points(price: float, lot: int) -> float:
    """Round-trip cost in POINTS for one lot of a stock future."""
    if not CHARGE_COSTS:
        return 0.0
    notional = price * lot
    brokerage = 2 * BROKERAGE_PER_LEG
    exch = EXCH_TXN * notional * 2
    sebi = SEBI_FEE * notional * 2
    gst = GST_RATE * (brokerage + exch + sebi)
    rupees = (brokerage + STT_SELL * notional + exch
              + STAMP_BUY * notional + sebi + gst)
    return rupees / lot + price * SLIP_PCT_STOCK * 2


def fetch(ticker: str, interval: str, days: int) -> pd.DataFrame | None:
    """Cached per stock+interval, with a retry when the rate limit bites."""
    os.makedirs(CACHE, exist_ok=True)
    safe = ticker.replace(".", "_").replace("&", "_")
    path = os.path.join(CACHE, f"{safe}_{interval}_{days}d.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if len(df) > 500:
                return df
        except Exception:
            pass

    per_day = {"ONE_MINUTE": 375, "FIVE_MINUTE": 75}[interval]
    step = max(1, int((500 * 0.8 / per_day) / 0.68))
    try:
        client = sd._session()
        token, _s, exch = sd.resolve_token(ticker)
    except Exception as exc:
        print(f"    ! {ticker}: {exc}")
        return None

    end = dt.datetime.now()
    cursor = end - dt.timedelta(days=days)
    frames = []
    while cursor < end:
        cend = min(cursor + dt.timedelta(days=step), end)
        for attempt in range(3):
            try:
                resp = client.getCandleData({
                    "exchange": exch, "symboltoken": token, "interval": interval,
                    "fromdate": cursor.strftime("%Y-%m-%d 09:15"),
                    "todate": cend.strftime("%Y-%m-%d 15:30"),
                })
                rows = resp.get("data") if isinstance(resp, dict) else None
                if rows:
                    f = pd.DataFrame(rows, columns=["ts", "Open", "High", "Low",
                                                    "Close", "Volume"])
                    f["ts"] = pd.to_datetime(f["ts"])
                    frames.append(f.set_index("ts").astype(float))
                break
            except Exception as exc:
                # "exceeding access rate" just needs a pause, not a failure
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                print(f"    ! {ticker} {cursor.date()}: {str(exc)[:60]}")
        cursor = cend
        time.sleep(0.4)

    if not frames:
        return None
    df = pd.concat(frames)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_csv(path)
    return df


def rsi(close: pd.Series, n: int = RSI_LEN) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def prepare(ticker: str, days: int) -> pd.DataFrame | None:
    m1 = fetch(ticker, "ONE_MINUTE", days)
    m5 = fetch(ticker, "FIVE_MINUTE", days)
    if m1 is None or m5 is None or len(m1) < 2000:
        return None
    m1 = m1.copy()
    m1["rsi1"] = rsi(m1["Close"])
    r5 = rsi(m5["Close"]).shift(1)          # known only after that candle closes
    m1["rsi5"] = r5.reindex(m1.index, method="ffill")
    m1["day"] = m1.index.normalize()
    return m1.dropna(subset=["rsi1", "rsi5"])


def signals(df) -> tuple[list[int], list[int]]:
    """Returns (short_entries, long_entries) for the REVERSAL reading:
    both RSI > 80 -> SELL, both RSI < 20 -> BUY."""
    r1, r5 = df["rsi1"].values, df["rsi5"].values
    shorts, longs = [], []
    for i in range(1, len(df) - 2):
        ob = r1[i] > OB and r5[i] > OB
        ob_p = r1[i - 1] > OB and r5[i - 1] > OB
        os_ = r1[i] < OS and r5[i] < OS
        os_p = r1[i - 1] < OS and r5[i - 1] < OS
        if ob and not ob_p:
            shorts.append(i)
        elif os_ and not os_p:
            longs.append(i)
    return shorts, longs


def simulate(df, entries, side: int, lot: int) -> list[dict]:
    close, high, low = df["Close"].values, df["High"].values, df["Low"].values
    day = df["day"].values
    n = len(df)
    out = []
    for i in entries:
        entry = float(close[i])
        if side > 0:
            stop, target = entry * (1 - STOP), entry * (1 + TARGET)
        else:
            stop, target = entry * (1 + STOP), entry * (1 - TARGET)
        res, pts, k = "eod", 0.0, i
        for k in range(i + 1, min(i + 1 + MAX_HOLD_MIN, n)):
            if day[k] != day[i]:
                k -= 1
                break
            hit_stop = (low[k] <= stop) if side > 0 else (high[k] >= stop)
            hit_tgt = (high[k] >= target) if side > 0 else (low[k] <= target)
            if hit_stop:                      # pessimistic: stop wins a tie
                pts, res = side * (stop - entry), "stop"
                break
            if hit_tgt:
                pts, res = side * (target - entry), "target"
                break
        if res == "eod":
            pts = side * (float(close[k]) - entry)
        c = cost_points(entry, lot)
        out.append({"pts": pts, "net": pts - c, "net_rs": (pts - c) * lot,
                    "res": res, "pct": (pts - c) / entry * 100})
    return out


def stats(rows):
    if not rows:
        return dict(n=0, win=0.0, pct=0.0, t=0.0, rs=0.0)
    a = np.array([r["pct"] for r in rows])
    sd_ = a.std(ddof=1) if len(a) > 1 else 0.0
    se = sd_ / math.sqrt(len(a)) if sd_ else 0.0
    return dict(n=len(a), win=100 * float((np.array([r["pts"] for r in rows]) > 0).mean()),
                pct=float(a.mean()), t=float(a.mean() / se) if se else 0.0,
                rs=float(np.sum([r["net_rs"] for r in rows])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--stocks", nargs="*", default=None)
    ap.add_argument("--no-cost", action="store_true")
    ap.add_argument("--draws", type=int, default=30)
    args = ap.parse_args()
    globals()["CHARGE_COSTS"] = not args.no_cost

    stocks = args.stocks or NIFTY50
    print(f"Dual-RSI REVERSAL on {len(stocks)} stock futures — "
          f"{args.days} days, target +{TARGET:.2%} / stop -{STOP:.2%}")
    print(f"Costs: {'IGNORED' if args.no_cost else 'included'}; stock-futures "
          f"slippage {SLIP_PCT_STOCK:.2%} per side\n")

    all_rows, per_stock, t0 = [], [], time.time()
    for j, tk in enumerate(stocks, 1):
        df = prepare(tk, args.days)
        if df is None:
            print(f"  [{j:2d}/{len(stocks)}] {tk:16s} no data")
            continue
        try:
            lot = sd.get_lot_size(tk) or 100
        except Exception:
            lot = 100
        sh, lo = signals(df)
        rows = simulate(df, sh, -1, lot) + simulate(df, lo, +1, lot)
        s = stats(rows)
        per_stock.append((tk, s))
        all_rows += rows
        el = time.time() - t0
        print(f"  [{j:2d}/{len(stocks)}] {tk:16s} lot {lot:4d} | "
              f"trades {s['n']:4d} | win {s['win']:5.1f}% | "
              f"avg {s['pct']:+6.3f}% | Rs {s['rs']:+10,.0f}   "
              f"({el / j * (len(stocks) - j) / 60:.0f} min left)")

    if not all_rows:
        print("\nNo trades. Check data fetched correctly.")
        return

    print("\n" + "=" * 74)
    agg = stats(all_rows)
    print(f"  ALL STOCKS POOLED: {agg['n']} trades | win {agg['win']:.1f}% | "
          f"avg {agg['pct']:+.3f}% per trade | t={agg['t']:+.2f}")
    print(f"  TOTAL P&L: Rs {agg['rs']:+,.0f}"
          + ("   (costs IGNORED)" if args.no_cost else ""))
    print("=" * 74)

    good = [(t, s) for t, s in per_stock if s["rs"] > 0]
    print(f"\n  {len(good)}/{len(per_stock)} stocks profitable")
    print("  Best 5:")
    for t, s in sorted(per_stock, key=lambda x: -x[1]["rs"])[:5]:
        print(f"    {t:16s} Rs {s['rs']:+10,.0f}  ({s['n']} trades, "
              f"{s['pct']:+.3f}%/trade)")
    print("  Worst 5:")
    for t, s in sorted(per_stock, key=lambda x: x[1]["rs"])[:5]:
        print(f"    {t:16s} Rs {s['rs']:+10,.0f}  ({s['n']} trades, "
              f"{s['pct']:+.3f}%/trade)")

    # permutation control on the pooled result
    print(f"\n  Control: {args.draws} random-entry runs, same exits")
    rng = np.random.default_rng(13)
    dist = []
    for _ in range(args.draws):
        vals = []
        for tk, s in per_stock:
            df = prepare(tk, args.days)
            if df is None or s["n"] == 0:
                continue
            try:
                lot = sd.get_lot_size(tk) or 100
            except Exception:
                lot = 100
            idx = sorted(int(x) for x in rng.integers(50, len(df) - 200, size=s["n"]))
            h = len(idx) // 2
            vals += simulate(df, idx[:h], -1, lot) + simulate(df, idx[h:], +1, lot)
        st = stats(vals)
        if st["n"]:
            dist.append(st["pct"])
    if dist:
        d = np.array(dist)
        pct = 100.0 * float((d < agg["pct"]).mean())
        lo_, hi_ = np.percentile(d, [5, 95])
        print(f"  signal {agg['pct']:+.3f}%  vs random {d.mean():+.3f}% "
              f"[5-95%: {lo_:+.3f}, {hi_:+.3f}]  ->  percentile {pct:.0f}%")
        print("  " + ("BEATS random" if pct > 95 else
                      "WORSE than random" if pct < 5 else
                      "no evidence of edge"))

    print("\n  Note: a few stocks looking good among 50 is expected from noise.")
    print("  The POOLED number and the control are what decide this, not the")
    print("  best-5 list — picking those after the fact is how backtests lie.")


if __name__ == "__main__":
    main()
