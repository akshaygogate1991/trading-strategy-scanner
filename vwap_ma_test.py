"""VWAP + MA Mean Reversion with ATR stop — the uploaded TradingView strategy.

THE ORIGINAL PINE SCRIPT
    vwap = ta.vwap(hlc3)
    ma   = ta.sma(close, 14)
    atr  = ta.atr(14)
    LONG  when close < vwap and close > ma   ... stop = close - 1.5*atr
    SHORT when close > vwap and close < ma   ... stop = close + 1.5*atr

It is a FUTURES strategy: it goes long and short the same instrument, with
stops in price terms. There is no profit target in the code at all — you exit
only on the stop, or when the opposite signal flips the position. So it is a
stop-and-reverse system that is always in the market.

WHY THIS NEEDS REAL FUTURES DATA
VWAP is VOLUME-weighted, and the Nifty INDEX reports zero volume (there is
nothing to trade in an index). Running this on index candles would silently
compute a time-weighted average instead and call it VWAP. So this script
fetches the actual NIFTY futures contracts, which have real traded volume, and
stitches the near-month contracts into one continuous series.

COSTS (verified October 2026 — these were wrong in earlier scripts)
    STT 0.05% on the SELL side, charged on FULL CONTRACT VALUE
    brokerage Rs 20/order, NSE txn 0.00183%, stamp 0.002% buy, GST 18% on fees
    => roughly Rs 950, about 15 Nifty points, per round trip on one lot.

ADDING A PROFIT TARGET
The original has none. Two ways to add one, and they are NOT equivalent:
    --target-pct 20     a fixed 20% of the share price
    --target-atr 2.0    2x the same ATR the stop uses, i.e. a 2:1 R:R
A fixed % ignores volatility; an ATR multiple fixes risk:reward, which is
what "does a target help?" actually asks. Use --grid to sweep it.

Run:  python vwap_ma_test.py --equity --grid --days 180
      python vwap_ma_test.py --equity --target-atr 2.0 --days 180
      python vwap_ma_test.py --fetch --months 12     (futures, if available)
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

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nifty_fut_data")
LOT_FALLBACK = 65

# --- the strategy's own inputs, exactly as in the Pine script ---
MA_LEN = 14
ATR_LEN = 14
ATR_MULT = 1.5

# --- costs, verified Oct 2026 ---
BROKERAGE_PER_LEG = 20.0
STT_SELL = 0.0005          # 0.05% on sell side, on full contract value
EXCH_TXN = 0.0000183
STAMP_BUY = 0.00002
SEBI_FEE = 0.000001
GST_RATE = 0.18
SLIPPAGE_PTS = 1.0
CHARGE_COSTS = True


def round_trip_points(price: float, lot: int) -> float:
    if not CHARGE_COSTS:
        return 0.0
    notional = price * lot
    brokerage = 2 * min(0.0003 * notional, BROKERAGE_PER_LEG)
    exch = EXCH_TXN * notional * 2
    sebi = SEBI_FEE * notional * 2
    gst = GST_RATE * (brokerage + exch + sebi)
    rupees = (brokerage + STT_SELL * notional + exch
              + STAMP_BUY * notional + sebi + gst)
    return rupees / lot + SLIPPAGE_PTS


# ------------------------------------------------------------------ data
LIQUID_STOCKS = ["RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS",
                 "INFY.NS", "TCS.NS", "AXISBANK.NS", "ITC.NS"]

# Equity INTRADAY costs differ from futures (verified Oct 2026):
#   STT 0.025% sell, NSE txn 0.00307%, stamp 0.003% buy, brokerage 0.03%/Rs20
EQ_STT_SELL = 0.00025
EQ_EXCH_TXN = 0.0000307
EQ_STAMP_BUY = 0.00003


def equity_round_trip_pct(price: float, qty: int) -> float:
    """Round-trip cost as a FRACTION of price, for intraday equity.

    Returned as a fraction rather than points so it is comparable across
    stocks of very different prices.
    """
    if not CHARGE_COSTS:
        return 0.0
    notional = price * qty
    brokerage = 2 * min(0.0003 * notional, BROKERAGE_PER_LEG)
    exch = EQ_EXCH_TXN * notional * 2
    sebi = SEBI_FEE * notional * 2
    gst = GST_RATE * (brokerage + exch + sebi)
    rupees = (brokerage + EQ_STT_SELL * notional + exch
              + EQ_STAMP_BUY * notional + sebi + gst)
    return rupees / notional + 0.0004      # + ~0.04% slippage on a liquid stock


def fetch_stock_1min(ticker: str, days: int) -> pd.DataFrame | None:
    """1-minute NSE cash candles — these DO carry real volume, and unlike
    futures the history is not deleted when a contract expires."""
    os.makedirs(CACHE, exist_ok=True)
    safe = ticker.replace(".", "_").replace("&", "_")
    path = os.path.join(CACHE, f"EQ_{safe}_{days}d.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if len(df) > 2000:
            print(f"  {ticker:14s} {len(df):,} candles cached")
            return df
    try:
        client = sd._session()
        token, _s, exch = sd.resolve_token(ticker)
    except Exception as exc:
        print(f"  {ticker:14s} ! {exc}")
        return None

    end = dt.date.today()
    cursor = end - dt.timedelta(days=days)
    frames, gap, n = [], 0.45, 0
    while cursor <= end:
        if cursor.weekday() < 5:
            for attempt in range(4):
                try:
                    resp = client.getCandleData({
                        "exchange": exch, "symboltoken": token,
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
                        n += len(f)
                    break
                except Exception as exc:
                    if "access rate" in str(exc).lower() and attempt < 3:
                        gap = min(gap + 0.05, 0.9)
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    break
            time.sleep(gap)
        cursor += dt.timedelta(days=1)
    if not frames:
        print(f"  {ticker:14s} no data returned")
        return None
    df = pd.concat(frames)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_csv(path)
    vol = 100 * float((df["Volume"] > 0).mean())
    print(f"  {ticker:14s} {len(df):,} candles, {vol:.0f}% with volume")
    return df


def near_month_contracts(n_months: int):
    """The last `n_months` NIFTY futures contracts, oldest first.

    Each contract is used only for the window ending at its own expiry, so the
    stitched series is always the near-month contract — which is where the
    volume is.
    """
    master = sd._instrument_master()
    rows = []
    for r in master:
        if (r.get("exch_seg") == "NFO" and r.get("name", "").upper() == "NIFTY"
                and r.get("instrumenttype") == "FUTIDX"):
            d = sd._parse_expiry(r.get("expiry", ""))
            if d:
                rows.append((d, r["symbol"], r["token"], int(float(r.get("lotsize") or 0))))
    rows.sort()
    today = dt.date.today()
    past = [x for x in rows if x[0] <= today]
    return past[-n_months:] if past else []


def fetch_contract(token: str, symbol: str, start: dt.date, end: dt.date):
    """1-minute candles for one contract between two dates, with retry."""
    client = sd._session()
    frames, cursor, gap = [], start, 0.45
    while cursor <= end:
        if cursor.weekday() < 5:
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
                        f = pd.DataFrame(rows, columns=["ts", "Open", "High", "Low",
                                                        "Close", "Volume"])
                        f["ts"] = pd.to_datetime(f["ts"])
                        frames.append(f.set_index("ts").astype(float))
                    break
                except Exception as exc:
                    if "access rate" in str(exc).lower() and attempt < 3:
                        gap = min(gap + 0.05, 0.9)
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    break
            time.sleep(gap)
        cursor += dt.timedelta(days=1)
    if not frames:
        return None
    df = pd.concat(frames)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    return df[~df.index.duplicated(keep="last")].sort_index()


def fetch_all(n_months: int):
    os.makedirs(CACHE, exist_ok=True)
    contracts = near_month_contracts(n_months + 1)
    if not contracts:
        print("No NIFTY futures contracts found in the instrument master.")
        return
    print(f"Found {len(contracts)} contracts. Fetching near-month windows...\n")
    prev_exp = None
    for i, (exp, sym, token, lot) in enumerate(contracts):
        path = os.path.join(CACHE, f"{sym}.csv")
        if os.path.exists(path):
            print(f"  {sym:22s} cached")
            prev_exp = exp
            continue
        start = (prev_exp + dt.timedelta(days=1)) if prev_exp else (exp - dt.timedelta(days=32))
        print(f"  {sym:22s} {start} -> {exp} (lot {lot})", flush=True)
        df = fetch_contract(token, sym, start, exp)
        if df is not None and len(df):
            df.to_csv(path)
            vol_ok = (df["Volume"] > 0).mean()
            print(f"      {len(df):,} candles, {100 * vol_ok:.0f}% have volume")
        else:
            print("      nothing returned")
        prev_exp = exp


def load_series() -> pd.DataFrame | None:
    if not os.path.isdir(CACHE):
        return None
    parts = []
    for f in sorted(os.listdir(CACHE)):
        if f.endswith(".csv"):
            parts.append(pd.read_csv(os.path.join(CACHE, f),
                                     index_col=0, parse_dates=True))
    if not parts:
        return None
    df = pd.concat(parts)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    return df[~df.index.duplicated(keep="last")].sort_index()


# ------------------------------------------------------------- indicators
def add_indicators(df: pd.DataFrame, ma_len: int, atr_len: int) -> pd.DataFrame:
    d = df.copy()
    d["day"] = d.index.normalize()
    hlc3 = (d["High"] + d["Low"] + d["Close"]) / 3.0
    vol = d["Volume"].where(d["Volume"] > 0, 0.0)
    # session VWAP: cumulative (price x volume) / cumulative volume, reset daily
    pv = (hlc3 * vol).groupby(d["day"]).cumsum()
    vv = vol.groupby(d["day"]).cumsum()
    d["vwap"] = pv / vv.replace(0, np.nan)
    d["ma"] = d["Close"].rolling(ma_len).mean()
    prev = d["Close"].shift(1)
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - prev).abs(),
                    (d["Low"] - prev).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(atr_len).mean()
    return d.dropna(subset=["vwap", "ma", "atr"])


def run(d: pd.DataFrame, lot: int, atr_mult: float,
        target_atr: float = 0.0, target_pct: float = 0.0) -> list[dict]:
    """Stop-and-reverse, exactly as the Pine script behaves: enter on signal,
    hold until stopped out or until the opposite signal flips the position."""
    close = d["Close"].values
    high, low = d["High"].values, d["Low"].values
    vwap, ma, atr = d["vwap"].values, d["ma"].values, d["atr"].values
    day = d["day"].values
    n = len(d)

    pos, entry, stop, entry_i = 0, 0.0, 0.0, 0
    target = None
    trades = []

    def close_pos(i, px, reason):
        nonlocal pos
        pts = pos * (px - entry)
        c = round_trip_points(entry, lot)
        trades.append({"i": entry_i, "pts": pts, "net": pts - c,
                       "rs": (pts - c) * lot, "res": reason,
                       "side": "LONG" if pos > 0 else "SHORT",
                       "mins": i - entry_i,
                       "ym": str(pd.Timestamp(day[entry_i]).to_period("M"))})
        pos = 0

    for i in range(1, n):
        new_day = day[i] != day[i - 1]
        if pos != 0 and new_day:                 # never carry overnight
            close_pos(i - 1, float(close[i - 1]), "eod")

        if pos > 0 and low[i] <= stop:
            close_pos(i, stop, "stop")
        elif pos < 0 and high[i] >= stop:
            close_pos(i, stop, "stop")
        elif pos > 0 and target is not None and high[i] >= target:
            close_pos(i, target, "target")
        elif pos < 0 and target is not None and low[i] <= target:
            close_pos(i, target, "target")

        long_sig = close[i] < vwap[i] and close[i] > ma[i]
        short_sig = close[i] > vwap[i] and close[i] < ma[i]

        if long_sig and pos <= 0:
            if pos < 0:
                close_pos(i, float(close[i]), "reverse")
            pos, entry, entry_i = 1, float(close[i]), i
            stop = entry - atr_mult * float(atr[i])
            target = _target_price(entry, 1, float(atr[i]), target_atr, target_pct)
        elif short_sig and pos >= 0:
            if pos > 0:
                close_pos(i, float(close[i]), "reverse")
            pos, entry, entry_i = -1, float(close[i]), i
            stop = entry + atr_mult * float(atr[i])
            target = _target_price(entry, -1, float(atr[i]), target_atr, target_pct)

    if pos != 0:
        close_pos(n - 1, float(close[n - 1]), "eod")
    return trades


def stats(rows):
    if not rows:
        return dict(n=0, win=0.0, gross=0.0, net=0.0, t=0.0, rs=0.0)
    g = np.array([r["pts"] for r in rows])
    a = np.array([r["net"] for r in rows])
    sd_ = a.std(ddof=1) if len(a) > 1 else 0.0
    se = sd_ / math.sqrt(len(a)) if sd_ else 0.0
    return dict(n=len(a), win=100 * float((g > 0).mean()), gross=float(g.mean()),
                net=float(a.mean()), t=float(a.mean() / se) if se else 0.0,
                rs=float(np.sum([r["rs"] for r in rows])))


def run_stocks(args):
    """The strategy on NSE cash equities, where VWAP is genuine.

    Angel One keeps no history for expired futures contracts, so a 12-month
    futures backtest is impossible. Equities have real volume AND full history,
    which makes this the only way to test the real indicator over a real span.
    """
    tickers = args.stocks or LIQUID_STOCKS
    tgt_desc = ("no target (original)" if not (args.target_atr or args.target_pct)
                else f"target {args.target_atr}x ATR" if args.target_atr
                else f"target +{args.target_pct}% of price")
    print("VWAP + MA mean reversion — NSE EQUITIES (stop and reverse)")
    print(f"  {len(tickers)} stocks, {args.days} days of 1-minute data")
    print(f"  MA {args.ma} | ATR {args.atr} x {args.atr_mult} | {tgt_desc}"
          + ("  | costs IGNORED" if args.no_cost else ""))
    print()

    frames = {}
    for tk in tickers:
        raw = fetch_stock_1min(tk, args.days)
        if raw is None or len(raw) < 3000:
            continue
        if (raw["Volume"] > 0).mean() < 0.8:
            print(f"  {tk}: too little volume — VWAP unreliable, skipping")
            continue
        frames[tk] = add_indicators(raw, args.ma, args.atr)
    if not frames:
        print("\nNo usable data.")
        return

    # ---- scale check: is the requested target even reachable? ----------
    scale_report(frames, args)

    if args.grid:
        run_grid(frames, args)
        return

    t_atr, t_pct = args.target_atr, args.target_pct / 100.0
    all_rows, per_stock = [], []
    for tk, d in frames.items():
        qty = max(1, int(100000 / float(d["Close"].mean())))   # ~Rs 1 lakh/trade
        rows = run_equity(d, qty, args.atr_mult, t_atr, t_pct)
        for r in rows:
            r["ticker"] = tk
        all_rows += rows
        per_stock.append((tk, stats_pct(rows)))

    if not all_rows:
        print("\nNo trades generated.")
        return

    print(f"\n{'stock':16s}{'trades':>8s}{'win%':>8s}{'gross%':>9s}"
          f"{'net%':>9s}{'t':>7s}")
    print("-" * 57)
    for tk, s in per_stock:
        print(f"{tk:16s}{s['n']:>8d}{s['win']:>8.1f}{s['gross']:>+9.3f}"
              f"{s['net']:>+9.3f}{s['t']:>+7.2f}")

    agg = stats_pct(all_rows)
    print("\n" + "=" * 57)
    print(f"  POOLED: {agg['n']} trades | win {agg['win']:.1f}% | "
          f"gross {agg['gross']:+.3f}% | net {agg['net']:+.3f}% | "
          f"t={agg['t']:+.2f}")
    print("=" * 57)

    print("\n  How trades ended:")
    for res in ("stop", "target", "reverse", "eod"):
        sub = [r["pct"] for r in all_rows if r["res"] == res]
        if sub:
            print(f"    {res:9s} {len(sub):6d} "
                  f"({100 * len(sub) / len(all_rows):3.0f}%)  "
                  f"avg {np.mean(sub):+7.3f}%")

    # control: random entries, SAME stop-and-target geometry
    rng = np.random.default_rng(43)
    dist = []
    for _ in range(args.draws):
        rows = []
        for tk, s in per_stock:
            d = frames[tk]
            qty = max(1, int(100000 / float(d["Close"].mean())))
            rows += random_equity(d, s["n"], qty, args.atr_mult, rng, t_atr, t_pct)
        st = stats_pct(rows)
        if st["n"]:
            dist.append(st["net"])
    if dist:
        dd = np.array(dist)
        pct = 100.0 * float((dd < agg["net"]).mean())
        lo_, hi_ = np.percentile(dd, [5, 95])
        print(f"\n  Control ({args.draws} random runs): signal {agg['net']:+.3f}% "
              f"vs random {dd.mean():+.3f}% [5-95%: {lo_:+.3f}, {hi_:+.3f}] "
              f"-> percentile {pct:.0f}%")
        print("  " + ("BEATS random" if pct > 95 else
                      "WORSE than random" if pct < 5 else
                      "no evidence of edge"))


def _target_price(entry: float, side: int, atr_v: float,
                  target_atr: float, target_pct: float) -> float | None:
    """Where to take profit, or None for the original no-target behaviour.

    Two ways to express it, because they mean very different things:
      target_pct  — a fixed % of price. "20%" means 20% of the share price.
      target_atr  — a multiple of the SAME ATR the stop uses, so the
                    risk:reward ratio is fixed regardless of volatility.
    """
    if target_atr > 0:
        return entry + side * target_atr * atr_v
    if target_pct > 0:
        return entry * (1 + side * target_pct)
    return None


def run_equity(d, qty: int, atr_mult: float,
               target_atr: float = 0.0, target_pct: float = 0.0) -> list[dict]:
    close, high, low = d["Close"].values, d["High"].values, d["Low"].values
    vwap, ma, atr = d["vwap"].values, d["ma"].values, d["atr"].values
    day = d["day"].values
    n = len(d)
    pos, entry, stop, entry_i = 0, 0.0, 0.0, 0
    target = None
    out = []

    def close_pos(i, px, reason):
        nonlocal pos
        pct = pos * (px - entry) / entry
        c = equity_round_trip_pct(entry, qty)
        out.append({"pct": 100 * pct, "net": 100 * (pct - c), "res": reason,
                    "ym": str(pd.Timestamp(day[entry_i]).to_period("M"))})
        pos = 0

    for i in range(1, n):
        if pos != 0 and day[i] != day[i - 1]:
            close_pos(i - 1, float(close[i - 1]), "eod")
        # stop is tested BEFORE target: if one bar spans both, the stop wins.
        # That is the pessimistic assumption, and it is the honest one — from
        # a single bar's high/low you cannot know which came first.
        if pos > 0 and low[i] <= stop:
            close_pos(i, stop, "stop")
        elif pos < 0 and high[i] >= stop:
            close_pos(i, stop, "stop")
        elif pos > 0 and target is not None and high[i] >= target:
            close_pos(i, target, "target")
        elif pos < 0 and target is not None and low[i] <= target:
            close_pos(i, target, "target")

        long_sig = close[i] < vwap[i] and close[i] > ma[i]
        short_sig = close[i] > vwap[i] and close[i] < ma[i]
        if long_sig and pos <= 0:
            if pos < 0:
                close_pos(i, float(close[i]), "reverse")
            pos, entry, entry_i = 1, float(close[i]), i
            stop = entry - atr_mult * float(atr[i])
            target = _target_price(entry, 1, float(atr[i]), target_atr, target_pct)
        elif short_sig and pos >= 0:
            if pos > 0:
                close_pos(i, float(close[i]), "reverse")
            pos, entry, entry_i = -1, float(close[i]), i
            stop = entry + atr_mult * float(atr[i])
            target = _target_price(entry, -1, float(atr[i]), target_atr, target_pct)
    if pos != 0:
        close_pos(n - 1, float(close[n - 1]), "eod")
    return out


def random_equity(d, n_trades: int, qty: int, atr_mult: float, rng,
                  target_atr: float = 0.0, target_pct: float = 0.0) -> list[dict]:
    """The control MUST use the same exit geometry as the signal, otherwise
    the comparison measures the target rather than the entry rule."""
    close, high, low = d["Close"].values, d["High"].values, d["Low"].values
    atr, day = d["atr"].values, d["day"].values
    n = len(d)
    out = []
    for _ in range(max(n_trades, 1)):
        i = int(rng.integers(5, n - 200))
        side = 1 if rng.random() < 0.5 else -1
        e = float(close[i])
        stp = e - side * atr_mult * float(atr[i])
        tgt = _target_price(e, side, float(atr[i]), target_atr, target_pct)
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
        pct = pts / e
        c = equity_round_trip_pct(e, qty)
        out.append({"pct": 100 * pct, "net": 100 * (pct - c), "res": "rand"})
    return out


def scale_report(frames: dict, args) -> None:
    """Before testing a target, check it is physically reachable.

    A profit target only means something relative to how far the price
    actually travels. This measures that from the data instead of assuming it.
    """
    print("\n  SCALE CHECK — how far does price actually move?")
    print(f"  {'stock':14s}{'ATR%':>8s}{'stop%':>8s}{'day range%':>12s}"
          f"{'best day%':>11s}")
    print("  " + "-" * 53)
    atr_pcts, rng_pcts, best = [], [], []
    for tk, d in frames.items():
        ap = float((d["atr"] / d["Close"]).mean())
        day_rng = d.groupby("day").apply(
            lambda g: (g["High"].max() - g["Low"].min()) / g["Open"].iloc[0],
            include_groups=False)
        atr_pcts.append(ap)
        rng_pcts.append(float(day_rng.mean()))
        best.append(float(day_rng.max()))
        print(f"  {tk:14s}{100 * ap:>8.3f}{100 * ap * args.atr_mult:>8.3f}"
              f"{100 * day_rng.mean():>12.2f}{100 * day_rng.max():>11.2f}")
    m_atr = float(np.mean(atr_pcts))
    m_rng = float(np.mean(rng_pcts))
    m_best = float(np.max(best))
    print(f"\n  Average 1-minute ATR is {100 * m_atr:.3f}% of price, so the "
          f"{args.atr_mult}x stop\n  sits {100 * m_atr * args.atr_mult:.3f}% away. "
          f"A typical whole day ranges {100 * m_rng:.2f}%;\n  the single best day "
          f"in this data ranged {100 * m_best:.2f}%.")
    if args.target_pct:
        need_atr = (args.target_pct / 100.0) / m_atr if m_atr else 0
        print(f"\n  A +{args.target_pct}% target is {need_atr:.0f}x the 1-minute ATR, "
              f"and {args.target_pct / (100 * m_rng):.0f}x a typical day's\n  "
              f"ENTIRE range. On an intraday system that closes by 15:30, it "
              f"can essentially\n  never be reached — so it would not change "
              f"the result, it would just\n  remove the target from the system. "
              f"Use --target-atr to size it against risk.")


def run_grid(frames: dict, args) -> None:
    """Sweep the target against the ATR stop, so risk:reward is explicit.

    With a stop at `atr_mult` x ATR, a target at `t` x ATR gives R:R = t/atr_mult
    and needs a win rate of atr_mult/(atr_mult + t) just to break even gross.
    """
    qtys = {tk: max(1, int(100000 / float(d["Close"].mean())))
            for tk, d in frames.items()}
    ladder = [0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 0.0]
    print(f"\n  TARGET SWEEP (stop fixed at {args.atr_mult}x ATR)")
    print(f"  {'target':>9s}{'R:R':>7s}{'need%':>8s}{'trades':>8s}"
          f"{'win%':>7s}{'gross%':>9s}{'net%':>9s}{'vs random':>11s}")
    print("  " + "-" * 68)
    rng = np.random.default_rng(43)
    for t in ladder:
        rows, counts = [], []
        for tk, d in frames.items():
            r = run_equity(d, qtys[tk], args.atr_mult, t, 0.0)
            counts.append((tk, len(r)))
            rows += r
        s = stats_pct(rows)
        if not s["n"]:
            continue
        rr = t / args.atr_mult if t else float("nan")
        need = 100 * args.atr_mult / (args.atr_mult + t) if t else float("nan")
        # control with the identical geometry
        dist = []
        for _ in range(max(args.draws // 3, 5)):
            rr_rows = []
            for tk, nn in counts:
                rr_rows += random_equity(frames[tk], nn, qtys[tk],
                                         args.atr_mult, rng, t, 0.0)
            st = stats_pct(rr_rows)
            if st["n"]:
                dist.append(st["net"])
        pctl = (100.0 * float((np.array(dist) < s["net"]).mean())
                if dist else float("nan"))
        label = f"{t:.2f}x ATR" if t else "none"
        print(f"  {label:>9s}{rr:>7.2f}{need:>8.1f}{s['n']:>8d}{s['win']:>7.1f}"
              f"{s['gross']:>+9.3f}{s['net']:>+9.3f}{pctl:>10.0f}%")
    print("\n  'need%' is the win rate required to break even GROSS at that R:R.")
    print("  Compare it to the win% actually achieved — that is the whole test.")
    print("  'vs random' above 95% would be a real edge; below 5% is worse than random.")


def stats_pct(rows):
    if not rows:
        return dict(n=0, win=0.0, gross=0.0, net=0.0, t=0.0)
    g = np.array([r["pct"] for r in rows])
    a = np.array([r["net"] for r in rows])
    sd_ = a.std(ddof=1) if len(a) > 1 else 0.0
    se = sd_ / math.sqrt(len(a)) if sd_ else 0.0
    return dict(n=len(a), win=100 * float((g > 0).mean()), gross=float(g.mean()),
                net=float(a.mean()), t=float(a.mean() / se) if se else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="download futures data")
    ap.add_argument("--stocks", nargs="*", default=None,
                    help="run on NSE equities instead of futures")
    ap.add_argument("--equity", action="store_true",
                    help="use the default liquid-stock list")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--ma", type=int, default=MA_LEN)
    ap.add_argument("--atr", type=int, default=ATR_LEN)
    ap.add_argument("--atr-mult", type=float, default=ATR_MULT)
    ap.add_argument("--target-pct", type=float, default=0.0,
                    help="profit target as a %% of price, e.g. 20 for +20%%")
    ap.add_argument("--target-atr", type=float, default=0.0,
                    help="profit target as a multiple of ATR (same ATR as the "
                         "stop), e.g. 2.0 for 2:1 reward:risk at a 1x stop")
    ap.add_argument("--grid", action="store_true",
                    help="sweep the target from 0.2x to 4x ATR and compare")
    ap.add_argument("--lot", type=int, default=None)
    ap.add_argument("--no-cost", action="store_true")
    ap.add_argument("--draws", type=int, default=30)
    args = ap.parse_args()
    globals()["CHARGE_COSTS"] = not args.no_cost

    if args.equity or args.stocks:
        run_stocks(args)
        return

    if args.fetch:
        fetch_all(args.months)
        print()

    raw = load_series()
    if raw is None:
        print("No futures data. Run:  python vwap_ma_test.py --fetch --months 12")
        return
    try:
        LOT = args.lot or sd.get_lot_size("^NSEI") or LOT_FALLBACK
    except Exception:
        LOT = args.lot or LOT_FALLBACK

    vol_share = 100 * float((raw["Volume"] > 0).mean())
    print(f"VWAP + MA mean reversion — NIFTY FUTURES (stop and reverse)")
    print(f"  {len(raw):,} one-minute candles, "
          f"{raw.index[0].date()} to {raw.index[-1].date()}")
    print(f"  {vol_share:.0f}% of candles carry real volume "
          f"{'(VWAP is valid)' if vol_share > 80 else '(TOO LITTLE VOLUME — VWAP unreliable)'}")
    print(f"  MA {args.ma} | ATR {args.atr} x {args.atr_mult} | lot {LOT}"
          + ("  | costs IGNORED" if args.no_cost else ""))

    d = add_indicators(raw, args.ma, args.atr)
    trades = run(d, LOT, args.atr_mult, args.target_atr, args.target_pct / 100.0)
    if not trades:
        print("\n  No trades generated.")
        return
    s = stats(trades)
    cost = round_trip_points(float(d["Close"].mean()), LOT)
    print(f"\n  RESULT: {s['n']} trades | win {s['win']:.1f}% | "
          f"gross {s['gross']:+.1f} pts | net {s['net']:+.1f} pts | "
          f"t={s['t']:+.2f}")
    print(f"  TOTAL : Rs {s['rs']:+,.0f}   (cost {cost:.1f} pts per round trip)")

    print("\n  How trades ended:")
    for res in ("stop", "target", "reverse", "eod"):
        sub = [r["pts"] for r in trades if r["res"] == res]
        if sub:
            print(f"    {res:9s} {len(sub):5d} ({100 * len(sub) / len(trades):3.0f}%)  "
                  f"avg {np.mean(sub):+8.1f} pts")

    print("\n  Month by month:")
    months = sorted({r["ym"] for r in trades})
    pos = 0
    for ym in months:
        sub = [r for r in trades if r["ym"] == ym]
        st = stats(sub)
        pos += st["rs"] > 0
        print(f"    {ym:10s} {st['n']:4d} trades | win {st['win']:5.1f}% | "
              f"net {st['net']:+7.1f} pts | Rs {st['rs']:+10,.0f}")
    print(f"    -> {pos} of {len(months)} months profitable")

    # control: same stop-and-reverse geometry, random entry bars
    rng = np.random.default_rng(41)
    dist = []
    close_v = d["Close"].values
    atr_v = d["atr"].values
    day_v = d["day"].values
    for _ in range(args.draws):
        rows = []
        for _k in range(s["n"]):
            i = int(rng.integers(5, len(d) - 200))
            side = 1 if rng.random() < 0.5 else -1
            e = float(close_v[i])
            stp = e - side * args.atr_mult * float(atr_v[i])
            pts = 0.0
            for j in range(i + 1, min(i + 200, len(d))):
                if day_v[j] != day_v[i]:
                    pts = side * (float(close_v[j - 1]) - e)
                    break
                if (side > 0 and d["Low"].values[j] <= stp) or \
                   (side < 0 and d["High"].values[j] >= stp):
                    pts = side * (stp - e)
                    break
            else:
                pts = side * (float(close_v[min(i + 200, len(d) - 1)]) - e)
            rows.append({"pts": pts, "net": pts - round_trip_points(e, LOT),
                         "rs": 0.0})
        st = stats(rows)
        if st["n"]:
            dist.append(st["net"])
    if dist:
        dd = np.array(dist)
        pct = 100.0 * float((dd < s["net"]).mean())
        lo_, hi_ = np.percentile(dd, [5, 95])
        print(f"\n  Control ({args.draws} random runs): signal {s['net']:+.1f} "
              f"vs random {dd.mean():+.1f} [5-95%: {lo_:+.1f}, {hi_:+.1f}] "
              f"-> percentile {pct:.0f}%")
        print("  " + ("BEATS random" if pct > 95 else
                      "WORSE than random" if pct < 5 else
                      "no evidence of edge"))

    print(f"\n  Gross is the signal; net is what reaches your account. With "
          f"{cost:.0f} points\n  of cost per trade, gross must exceed that "
          f"before anything is left.")


if __name__ == "__main__":
    main()
