"""NIFTY FUTURES intraday: dual-timeframe RSI extremes. 2-year backtest.

THE RULES
  * Nifty 50 futures, long and short
  * RSI on 1-minute AND 5-minute candles
  * Overbought 80, oversold 20
  * BOTH above 80  -> BUY  (momentum long)
  * BOTH below 20  -> SELL (momentum short)
  * Take profit +1%
  * Stop-loss "50%"

WHY FUTURES IS A CLEANER TEST THAN OPTIONS
No premium, no theta, no implied volatility, no Black-Scholes. P&L is simply
points moved x lot size. Every option backtest we ran depended on modelling a
premium we could not observe; this one depends on nothing but the price series.
Whatever it says is far more trustworthy.

"STOP-LOSS 50%" — WHAT IT CAN MEAN ON FUTURES
There is no premium to lose 50% of, so the script tests several readings:
    0.50% stop  = half the 1% target  -> 2:1 reward:risk   <- most likely
    0.33% stop                        -> 3:1
    0.25% stop                        -> 4:1
    1.00% stop                        -> 1:1
Each needs a different win rate to break even, and the table shows that figure
so you can judge each honestly rather than by its total.

COSTS ARE REAL AND INCLUDED
Brokerage, STT, exchange and stamp charges plus slippage run roughly 5 Nifty
points per round trip on one lot. At 1% (~226 points) that is about 2% of the
target — small, but it decides marginal cases, so net columns are shown.

No lookahead: the 5-minute RSI is shifted one bar, so it is only used after
that candle has actually closed.

Run:  python rsi_futures_test.py                 (2 years, cached after first run)
      python rsi_futures_test.py --days 365
      python rsi_futures_test.py --ob 70 --os 30  (if 80/20 is too rare)
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

# Which index future to test. Set by --symbol; Bank Nifty is roughly 1.4x as
# volatile as Nifty, so the same 0.25% target is reached far more often - which
# is exactly why it is worth testing separately rather than assumed similar.
SYMBOLS = {"nifty": ("^NSEI", 65), "banknifty": ("^NSEBANK", 35)}
TICKER = "^NSEI"
LOT_FALLBACK = 65           # used only if the instrument master is unreachable
MAX_HOLD_MIN = 180          # 3 hours max; never held overnight
RSI_LEN = 14

# ---------------------------------------------------------------- costs
# Real Indian futures costs, not a guess. Every component below is charged on
# NOTIONAL (price x lot), so cost in POINTS depends on the index level - which
# is why this is computed per trade rather than fixed.
BROKERAGE_PER_LEG = 20.0    # discount broker, flat per order
STT_SELL = 0.0005        # 0.05% on the SELL side (verified Oct 2026, was 0.02%)
EXCH_TXN = 0.0000183     # NSE futures transaction charge, each side
STAMP_BUY = 0.00002         # 0.002% on the buy side
SEBI_FEE = 0.000001         # Rs 10 per crore, each side
GST_RATE = 0.18             # on brokerage + exchange + SEBI
SLIPPAGE_PTS = 1.0          # entry + exit combined; Nifty futures is liquid
CHARGE_COSTS = True         # flipped off by --no-cost
MARGIN_PCT = 0.15           # SPAN + exposure for Nifty futures, approximate


def round_trip_cost(price: float, lot: int) -> tuple[float, float]:
    """Returns (rupees, points) for one complete round trip on one lot.

    At Nifty ~23,000 with a 65 lot this works out near Rs 450, or about 7
    points - versus the flat 5 points assumed earlier. STT alone is roughly
    Rs 300 of it, which dwarfs the Rs 20 brokerage people usually focus on.
    """
    notional = price * lot
    brokerage = 2 * BROKERAGE_PER_LEG
    exch = EXCH_TXN * notional * 2
    sebi = SEBI_FEE * notional * 2
    gst = GST_RATE * (brokerage + exch + sebi)
    rupees = (brokerage + STT_SELL * notional + exch
              + STAMP_BUY * notional + sebi + gst)
    return rupees, rupees / lot + SLIPPAGE_PTS


def index_lot_size(override: int | None = None) -> int:
    """Read the live NSE lot size from Angel One's instrument master.

    Hardcoding this is how backtests drift from reality: NSE has revised the
    Nifty lot size several times (75 -> 25 -> 75 -> ...), and every rupee
    figure scales directly with it.
    """
    if override:
        return override
    try:
        lot = sd.get_lot_size(TICKER)
        if lot and lot > 0:
            return int(lot)
    except Exception:
        pass
    return LOT_FALLBACK

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rsi_data")

# Grid searched so the whole landscape is visible, not one cherry-picked cell.
GRID_TARGETS = [0.0010, 0.0025, 0.0050, 0.0075, 0.0100]
GRID_STOPS = [0.0025, 0.0050, 0.0075, 0.0100]


# --------------------------------------------------------------- data
def fetch(interval: str, days: int, refresh: bool = False) -> pd.DataFrame:
    """Angel One caps every request at 500 candles, so long spans are chunked."""
    os.makedirs(CACHE, exist_ok=True)
    tag = TICKER.replace("^", "")
    path = os.path.join(CACHE, f"{tag}_{interval}_{days}d.csv")
    if os.path.exists(path) and not refresh:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if len(df) > 1000:
            print(f"  {interval}: {len(df):,} candles from cache")
            return df

    per_day = {"ONE_MINUTE": 375, "FIVE_MINUTE": 75}[interval]
    step = max(1, int((500 * 0.8 / per_day) / 0.68))   # calendar days per call
    client = sd._session()
    token, _sym, exch = sd.resolve_token(TICKER)

    end = dt.datetime.now()
    cursor = end - dt.timedelta(days=days)
    frames, t0, calls = [], time.time(), 0
    total_calls = max(1, int((end - cursor).days / step))
    print(f"  {interval}: ~{total_calls} requests at {step} day(s) each...")

    # Angel One allows ~3 history requests/second. A flat 0.35s gap sits right
    # on that ceiling, so bursts get refused with "exceeding access rate" — and
    # without a retry those chunks were silently dropped, which is why earlier
    # pulls came back ~8% short. Pace it slightly slower and retry on refusal.
    gap = 0.45
    while cursor < end:
        cend = min(cursor + dt.timedelta(days=step), end)
        if cursor.weekday() >= 5:            # Sat/Sun: no session, skip the call
            cursor = cend
            continue

        for attempt in range(4):
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
                if "access rate" in str(exc).lower() and attempt < 3:
                    gap = min(gap + 0.05, 0.9)          # ease off permanently
                    time.sleep(1.5 * (attempt + 1))      # and back off now
                    continue
                if attempt == 3:
                    print(f"    ! {cursor.date()} gave up: {str(exc)[:50]}")
                break

        calls += 1
        if calls % 50 == 0:
            el = time.time() - t0
            print(f"    {calls}/{total_calls} ({el:.0f}s, "
                  f"~{el / calls * max(total_calls - calls, 0):.0f}s left, "
                  f"gap {gap:.2f}s)")
            # Progress is saved to a .partial file, NEVER to the real cache
            # path — otherwise an interrupted run would look like a complete
            # one next time and silently backtest on half the data.
            if frames:
                part = pd.concat(frames)
                part.index = pd.DatetimeIndex(part.index).tz_localize(None)
                part[~part.index.duplicated(keep="last")].sort_index().to_csv(
                    path + ".partial")
        cursor = cend
        time.sleep(gap)

    if not frames:
        raise RuntimeError(f"No {interval} data returned — check credentials.")
    df = pd.concat(frames)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_csv(path)                       # only NOW is the cache complete
    if os.path.exists(path + ".partial"):
        os.remove(path + ".partial")

    sessions = df.index.normalize().nunique()
    expected = {"ONE_MINUTE": 375, "FIVE_MINUTE": 75}[interval] * sessions
    pct = 100.0 * len(df) / expected if expected else 0.0
    print(f"    {len(df):,} candles over {sessions} sessions "
          f"({pct:.0f}% of a full set)")
    if pct < 90:
        print("    WARNING: gaps remain — some chunks were refused. Re-run "
              "with --refresh to fill them in.")
    return df


def rsi(close: pd.Series, n: int = RSI_LEN) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def build(days: int, refresh: bool, rsi_len: int) -> pd.DataFrame:
    m1 = fetch("ONE_MINUTE", days, refresh)
    m5 = fetch("FIVE_MINUTE", days, refresh)
    m1["rsi1"] = rsi(m1["Close"], rsi_len)
    m5["rsi5"] = rsi(m5["Close"], rsi_len)
    # shift(1): a 5-minute RSI is only known once that candle has CLOSED.
    # Skipping this lets the backtest see the future and invents an edge.
    m1["rsi5"] = m5["rsi5"].shift(1).reindex(m1.index, method="ffill")
    m1["day"] = m1.index.normalize()
    return m1.dropna(subset=["rsi1", "rsi5"])


def signals(df, ob, os_) -> tuple[list[int], list[int]]:
    """Fire on the bar BOTH cross into the zone, not every bar they sit there."""
    r1, r5 = df["rsi1"].values, df["rsi5"].values
    longs, shorts = [], []
    for i in range(1, len(df) - 2):
        ob_now = r1[i] > ob and r5[i] > ob
        ob_prev = r1[i - 1] > ob and r5[i - 1] > ob
        os_now = r1[i] < os_ and r5[i] < os_
        os_prev = r1[i - 1] < os_ and r5[i - 1] < os_
        if ob_now and not ob_prev:
            longs.append(i)
        elif os_now and not os_prev:
            shorts.append(i)
    return longs, shorts


def simulate(df, entries, side: int, stop_pct: float,
             target_pct: float, lot: int = LOT_FALLBACK) -> list[dict]:
    """side +1 long, -1 short. Returns points per trade.

    Pessimistic tie-break: if a single minute's range contains both the stop
    and the target, the STOP is taken. Over thousands of trades this matters.
    """
    close, high, low = df["Close"].values, df["High"].values, df["Low"].values
    day = df["day"].values
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
            if day[k] != day[i]:                  # square off, never overnight
                k -= 1
                break
            hit_stop = (low[k] <= stop) if side > 0 else (high[k] >= stop)
            hit_tgt = (high[k] >= target) if side > 0 else (low[k] <= target)
            if hit_stop:
                pts, res = side * (stop - entry), "stop"
                break
            if hit_tgt:
                pts, res = side * (target - entry), "target"
                break
        if res == "eod":
            pts = side * (float(close[k]) - entry)
        # costs scale with notional, so charge them at THIS trade's price
        rup, cost_pts = round_trip_cost(entry, lot)
        if not CHARGE_COSTS:
            rup, cost_pts = 0.0, 0.0
        out.append({"i": i, "entry": entry,
                    "pts": float(pts), "net": float(pts) - cost_pts,
                    "cost_pts": cost_pts, "cost_rs": rup,
                    "res": res, "mins": k - i,
                    "side": "LONG" if side > 0 else "SHORT"})
    return out


def equity_report(rows, lot: int) -> dict:
    """Capital actually needed: margin for one lot PLUS the worst drawdown.

    Peak margin matters more than average: you must have the cash on the worst
    day, not the typical one. And a strategy is only survivable if you can fund
    its deepest losing streak without being forced out.
    """
    if not rows:
        return {}
    ordered = sorted(rows, key=lambda r: r["i"])
    pnl = np.array([r["net"] for r in ordered]) * lot
    curve = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[0.0], curve]))[1:]
    dd = curve - peak
    max_price = max(r["entry"] for r in ordered)
    margin = max_price * lot * MARGIN_PCT
    return {
        "total": float(curve[-1]),
        "max_dd": float(dd.min()),
        "margin": float(margin),
        "capital": float(margin + abs(dd.min())),
        "best": float(pnl.max()),
        "worst": float(pnl.min()),
        "curve": curve,
    }


def stats(rows, lot: int = LOT_FALLBACK):
    if not rows:
        return dict(n=0, win=0.0, avg=0.0, net=0.0, t=0.0, tot=0.0,
                    rupees=0.0, cost_rs=0.0)
    a = np.array([r["pts"] for r in rows])
    nt = np.array([r["net"] for r in rows])
    sd_ = nt.std(ddof=1) if len(nt) > 1 else 0.0
    se = sd_ / math.sqrt(len(nt)) if sd_ else 0.0
    return dict(n=len(a), win=100 * float((a > 0).mean()),
                avg=float(a.mean()), net=float(nt.mean()),
                t=float(nt.mean() / se) if se else 0.0,
                tot=float(nt.sum()), rupees=float(nt.sum()) * lot,
                cost_rs=float(np.sum([r["cost_rs"] for r in rows])))


def breakdown(rows) -> None:
    """How trades ACTUALLY ended — the thing a win rate hides.

    A win rate counts any positive result, including a trade that drifted a
    few points and timed out. If most trades end at the close rather than at
    the target or the stop, the 'breakeven win rate' implied by the
    reward:risk is meaningless, because real wins are nothing like full size.
    That is how a 55% win rate still loses money.
    """
    if not rows:
        return
    print(f"    {'outcome':10s}{'count':>7s}{'share':>8s}{'avg pts':>10s}")
    for res in ("target", "stop", "eod"):
        sub = [r["pts"] for r in rows if r["res"] == res]
        if not sub:
            continue
        print(f"    {res:10s}{len(sub):7d}{100 * len(sub) / len(rows):7.0f}%"
              f"{np.mean(sub):+10.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=730, help="calendar days (default 2y)")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--rsi-len", type=int, default=RSI_LEN)
    ap.add_argument("--ob", type=float, default=80.0)
    ap.add_argument("--os", dest="os_", type=float, default=20.0)
    ap.add_argument("--draws", type=int, default=50)
    ap.add_argument("--target", type=float, default=0.25,
                    help="profit target as a %% of entry (default 0.25)")
    ap.add_argument("--stop", type=float, default=0.50,
                    help="stop-loss as a %% of entry (default 0.50)")
    ap.add_argument("--lot", type=int, default=None,
                    help="override lot size (default: read from Angel One)")
    ap.add_argument("--no-cost", action="store_true",
                    help="ignore ALL costs - pure strategy result")
    ap.add_argument("--reverse", action="store_true",
                    help="treat extremes as REVERSALS: >80 sell, <20 buy")
    ap.add_argument("--symbol", choices=["nifty", "banknifty"], default="nifty")
    args = ap.parse_args()
    globals()["CHARGE_COSTS"] = not args.no_cost
    globals()["TICKER"], fallback = SYMBOLS[args.symbol]
    globals()["LOT_FALLBACK"] = fallback
    TGT, STP = args.target / 100.0, args.stop / 100.0
    LOT = index_lot_size(args.lot)

    print(f"{args.symbol.upper()} FUTURES — dual RSI({args.rsi_len}) 1min + 5min, "
          f"OB {args.ob:.0f} / OS {args.os_:.0f}")
    print(f"  TARGET  +{args.target:.2f}%   (profit, exit here)")
    print(f"  STOP    -{args.stop:.2f}%   (loss, exit here)")
    print(f"  reward:risk {TGT / STP:.2f}:1  ->  need "
          f"{100 * STP / (STP + TGT):.1f}% of trades to win just to break even")
    print(f"  lot size {LOT} "
          f"({'your override' if args.lot else 'from Angel One instrument master'})")

    df = build(args.days, args.refresh, args.rsi_len)
    print(f"\n  {len(df):,} one-minute bars, "
          f"{df.index[0].date()} to {df.index[-1].date()}")
    longs, shorts = signals(df, args.ob, args.os_)
    print(f"  signals: {len(longs)} LONG (both RSI>{args.ob:.0f}), "
          f"{len(shorts)} SHORT (both RSI<{args.os_:.0f})\n")
    if not (longs or shorts):
        print("  No signals. Both RSIs rarely reach the same extreme at once —")
        print("  try:  python rsi_futures_test.py --ob 70 --os 30")
        return

    # ---------------- the combination you asked for ----------------
    # MOMENTUM reading (default): RSI>80 on both -> BUY, RSI<20 on both -> SELL.
    # REVERSAL reading (--reverse): the opposite, which is how RSI extremes are
    # more commonly taught. Both are tested because your description could mean
    # either, and the answer is not obvious in advance.
    a_side, b_side = (+1, -1) if not args.reverse else (-1, +1)
    print(f"  direction: {'REVERSAL' if args.reverse else 'MOMENTUM'} — "
          f"RSI>{args.ob:.0f} both = {'SELL' if args.reverse else 'BUY'}, "
          f"RSI<{args.os_:.0f} both = {'BUY' if args.reverse else 'SELL'}")
    if not CHARGE_COSTS:
        print("  costs: IGNORED (--no-cost) — gross strategy result only")
    print()

    rows = (simulate(df, longs, a_side, STP, TGT, LOT)
            + simulate(df, shorts, b_side, STP, TGT, LOT))
    s = stats(rows, LOT)
    need = 100.0 * STP / (STP + TGT)
    cost_pts = float(np.mean([r["cost_pts"] for r in rows])) if rows else 0.0
    cost_rs = float(np.mean([r["cost_rs"] for r in rows])) if rows else 0.0
    print(f"  RESULT: trades {s['n']} | win {s['win']:.1f}% "
          f"(need {need:.1f}%) | gross {s['avg']:+.1f} pts "
          f"| net {s['net']:+.1f} pts | t={s['t']:+.2f}")
    print(f"  TOTAL : Rs {s['rupees']:+,.0f} over {args.days} days, "
          f"of which Rs {s['cost_rs']:,.0f} went on costs")
    avg_px = float(np.mean([r["entry"] for r in rows])) if rows else 1.0
    print(f"  cost per round trip: Rs {cost_rs:,.0f} = {cost_pts:.1f} points "
          f"= {100 * cost_pts / avg_px:.3f}% of price")
    print(f"    (brokerage Rs {2 * BROKERAGE_PER_LEG:.0f}; the rest is STT, "
          f"exchange, GST, stamp, slippage)")
    print(f"    costs eat {100 * (cost_pts / avg_px) / TGT:.0f}% of every "
          f"winning trade at a +{args.target:.2f}% target\n")
    # ---------------- capital required and the verdict ----------------
    eq = equity_report(rows, LOT)
    if eq:
        print("  CAPITAL AND PROFIT (1 lot, one position at a time)")
        print(f"    margin for 1 lot          Rs {eq['margin']:>12,.0f}   "
              f"({MARGIN_PCT:.0%} of notional at the highest price seen)")
        print(f"    worst drawdown            Rs {eq['max_dd']:>12,.0f}")
        print(f"    CAPITAL YOU MUST HAVE     Rs {eq['capital']:>12,.0f}   "
              f"(margin + worst drawdown)")
        print(f"    best single trade         Rs {eq['best']:>12,.0f}")
        print(f"    worst single trade        Rs {eq['worst']:>12,.0f}")
        print(f"    TOTAL P&L over {args.days}d     Rs {eq['total']:>12,.0f}")
        ret = 100 * eq["total"] / eq["capital"] if eq["capital"] else 0.0
        yrs = max(args.days / 365.0, 0.01)
        print(f"    return on capital        {ret:>12.1f}%   "
              f"over {yrs:.1f} years = {ret / yrs:+.1f}% a year")
        print(f"\n    VERDICT: "
              f"{'PROFITABLE' if eq['total'] > 0 else 'LOSS-MAKING'} "
              f"Rs {eq['total']:+,.0f}"
              + ("" if CHARGE_COSTS else "   <-- and this IGNORES ALL COSTS"))
        print()

    print("    How these trades ACTUALLY ended:")
    breakdown(rows)
    print("\n    A win rate counts ANY positive result, including a trade that")
    print("    drifted a few points and timed out. If 'eod' dominates below,")
    print("    the breakeven figure implied by reward:risk does not apply —")
    print("    real wins are nowhere near full size. That is how a 55% win")
    print("    rate still loses money.\n")

    # ---------------- the whole landscape ----------------
    print(f"  POINTS PER TRADE ({'after costs' if CHARGE_COSTS else 'GROSS, no costs'})"
          f" — target across, stop down")
    hdr = f"{'':>12s}" + "".join(f"{f'+{t * 100:.2f}%':>11s}" for t in GRID_TARGETS)
    print(hdr)
    print("-" * len(hdr))
    for sp in GRID_STOPS:
        line = f"{f'-{sp * 100:.2f}% stop':>12s}"
        for tg in GRID_TARGETS:
            # must use the SAME direction as the headline result - the first
            # version hardcoded +1/-1, so --reverse printed momentum numbers
            r = (simulate(df, longs, a_side, sp, tg, LOT)
                 + simulate(df, shorts, b_side, sp, tg, LOT))
            line += f"{stats(r, LOT)['net']:+11.1f}"
        print(line)
    print("\n  Every cell is after costs. A grid is shown deliberately: one")
    print("  positive cell among twenty is what you would expect from noise,")
    print("  so only a BLOCK of positive cells means anything.")

    # ---------------- control: same exits, random entry times ----------------
    print(f"\n  Control: {args.draws} random-entry runs at YOUR setting")
    rng = np.random.default_rng(5)
    want = len(longs) + len(shorts)
    dist = []
    for _ in range(args.draws):
        idx = sorted(int(x) for x in rng.integers(400, len(df) - 200, size=want))
        half = len(idx) // 2
        rr = (simulate(df, idx[:half], a_side, STP, TGT, LOT)
              + simulate(df, idx[half:], b_side, STP, TGT, LOT))
        st = stats(rr, LOT)
        if st["n"]:
            dist.append(st["net"])
    if dist:
        d = np.array(dist)
        real = s["net"]
        pct = 100.0 * float((d < real).mean())
        lo, hi = np.percentile(d, [5, 95])
        print(f"  signal {real:+.1f} pts  vs random {d.mean():+.1f} pts "
              f"[5-95%: {lo:+.1f}, {hi:+.1f}]  ->  percentile {pct:.0f}%")
        verdict = ("BEATS random — the RSI extremes carry information"
                   if pct > 95 else
                   "WORSE than random" if pct < 5 else
                   "no evidence of edge — RSI extremes add nothing over "
                   "random entry")
        print(f"  {verdict}")

    print("\n  Futures needs no option pricing model, so unlike every earlier")
    print("  test here, nothing above depends on assumptions about premiums.")


if __name__ == "__main__":
    main()
