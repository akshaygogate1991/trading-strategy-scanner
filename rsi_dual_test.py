"""Nifty intraday: dual-timeframe RSI extremes — backtest before trading it.

THE RULES AS GIVEN
  * Nifty 50 only
  * Two RSI readings: one on 1-minute candles, one on 5-minute
  * Overbought 80, oversold 20
  * BOTH above 80  -> buy CALL
  * BOTH below 20  -> buy PUT
  * Take profit at +1%
  * Stop-loss 50%

THREE THINGS I HAD TO INTERPRET — check these before trusting any number
-----------------------------------------------------------------------
1. "+1% profit" vs "50% stop" are on DIFFERENT SCALES, so the script tests
   several readings side by side:
       - +1% on NIFTY itself (226 points at 22,600 — a big intraday move)
       - +25% / +50% / +100% on the PREMIUM
   The stop is always -50% of premium. If 1% meant premium, a +1% target
   against a -50% stop would need a ~98% win rate just to break even, so
   that reading is almost certainly not what you meant.

2. Your screenshot shows "RSI divergence 5" and "RSI divergence 1" — those
   are RSI LENGTHS, not timeframes. The bottom panel is a square wave
   flipping 0/100, which is what RSI(length=1) always looks like: it only
   says "was this candle up or down". It carries no information beyond the
   candle's colour. This script uses genuine 1-minute and 5-minute
   TIMEFRAMES with RSI length 14, which is what I think you meant. Set
   --rsi-len to change it.

3. RSI above 80 on both timeframes is a MOMENTUM entry, not a reversal one.
   You are buying after a sharp move, when the option is already expensive
   and IV is elevated. That is a real strategy family, but it is the
   opposite of how RSI is usually taught, so it is worth being deliberate
   about it.

No lookahead: the 5-minute RSI is forward-filled onto 1-minute bars only
after that 5-minute candle has closed.

Run:  python rsi_dual_test.py
      python rsi_dual_test.py --days 90 --refresh
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

TICKER = "^NSEI"
LOT = 75
RSI_LEN = 14
OVERBOUGHT, OVERSOLD = 80.0, 20.0
STOP_PREMIUM = -0.50          # your 50% stop, on the premium
DAYS_TO_EXPIRY = 3            # a weekly Nifty option mid-week
RISK_FREE = 0.065
IV_FLOOR, IV_CAP = 0.08, 0.60
COST_PCT = 0.02               # brokerage + spread per round trip, index options
MAX_HOLD_MIN = 120            # abandon after 2 hours; also forces same-day exit

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rsi_data")

# The exit variants tested. "under" = % move in NIFTY, "prem" = % of premium.
TARGETS = [
    ("+1% NIFTY (your rule, read as index)", "under", 0.01),
    ("+0.5% NIFTY", "under", 0.005),
    ("+25% premium", "prem", 0.25),
    ("+50% premium", "prem", 0.50),
    ("+100% premium", "prem", 1.00),
    ("no target (EOD / 2h only)", "prem", None),
]


# --------------------------------------------------------------- data
def _chunk(interval: str) -> int:
    """Calendar days per request, staying under the 500-candle cap."""
    per_day = {"ONE_MINUTE": 375, "FIVE_MINUTE": 75}[interval]
    return max(1, int((500 * 0.8 / per_day) / 0.68))


def fetch(interval: str, days: int, refresh: bool = False) -> pd.DataFrame:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"NIFTY_{interval}_{days}d.csv")
    if os.path.exists(path) and not refresh:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if len(df) > 500:
            print(f"  {interval}: {len(df)} cached candles")
            return df

    client = sd._session()
    token, _sym, exch = sd.resolve_token(TICKER)
    step = _chunk(interval)
    end = dt.datetime.now()
    start = end - dt.timedelta(days=days)
    frames, cursor, n = [], start, 0
    print(f"  {interval}: fetching ~{days}d in {step}-day chunks...")
    while cursor < end:
        cend = min(cursor + dt.timedelta(days=step), end)
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
                n += len(f)
        except Exception as exc:
            print(f"    ! {cursor.date()}: {exc}")
        cursor = cend
        time.sleep(0.35)                      # stay under 180 req/min
    if not frames:
        raise RuntimeError(f"No {interval} data returned")
    df = pd.concat(frames)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_csv(path)
    print(f"    got {len(df)} candles")
    return df


# --------------------------------------------------------------- indicators
def rsi(close: pd.Series, n: int = RSI_LEN) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def bs(S, K, T, iv, kind, r=RISK_FREE):
    if T <= 0 or iv <= 0:
        return max(0.0, S - K) if kind == "C" else max(0.0, K - S)
    d1 = (math.log(S / K) + (r + iv * iv / 2) * T) / (iv * math.sqrt(T))
    d2 = d1 - iv * math.sqrt(T)
    N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    if kind == "C":
        return S * N(d1) - K * math.exp(-r * T) * N(d2)
    return K * math.exp(-r * T) * N(-d2) - S * N(-d1)


def build(days: int, refresh: bool, rsi_len: int) -> pd.DataFrame:
    m1 = fetch("ONE_MINUTE", days, refresh)
    m5 = fetch("FIVE_MINUTE", days, refresh)

    m1["rsi1"] = rsi(m1["Close"], rsi_len)
    m5["rsi5"] = rsi(m5["Close"], rsi_len)

    # A 5-minute RSI is only known once that candle CLOSES, so shift it one
    # bar before mapping onto 1-minute data. Without this the backtest reads
    # the future and every result is worthless.
    s5 = m5["rsi5"].shift(1)
    m1["rsi5"] = s5.reindex(m1.index, method="ffill")

    # realised volatility from 1-minute returns, annualised, for pricing
    ret = np.log(m1["Close"] / m1["Close"].shift(1))
    m1["iv"] = (ret.rolling(375).std() * math.sqrt(252 * 375)).clip(IV_FLOOR, IV_CAP)
    m1["day"] = m1.index.normalize()
    return m1.dropna(subset=["rsi1", "rsi5", "iv"])


def signals(df: pd.DataFrame) -> tuple[list[int], list[int]]:
    """Fire on the BAR BOTH cross into the zone, not every bar they stay there."""
    r1, r5 = df["rsi1"].values, df["rsi5"].values
    calls, puts = [], []
    for i in range(1, len(df) - 1):
        both_ob = r1[i] > OVERBOUGHT and r5[i] > OVERBOUGHT
        both_ob_prev = r1[i - 1] > OVERBOUGHT and r5[i - 1] > OVERBOUGHT
        both_os = r1[i] < OVERSOLD and r5[i] < OVERSOLD
        both_os_prev = r1[i - 1] < OVERSOLD and r5[i - 1] < OVERSOLD
        if both_ob and not both_ob_prev:
            calls.append(i)
        elif both_os and not both_os_prev:
            puts.append(i)
    return calls, puts


def simulate(df, entries, kind, tgt_mode, tgt_val) -> list[dict]:
    """kind 'C' or 'P'. Pessimistic: if stop and target are both reachable
    inside one minute, the stop is taken."""
    close, high, low = df["Close"].values, df["High"].values, df["Low"].values
    iv = df["iv"].values
    day = df["day"].values
    n = len(df)
    out = []
    for i in entries:
        S0 = float(close[i])
        K = round(S0 / 50.0) * 50.0                 # Nifty strikes step 50
        v = float(iv[i])
        T0 = DAYS_TO_EXPIRY / 365.0
        p0 = bs(S0, K, T0, v, kind)
        if p0 <= 0.5:
            continue
        res, pnl = "eod", 0.0
        for k in range(i + 1, min(i + 1 + MAX_HOLD_MIN, n)):
            if day[k] != day[i]:                    # never hold overnight
                k -= 1
                break
            T = max(T0 - (k - i) / (375 * 252), 1e-6)
            vv = float(iv[k]) if np.isfinite(iv[k]) else v
            # worst price first (stop), then best (target)
            bad = bs(float(low[k] if kind == "C" else high[k]), K, T, vv, kind)
            good = bs(float(high[k] if kind == "C" else low[k]), K, T, vv, kind)
            if bad / p0 - 1 <= STOP_PREMIUM:
                pnl, res = STOP_PREMIUM, "stop"
                break
            if tgt_val is not None:
                if tgt_mode == "prem" and good / p0 - 1 >= tgt_val:
                    pnl, res = tgt_val, "target"
                    break
                if tgt_mode == "under":
                    moved = ((high[k] - S0) / S0 if kind == "C"
                             else (S0 - low[k]) / S0)
                    if moved >= tgt_val:
                        hit = S0 * (1 + tgt_val) if kind == "C" else S0 * (1 - tgt_val)
                        pnl, res = bs(hit, K, T, vv, kind) / p0 - 1, "target"
                        break
        else:
            k = min(i + MAX_HOLD_MIN, n - 1)
        if res == "eod":
            T = max(T0 - (k - i) / (375 * 252), 1e-6)
            pnl = bs(float(close[k]), K, T, v, kind) / p0 - 1
        out.append({"pnl": float(pnl), "res": res, "mins": k - i,
                    "premium": p0, "kind": kind})
    return out


def stats(rows):
    if not rows:
        return dict(n=0, win=0, avg=0, t=0, tot=0)
    a = np.array([r["pnl"] for r in rows])
    sd_ = a.std(ddof=1) if len(a) > 1 else 0.0
    se = sd_ / math.sqrt(len(a)) if sd_ else 0.0
    return dict(n=len(a), win=100 * float((a > 0).mean()),
                avg=100 * float(a.mean()), t=float(a.mean() / se) if se else 0.0,
                tot=100 * float(a.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--rsi-len", type=int, default=RSI_LEN)
    ap.add_argument("--draws", type=int, default=50)
    args = ap.parse_args()

    print(f"Nifty dual-RSI test — 1min + 5min, RSI({args.rsi_len}), "
          f"OB {OVERBOUGHT:.0f} / OS {OVERSOLD:.0f}, stop {STOP_PREMIUM:.0%}\n")
    df = build(args.days, args.refresh, args.rsi_len)
    print(f"\n  {len(df)} usable 1-minute bars, "
          f"{df.index[0].date()} to {df.index[-1].date()}")

    calls, puts = signals(df)
    print(f"  signals: {len(calls)} CALL (both RSI>80), "
          f"{len(puts)} PUT (both RSI<20)\n")
    if not calls and not puts:
        print("  No signals at all. Both RSIs rarely hit the same extreme — try")
        print("  OB/OS of 70/30, or a shorter RSI length.")
        return

    hdr = f"{'exit rule':38s}{'n':>6s}{'win%':>8s}{'avg%':>9s}{'t':>7s}{'net%':>9s}"
    print(hdr)
    print("-" * len(hdr))
    for name, mode, val in TARGETS:
        rows = (simulate(df, calls, "C", mode, val)
                + simulate(df, puts, "P", mode, val))
        s = stats(rows)
        flag = "  <-- significant" if abs(s["t"]) >= 2 else ""
        print(f"{name:38s}{s['n']:6d}{s['win']:8.1f}{s['avg']:+9.2f}"
              f"{s['t']:+7.2f}{s['avg'] - COST_PCT * 100:+9.2f}{flag}")

    # ---- the control: same exits, random entry times ----
    print(f"\n  Control: {args.draws} random-entry runs "
          f"(+50% premium target, same stop)")
    rng = np.random.default_rng(3)
    dist = []
    want = len(calls) + len(puts)
    for _ in range(args.draws):
        idx = sorted(int(x) for x in rng.integers(400, len(df) - 150, size=want))
        half = len(idx) // 2
        rows = (simulate(df, idx[:half], "C", "prem", 0.50)
                + simulate(df, idx[half:], "P", "prem", 0.50))
        s = stats(rows)
        if s["n"]:
            dist.append(s["avg"])
    if dist:
        d = np.array(dist)
        real = stats(simulate(df, calls, "C", "prem", 0.50)
                     + simulate(df, puts, "P", "prem", 0.50))["avg"]
        pct = 100.0 * float((d < real).mean())
        lo, hi = np.percentile(d, [5, 95])
        print(f"  signal {real:+.2f}%  vs random {d.mean():+.2f}% "
              f"[5-95%: {lo:+.1f}, {hi:+.1f}]  ->  percentile {pct:.0f}%")
        print("  " + ("BEATS random" if pct > 95 else
                      "WORSE than random" if pct < 5 else
                      "no evidence of edge — the RSI extremes are not adding "
                      "information"))

    print(f"\n  Costs of ~{COST_PCT * 100:.0f}% per round trip are in 'net%'.")
    print("  Intraday option spreads on Nifty are tight, but at this trade")
    print("  frequency costs compound fast — check 'net%', not 'avg%'.")


if __name__ == "__main__":
    main()
