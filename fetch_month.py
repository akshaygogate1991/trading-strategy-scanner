"""Fetch ONE month of Nifty futures 1-minute + 5-minute data, verify it, save it.

WHY MONTH BY MONTH
A single two-year pull is ~730 requests in one burst, which trips Angel One's
rate limit and leaves silent gaps. One month is ~21 trading days = ~26 requests,
paced slowly enough that refusals are rare — and you can SEE the completeness of
each month before moving on, instead of discovering holes after the backtest.

Each month lands in its own file, so a failed month only costs that month.

USAGE — one month at a time, newest first
    python fetch_month.py --month 2026-09
    python fetch_month.py --month 2026-08
    python fetch_month.py --month 2026-07
    ... back to 2025-10

Check progress at any time:
    python fetch_month.py --list

Re-do a month that came back incomplete:
    python fetch_month.py --month 2026-07 --refresh

When all twelve are done:
    python rsi_month_test.py
"""
from __future__ import annotations

import argparse
import calendar
import datetime as dt
import os
import time

import pandas as pd

import smartapi_data as sd

TICKER = "^NSEI"
MONTHS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nifty_months")
PER_SESSION = {"ONE_MINUTE": 375, "FIVE_MINUTE": 75}
# Deliberately slower than the limit allows. At ~26 requests a month the extra
# seconds cost nothing, and refusals cost a re-run.
GAP = 0.7
STEP_DAYS = {"ONE_MINUTE": 1, "FIVE_MINUTE": 7}


def month_bounds(ym: str) -> tuple[dt.date, dt.date]:
    y, m = (int(x) for x in ym.split("-"))
    last = calendar.monthrange(y, m)[1]
    return dt.date(y, m, 1), dt.date(y, m, last)


def path_for(interval: str, ym: str) -> str:
    return os.path.join(MONTHS, f"NIFTY_{interval}_{ym}.csv")


def fetch_interval(interval: str, ym: str, refresh: bool) -> pd.DataFrame | None:
    out_path = path_for(interval, ym)
    if os.path.exists(out_path) and not refresh:
        df = pd.read_csv(out_path, index_col=0, parse_dates=True)
        print(f"  {interval:13s} already have {len(df):,} candles "
              f"(use --refresh to redo)")
        return df

    start, end = month_bounds(ym)
    today = dt.date.today()
    if start > today:
        print(f"  {interval:13s} {ym} is in the future — nothing to fetch")
        return None
    end = min(end, today)

    client = sd._session()
    token, _sym, exch = sd.resolve_token(TICKER)
    step = STEP_DAYS[interval]

    frames, refused, asked = [], 0, 0
    cursor = start
    while cursor <= end:
        cend = min(cursor + dt.timedelta(days=step - 1), end)
        if step == 1 and cursor.weekday() >= 5:      # skip Sat/Sun
            cursor = cend + dt.timedelta(days=1)
            continue
        asked += 1
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
                    time.sleep(2.0 * (attempt + 1))
                    continue
                if attempt == 3:
                    refused += 1
                    print(f"    ! {cursor} refused after 4 tries")
                break
        cursor = cend + dt.timedelta(days=1)
        time.sleep(GAP)

    if not frames:
        print(f"  {interval:13s} NOTHING returned — check your connection")
        return None

    df = pd.concat(frames)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    os.makedirs(MONTHS, exist_ok=True)
    df.to_csv(out_path)

    # Judge completeness PER SESSION, not against a fixed 375.
    # A session is not reliably 375 one-minute candles: the 09:15 and 15:30
    # boundaries vary, and half-days exist. What actually signals a problem is
    # a session that is far shorter than its neighbours, or a refused request.
    per_day = df.groupby(df.index.normalize()).size()
    sessions = len(per_day)
    typical = float(per_day.median())
    short = per_day[per_day < 0.80 * typical]
    pct = 100.0 * len(df) / (PER_SESSION[interval] * sessions) if sessions else 0.0

    if refused == 0 and len(short) == 0:
        status = "OK"
    elif refused:
        status = f"INCOMPLETE — {refused} request(s) refused, re-run --refresh"
    else:
        status = f"CHECK — {len(short)} short session(s)"

    print(f"  {interval:13s} {len(df):>7,} candles | {sessions:2d} sessions | "
          f"median {typical:.0f}/session | {pct:5.1f}% of nominal | {status}")
    if len(short):
        for d_, c_ in short.items():
            print(f"      short: {d_.date()} only {c_} candles "
                  f"(typical {typical:.0f}) — may be a half-day")
    return df


def list_months() -> None:
    os.makedirs(MONTHS, exist_ok=True)
    files = sorted(os.listdir(MONTHS))
    yms = sorted({f.split("_")[-1].replace(".csv", "") for f in files
                  if f.endswith(".csv")}, reverse=True)
    if not yms:
        print("Nothing downloaded yet. Start with the most recent full month:")
        print("    python fetch_month.py --month 2026-09")
        return
    print(f"{'month':10s}{'1-min':>12s}{'5-min':>10s}{'sessions':>10s}  status")
    print("-" * 52)
    total = 0
    for ym in yms:
        p1, p5 = path_for("ONE_MINUTE", ym), path_for("FIVE_MINUTE", ym)
        n1 = n5 = sess = 0
        if os.path.exists(p1):
            d1 = pd.read_csv(p1, index_col=0, parse_dates=True)
            n1, sess = len(d1), d1.index.normalize().nunique()
        if os.path.exists(p5):
            n5 = len(pd.read_csv(p5, index_col=0, parse_dates=True))
        # a month is fine if it has both files and a sane candles-per-session
        per_sess = (n1 / sess) if sess else 0
        ok = sess >= 15 and per_sess >= 300 and n5 > 0
        total += n1
        print(f"{ym:10s}{n1:>12,}{n5:>10,}{sess:>10d}  "
              f"{'OK' if ok else 'CHECK — re-run with --refresh'}")
    print(f"\n{len(yms)} months, {total:,} one-minute candles total")
    if len(yms) >= 12:
        print("\nTwelve months done — now run:  python rsi_month_test.py")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", help="YYYY-MM, e.g. 2026-09")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list or not args.month:
        list_months()
        return

    print(f"\nNifty futures — {args.month}")
    for interval in ("ONE_MINUTE", "FIVE_MINUTE"):
        fetch_interval(interval, args.month, args.refresh)

    # suggest the previous month so you can just keep going
    y, m = (int(x) for x in args.month.split("-"))
    prev = f"{y - 1}-12" if m == 1 else f"{y}-{m - 1:02d}"
    print(f"\nNext:  python fetch_month.py --month {prev}")
    print(f"Check: python fetch_month.py --list")


if __name__ == "__main__":
    main()
