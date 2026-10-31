"""How far is options_backtest.py's MODEL from REAL Nifty option prices?

options_backtest.py has to invent option prices (expired contracts are deleted,
so real history does not exist). This script measures the size of that
invention on the contracts that ARE live: Angel One serves a live option's own
1-minute history back to its listing date.

It fetches, for the last few trading days:
  * Nifty index 1-minute candles
  * 1-minute candles for the live weekly CE/PE contracts near the money
  * daily India VIX
and reports three things:

  1. IV RATIO — the real option's implied volatility divided by the previous
     day's VIX, for at-the-money contracts, by days to expiry. This is the
     --iv-mult to pass to options_backtest.py.
  2. LEVEL ERROR — model premium vs real premium with IV = VIX x that ratio.
  3. DECAY CHECK — the real change in premium over 30/60/120 minutes against
     the model's prediction, using both the calendar clock (the first
     version's bug) and the trading-time clock (the fix). The clock whose
     average error is nearer zero is the one that matches the market. This is
     the number that decides whether the backtest's P&L can be trusted.

Credentials: read from .streamlit/secrets.toml by smartapi_data.py, as always.
Nothing here prints them — the SmartApi library's habit of logging request
headers on a failed login is filtered out below.

Run:  python option_model_check.py               (fetch + analyse, 5 days)
      python option_model_check.py --days 8 --strikes 4
      python option_model_check.py --no-fetch    (re-analyse the cache)
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import time

import numpy as np
import pandas as pd

import options_backtest as OB
import rsi_month_test as R
import smartapi_data as sd

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opt_check_data")


# ---------------------------------------------------------------- no secrets
class _HideRequestDetails(logging.Filter):
    """SmartApi logs the full request on failure — API key, client code, MPIN
    and TOTP included. Keep only the 'Response:' part, which is the error."""
    def filter(self, record):
        msg = record.getMessage()
        if "Headers:" in msg or "Request:" in msg:
            tail = msg.split("Response:", 1)[1].strip() if "Response:" in msg else ""
            record.msg = "SmartApi request failed (request details hidden). " + tail
            record.args = ()
        return True


try:
    import logzero
    logzero.logger.addFilter(_HideRequestDetails())
except ImportError:
    pass


# ---------------------------------------------------------------- fetch
def fetch_days(exchange: str, token: str, days: list[dt.date]) -> pd.DataFrame | None:
    client = sd._session()
    frames, gap = [], 0.45
    for day in days:
        for attempt in range(4):
            try:
                resp = client.getCandleData({
                    "exchange": exchange, "symboltoken": token,
                    "interval": "ONE_MINUTE",
                    "fromdate": day.strftime("%Y-%m-%d 09:15"),
                    "todate": day.strftime("%Y-%m-%d 15:30"),
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
    if not frames:
        return None
    df = pd.concat(frames)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    return df[~df.index.duplicated(keep="last")].sort_index()


def recent_weekdays(n: int) -> list[dt.date]:
    out, x = [], dt.date.today()
    while len(out) < n:
        if x.weekday() < 5:
            out.append(x)
        x -= dt.timedelta(days=1)
    return sorted(out)


def vix_token():
    for r in sd._instrument_master():
        if (r.get("exch_seg") == "NSE"
                and r.get("name", "").upper().replace(" ", "") == "INDIAVIX"):
            return r["token"]
    return None


def fetch_vix_daily(days_back: int) -> pd.Series | None:
    tok = vix_token()
    if tok:
        client = sd._session()
        end = dt.date.today()
        resp = client.getCandleData({
            "exchange": "NSE", "symboltoken": tok, "interval": "ONE_DAY",
            "fromdate": (end - dt.timedelta(days=days_back)).strftime("%Y-%m-%d 09:15"),
            "todate": end.strftime("%Y-%m-%d 15:30"),
        })
        rows = resp.get("data") if isinstance(resp, dict) else None
        if rows:
            f = pd.DataFrame(rows, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
            f["ts"] = pd.to_datetime(f["ts"]).dt.tz_localize(None).dt.normalize()
            return f.set_index("ts")["Close"].astype(float)
    print("  ! India VIX not available from Angel One; using vix_data/ (may be stale)")
    return OB.load_vix()


def option_contracts(spot: float, n_expiries: int, n_strikes: int):
    today = dt.date.today()
    rows = []
    for r in sd._instrument_master():
        if (r.get("exch_seg") == "NFO" and r.get("name", "").upper() == "NIFTY"
                and r.get("instrumenttype") == "OPTIDX"):
            e = sd._parse_expiry(r.get("expiry", ""))
            if e and e >= today:
                rows.append((e, float(r["strike"]) / 100.0, r["symbol"][-2:],
                             r["symbol"], r["token"]))
    expiries = sorted({e for e, *_ in rows})[:n_expiries]
    atm = round(spot / OB.STRIKE_STEP) * OB.STRIKE_STEP
    wanted = {atm + k * OB.STRIKE_STEP for k in range(-n_strikes, n_strikes + 1)}
    return [r for r in rows if r[0] in expiries and r[1] in wanted
            and r[2] in ("CE", "PE")]


def do_fetch(args):
    os.makedirs(CACHE, exist_ok=True)
    days = recent_weekdays(args.days)
    print(f"Fetching {days[0]} to {days[-1]} ({len(days)} weekdays)")

    tok, _, exch = sd.resolve_token("^NSEI")
    spot = fetch_days(exch, tok, days)
    if spot is None:
        print("  no Nifty index candles returned — stopping")
        return False
    spot.to_csv(os.path.join(CACHE, "NIFTY_INDEX.csv"))
    print(f"  Nifty index: {len(spot):,} candles")

    vix = fetch_vix_daily(days_back=args.days * 2 + 20)
    vix.to_csv(os.path.join(CACHE, "INDIA_VIX_daily.csv"))

    cons = option_contracts(float(spot["Close"].iloc[-1]), args.expiries, args.strikes)
    print(f"  {len(cons)} option contracts "
          f"({len(cons) * len(days)} requests, ~{len(cons) * len(days) * 0.5 / 60:.0f} min)")
    for e, k, kind, sym, token in sorted(cons):
        path = os.path.join(CACHE, f"{sym}.csv")
        df = fetch_days("NFO", token, days)
        if df is None:
            print(f"    {sym:28s} no data")
            continue
        df["expiry"], df["strike"], df["kind"] = e.isoformat(), k, kind
        df.to_csv(path)
        print(f"    {sym:28s} {len(df):5d} candles")
    return True


# ---------------------------------------------------------------- analyse
def implied_vol(price, S, K, T, kind):
    """Bisection, vectorised. NaN where the price is outside no-arbitrage bounds."""
    lo, hi = np.full_like(price, 0.01), np.full_like(price, 2.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        p = OB.bs_price(S, K, T, mid, kind)
        lo = np.where(p < price, mid, lo)
        hi = np.where(p >= price, mid, hi)
    iv = (lo + hi) / 2
    ok = (iv > 0.011) & (iv < 1.99)
    return np.where(ok, iv, np.nan)


def units_to_expiry(ts: pd.DatetimeIndex, expiry: dt.date, gap: float) -> np.ndarray:
    """Same trading-time clock as options_backtest.prepare (weekday calendar)."""
    mod = ts.hour * 60 + ts.minute
    n_after = np.array([np.busday_count(t.date() + dt.timedelta(days=1),
                                        expiry + dt.timedelta(days=1)) for t in ts])
    return (930 - mod) + n_after * (375 + gap)


def load_cache():
    if not os.path.isdir(CACHE):
        return None
    spot_p = os.path.join(CACHE, "NIFTY_INDEX.csv")
    if not os.path.exists(spot_p):
        return None
    spot = pd.read_csv(spot_p, index_col=0, parse_dates=True)["Close"]
    vix = pd.read_csv(os.path.join(CACHE, "INDIA_VIX_daily.csv"),
                      index_col=0, parse_dates=True).iloc[:, 0].sort_index()
    opts = []
    for f in sorted(os.listdir(CACHE)):
        if f.startswith("NIFTY") and f.endswith(("CE.csv", "PE.csv")):
            o = pd.read_csv(os.path.join(CACHE, f), index_col=0, parse_dates=True)
            o["symbol"] = f[:-4]
            opts.append(o)
    return spot, vix, opts


def analyse(args):
    loaded = load_cache()
    if loaded is None or not loaded[2]:
        print("Nothing cached yet. Run without --no-fetch.")
        return
    spot, vix, opts = loaded
    m = R.load_all()
    gap = OB.overnight_weight(m[0]) if m else 469.0
    vdates = np.array([x.date() for x in vix.index])

    recs = []
    for o in opts:
        o = o[o["Volume"] > 0].join(spot.rename("S"), how="inner")
        if len(o) < 30:
            continue
        e = dt.date.fromisoformat(str(o["expiry"].iloc[0]))
        K = float(o["strike"].iloc[0])
        kind = 1 if o["kind"].iloc[0] == "CE" else -1
        exp_min = np.datetime64(dt.datetime.combine(e, dt.time(15, 30)), "m").astype(np.int64)
        tmin = o.index.values.astype("datetime64[m]").astype(np.int64)
        f = pd.DataFrame(index=o.index)
        f["P"], f["S"], f["K"], f["kind"] = o["Close"].values, o["S"].values, K, kind
        f["Tcal"] = (exp_min - tmin) / OB.MIN_PER_YEAR
        f["U"] = units_to_expiry(o.index, e, gap)
        j = np.searchsorted(vdates, o.index.date) - 1
        f["vix"] = np.where(j >= 0, vix.values[np.maximum(j, 0)] / 100.0, np.nan)
        f["dte"] = [np.busday_count(t.date(), e) for t in o.index]
        f["sym"] = o["symbol"].iloc[0]
        recs.append(f)
    if not recs:
        print("No option candles with volume overlap the index candles.")
        return
    f = pd.concat(recs)
    f = f[(f["Tcal"] > 0) & np.isfinite(f["vix"])]
    f["iv"] = implied_vol(f["P"].values, f["S"].values, f["K"].values,
                          f["Tcal"].values, f["kind"].values)
    f["ratio"] = f["iv"] / f["vix"]
    atm = f[(f["S"] - f["K"]).abs() <= OB.STRIKE_STEP / 2]

    print(f"\n{len(f):,} option-minutes with trades across {f['sym'].nunique()} contracts, "
          f"{f.index.min():%Y-%m-%d} to {f.index.max():%Y-%m-%d}")
    print(f"overnight-gap weight used for the trading clock: {gap:.0f} minutes")

    # ---- 1. IV ratio
    print("\n1. REAL ATM IMPLIED VOL / PREVIOUS-DAY VIX  (= --iv-mult)")
    print(f"   {'days to expiry':16s}{'minutes':>9s}{'median':>9s}{'25%':>8s}{'75%':>8s}")
    for dte, g in atm.dropna(subset=["ratio"]).groupby("dte"):
        q = g["ratio"].quantile([0.25, 0.5, 0.75]).values
        print(f"   {dte:<16d}{len(g):9d}{q[1]:9.2f}{q[0]:8.2f}{q[2]:8.2f}")
    held = atm[atm["dte"] >= 1]["ratio"].dropna()
    mult = float(held.median()) if len(held) else float("nan")
    print(f"   -> suggested --iv-mult {mult:.2f} (ATM, excluding expiry day, which "
          f"the backtest does not trade)")

    # ---- 2. level error
    if np.isfinite(mult):
        a = atm[atm["dte"] >= 1].copy()
        a["model"] = OB.bs_price(a["S"].values, a["K"].values, a["Tcal"].values,
                                 a["vix"].values * mult, a["kind"].values)
        err = a["model"] - a["P"]
        print("\n2. LEVEL ERROR, model minus real, ATM (premium points)")
        print(f"   median {err.median():+.2f} | middle half {err.quantile(.25):+.2f} "
              f"to {err.quantile(.75):+.2f} | real premium median {a['P'].median():.1f}")

    # ---- 3. decay check
    print("\n3. PREMIUM CHANGE OVER THE HOLD — real minus model (premium points)")
    print("   IV fixed at the real implied vol at the start, so only the clock "
          "differs.")
    print(f"   {'hold':>6s}{'samples':>9s}{'real avg':>10s}{'calendar':>11s}"
          f"{'trading':>10s}   (model error: nearer 0 is better)")
    starts = atm[(atm["dte"] >= 1) & np.isfinite(atm["iv"])]
    for h in (30, 60, 120):
        rows = []
        for sym, g in f.groupby("sym"):
            s0 = starts[starts["sym"] == sym]
            s0 = s0[s0.index.minute % 15 == 0]
            for t0, r0 in s0.iterrows():
                t1 = t0 + pd.Timedelta(minutes=h)
                if t1 not in g.index or t1.date() != t0.date():
                    continue
                r1 = g.loc[t1]
                cal = OB.bs_price(r1["S"], r0["K"], r1["Tcal"], r0["iv"], r0["kind"])
                trd = OB.bs_price(r1["S"], r0["K"], r0["Tcal"] * r1["U"] / r0["U"],
                                  r0["iv"], r0["kind"])
                real = r1["P"] - r0["P"]
                rows.append((real, real - (cal - r0["P"]), real - (trd - r0["P"])))
        if not rows:
            print(f"   {h:4d}m   no pairs")
            continue
        x = np.array(rows)
        print(f"   {h:4d}m{len(x):9d}{x[:, 0].mean():+10.2f}{x[:, 1].mean():+11.2f}"
              f"{x[:, 2].mean():+10.2f}")
    print("\n   Negative = real premiums lost MORE than the model said. If the "
          "trading-clock\n   column is clearly negative too, the backtest is still "
          "too kind to option buyers.")
    print("\nPaste this output back. A few days is a small sample — treat it as "
          "a calibration,\nnot a verdict, and re-run it next week to see if it "
          "holds.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=int, default=5, help="recent weekdays to fetch")
    ap.add_argument("--expiries", type=int, default=2, help="nearest weekly expiries")
    ap.add_argument("--strikes", type=int, default=3,
                    help="strikes each side of the money")
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args()
    if not args.no_fetch and not do_fetch(args):
        return
    analyse(args)


if __name__ == "__main__":
    main()
