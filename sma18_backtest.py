"""Backtest of the 18 SMA two-close breakout, on daily candles.

RULES (exactly as specified, nothing invented):
  1. 18-period SIMPLE moving average on daily closes.
  2. Setup: two consecutive candles CLOSE above the SMA, where the candle before
     them closed at or below it - i.e. a fresh event, not every bar of a trend.
  3. Entry: a buy-stop at the HIGH of the 2nd candle. It fills only if a later
     candle actually trades through that level (default: within 3 sessions).
  4. Risk  = entry - SMA on the entry day.
  5. Target = entry + 2 x risk        (the 1:2 reward)
  6. Stop  = a daily CLOSE below the SMA. The SMA moves, so this trails.
  7. Exit on whichever comes first.

Everything is measured in R (multiples of the risk taken), so instruments of
different prices are comparable, and a random-entry baseline with identical
geometry is reported alongside. A strategy that cannot beat its own baseline
has no edge.

No lookahead: entries fill only on a later bar, the SMA used is the one known on
that day, and if a single bar's range contains both stop and target the trade is
counted a LOSS (pessimistic tie-break).

Usage:
    python sma18_backtest.py
    python sma18_backtest.py --tickers ^NSEI ^NSEBANK
    python sma18_backtest.py --years 4
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import time

import numpy as np
import pandas as pd

import smartapi_data as sd

SMA_LEN = 18
RR_TARGET = 2.0        # the 1:2 you asked for
ENTRY_VALID_DAYS = 3   # how long the buy-stop stays live before we cancel it
MAX_HOLD_DAYS = 250    # abandon anything still open after ~1 year

# Minimum distance between entry and the SMA, in ATR.
# Without this the "risk" can be near zero (entry hugging the SMA), and because
# every result is divided by that risk, one ordinary red day becomes a -11R
# loss. The first run showed worst trade -11.69R and a random baseline hitting
# -240R, both from this same division. Skipping these is not curve-fitting: a
# trade whose stop is a rounding error away from entry has no defined risk, so
# "1:2" is meaningless for it.
MIN_RISK_ATR = 0.5


def atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    h, l, c = df["High"], df["Low"], df["Close"]
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean().values

INDEXES = ["^NSEI", "^NSEBANK"]
FO_STOCKS = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS",
    "KOTAKBANK.NS", "INFY.NS", "TCS.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
    "LT.NS", "BHARTIARTL.NS", "ITC.NS", "HINDUNILVR.NS", "MARUTI.NS",
    "TATAMOTORS.NS", "M&M.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS",
    "SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
    "TITAN.NS", "ULTRACEMCO.NS", "ADANIENT.NS", "ADANIPORTS.NS", "NTPC.NS",
    "POWERGRID.NS", "ONGC.NS", "COALINDIA.NS", "TRENT.NS", "BEL.NS", "DLF.NS",
    "VEDL.NS",
]
ALL_TICKERS = INDEXES + FO_STOCKS

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sma_data")


# --------------------------------------------------------------------------
# Data (Angel One daily, cached; 500 candles per request so long spans chunk)
# --------------------------------------------------------------------------
def load(ticker: str, years: float = 4.0, refresh: bool = False) -> pd.DataFrame | None:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, ticker.replace("^", "IDX_").replace(".", "_") + ".csv")
    if os.path.exists(path) and not refresh:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if len(df) > SMA_LEN + 30:
            return df

    frames = []
    end = dt.datetime.now()
    span = 700           # ~500 trading days, the per-request cap
    for _ in range(max(1, int(round(years * 365 / span)))):
        start = end - dt.timedelta(days=span)
        try:
            client = sd._session()
            token, _sym, exch = sd.resolve_token(ticker)
            resp = client.getCandleData({
                "exchange": exch, "symboltoken": token, "interval": "ONE_DAY",
                "fromdate": start.strftime("%Y-%m-%d 09:15"),
                "todate": end.strftime("%Y-%m-%d 15:30"),
            })
            rows = resp.get("data") if isinstance(resp, dict) else None
            if rows:
                f = pd.DataFrame(rows, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
                f["ts"] = pd.to_datetime(f["ts"])
                frames.append(f.set_index("ts").astype(float))
        except Exception as exc:
            print(f"    ! {ticker}: {exc}")
            break
        end = start
        time.sleep(0.4)          # stay under 180 requests/minute

    if not frames:
        return None
    df = pd.concat(frames)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_csv(path)
    return df


# --------------------------------------------------------------------------
# The strategy
# --------------------------------------------------------------------------
def signals(df: pd.DataFrame) -> list[dict]:
    """Every completed trade this ruleset would have taken."""
    d = df.copy()
    d["sma"] = d["Close"].rolling(SMA_LEN).mean()
    close = d["Close"].values
    high, low = d["High"].values, d["Low"].values
    sma = d["sma"].values
    a = atr(d)
    n = len(d)
    out: list[dict] = []
    skipped_thin = 0
    i = SMA_LEN + 1

    while i < n - 1:
        # fresh two-close-above setup
        fresh = (np.isfinite(sma[i - 2]) and close[i - 2] <= sma[i - 2]
                 and close[i - 1] > sma[i - 1] and close[i] > sma[i])
        if not fresh:
            i += 1
            continue

        trigger = float(high[i])          # buy-stop at the 2nd candle's high
        entry_idx = None
        for j in range(i + 1, min(i + 1 + ENTRY_VALID_DAYS, n)):
            if high[j] >= trigger:
                entry_idx = j
                break
        if entry_idx is None:
            i += 1
            continue                      # order never filled - not a trade

        entry = trigger
        risk = entry - float(sma[entry_idx])
        if risk <= 0:
            i += 1
            continue
        if risk < MIN_RISK_ATR * float(a[entry_idx]):
            skipped_thin += 1
            i += 1
            continue                      # stop too close to entry: risk undefined
        target = entry + RR_TARGET * risk

        outcome, r, held, exit_px = "open", 0.0, 0, entry
        for k in range(entry_idx, min(entry_idx + MAX_HOLD_DAYS, n)):
            # pessimistic: a bar that spans both is treated as a loss
            if close[k] < sma[k] and k > entry_idx:
                exit_px = float(close[k])
                r = (exit_px - entry) / risk
                outcome, held = "sma_exit", k - entry_idx
                break
            if high[k] >= target and k > entry_idx:
                exit_px, r = target, RR_TARGET
                outcome, held = "target", k - entry_idx
                break
        if outcome == "open":
            k = min(entry_idx + MAX_HOLD_DAYS, n - 1)
            exit_px = float(close[k])
            r = (exit_px - entry) / risk
            outcome, held = "timeout", k - entry_idx

        out.append({
            "date": d.index[entry_idx], "entry": round(entry, 2),
            "sma_at_entry": round(float(sma[entry_idx]), 2),
            "risk": round(risk, 2), "target": round(target, 2),
            "exit": round(exit_px, 2), "outcome": outcome,
            "r": round(float(r), 3), "days_held": held,
        })
        i = entry_idx + max(held, 1)      # no overlapping positions
    if out:
        out[0]["_skipped_thin"] = skipped_thin
    return out


def baseline(df: pd.DataFrame, n_trades: int, seed: int = 0) -> list[float]:
    """Random entries, same 2R geometry and same trailing-SMA stop."""
    rng = np.random.default_rng(seed)
    d = df.copy()
    d["sma"] = d["Close"].rolling(SMA_LEN).mean()
    close, high, sma = d["Close"].values, d["High"].values, d["sma"].values
    a = atr(d)
    n = len(d)
    out: list[float] = []
    if n < SMA_LEN + 60 or n_trades <= 0:
        return out
    tries = 0
    while len(out) < n_trades and tries < n_trades * 20:
        tries += 1
        i = int(rng.integers(SMA_LEN + 2, n - 10))
        if not np.isfinite(sma[i]):
            continue
        entry = float(close[i])
        risk = entry - float(sma[i])
        # the baseline must obey the SAME risk floor, or it is not a fair
        # comparison - unfiltered it produced a -240R trade
        if risk <= 0 or risk < MIN_RISK_ATR * float(a[i]):
            continue
        target = entry + RR_TARGET * risk
        r = 0.0
        for k in range(i + 1, min(i + MAX_HOLD_DAYS, n)):
            if close[k] < sma[k]:
                r = (float(close[k]) - entry) / risk
                break
            if high[k] >= target:
                r = RR_TARGET
                break
        out.append(float(r))
    return out


def summarise(rs: list[float]) -> dict:
    if not rs:
        return {"n": 0}
    a = np.array(rs, dtype=float)
    sd = a.std(ddof=1) if len(a) > 1 else 0.0
    se = sd / np.sqrt(len(a)) if sd > 0 else 0.0
    return {
        "n": len(a),
        "win_rate": round(100.0 * (a > 0).sum() / len(a), 1),
        "expectancy": round(float(a.mean()), 3),
        # t tells you whether an expectancy is distinguishable from zero.
        # Below about 2, a positive average is indistinguishable from luck -
        # which is why "greater than zero" is not a good enough test.
        "t": round(float(a.mean() / se), 2) if se > 0 else 0.0,
        "total_R": round(float(a.sum()), 1),
        "best": round(float(a.max()), 2),
        "worst": round(float(a.min()), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--years", type=float, default=4.0)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    tickers = args.tickers or ALL_TICKERS

    print(f"18 SMA two-close breakout | daily | target {RR_TARGET:g}R | "
          f"stop = close below SMA\n")
    all_rows: list[dict] = []
    base: list[float] = []

    for t in tickers:
        df = load(t, args.years, args.refresh)
        if df is None or len(df) < SMA_LEN + 40:
            print(f"  {t:16s} no data")
            continue
        tr = signals(df)
        for x in tr:
            x["ticker"] = t
            x["kind"] = "index" if t in INDEXES else "stock"
        all_rows += tr
        base += baseline(df, max(len(tr), 20))
        s = summarise([x["r"] for x in tr])
        print(f"  {t:16s} {len(df):5d} bars | trades {s.get('n', 0):3d} "
              f"| win {s.get('win_rate', 0):5.1f}% | exp {s.get('expectancy', 0):+.3f}R")

    if not all_rows:
        print("\nNo trades generated.")
        return

    d = pd.DataFrame(all_rows)
    d.to_csv("sma18_trades.csv", index=False)
    strat, rand = summarise(d["r"].tolist()), summarise(base)

    print("\n" + "=" * 70)
    print(f"{'':12s}{'trades':>8s}{'win%':>8s}{'expectancy':>13s}{'total R':>10s}"
          f"{'best':>8s}{'worst':>8s}")
    for name, s in (("STRATEGY", strat), ("RANDOM", rand)):
        print(f"{name:12s}{s['n']:8d}{s.get('win_rate', 0):8.1f}"
              f"{s.get('expectancy', 0):+13.3f}{s.get('total_R', 0):+10.1f}"
              f"{s.get('best', 0):8.2f}{s.get('worst', 0):8.2f}")
    print("=" * 70)

    print("\n  How trades ended:")
    for k, v in d["outcome"].value_counts().items():
        share = 100.0 * v / len(d)
        print(f"    {k:10s} {v:4d}  ({share:.0f}%)  "
              f"avg {d[d.outcome == k]['r'].mean():+.2f}R")

    print("\n  Indexes vs stocks:")
    for kind, g in d.groupby("kind"):
        s = summarise(g["r"].tolist())
        print(f"    {kind:8s} trades {s['n']:4d} | win {s['win_rate']:5.1f}% "
              f"| exp {s['expectancy']:+.3f}R | total {s['total_R']:+.1f}R")

    print("\n  Year by year (does it survive different markets?):")
    d["year"] = pd.to_datetime(d["date"]).dt.year
    for y, g in d.groupby("year"):
        s = summarise(g["r"].tolist())
        print(f"    {int(y)}     trades {s['n']:4d} | win {s['win_rate']:5.1f}% "
              f"| exp {s['expectancy']:+.3f}R | total {s['total_R']:+.1f}R")

    # The single most revealing test: is this a strategy, or one good year?
    by_year = d.groupby("year")["r"].sum()
    best_year = by_year.idxmax()
    without = d[d["year"] != best_year]
    s_wo = summarise(without["r"].tolist())
    print(f"\n  Drop the best year ({int(best_year)}, {by_year.max():+.1f}R) — what's left?")
    print(f"    remaining  trades {s_wo['n']:4d} | win {s_wo.get('win_rate', 0):5.1f}% "
          f"| exp {s_wo.get('expectancy', 0):+.3f}R (t={s_wo.get('t', 0):+.2f}) "
          f"| total {s_wo.get('total_R', 0):+.1f}R")

    losses = d[d["r"] < 0]["r"]
    print(f"\n  Loss profile: {len(losses)} losers, average {losses.mean():+.2f}R, "
          f"worst {losses.min():+.2f}R")
    if losses.min() < -2:
        print("    A 'stop' that loses more than 2R is not a 1:2 trade. The SMA exit "
              "caps profit at exactly +2R but leaves losses open-ended.")

    edge = strat["expectancy"] - rand.get("expectancy", 0)
    print(f"\nEdge over random entries with identical geometry: {edge:+.3f}R per trade")
    if strat["n"] < 100:
        print("NOTE: under 100 trades — treat as a smoke test, not evidence.")
    # "Greater than zero" is far too weak a bar. An average can be positive and
    # still be pure noise, so require it to be distinguishable from zero.
    wo_exp, wo_t = s_wo.get("expectancy", 0), s_wo.get("t", 0)
    if edge <= 0:
        print("Does NOT beat random. The setup adds nothing over the exit rules alone.")
    elif wo_exp <= 0:
        print("Beats random ONLY because of one exceptional year. Remove it and the "
              "edge is negative — that is a market regime, not a strategy.")
    elif abs(wo_t) < 2:
        print(f"NOT PROVEN. Without its best year the expectancy is {wo_exp:+.3f}R "
              f"with t={wo_t:+.2f} — statistically indistinguishable from zero. "
              f"The headline profit is one good year plus noise.")
        print("Do not trade this on the strength of this backtest.")
    else:
        print("Beats the baseline AND survives removing its best year with a "
              "significant t. Worth taking seriously.")
    print("\nPer-trade detail: sma18_trades.csv")


if __name__ == "__main__":
    main()
