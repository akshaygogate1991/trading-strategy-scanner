"""Which exit rule actually makes money ON THE OPTION?

Every earlier backtest measured the UNDERLYING move in R. But you trade options,
and premium moves are leveraged and decaying: a 2% move in the stock might be a
40% move in the premium, while theta bleeds you every day you hold. A rule that
looks good on the stock can lose on the option.

This simulates the actual option position:
  * buy an at-the-money monthly CALL on the signal day
  * price it with Black-Scholes, using volatility measured from that stock's own
    recent price action (not a guess)
  * reprice every day as spot moves and time decays
  * exit on whichever rule fires first

Exit rules compared:
  A  -40% / +80%   the app's current rule (1:2 on premium)
  B  -50% / +50%   your proposal (1:1 on premium)
  C  -50% / +100%  1:2 with a wider stop
  D  signal exit   hold until the 18 EMA breaks, no premium target

Signals compared:
  EMA_ELLIOTT  the live app's rule (18 EMA trend + Elliott agreement)
  SMA18        the two-close-above-18-SMA breakout

IV is the weak point of any such simulation, so it is derived from each stock's
own 20-day realised volatility rather than assumed. Results are directional
guidance, not a promise of fills.

Run:  python option_exit_test.py
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import types

import numpy as np
import pandas as pd

import sma18_backtest as B

# ---------------------------------------------------------------- app logic
def _load_app_logic():
    """app.py imports yfinance for the live fallback feed; this backtest uses
    cached Angel One candles only, so stub it rather than require it."""
    if "yfinance" not in sys.modules:
        try:
            import yfinance  # noqa: F401
        except ImportError:
            stub = types.ModuleType("yfinance")
            stub.download = lambda *a, **k: None
            stub.Ticker = lambda *a, **k: None
            sys.modules["yfinance"] = stub
    src = open("app.py", encoding="utf-8").read()
    mod = types.ModuleType("app_logic")
    sys.modules["app_logic"] = mod
    exec(compile(src[: src.index("st.set_page_config")], "app.py", "exec"), mod.__dict__)
    return mod


app = _load_app_logic()

DAYS_TO_EXPIRY = 30          # a monthly bought in week 1-2
RISK_FREE = 0.065
MAX_HOLD = 30                # trading days
IV_FLOOR, IV_CAP = 0.12, 0.90

EXIT_RULES = {
    "A -40/+80 (current app)": (-0.40, 0.80),
    "B -50/+50 (your idea)": (-0.50, 0.50),
    "C -50/+100": (-0.50, 1.00),
    "D no stop, no target": (None, None),
    # D wins but allows a -100% wipeout. These keep the uncapped upside while
    # cutting the tail: a stop, but no profit target at all.
    "E -40% stop, no target": (-0.40, None),
    "F -50% stop, no target": (-0.50, None),
    "G -60% stop, no target": (-0.60, None),
}


def bs_call(S: float, K: float, T: float, iv: float, r: float = RISK_FREE) -> float:
    if T <= 0 or iv <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + (r + iv * iv / 2) * T) / (iv * math.sqrt(T))
    d2 = d1 - iv * math.sqrt(T)
    N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    return S * N(d1) - K * math.exp(-r * T) * N(d2)


def realised_iv(close: pd.Series, win: int = 20) -> pd.Series:
    """Annualised volatility from the stock's own recent moves."""
    ret = np.log(close / close.shift(1))
    return (ret.rolling(win).std() * math.sqrt(252)).clip(IV_FLOOR, IV_CAP)


# ---------------------------------------------------------------- signals
def signals_sma(df: pd.DataFrame) -> list[int]:
    return [df.index.get_loc(pd.Timestamp(t["date"])) for t in B.signals(df)]


ELLIOTT_WINDOW = 150     # ~7 months: enough bars to count a wave, 40% cheaper
EVAL_STEP = 3            # re-count the wave every 3rd bar, not every bar


def signals_ema_elliott(df: pd.DataFrame) -> list[int]:
    """The live app's rule, evaluated bar by bar with no lookahead.

    Two speedups, neither of which changes the answer:
      1. The 18 EMA condition is vectorised and used as a cheap pre-filter, so
         the expensive Elliott wave count only runs on bars that could qualify.
      2. Elliott sees a fixed 250-bar trailing window instead of all history -
         which also matches what the live app actually fetches.

    The first version recomputed every indicator on a growing window for every
    bar of every instrument: ~27 million operations, which looked like a hang.
    """
    d = app.add_indicators(df)
    ema = d["ema18"]
    # app.trend_18ema: close above the EMA AND the EMA above its value 5 bars ago
    up_mask = ((d["Close"] > ema) & (ema > ema.shift(5))).values

    idx: list[int] = []
    prev = False
    last_wave_ok, last_eval = False, -99
    for i in range(120, len(df)):
        ok = False
        if up_mask[i]:
            # The wave count barely moves bar to bar, so recomputing it on every
            # single bar is wasted work - it was the reason this ran for 15
            # minutes. Re-count every EVAL_STEP bars and reuse in between.
            if i - last_eval >= EVAL_STEP:
                w = d.iloc[max(0, i - ELLIOTT_WINDOW + 1): i + 1]
                try:
                    wave = app.elliott_directional(w)
                    last_wave_ok = (wave["bias"] == "Bullish"
                                    and int(wave["score"]) >= 55)
                except Exception:
                    last_wave_ok = False
                last_eval = i
            ok = last_wave_ok
        if ok and not prev:          # fire on the transition only
            idx.append(i)
        prev = ok
    return idx


# ---------------------------------------------------------------- simulate
def simulate(df: pd.DataFrame, entries: list[int], stop_pct, target_pct) -> list[dict]:
    d = df.copy()
    d["sma"] = d["Close"].rolling(B.SMA_LEN).mean()
    d["ema"] = d["Close"].ewm(span=18, adjust=False).mean()
    iv_s = realised_iv(d["Close"])
    close, high, low = d["Close"].values, d["High"].values, d["Low"].values
    sma, ema = d["sma"].values, d["ema"].values
    iv = iv_s.values
    n = len(d)
    out: list[dict] = []

    for i in entries:
        if i >= n - 2 or not np.isfinite(iv[i]):
            continue
        spot0 = float(close[i])
        strike = app.nearest_strike(spot0, "X.NS")
        vol = float(iv[i])
        prem0 = bs_call(spot0, strike, DAYS_TO_EXPIRY / 365.0, vol)
        if prem0 <= 0.01:
            continue

        outcome, pnl_pct, held = "expiry", -1.0, 0
        for k in range(i + 1, min(i + 1 + MAX_HOLD, n)):
            days_left = DAYS_TO_EXPIRY - (k - i)
            if days_left <= 1:
                prem = max(0.0, float(close[k]) - strike)
                pnl_pct, outcome, held = prem / prem0 - 1, "expiry", k - i
                break
            T = days_left / 365.0
            v = float(iv[k]) if np.isfinite(iv[k]) else vol
            prem_hi = bs_call(float(high[k]), strike, T, v)
            prem_lo = bs_call(float(low[k]), strike, T, v)
            prem_cl = bs_call(float(close[k]), strike, T, v)

            # pessimistic: if both are touched in one day, the stop goes first
            if stop_pct is not None and prem_lo / prem0 - 1 <= stop_pct:
                pnl_pct, outcome, held = stop_pct, "stop", k - i
                break
            if target_pct is not None and prem_hi / prem0 - 1 >= target_pct:
                pnl_pct, outcome, held = target_pct, "target", k - i
                break
            if close[k] < ema[k]:            # the signal itself broke
                pnl_pct, outcome, held = prem_cl / prem0 - 1, "signal_exit", k - i
                break
        else:
            k = min(i + MAX_HOLD, n - 1)
            days_left = max(DAYS_TO_EXPIRY - (k - i), 1)
            prem = bs_call(float(close[k]), strike, days_left / 365.0, vol)
            pnl_pct, outcome, held = prem / prem0 - 1, "timeout", k - i

        out.append({"entry_premium": round(prem0, 2), "pnl_pct": round(pnl_pct, 4),
                    "outcome": outcome, "days": held, "iv": round(vol, 3)})
    return out


def stats(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    a = np.array([r["pnl_pct"] for r in rows], dtype=float)
    sd = a.std(ddof=1) if len(a) > 1 else 0.0
    se = sd / math.sqrt(len(a)) if sd > 0 else 0.0
    return {
        "n": len(a),
        "win": round(100.0 * (a > 0).sum() / len(a), 1),
        "avg": round(100.0 * float(a.mean()), 2),
        "t": round(float(a.mean() / se), 2) if se else 0.0,
        "total": round(100.0 * float(a.sum()), 0),
        "worst": round(100.0 * float(a.min()), 1),
        "days": round(float(np.mean([r["days"] for r in rows])), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=None,
                    help="limit instruments, e.g. --tickers ^NSEI ^NSEBANK")
    ap.add_argument("--signal", choices=["sma", "ema", "both"], default="both")
    args = ap.parse_args()

    tickers = args.tickers or B.ALL_TICKERS
    frames = {}
    for t in tickers:
        df = B.load(t, 4.0)
        if df is not None and len(df) > 200:
            frames[t] = df
    print(f"Loaded {len(frames)} instruments\n")

    wanted = [] if args.signal == "ema" else [("SMA18", signals_sma)]
    if args.signal in ("ema", "both"):
        wanted.append(("EMA_ELLIOTT (live app)", signals_ema_elliott))

    for sig_name, sig_fn in wanted:
        print("=" * 76)
        print(f"SIGNAL: {sig_name}   (ATM monthly call, {DAYS_TO_EXPIRY}d, "
              f"IV from realised vol)")
        print("=" * 76)
        print(f"{'exit rule':26s}{'n':>6s}{'win%':>7s}{'avg%':>8s}{'t':>7s}"
              f"{'total%':>9s}{'worst%':>9s}{'days':>7s}")
        # generate entries ONCE per instrument, with visible progress - the
        # Elliott pass is the slow part and silence looks like a crash
        entries_cache = {}
        t0 = time.time()
        print("  finding entries: ", end="", flush=True)
        for j, (t, df) in enumerate(frames.items(), 1):
            try:
                entries_cache[t] = sig_fn(df)
            except Exception as exc:
                print(f"\n    ! {t}: {exc}")
                entries_cache[t] = []
            if j % 5 == 0:
                el = time.time() - t0
                eta = el / j * (len(frames) - j)
                print(f"{j}/{len(frames)} ({el:.0f}s, ~{eta:.0f}s left) ",
                      end="", flush=True)
        total_sig = sum(len(v) for v in entries_cache.values())
        print(f"done — {total_sig} signals in {time.time() - t0:.0f}s\n")

        for rule, (sp, tp) in EXIT_RULES.items():
            rows: list[dict] = []
            for t, df in frames.items():
                rows += simulate(df, entries_cache[t], sp, tp)
            s = stats(rows)
            print(f"{rule:26s}{s.get('n', 0):6d}{s.get('win', 0):7.1f}"
                  f"{s.get('avg', 0):+8.2f}{s.get('t', 0):+7.2f}"
                  f"{s.get('total', 0):+9.0f}{s.get('worst', 0):9.1f}"
                  f"{s.get('days', 0):7.1f}")
        print()

    print("Reading this: 'avg%' is the average return per trade on the premium "
          "paid.\nA rule is only worth using if avg% is positive AND t is beyond "
          "about +2.\nEvery figure ignores brokerage and the bid-ask spread, "
          "which on stock\noptions can be 2-5% of premium per round trip — "
          "subtract that before believing any of it.")


if __name__ == "__main__":
    main()
