"""Stop x Target grid for three signals, so the comparison is visible at once.

SIGNALS
  1. EMA_ELLIOTT   the live app's rule (18 EMA trend + Elliott Wave bullish)
  2. SMA18         two consecutive closes above the 18 SMA, entry at 2nd high
  3. BOTH          an SMA18 entry taken ONLY when EMA+Elliott also confirms up

For each signal every combination of stop and target is simulated on an ATM
monthly call, priced with Black-Scholes and repriced daily as spot moves and
time decays. Cells show average % return on the premium paid.

The expensive part (the bar-by-bar Elliott count) is cached to disk after the
first run, so re-running is fast.

Run:  python exit_matrix.py
      python exit_matrix.py --tickers ^NSEI ^NSEBANK      (index only)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import types

import numpy as np
import pandas as pd

import sma18_backtest as B

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

DAYS_TO_EXPIRY = 30
RISK_FREE = 0.065
MAX_HOLD = 30
IV_FLOOR, IV_CAP = 0.12, 0.90
ELLIOTT_WINDOW = 150
EVAL_STEP = 3
COST_PCT = 0.03          # brokerage + bid/ask, subtracted in the net table

STOPS = [-0.30, -0.40, -0.50, -0.60, None]
TARGETS = [0.50, 0.80, 1.00, 1.50, 2.00, None]

MASK_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "ema_elliott_mask.json")


def bs_call(S, K, T, iv, r=RISK_FREE):
    if T <= 0 or iv <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + (r + iv * iv / 2) * T) / (iv * math.sqrt(T))
    d2 = d1 - iv * math.sqrt(T)
    N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    return S * N(d1) - K * math.exp(-r * T) * N(d2)


def realised_iv(close, win=20):
    ret = np.log(close / close.shift(1))
    return (ret.rolling(win).std() * math.sqrt(252)).clip(IV_FLOOR, IV_CAP)


# ------------------------------------------------------------------ signals
_masks: dict = {}


def _load_mask_cache():
    global _masks
    if os.path.exists(MASK_CACHE):
        try:
            _masks = json.load(open(MASK_CACHE))
        except Exception:
            _masks = {}


def _save_mask_cache():
    try:
        json.dump(_masks, open(MASK_CACHE, "w"))
    except Exception:
        pass


def ema_elliott_mask(ticker: str, df: pd.DataFrame) -> np.ndarray:
    """Per-bar: does the live app's rule say 'bullish' here? Cached to disk."""
    key = f"{ticker}|{len(df)}"
    if key in _masks:
        return np.array(_masks[key], dtype=bool)

    d = app.add_indicators(df)
    ema = d["ema18"]
    up = ((d["Close"] > ema) & (ema > ema.shift(5))).values
    out = np.zeros(len(df), dtype=bool)
    last_ok, last_eval = False, -99
    for i in range(120, len(df)):
        if not up[i]:
            continue
        if i - last_eval >= EVAL_STEP:
            w = d.iloc[max(0, i - ELLIOTT_WINDOW + 1): i + 1]
            try:
                wave = app.elliott_directional(w)
                last_ok = wave["bias"] == "Bullish" and int(wave["score"]) >= 55
            except Exception:
                last_ok = False
            last_eval = i
        out[i] = last_ok
    _masks[key] = out.tolist()
    return out


def entries_ema(ticker, df):
    m = ema_elliott_mask(ticker, df)
    return [i for i in range(1, len(m)) if m[i] and not m[i - 1]]


def entries_sma(ticker, df):
    return [df.index.get_loc(pd.Timestamp(t["date"])) for t in B.signals(df)]


def entries_both(ticker, df):
    m = ema_elliott_mask(ticker, df)
    return [i for i in entries_sma(ticker, df) if i < len(m) and m[i]]


# ------------------------------------------------------------------ simulate
def simulate(df, entries, stop_pct, target_pct):
    d = df.copy()
    d["ema"] = d["Close"].ewm(span=18, adjust=False).mean()
    iv = realised_iv(d["Close"]).values
    close, high, low = d["Close"].values, d["High"].values, d["Low"].values
    ema = d["ema"].values
    n = len(d)
    out = []
    for i in entries:
        if i >= n - 2 or not np.isfinite(iv[i]):
            continue
        spot0 = float(close[i])
        strike = app.nearest_strike(spot0, "X.NS")
        vol = float(iv[i])
        p0 = bs_call(spot0, strike, DAYS_TO_EXPIRY / 365.0, vol)
        if p0 <= 0.01:
            continue
        pnl = -1.0
        for k in range(i + 1, min(i + 1 + MAX_HOLD, n)):
            left = DAYS_TO_EXPIRY - (k - i)
            if left <= 1:
                pnl = max(0.0, float(close[k]) - strike) / p0 - 1
                break
            T, v = left / 365.0, (float(iv[k]) if np.isfinite(iv[k]) else vol)
            if stop_pct is not None and bs_call(float(low[k]), strike, T, v) / p0 - 1 <= stop_pct:
                pnl = stop_pct
                break
            if target_pct is not None and bs_call(float(high[k]), strike, T, v) / p0 - 1 >= target_pct:
                pnl = target_pct
                break
            if close[k] < ema[k]:
                pnl = bs_call(float(close[k]), strike, T, v) / p0 - 1
                break
        else:
            k = min(i + MAX_HOLD, n - 1)
            left = max(DAYS_TO_EXPIRY - (k - i), 1)
            pnl = bs_call(float(close[k]), strike, left / 365.0, vol) / p0 - 1
        out.append(float(pnl))
    return out


def stat(vals):
    if not vals:
        return 0.0, 0.0, 0
    a = np.array(vals)
    sd = a.std(ddof=1) if len(a) > 1 else 0.0
    se = sd / math.sqrt(len(a)) if sd else 0.0
    return 100 * float(a.mean()), (float(a.mean() / se) if se else 0.0), len(a)


def label(v, kind):
    if v is None:
        return "none"
    return f"{'+' if kind == 'tgt' else ''}{int(v * 100)}%"


def grid(name, frames, entry_fn):
    ent = {t: entry_fn(t, df) for t, df in frames.items()}
    total = sum(len(v) for v in ent.values())
    print("\n" + "=" * 92)
    print(f"  {name}   —   {total} signals")
    print("=" * 92)

    header = f"{'stop':>10s}" + "".join(f"{'tgt ' + label(x, 'tgt'):>13s}" for x in TARGETS)
    print(header)
    print("-" * len(header))
    best = (None, -999, 0, 0)
    rows = {}
    for s in STOPS:
        line = f"{label(s, 'stop'):>10s}"
        for tg in TARGETS:
            vals = []
            for t, df in frames.items():
                vals += simulate(df, ent[t], s, tg)
            avg, tt, n = stat(vals)
            rows[(s, tg)] = (avg, tt, n)
            line += f"{avg:+9.2f} " + (f"({tt:+.1f})" if abs(tt) >= 2 else "      ")
            if avg > best[1]:
                best = ((s, tg), avg, tt, n)
        print(line)
    print("\n  (t-stat shown only where |t| >= 2, i.e. unlikely to be chance)")
    s, tg = best[0]
    print(f"  BEST: stop {label(s, 'stop')}, target {label(tg, 'tgt')} → "
          f"{best[1]:+.2f}% per trade (t={best[2]:+.2f}, n={best[3]})")
    print(f"  after ~{COST_PCT * 100:.0f}% costs: {best[1] - COST_PCT * 100:+.2f}% per trade")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=None)
    args = ap.parse_args()
    _load_mask_cache()

    frames = {}
    for t in (args.tickers or B.ALL_TICKERS):
        df = B.load(t, 4.0)
        if df is not None and len(df) > 200:
            frames[t] = df
    print(f"Loaded {len(frames)} instruments. "
          f"First run computes the Elliott mask (~7 min), then it is cached.")

    t0 = time.time()
    r_ema = grid("1. EMA + ELLIOTT  (the live app's signal)", frames, entries_ema)
    _save_mask_cache()
    r_sma = grid("2. SMA18 only  (two closes above the 18 SMA)", frames, entries_sma)
    r_both = grid("3. BOTH  (SMA18 entry, only when EMA+Elliott also confirms up)",
                  frames, entries_both)
    _save_mask_cache()

    # side-by-side on the rules that matter
    print("\n" + "=" * 92)
    print("  SIDE BY SIDE — average % per trade on the premium")
    print("=" * 92)
    picks = [(-0.40, 0.80), (-0.50, 0.50), (-0.50, 1.00),
             (-0.60, None), (-0.50, None), (None, None)]
    print(f"{'stop / target':>22s}{'EMA+Elliott':>16s}{'SMA18':>14s}{'BOTH':>14s}")
    print("-" * 68)
    for s, tg in picks:
        nm = f"{label(s, 'stop')} / {label(tg, 'tgt')}"
        a = r_ema.get((s, tg), (0, 0, 0))
        b = r_sma.get((s, tg), (0, 0, 0))
        c = r_both.get((s, tg), (0, 0, 0))
        print(f"{nm:>22s}{a[0]:+13.2f}  {b[0]:+13.2f} {c[0]:+13.2f}")
    print(f"\n  n =                  {r_ema.get((-0.4, 0.8), (0, 0, 0))[2]:>10d}  "
          f"{r_sma.get((-0.4, 0.8), (0, 0, 0))[2]:>13d} "
          f"{r_both.get((-0.4, 0.8), (0, 0, 0))[2]:>13d}")
    print(f"\nDone in {time.time() - t0:.0f}s. Costs of ~{COST_PCT * 100:.0f}% per "
          f"round trip are NOT deducted in the tables above.")


if __name__ == "__main__":
    main()
