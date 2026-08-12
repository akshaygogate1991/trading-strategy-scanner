"""Anchored VWAP + 18 EMA + Fair Value Gap — long only, daily, on option P&L.

WHY ANCHORED VWAP
Session VWAP resets every morning, so on a daily chart every candle contains
exactly one VWAP value and the indicator means nothing. Anchored VWAP instead
starts at a chosen event and accumulates from there, which works on any
timeframe. Here it is anchored to the lowest low of the trailing 60 sessions -
the most recent significant bottom - and re-anchors when a new one forms.

Interpretation: anchored VWAP is the average price everyone who bought since
that low has paid. Above it, buyers since the bottom are collectively in
profit; below it they are underwater.

WHAT IS TESTED (each adds one filter, so you can see what each one earns)
  1  VWAP only        close above the anchored VWAP
  2  VWAP + EMA       ... and above the 18 EMA
  3  VWAP + EMA + FVG ... and an unfilled bullish gap sits below as support
  4  EMA only         the app's existing trend filter, as a reference point

Exit uses the rule the earlier tests supported: -60% stop, NO profit target,
close when price shuts below the 18 EMA.

Run:  python vwap_strategy_test.py
"""
from __future__ import annotations

import argparse
import math
import sys
import types

import numpy as np
import pandas as pd

import sma18_backtest as B

def _load_app_logic():
    """Import app.py's indicator functions without needing its live-data deps.

    app.py imports yfinance for the live app's fallback feed. This backtest
    reads cached Angel One candles and never touches Yahoo, so a missing
    yfinance should not stop it - stub the module rather than require it.
    """
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

ANCHOR_LOOKBACK = 60      # sessions used to find the anchor low
DAYS_TO_EXPIRY = 30
RISK_FREE = 0.065
MAX_HOLD = 30
STOP_PCT = -0.60          # the exit rule the earlier tests supported
COST_PCT = 0.03


def bs_call(S, K, T, iv, r=RISK_FREE):
    if T <= 0 or iv <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + (r + iv * iv / 2) * T) / (iv * math.sqrt(T))
    d2 = d1 - iv * math.sqrt(T)
    N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    return S * N(d1) - K * math.exp(-r * T) * N(d2)


def anchored_vwap(df: pd.DataFrame, lookback: int = ANCHOR_LOOKBACK) -> np.ndarray:
    """VWAP accumulated from the most recent significant low.

    Re-anchors whenever a new trailing low is made, so it always measures the
    average cost of buyers since the latest bottom.
    """
    low = df["Low"].values
    tp = ((df["High"] + df["Low"] + df["Close"]) / 3).values      # typical price
    vol = df["Volume"].values.astype(float)
    vol = np.where(vol <= 0, 1.0, vol)                            # indices report 0
    n = len(df)
    out = np.full(n, np.nan)
    anchor = 0
    for i in range(n):
        if i >= lookback:
            j = int(np.argmin(low[i - lookback + 1: i + 1])) + i - lookback + 1
            if j > anchor:
                anchor = j                                        # new low -> re-anchor
        pv = np.sum(tp[anchor: i + 1] * vol[anchor: i + 1])
        vv = np.sum(vol[anchor: i + 1])
        out[i] = pv / vv if vv > 0 else np.nan
    return out


def bullish_fvg_support(df: pd.DataFrame) -> np.ndarray:
    """True where an UNFILLED bullish fair value gap sits below current price.

    FVG = the 3-candle pattern from the course: low of candle 3 above high of
    candle 1. It stays usable until price trades back through it.
    """
    high, low, close = df["High"].values, df["Low"].values, df["Close"].values
    n = len(df)
    out = np.zeros(n, dtype=bool)
    gaps: list[list[float]] = []          # [bottom, top]
    for i in range(2, n):
        if low[i] > high[i - 2]:
            gaps.append([high[i - 2], low[i]])
        gaps = [g for g in gaps if low[i] > g[0]]      # drop filled gaps
        out[i] = any(g[1] < close[i] for g in gaps)    # a gap sits below price
    return out


def build_masks(df: pd.DataFrame) -> dict:
    d = app.add_indicators(df)
    ema = d["ema18"].values
    close = d["Close"].values
    vwap = anchored_vwap(df)
    fvg = bullish_fvg_support(df)
    above_vwap = close > vwap
    above_ema = (close > ema) & (ema > pd.Series(ema).shift(5).values)
    return {
        "1 VWAP only": above_vwap,
        "2 VWAP + EMA": above_vwap & above_ema,
        "3 VWAP + EMA + FVG": above_vwap & above_ema & fvg,
        "4 EMA only (reference)": above_ema,
    }


def entries_from(mask: np.ndarray) -> list[int]:
    """Fire on the transition into the condition, not every bar it stays true."""
    m = np.nan_to_num(mask.astype(float), nan=0.0).astype(bool)
    return [i for i in range(121, len(m)) if m[i] and not m[i - 1]]


def entries_random(n_bars: int, count: int, seed: int) -> list[int]:
    """Control group: buy calls on random days, same exit rule.

    THE decisive comparison. These are long-only calls over 2022-2026, when
    Indian markets rose substantially, so a good exit rule alone will show a
    profit regardless of entry. If random entries score close to the signals,
    the signals are adding nothing and we are measuring the market, not a
    strategy.
    """
    if n_bars < 200 or count <= 0:
        return []
    rng = np.random.default_rng(seed)
    return sorted(int(x) for x in rng.integers(121, n_bars - 35, size=count))


def simulate(df: pd.DataFrame, entries: list[int]) -> list[float]:
    d = df.copy()
    d["ema"] = d["Close"].ewm(span=18, adjust=False).mean()
    ret = np.log(d["Close"] / d["Close"].shift(1))
    iv = (ret.rolling(20).std() * math.sqrt(252)).clip(0.12, 0.90).values
    close, high, low = d["Close"].values, d["High"].values, d["Low"].values
    ema = d["ema"].values
    n = len(d)
    out: list[float] = []
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
            T = left / 365.0
            v = float(iv[k]) if np.isfinite(iv[k]) else vol
            if bs_call(float(low[k]), strike, T, v) / p0 - 1 <= STOP_PCT:
                pnl = STOP_PCT
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


def stat(v):
    if not v:
        return 0.0, 0.0, 0, 0.0
    a = np.array(v)
    sd = a.std(ddof=1) if len(a) > 1 else 0.0
    se = sd / math.sqrt(len(a)) if sd else 0.0
    return (100 * float(a.mean()), float(a.mean() / se) if se else 0.0,
            len(a), 100.0 * (a > 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=None)
    args = ap.parse_args()

    frames = {}
    for t in (args.tickers or B.ALL_TICKERS):
        df = B.load(t, 4.0)
        if df is not None and len(df) > 200:
            frames[t] = df
    print(f"Anchored VWAP + EMA + FVG — long only, daily, ATM monthly call")
    print(f"Exit: {int(STOP_PCT * 100)}% stop, no target, close below 18 EMA. "
          f"{len(frames)} instruments.\n")

    results: dict = {}
    for seed, (t, df) in enumerate(frames.items()):
        masks = build_masks(df)
        for name, mask in masks.items():
            results.setdefault(name, []).extend(simulate(df, entries_from(mask)))
        # control: same number of trades as the EMA filter, chosen at random
        n_ref = len(entries_from(masks["4 EMA only (reference)"]))
        results.setdefault("0 RANDOM entries (control)", []).extend(
            simulate(df, entries_random(len(df), n_ref, seed)))

    print(f"{'filter':26s}{'trades':>8s}{'win%':>8s}{'avg%':>9s}{'t':>8s}{'net%':>9s}")
    print("-" * 68)
    for name in sorted(results):
        avg, t_, n, win = stat(results[name])
        flag = "  <-- significant" if abs(t_) >= 2 else ""
        print(f"{name:26s}{n:8d}{win:8.1f}{avg:+9.2f}{t_:+8.2f}"
              f"{avg - COST_PCT * 100:+9.2f}{flag}")

    ctrl, _, _, _ = stat(results.get("0 RANDOM entries (control)", []))
    print("\n" + "=" * 68)
    print("  THE QUESTION THAT MATTERS: does any filter beat random entry?")
    print("=" * 68)
    for name in sorted(results):
        if name.startswith("0 "):
            continue
        avg, _, _, _ = stat(results[name])
        print(f"    {name:26s} {avg - ctrl:+7.2f}% vs random")
    print("\n  If these gaps are near zero, the indicators are not the source of")
    print("  the return - the exit rule and a rising market are. That would mean")
    print("  adding VWAP or FVG to the app buys you nothing.")

    print("\n  'net%' subtracts ~3% for brokerage and bid-ask per round trip.")
    print("  Compare each line to '4 EMA only' — that is what your app already")
    print("  does. A filter is only worth adding if it beats that row, and does")
    print("  so with |t| >= 2. Otherwise it is complexity without benefit.")


if __name__ == "__main__":
    main()
