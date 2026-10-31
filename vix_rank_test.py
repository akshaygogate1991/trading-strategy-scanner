"""India VIX rank — buy Nifty options when volatility is genuinely CHEAP.

WHY THIS TEST IS VALID WHERE THE AUGUST ONE WAS NOT
That attempt used 20-day REALISED volatility as a stand-in for implied vol. It
failed badly: realised vol lags, so it rises AFTER a move, and the model paid
option holders twice for the same event. Random straddles "returned" 31% a
month, which is impossible, and the error was correlated with the signal (we
selected volatility lows, exactly where a lagging measure is most wrong).

India VIX is the real thing — the market's own 30-day implied volatility for
Nifty, published daily by NSE. Options are priced here at the VIX of the day
they are bought and re-priced at the VIX of the day they are sold. That means:

  * the variance risk premium is PAID, not ignored
  * IV crush after a move is captured, because VIX actually falls
  * no circularity: VIX is an independent series, not derived from the path

THE HYPOTHESIS
Volatility mean-reverts. When India VIX sits near the bottom of its own
one-year range, options are cheap; when VIX normalises, they gain value even if
Nifty goes nowhere. This is a PRICING edge, not a forecast of direction.

THE HONEST PRIOR
Implied volatility exceeds subsequent realised volatility most of the time -
that is why option sellers profit on average and buyers do not. Buying at low
IV should make you LESS wrong; whether it makes you right is the open question.
I would not expect this to work, but it has never been properly tested.

WHAT IS TESTED
  A  STRADDLE bought at LOW VIX rank   <- the clean test, no direction
  B  STRADDLE bought at HIGH VIX rank  <- should be worse if the idea holds
  C  CALL at low VIX rank              <- adds direction risk
  D  random entry control, many runs

Run:  python vix_rank_test.py
      python vix_rank_test.py --hold 10 --low 20 --high 80
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vix_data")
RANK_WIN = 252              # rank VIX against its own trailing year
DAYS_TO_EXPIRY = 30
RISK_FREE = 0.065
LOT = 65
COST_PCT = 0.015            # ~1.5% of premium, round trip, index options


def fetch_series(years: float, refresh: bool) -> pd.DataFrame | None:
    """Daily Nifty close + daily India VIX close, aligned.

    Yahoo is used for both because India VIX history is reliably available
    there and this test needs only daily closes.
    """
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"nifty_vix_{years:.0f}y.csv")
    if os.path.exists(path) and not refresh:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if len(df) > RANK_WIN + 60:
            print(f"  {len(df)} daily rows from cache")
            return df
    try:
        import yfinance as yf
    except ImportError:
        print("Needs yfinance:  python -m pip install yfinance")
        return None

    period = f"{int(years)}y"
    print(f"  downloading Nifty and India VIX ({period})...")
    nif = yf.download("^NSEI", period=period, interval="1d",
                      auto_adjust=True, progress=False)
    vix = yf.download("^INDIAVIX", period=period, interval="1d",
                      auto_adjust=True, progress=False)
    if nif is None or vix is None or nif.empty or vix.empty:
        print("  download failed — check the connection")
        return None
    for d in (nif, vix):
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)

    df = pd.DataFrame({
        "Close": nif["Close"], "High": nif["High"], "Low": nif["Low"],
        "VIX": vix["Close"],
    }).dropna()
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df.to_csv(path)
    print(f"  {len(df)} daily rows "
          f"({df.index[0].date()} to {df.index[-1].date()})")
    return df


def bs(S, K, T, iv, kind, r=RISK_FREE):
    if T <= 0 or iv <= 0:
        return max(0.0, S - K) if kind == "C" else max(0.0, K - S)
    d1 = (math.log(S / K) + (r + iv * iv / 2) * T) / (iv * math.sqrt(T))
    d2 = d1 - iv * math.sqrt(T)
    N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    if kind == "C":
        return S * N(d1) - K * math.exp(-r * T) * N(d2)
    return K * math.exp(-r * T) * N(-d2) - S * N(-d1)


def vix_rank(vix: pd.Series, win: int = RANK_WIN) -> pd.Series:
    """Where today's VIX sits inside its own trailing-year range, 0-100."""
    return vix.rolling(win).apply(
        lambda w: 100.0 * (w[-1] - np.min(w)) / (np.max(w) - np.min(w) + 1e-9),
        raw=True)


def entries(df: pd.DataFrame, low: bool, lo_th: float, hi_th: float) -> list[int]:
    """Fire on the day rank CROSSES into the cheap (or expensive) zone."""
    r = df["rank"].values
    out = []
    for i in range(RANK_WIN + 2, len(df) - 35):
        if not np.isfinite(r[i]) or not np.isfinite(r[i - 1]):
            continue
        if low and r[i] < lo_th <= r[i - 1]:
            out.append(i)
        elif (not low) and r[i] > hi_th >= r[i - 1]:
            out.append(i)
    return out


def simulate(df, idx, kind: str, hold: int) -> list[float]:
    """kind 'STRADDLE' or 'CALL'. Returns % P&L on premium paid.

    Entry priced at THAT DAY's India VIX; exit priced at the VIX on the exit
    day. Buying expensive and selling cheap is exactly what an option buyer
    suffers, and using the real series is what makes it show up here.
    """
    close, vix = df["Close"].values, df["VIX"].values
    n = len(df)
    out = []
    for i in idx:
        S0, v0 = float(close[i]), float(vix[i]) / 100.0
        K = round(S0 / 50.0) * 50.0
        T0 = DAYS_TO_EXPIRY / 365.0
        if kind == "STRADDLE":
            p0 = bs(S0, K, T0, v0, "C") + bs(S0, K, T0, v0, "P")
        else:
            p0 = bs(S0, K, T0, v0, "C")
        if p0 <= 0.01:
            continue
        k = min(i + hold, n - 1)
        left = max(DAYS_TO_EXPIRY - (k - i), 1) / 365.0
        vk = float(vix[k]) / 100.0          # the REAL IV on the exit day
        Sk = float(close[k])
        if kind == "STRADDLE":
            p1 = bs(Sk, K, left, vk, "C") + bs(Sk, K, left, vk, "P")
        else:
            p1 = bs(Sk, K, left, vk, "C")
        out.append(p1 / p0 - 1 - COST_PCT)
    return out


def stat(v):
    if not v:
        return 0.0, 0.0, 0, 0.0
    a = np.array(v, dtype=float)
    sd = a.std(ddof=1) if len(a) > 1 else 0.0
    se = sd / math.sqrt(len(a)) if sd else 0.0
    return (100 * float(a.mean()), float(a.mean() / se) if se else 0.0,
            len(a), 100.0 * float((a > 0).mean()))


def show(label, vals):
    m, t, n, w = stat(vals)
    flag = "  <-- significant" if abs(t) >= 2 else ""
    print(f"  {label:36s} n={n:4d}  win {w:5.1f}%  avg {m:+7.2f}%  "
          f"t={t:+5.2f}{flag}")
    return m, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=8)
    ap.add_argument("--hold", type=int, default=10, help="trading days held")
    ap.add_argument("--low", type=float, default=25.0)
    ap.add_argument("--high", type=float, default=75.0)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    print("India VIX rank — buying Nifty options when IV is genuinely cheap")
    df = fetch_series(args.years, args.refresh)
    if df is None:
        return
    df = df.copy()
    df["rank"] = vix_rank(df["VIX"])
    df = df.dropna(subset=["rank"])

    lo_idx = entries(df, True, args.low, args.high)
    hi_idx = entries(df, False, args.low, args.high)
    print(f"\n  VIX today {df['VIX'].iloc[-1]:.2f}, "
          f"rank {df['rank'].iloc[-1]:.0f} of its past year")
    print(f"  signals: {len(lo_idx)} cheap (rank<{args.low:.0f}), "
          f"{len(hi_idx)} expensive (rank>{args.high:.0f})")
    print(f"  holding {args.hold} trading days, {DAYS_TO_EXPIRY}d options, "
          f"costs {COST_PCT:.1%} of premium\n")
    if not lo_idx:
        print("  No cheap-VIX signals in this window.")
        return

    print("THE CLEAN TEST — straddles (direction removed)")
    s_lo = simulate(df, lo_idx, "STRADDLE", args.hold)
    s_hi = simulate(df, hi_idx, "STRADDLE", args.hold)
    m_lo, _ = show("A  STRADDLE at LOW VIX rank", s_lo)
    m_hi, _ = show("B  STRADDLE at HIGH VIX rank", s_hi)

    print("\nWITH DIRECTION — calls only")
    c_lo = simulate(df, lo_idx, "CALL", args.hold)
    c_hi = simulate(df, hi_idx, "CALL", args.hold)
    show("C  CALL at LOW VIX rank", c_lo)
    show("D  CALL at HIGH VIX rank", c_hi)

    # ---- control: random entry days, same instrument and holding period ----
    print(f"\nCONTROL — {args.draws} random-entry runs (straddles)")
    rng = np.random.default_rng(31)
    dist = []
    for _ in range(args.draws):
        idx = sorted(int(x) for x in
                     rng.integers(RANK_WIN + 2, len(df) - 35, size=len(lo_idx)))
        r = simulate(df, idx, "STRADDLE", args.hold)
        if r:
            dist.append(100 * float(np.mean(r)))
    if dist:
        d = np.array(dist)
        pct = 100.0 * float((d < m_lo).mean())
        lo_p, hi_p = np.percentile(d, [5, 95])
        print(f"  cheap-VIX {m_lo:+.2f}%  vs random {d.mean():+.2f}% "
              f"[5-95%: {lo_p:+.2f}, {hi_p:+.2f}]  ->  percentile {pct:.0f}%")

        print("\n  SANITY CHECK: random straddles came out "
              f"{d.mean():+.2f}%.")
        if d.mean() > 5:
            print("  That is implausibly high — buying options at random should")
            print("  LOSE, because implied vol exceeds realised vol on average.")
            print("  If you see this, the model is still mispricing and no")
            print("  result here should be trusted.")
        else:
            print("  Negative or near zero is what we expect and want to see:")
            print("  it means the variance risk premium is being charged, so")
            print("  the comparison above is a fair one.")

    print("\n" + "=" * 70)
    edge = m_lo - m_hi
    print(f"  Cheap minus expensive (straddles): {edge:+.2f}%")
    if dist and pct > 95 and edge > 0:
        print("  Cheap volatility BEATS random and beats expensive volatility.")
        print("  This is a pricing edge, and the first thing in this project")
        print("  to survive a proper control.")
    elif edge > 0:
        print("  Points the right way but does not clear the 95% bar.")
        print("  Suggestive, not proven.")
    else:
        print("  No edge. Even with real implied volatility, buying options")
        print("  when VIX is low does not pay on Nifty.")
    print("=" * 70)


if __name__ == "__main__":
    main()
