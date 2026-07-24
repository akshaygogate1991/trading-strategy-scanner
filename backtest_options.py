"""2-year backtest: regime-based defined-risk option spreads on Nifty / Bank Nifty.

WHAT IS REAL vs MODELED
- Real: index daily candles (Angel One SmartAPI, Yahoo fallback), regime signals,
  index moves over each holding period.
- Modeled: option premiums. Debit spreads assumed to cost 45% of strike width;
  iron condor credit assumed 33% of wing width. These are standard ballpark
  levels for near-month NSE index options; real fills vary with volatility.

STRATEGY UNDER TEST (all defined-risk, max loss known at entry)
  Uptrend   -> Bull Call Spread  (ATM + 2% wide)
  Downtrend -> Bear Put Spread   (ATM + 2% wide)
  Range     -> Iron Condor       (short +/-1.5%, wings +/-3.0%)
Entries every 10 trading days (non-overlapping), held ~10 sessions.

Run:  python backtest_options.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HOLD = 10          # trading days per trade
STEP = 10          # days between entries (non-overlapping)
WIDTH = 2.0        # debit spread width, % of spot
DEBIT = 0.45 * WIDTH          # modeled cost of debit spreads (% of spot)
C_SHORT = 1.5      # condor short strikes, % from spot
C_WING = 1.5       # condor wing width beyond shorts, %
C_CREDIT = 0.33 * C_WING      # modeled condor credit (% of spot)


def get_data(ticker: str) -> pd.DataFrame:
    try:
        import smartapi_data as sd

        df = sd.get_history(ticker, "1d", 760)
        if df is not None and len(df) > 400:
            print(f"  {ticker}: {len(df)} days from Angel One (exact NSE)")
            return df
    except Exception as exc:
        print(f"  {ticker}: Angel One unavailable ({exc}); trying Yahoo...")
    import yfinance as yf

    df = yf.download(ticker, period="2y", interval="1d", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    print(f"  {ticker}: {len(df)} days from Yahoo Finance")
    return df.dropna(subset=["Close"])


def classify_regimes(df: pd.DataFrame) -> pd.Series:
    """Daily regime: 'up' / 'down' / 'range' using the scanner's core logic."""
    close = df["Close"]
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema50_prev = ema50.shift(5)

    weekly = close.resample("W-FRI").last().dropna()
    wma = weekly.ewm(span=20, adjust=False).mean()
    w_up = (weekly > wma) & (wma > wma.shift(3))
    w_down = (weekly < wma) & (wma < wma.shift(3))
    w_up_d = w_up.reindex(df.index, method="ffill").fillna(False).astype(bool)
    w_down_d = w_down.reindex(df.index, method="ffill").fillna(False).astype(bool)

    up = (close > ema50) & (ema50 > ema50_prev) & w_up_d
    down = (close < ema50) & (ema50 < ema50_prev) & w_down_d
    regime = pd.Series("range", index=df.index)
    regime[up] = "up"
    regime[down] = "down"
    return regime


def spread_pnl(move: float, direction: str) -> tuple[float, float]:
    """(pnl, risk) in % of spot for a 2%-wide debit spread. R = pnl/risk."""
    m = move if direction == "bull" else -move
    value = float(np.clip(m, 0.0, WIDTH))
    return value - DEBIT, DEBIT


def condor_pnl(move: float) -> tuple[float, float]:
    loss_leg = float(np.clip(abs(move) - C_SHORT, 0.0, C_WING))
    max_loss = C_WING - C_CREDIT
    return C_CREDIT - loss_leg, max_loss


def run(ticker: str) -> pd.DataFrame:
    df = get_data(ticker)
    regime = classify_regimes(df)
    close = df["Close"]

    trades = []
    i = 60  # warm-up for indicators
    while i + HOLD < len(df):
        r = regime.iloc[i]
        s0, s1 = float(close.iloc[i]), float(close.iloc[i + HOLD])
        move = (s1 - s0) / s0 * 100.0

        if r == "up":
            pnl, risk = spread_pnl(move, "bull")
            structure = "Bull Call Spread"
        elif r == "down":
            pnl, risk = spread_pnl(move, "bear")
            structure = "Bear Put Spread"
        else:
            pnl, risk = condor_pnl(move)
            structure = "Iron Condor"

        base_pnl, base_risk = spread_pnl(move, "bull")  # baseline: always bullish
        trades.append(
            {
                "date": df.index[i].date(),
                "regime": r,
                "structure": structure,
                "move_%": round(move, 2),
                "R": round(pnl / risk, 2),
                "baseline_R": round(base_pnl / base_risk, 2),
            }
        )
        i += STEP
    return pd.DataFrame(trades)


def report(t: pd.DataFrame, label: str) -> None:
    print(f"\n===== {label}: {len(t)} trades over ~2 years =====")
    for name, grp in [("ALL", t)] + list(t.groupby("regime")):
        if isinstance(name, str) and name != "ALL":
            name = f"regime={name} ({grp['structure'].iloc[0]})"
        wins = (grp["R"] > 0).mean() * 100
        print(
            f"  {name:38s} trades={len(grp):3d}  win%={wins:5.1f}  "
            f"avgR={grp['R'].mean():+.2f}  totalR={grp['R'].sum():+.1f}"
        )
    eq = t["R"].cumsum()
    dd = (eq - eq.cummax()).min()
    beq = t["baseline_R"].cumsum()
    bdd = (beq - beq.cummax()).min()
    print(f"  Strategy : totalR={eq.iloc[-1]:+.1f}  maxDrawdown={dd:.1f}R")
    print(f"  Baseline (always-bull, no regime filter): totalR={beq.iloc[-1]:+.1f}  maxDD={bdd:.1f}R")


def main() -> None:
    print("Fetching 2 years of index data...")
    all_t = []
    for ticker, label in (("^NSEI", "NIFTY 50"), ("^NSEBANK", "BANK NIFTY")):
        try:
            t = run(ticker)
        except Exception as exc:
            print(f"  {label}: FAILED -> {exc}")
            continue
        all_t.append(t)
        report(t, label)
    if len(all_t) == 2:
        report(pd.concat(all_t, ignore_index=True), "COMBINED (both indices)")
    print(
        "\nNOTE: premiums are modeled (debit=45% of width, condor credit=33% of wing);"
        "\nreal results vary with volatility, fills, and costs. Decision-support only,"
        "\nnot financial advice. If totals beat the baseline with smaller drawdown,"
        "\nthe regime engine is adding value."
    )


if __name__ == "__main__":
    main()
