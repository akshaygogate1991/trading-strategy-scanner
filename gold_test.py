"""Gold for an Indian investor: does a trend rule help, and does gold help the momentum portfolio?

Gold in rupees = COMEX gold (GC=F) x USD/INR (INR=X), daily from Yahoo - what a gold ETF
(GOLDBEES) or MCX gold tracks, before the ETF's small fee. GOLDBEES itself is shown as a check.

  1. Trend rules on gold (else cash), signal at a close, traded the next day, 0.1 % per switch:
       above the 200-day average  |  10-month average (month-end)  |  12-month return > 0 (month-end)
       |  both 200-day and 12-month
  2. Gold next to the momentum paper portfolio (top200 n20, monthly, 2014-2026):
       a. the cash the portfolio holds when fewer than 20 stocks qualify goes into gold
       b. a fixed 20 % in gold, 80 % momentum, rebalanced monthly
       c. as b, but the 20 % only in gold when gold's 12-month return is positive (else cash)
  Same checks as before: before / from 2020, t of the monthly difference vs the base.

Run: python gold_test.py   -> logs/gold.txt
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
CACHE = HERE / "swing_data" / "gold_inr_daily.csv"
SWITCH_COST = 0.001
SPLIT = pd.Timestamp("2020-01-01")


def yahoo(ticker: str) -> pd.Series:
    import yfinance as yf
    d = yf.download(ticker, period="max", interval="1d", auto_adjust=True, progress=False)
    d.columns = d.columns.get_level_values(0)
    s = d["Close"].dropna()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    return s


def gold() -> pd.DataFrame:
    if not CACHE.exists():
        g, fx, etf = yahoo("GC=F"), yahoo("INR=X"), yahoo("GOLDBEES.NS")
        fx = fx[(fx > 30) & (fx < 200)]                       # a few bad ticks in the old FX data
        df = pd.DataFrame({"usd": g, "fx": fx}).dropna()
        df["inr"] = df.usd * df.fx
        df = df.join(etf.rename("goldbees"), how="left")
        CACHE.parent.mkdir(exist_ok=True)
        df.to_csv(CACHE)
    return pd.read_csv(CACHE, index_col=0, parse_dates=True)


def stats(r: pd.Series, per_year: int) -> dict:
    eq = (1 + r).cumprod()
    yrs = len(r) / per_year
    half = {k: ((1 + p).prod() ** (per_year / len(p)) - 1) * 100 if len(p) else float("nan")
            for k, p in (("1st", r[r.index < SPLIT]), ("2nd", r[r.index >= SPLIT]))}
    return {"CAGR %": (eq.iloc[-1] ** (1 / yrs) - 1) * 100, "MaxDD %": (eq / eq.cummax() - 1).min() * 100,
            "Sharpe": r.mean() / r.std() * math.sqrt(per_year), "CAGR 1st": half["1st"], "CAGR 2nd": half["2nd"],
            "worst yr %": ((1 + r).groupby(r.index.year).prod() - 1).min() * 100}


def tdiff(a: pd.Series, b: pd.Series) -> float:
    d = (a - b).dropna()
    return d.mean() / (d.std() / math.sqrt(len(d))) if d.std() > 0 else float("nan")


def trend_rules(px: pd.Series) -> list[str]:
    ret = px.pct_change().fillna(0)
    month_end = px.groupby(px.index.to_period("M")).tail(1).index
    sma200 = px.rolling(200).mean()
    m_px = px.loc[month_end]
    sma10m = m_px.rolling(10).mean()
    mom12 = m_px / m_px.shift(12) - 1

    def monthly(sig_m):                               # month-end signal held through the next month
        return sig_m.reindex(px.index).ffill().fillna(0)

    sig = {"buy and hold": pd.Series(1.0, index=px.index),
           "above 200-day average": (px > sma200).astype(float),
           "10-month average": monthly((m_px > sma10m).astype(float)),
           "12-month return > 0": monthly((mom12 > 0).astype(float)),
           "200-day and 12-month": ((px > sma200) & (monthly((mom12 > 0).astype(float)) > 0)).astype(float)}
    start = px.index[260]
    rows, base_m = [], None
    for name, s in sig.items():
        pos = s.shift(1).fillna(0)                    # known at the close, held from the next day
        r = (pos * ret - pos.diff().abs().fillna(0) * SWITCH_COST)[start:]
        m = (1 + r).groupby(r.index.to_period("M")).prod() - 1
        base_m = m if base_m is None else base_m
        st = stats(r, 252)
        rows.append({"rule": name, **st, "in gold %": pos[start:].mean() * 100,
                     "switches/yr": pos[start:].diff().abs().sum() / (len(r) / 252), "t vs hold": tdiff(m, base_m)})
    t = pd.DataFrame(rows).set_index("rule").round(2)
    return [f"\n=== 1. Gold in rupees, trend rules vs holding, {start.date()} to {px.index[-1].date()} ===",
            t.to_string()]


def with_momentum(px: pd.Series) -> list[str]:
    from momentum_stop_test import CAPITAL, N, PER, UNIVERSE, load, picker
    from swing_screener_test import rebalance_days, simulate
    close, raw_close, opn, low, value, nifty = load()
    del low
    days = rebalance_days(close)
    picks = picker(close, raw_close, value, days)
    sim = simulate(lambda d: picks[d], close, opn, days, PER)
    mom = sim["net"] / CAPITAL
    cash = (N - sim["n"]) / N
    g = px.reindex(px.index.union(days)).ffill()
    gold_m = pd.Series({d: g.loc[days[i + 1]] / g.loc[d] - 1 for i, d in enumerate(days[:-1])}).reindex(mom.index)
    gm = px.groupby(px.index.to_period("M")).last()
    g12 = pd.Series({d: (gm.loc[:pd.Period(d, "M") - 1].iloc[-1] / gm.loc[:pd.Period(d, "M") - 13].iloc[-1] - 1)
                     for d in mom.index})
    on = (g12 > 0).astype(float)
    versions = {"momentum alone (paper job)": mom,
                "a. spare cash into gold": mom + cash * gold_m - cash * 2 * SWITCH_COST * (cash.diff().abs() > 0),
                "b. 80 % momentum + 20 % gold": 0.8 * mom + 0.2 * gold_m,
                "c. 80 % + 20 % gold only if gold trend up": 0.8 * mom + 0.2 * on * gold_m,
                "gold alone (same months)": gold_m}
    rows = []
    for name, r in versions.items():
        r = r.dropna()
        rows.append({"version": name, **stats(r, 12), "t vs momentum": tdiff(r, mom)})
    corr = mom.corr(gold_m)
    bad = mom.nsmallest(12).index
    t = pd.DataFrame(rows).set_index("version").round(2)
    return [f"\n=== 2. Gold next to the momentum portfolio, {mom.index[0].date()} to {mom.index[-1].date()} "
            f"({len(mom)} months) ===", t.to_string(),
            f"\nCorrelation of monthly returns, momentum vs gold: {corr:+.2f}. "
            f"Average spare cash: {cash.mean() * 100:.0f} % of the portfolio.",
            f"In momentum's 12 worst months (avg {mom[bad].mean() * 100:+.1f} %), gold averaged "
            f"{gold_m[bad].mean() * 100:+.1f} % and rose in {(gold_m[bad] > 0).sum()} of 12."]


def main() -> None:
    df = gold()
    px = df["inr"]
    lines = [f"Gold in rupees (COMEX x USD/INR) {px.index[0].date()} to {px.index[-1].date()}; "
             f"GOLDBEES from {df.goldbees.first_valid_index().date()}"]
    both = df.dropna(subset=["goldbees"])
    yrs = (both.index[-1] - both.index[0]).days / 365.25
    lines.append(f"Check over GOLDBEES's life: gold in rupees {((both.inr.iloc[-1] / both.inr.iloc[0]) ** (1 / yrs) - 1) * 100:.1f} %/yr, "
                 f"GOLDBEES {((both.goldbees.iloc[-1] / both.goldbees.iloc[0]) ** (1 / yrs) - 1) * 100:.1f} %/yr")
    lines += trend_rules(px)
    lines += with_momentum(px)
    lines += ["", "1st = before 2020, 2nd = from 2020. t = t-stat of the monthly difference (2+ needed). "
              f"Costs {SWITCH_COST * 100:.1f} % per switch; momentum net of its usual costs."]
    text = "\n".join(lines)
    print(text)
    (HERE / "logs").mkdir(exist_ok=True)
    (HERE / "logs" / "gold.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
