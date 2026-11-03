"""Price-action add-ons to the monthly momentum portfolio.

Base = the paper job (top 20 by 12-1 month return among the 200 most traded,
above the 200-day average, monthly, Rs 5 lakh, same costs). Four ideas from the
literature / practice, each tried as a filter or a different ranking:

  1. 52-week high (George & Hwang 2004): price within X% of its 52-week high,
     or rank by price / 52-week high.
  2. Frog in the pan (Da, Gurun & Warachka 2014): among the strongest 40 / 60,
     take the 20 whose gain came most smoothly (more up days than down days over
     the 12-1 month window).
  3. Volatility-adjusted (NSE Nifty200 Momentum 30 style): 6-1 and 12-1 month
     returns each divided by yearly volatility, z-scored, averaged.
  4. Minervini trend template: price > 50 > 150 > 200-day averages, 200-day
     average rising over a month, price within 25 % of the 52-week high and 30 %+
     above the 52-week low.

12 versions, so one looking good can be luck: keep one only if it beats the base
in both halves and its monthly difference vs the base has t >= 2.

Run: python momentum_priceaction_test.py   -> logs/momentum_priceaction.txt
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from momentum_stop_test import CAPITAL, N, PER, UNIVERSE, load
from swing_pit_test import MIN_PRICE
from swing_screener_test import SPLIT, rebalance_days, simulate, stats


def month_rows(df, months):
    return df.loc[months].astype("float32")


def main() -> None:
    close, raw_close, opn, low, value, nifty = load()
    del low
    months = rebalance_days(close)

    # everything needed at month-ends only (keeps memory down)
    M = {}
    M["bars"] = month_rows(raw_close.notna().cumsum(), months)
    M["recent"] = month_rows(raw_close.notna().rolling(5, min_periods=1).max(), months)
    M["liq"] = month_rows(value.rolling(126, min_periods=100).median(), months)
    del value
    for n in (50, 150, 200):
        M[f"sma{n}"] = month_rows(close.rolling(n, min_periods=n).mean(), months)
    M["sma200_prev"] = month_rows(close.rolling(200, min_periods=200).mean().shift(21), months)
    M["hi252"] = month_rows(close.rolling(252, min_periods=200).max(), months)
    M["lo252"] = month_rows(close.rolling(252, min_periods=200).min(), months)
    M["ret12_1"] = month_rows(close.shift(21) / close.shift(252) - 1, months)
    M["ret6_1"] = month_rows(close.shift(21) / close.shift(126) - 1, months)
    dr = close.pct_change(fill_method=None)
    M["vol"] = month_rows(dr.rolling(252, min_periods=200).std() * math.sqrt(252), months)
    # frog in the pan: (share of down days - share of up days) over the 12-1 window, lower = smoother
    up = (dr > 0).astype("float32").shift(21).rolling(231, min_periods=180).mean()
    dn = (dr < 0).astype("float32").shift(21).rolling(231, min_periods=180).mean()
    M["fip"] = month_rows(dn - up, months)
    del dr, up, dn
    M["close"] = month_rows(close, months)

    def base_set(day):
        """Universe and the plain eligible list (above the 200-day average), best 12-1 first."""
        ok = (M["bars"].loc[day] >= 260) & (M["recent"].loc[day] > 0) & (M["close"].loc[day] >= MIN_PRICE)
        u = M["liq"].loc[day][ok].dropna().nlargest(UNIVERSE).index
        c = M["close"].loc[day, u]
        above = c > M["sma200"].loc[day, u]
        r = M["ret12_1"].loc[day, u][above].dropna().sort_values(ascending=False)
        return u, r

    def v_base(day):
        return base_set(day)[1].index[:N].tolist()

    def v_near_high(x):
        def pick(day):
            u, r = base_set(day)
            near = M["close"].loc[day, r.index] / M["hi252"].loc[day, r.index] >= 1 - x
            return r[near].index[:N].tolist()
        return pick

    def v_rank_high(day):
        u, r = base_set(day)
        r = r[r > 0]
        ratio = (M["close"].loc[day, r.index] / M["hi252"].loc[day, r.index]).dropna()
        return ratio.sort_values(ascending=False).index[:N].tolist()

    def v_mix_high(day):
        u, r = base_set(day)
        ratio = (M["close"].loc[day, r.index] / M["hi252"].loc[day, r.index])
        score = r.rank(pct=True) + ratio.rank(pct=True)
        return score.dropna().sort_values(ascending=False).index[:N].tolist()

    def v_fip(k):
        def pick(day):
            u, r = base_set(day)
            top = r.index[:k]
            return M["fip"].loc[day, top].dropna().sort_values().index[:N].tolist()
        return pick

    def v_voladj(both):
        def pick(day):
            u, r = base_set(day)
            names = r.index
            vol = M["vol"].loc[day, names]
            z = lambda s: (s - s.mean()) / s.std()
            score = z(M["ret12_1"].loc[day, names] / vol)
            if both:
                score = (score + z(M["ret6_1"].loc[day, names] / vol)) / 2
            return score.dropna().sort_values(ascending=False).index[:N].tolist()
        return pick

    def v_minervini(day):
        u, r = base_set(day)
        n = r.index
        c = M["close"].loc[day, n]
        ok = ((c > M["sma50"].loc[day, n]) & (M["sma50"].loc[day, n] > M["sma150"].loc[day, n])
              & (M["sma150"].loc[day, n] > M["sma200"].loc[day, n])
              & (M["sma200"].loc[day, n] > M["sma200_prev"].loc[day, n])
              & (c >= 0.75 * M["hi252"].loc[day, n]) & (c >= 1.30 * M["lo252"].loc[day, n]))
        return r[ok].index[:N].tolist()

    versions = [("base (paper job)", v_base),
                ("1. within 10% of 52w high", v_near_high(0.10)),
                ("1. within 20% of 52w high", v_near_high(0.20)),
                ("1. within 25% of 52w high", v_near_high(0.25)),
                ("1. rank by price/52w high", v_rank_high),
                ("1. rank by 12-1 + 52w high", v_mix_high),
                ("2. frog in pan, best 20 of 40", v_fip(40)),
                ("2. frog in pan, best 20 of 60", v_fip(60)),
                ("3. vol-adjusted 12-1", v_voladj(False)),
                ("3. vol-adjusted 6-1 + 12-1 (NSE style)", v_voladj(True)),
                ("4. Minervini trend template", v_minervini)]

    nc = nifty["Close"]
    nifty_m = pd.Series({d: nc.loc[:months[i + 1]].iloc[-1] / nc.loc[:d].iloc[-1] - 1
                         for i, d in enumerate(months[:-1])})
    rows, base = [], None
    for name, fn in versions:
        sim = simulate(fn, close, opn, months, PER)
        if base is None:
            base = sim
        s = stats(sim["net"], CAPITAL, nifty_m)
        s1 = stats(sim["net"][sim.index < SPLIT], CAPITAL, nifty_m)
        s2 = stats(sim["net"][sim.index >= SPLIT], CAPITAL, nifty_m)
        diff = (sim["net"] - base["net"]) / CAPITAL
        t = diff.mean() / (diff.std() / math.sqrt(len(diff))) if diff.std() > 0 else float("nan")
        d1, d2 = diff[diff.index < SPLIT].mean() * 1200, diff[diff.index >= SPLIT].mean() * 1200
        yrs = len(sim) / 12
        yr = sim["net"].groupby(sim.index.year).sum() / CAPITAL * 100
        rows.append({"version": name, "CAGR %": s["CAGR %"], "MaxDD %": s["MaxDD %"], "Sharpe": s["Sharpe"],
                     "worst yr %": yr.min(), "CAGR 1st": s1["CAGR %"], "CAGR 2nd": s2["CAGR %"],
                     "vs base 1st %/yr": d1, "vs base 2nd %/yr": d2, "t vs base": t,
                     "avg stocks": sim["n"].mean(), "buys/yr": sim["buys"].sum() / yrs})
        print(f"  {name}: CAGR {s['CAGR %']:.1f}%, max DD {s['MaxDD %']:.1f}%, t vs base {t:+.2f}", flush=True)

    table = pd.DataFrame(rows).set_index("version")
    text = "\n".join([
        f"\n=== Momentum + price action, Rs 5 lakh, net of costs, {months[0].date()} to {months[-2].date()} ===",
        table.round(2).to_string(),
        "\n1st = before 2020, 2nd = from 2020. vs base = average yearly difference in each half; "
        "t vs base = t-stat of the monthly difference. Empty places sit in cash (avg stocks < 20)."])
    print(text)
    Path("logs").mkdir(exist_ok=True)
    Path("logs/momentum_priceaction.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
