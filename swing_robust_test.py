"""Momentum 12-1 with more stocks and a market trend filter.

swing_pit_test.py showed momentum with 5 stocks beats random picks but falls up
to 58%. This tests the two usual fixes, on the same point-in-time universe:
  * more stocks: 5, 10 or 20, sharing the same Rs 5 lakh equally
  * trend filter: at a month-end close with the Nifty 50 below its 200-day
    average, hold no stocks (cash) for that month

12 versions are tested, so one of them looking good can be luck. Only trust a
version that is good in both halves (before and from 2020) and not just one.

Needs swing_data/prices_all.pkl (python fetch_swing_data.py --all).
Run:  python swing_robust_test.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from swing_pit_test import MIN_PRICE, load
from swing_screener_test import SPLIT, rebalance_days, simulate, stats

CAPITAL = 500_000
RANDOM_RUNS = 200


def main() -> None:
    close, raw_close, opn, value, nifty = load()
    days = rebalance_days(close)
    print(f"{close.shape[1]} stocks, {len(days) - 1} monthly rebalances "
          f"{days[0].date()} to {days[-2].date()}")

    bars = raw_close.notna().cumsum()
    traded_recently = raw_close.notna().rolling(5, min_periods=1).max().astype(bool)
    liq = value.rolling(126, min_periods=100).median()
    sma200 = close.rolling(200, min_periods=200).mean()
    ret12_1 = close.shift(21) / close.shift(252) - 1

    nc = nifty["Close"]
    nifty_up = {d: nc.loc[:d].iloc[-1] > nc.loc[:d].tail(200).mean() for d in days}
    nifty_m = pd.Series({d: nc.loc[:days[i + 1]].iloc[-1] / nc.loc[:d].iloc[-1] - 1
                         for i, d in enumerate(days[:-1])})
    print(f"Nifty below its 200-day average at {sum(not v for v in nifty_up.values())} "
          f"of {len(days)} month-ends")

    def universe(day, size):
        ok = (bars.loc[day] >= 260) & traded_recently.loc[day] & (close.loc[day] >= MIN_PRICE)
        return liq.loc[day][ok].dropna().nlargest(size).index

    unis = {size: {d: universe(d, size) for d in days} for size in (200, 100)}

    def momentum(size, n, filt):
        uni = unis[size]

        def pick(day):
            if filt and not nifty_up[day]:
                return []
            u = uni[day]
            ok = close.loc[day, u] > sma200.loc[day, u]
            return ret12_1.loc[day, u][ok].dropna().nlargest(n).index.tolist()
        return pick

    first = lambda ix: ix < SPLIT
    second = lambda ix: ix >= SPLIT
    rows, sims = [], {}
    for size in (200, 100):
        for n in (5, 10, 20):
            per = CAPITAL / n
            rng = np.random.default_rng(7)
            rand = []
            for _ in range(RANDOM_RUNS):
                draws = {d: list(rng.choice(unis[size][d], n, replace=False)) for d in days[:-1]}
                rand.append(stats(simulate(lambda d: draws[d], close, opn, days, per)["net"],
                                  CAPITAL, nifty_m)["CAGR %"])
            rand = np.array(rand)
            for filt in (False, True):
                name = f"top{size} n{n}{' +filter' if filt else ''}"
                sim = simulate(momentum(size, n, filt), close, opn, days, per)
                sims[name] = sim
                s = stats(sim["net"], CAPITAL, nifty_m)
                s1 = stats(sim["net"][first(sim.index)], CAPITAL, nifty_m)
                s2 = stats(sim["net"][second(sim.index)], CAPITAL, nifty_m)
                yr = sim["net"].groupby(sim.index.year).sum() / CAPITAL * 100
                rows.append({"version": name, "CAGR %": s["CAGR %"], "Sharpe": s["Sharpe"],
                             "MaxDD %": s["MaxDD %"], "Excess/yr": s["Excess/yr %"],
                             "t": s["t (excess)"], "t 1st": s1["t (excess)"], "t 2nd": s2["t (excess)"],
                             "CAGR 1st": s1["CAGR %"], "CAGR 2nd": s2["CAGR %"],
                             "worst yr": yr.min(), "costs/yr": sim["cost"].sum() / (len(sim) / 12),
                             "rand pct": (rand < s["CAGR %"]).mean() * 100,
                             "rand med": float(np.median(rand))})
                print(f"  {name}: CAGR {s['CAGR %']:.1f}%, max DD {s['MaxDD %']:.1f}%")

    table = pd.DataFrame(rows).set_index("version")
    sn = stats(nifty_m * CAPITAL, CAPITAL)
    out = ["\n=== Momentum 12-1, Rs 5 lakh, net of costs (153 months, Feb 2014 - Oct 2026) ===",
           table.round(2).to_string(),
           f"\nNifty 50 buy and hold: CAGR {sn['CAGR %']:.1f}%, Sharpe {sn['Sharpe']:.2f}, "
           f"max DD {sn['MaxDD %']:.1f}%",
           "t = excess over Nifty per month, t-stat; 1st = before 2020, 2nd = from 2020.",
           "rand pct = share of random runs (same universe and stock count, no filter) "
           "with lower CAGR; rand med = their median CAGR."]

    out.append("\n=== Calendar years, net % ===")
    yr = pd.DataFrame({n: s["net"].groupby(s.index.year).sum() / CAPITAL * 100 for n, s in sims.items()})
    yr["nifty50"] = ((1 + nifty_m).groupby(nifty_m.index.year).prod() - 1) * 100
    out.append(yr.round(1).T.to_string())

    last = days[-1]
    out.append(f"\nNifty above its 200-day average at {last.date()}: {nifty_up[last]}")
    for name in ("top100 n10 +filter", "top100 n20 +filter", "top200 n10 +filter"):
        size, n = int(name[3:6]), int(name.split(" n")[1].split()[0])
        picks = momentum(size, n, True)(last)
        out.append(f"{name} picks: " + ", ".join(t.replace(".NS", "") for t in picks))

    text = "\n".join(out)
    print(text)
    Path("logs").mkdir(exist_ok=True)
    Path("logs/swing_robust.txt").write_text(text, encoding="utf-8")
    pd.concat(sims, names=["version"]).to_csv("logs/swing_robust_months.csv")


if __name__ == "__main__":
    main()
