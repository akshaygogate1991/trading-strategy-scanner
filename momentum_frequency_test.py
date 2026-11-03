"""Check momentum more often than once a month?

Same portfolio as the paper job (top 20 by 12-1 month return among the 200 most
traded, above the 200-day average, Rs 5 lakh, same costs). Versions:

  monthly             the paper job: full top 20 at each month-end
  every 2 weeks       full top 20 every second Friday-ish (last trading day of the week)
  weekly              full top 20 every week
  monthly + weekly exits
                      top 20 at month-end; every week a held stock is sold (cash until
                      the month-end) if it fell out of the top K or below its 200-day
                      average. K = 20, 40, 60.
  monthly, buffer K   at month-end keep a held stock while it is still in the top K
                      (not just the top 20); fill the free places from the top.
                      Fewer trades. K = 30, 40, 60.

Run: python momentum_frequency_test.py   -> logs/momentum_frequency.txt
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from momentum_stop_test import CAPITAL, N, PER, UNIVERSE, load
from swing_pit_test import MIN_PRICE
from swing_screener_test import SPLIT, rebalance_days, simulate, stats


def main() -> None:
    close, raw_close, opn, low, value, nifty = load()
    months = rebalance_days(close)
    idx = close.index
    wk = pd.Series(idx, index=idx)
    weeks = [d for d in wk.groupby(idx.to_period("W")).last() if months[0] <= d <= months[-1]]
    weeks2 = weeks[::2]
    month_set = set(months)

    bars = raw_close.notna().cumsum()
    recent = raw_close.notna().rolling(5, min_periods=1).max().astype(bool)
    liq = value.rolling(126, min_periods=100).median()
    sma200 = close.rolling(200, min_periods=200).mean()
    ret12_1 = close.shift(21) / close.shift(252) - 1
    cache = {}

    def ranked(day):
        """Eligible stocks, best 12-1 return first."""
        if day not in cache:
            ok = (bars.loc[day] >= 260) & recent.loc[day] & (close.loc[day] >= MIN_PRICE)
            u = liq.loc[day][ok].dropna().nlargest(UNIVERSE).index
            up = close.loc[day, u] > sma200.loc[day, u]
            cache[day] = ret12_1.loc[day, u][up].dropna().sort_values(ascending=False).index.tolist()
        return cache[day]

    def full(day):
        return ranked(day)[:N]

    def weekly_exits(k):
        state = {"held": []}

        def pick(day):
            if day in month_set:
                state["held"] = full(day)
            else:
                top = set(ranked(day)[:k])
                state["held"] = [t for t in state["held"] if t in top]
            return state["held"]
        return pick

    def buffer(k):
        state = {"held": []}

        def pick(day):
            r = ranked(day)
            keep = [t for t in state["held"] if t in set(r[:k])]
            for t in r:
                if len(keep) >= N:
                    break
                if t not in keep:
                    keep.append(t)
            state["held"] = keep
            return keep
        return pick

    nc = nifty["Close"]
    nifty_m = pd.Series({d: nc.loc[:months[i + 1]].iloc[-1] / nc.loc[:d].iloc[-1] - 1
                         for i, d in enumerate(months[:-1])})

    def to_months(sim):
        """Sum the P&L of each period into the month it started in (labelled like the monthly run)."""
        key = pd.Series(sim.index, index=sim.index).map(lambda d: max(m for m in months if m <= d))
        return sim.groupby(key.values)[["gross", "cost", "net", "buys"]].sum()

    versions = [("monthly (paper job)", full, months),
                ("every 2 weeks", full, sorted(set(weeks2) | {months[-1]})),
                ("weekly", full, sorted(set(weeks) | {months[-1]}))]
    versions += [(f"monthly + weekly exits, top {k}", weekly_exits(k), sorted(set(weeks) | month_set))
                 for k in (20, 40, 60)]
    versions += [(f"monthly, buffer top {k}", buffer(k), months) for k in (30, 40, 60)]

    base = None
    rows = []
    for name, fn, days in versions:
        sim = to_months(simulate(fn, close, opn, days, PER))
        sim = sim.reindex(months[:-1]).fillna(0)
        if base is None:
            base = sim
        s = stats(sim["net"], CAPITAL, nifty_m)
        s1 = stats(sim["net"][sim.index < SPLIT], CAPITAL, nifty_m)
        s2 = stats(sim["net"][sim.index >= SPLIT], CAPITAL, nifty_m)
        diff = (sim["net"] - base["net"]) / CAPITAL
        t = diff.mean() / (diff.std() / math.sqrt(len(diff))) if diff.std() > 0 else float("nan")
        yrs = len(sim) / 12
        rows.append({"version": name, "CAGR %": s["CAGR %"], "MaxDD %": s["MaxDD %"], "Sharpe": s["Sharpe"],
                     "CAGR 1st": s1["CAGR %"], "CAGR 2nd": s2["CAGR %"],
                     "buys/yr": sim["buys"].sum() / yrs, "costs/yr Rs": sim["cost"].sum() / yrs,
                     "t vs monthly": t})
        print(f"  {name}: CAGR {s['CAGR %']:.1f}%, max DD {s['MaxDD %']:.1f}%, {sim['buys'].sum() / yrs:.0f} buys/yr")

    table = pd.DataFrame(rows).set_index("version")
    text = "\n".join([
        f"\n=== Momentum top200 n20, Rs 5 lakh, net of costs, {months[0].date()} to {months[-2].date()} ===",
        table.round(2).to_string(),
        "\n1st = before 2020, 2nd = from 2020. t vs monthly = t-stat of the monthly difference "
        "against the paper job's monthly version."])
    print(text)
    Path("logs").mkdir(exist_ok=True)
    Path("logs/momentum_frequency.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
