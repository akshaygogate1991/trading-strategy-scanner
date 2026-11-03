"""Does a stop-loss help the monthly momentum portfolio?

The paper portfolio (momentum_paper.py) is the "top200 n20" version from
swing_robust_test.py: every month-end, the 20 stocks with the best 12-1 month
return among the 200 most traded, above their 200-day average; equal money,
bought at the next open, held until the next month's open. Rs 5 lakh, same costs.

Here a stock can also be sold in the middle of the month, then sits in cash until
the next rebalance (and is bought again then if it is still a pick):

  close stops   checked on the daily close, sold at the NEXT day's open. This is
                what the evening paper job could do (it sees closes only).
  intraday      a stop order: sold when the day's low touches the stop, at the stop
                price (or the open, if it gaps below). Needs a broker order.
  fixed X%      X% below the price the stock was bought at
  trailing X%   X% below the highest close since it was bought
  portfolio X%  all 20 sold if the portfolio is down X% in the month (close check)

Many versions are tested, so a small gain can be luck. A stop is only worth adding
if it is better in both halves (before / from 2020) and the monthly difference
against "no stop" has a t of 2 or more.

Needs swing_data/prices_all.pkl.   Run: python momentum_stop_test.py
"""
from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from swing_pit_test import MIN_PRICE
from swing_screener_test import SPLIT, buy_cost, rebalance_days, sell_cost, simulate, stats

DATA = Path(__file__).parent / "swing_data"
INDEX = "^NSEI"
CAPITAL, N, UNIVERSE = 500_000, 20, 200
PER = CAPITAL / N


def load():
    with open(DATA / "prices_all.pkl", "rb") as f:
        prices = pickle.load(f)
    nifty = prices.pop(INDEX)
    frame = lambda col: pd.DataFrame({t: d[col] for t, d in prices.items()}).sort_index()
    close = frame("Close")
    raw_close = close.copy()
    close = close.ffill(limit=5)
    opn, low = frame("Open").reindex(close.index), frame("Low").reindex(close.index)
    value = (frame("Close") * frame("Volume")).reindex(close.index)
    return close, raw_close, opn, low, value, nifty


def picker(close, raw_close, value, days):
    bars = raw_close.notna().cumsum()
    recent = raw_close.notna().rolling(5, min_periods=1).max().astype(bool)
    liq = value.rolling(126, min_periods=100).median()
    sma200 = close.rolling(200, min_periods=200).mean()
    ret12_1 = close.shift(21) / close.shift(252) - 1
    picks = {}
    for day in days:
        ok = (bars.loc[day] >= 260) & recent.loc[day] & (close.loc[day] >= MIN_PRICE)
        u = liq.loc[day][ok].dropna().nlargest(UNIVERSE).index
        up = close.loc[day, u] > sma200.loc[day, u]
        picks[day] = ret12_1.loc[day, u][up].dropna().nlargest(N).index.tolist()
    return picks


def run(picks, close, opn, low, days, kind="none", pct=0.0):
    """Monthly rupee P&L with an optional stop. kind: none, fixed, trail, ifixed, itrail, port."""
    idx = close.index
    C, O, L = close.to_numpy(), opn.to_numpy(), low.to_numpy()
    col = {t: j for j, t in enumerate(close.columns)}
    held = {}                                    # ticker -> (entry price, highest close since entry)
    rows, stops = [], 0
    for i, day in enumerate(days[:-1]):
        a = idx.get_loc(day) + 1
        b = idx.get_loc(days[i + 1]) + 1
        b = min(b, len(idx) - 1)
        names = [t for t in picks[day] if not math.isnan(O[a, col[t]]) and O[a, col[t]] > 0]
        cost = sum(buy_cost(PER) for t in names if t not in held)
        cost += sum(sell_cost(PER) for t in held if t not in names)
        held = {t: held.get(t, (O[a, col[t]], O[a, col[t]])) for t in names}
        exit_px, out_day = {}, {}
        # day by day; a stop sells at the next open (close checks) or at the stop (intraday)
        port_out = None
        for k in range(a, b):
            for t in names:
                if t in exit_px:
                    continue
                j = col[t]
                entry, peak = held[t]
                ref = peak if kind in ("trail", "itrail") else entry
                level = ref * (1 - pct)
                if kind in ("ifixed", "itrail") and not math.isnan(L[k, j]) and L[k, j] <= level:
                    o = O[k, j] if not math.isnan(O[k, j]) else level
                    exit_px[t], out_day[t] = min(level, o), k
                    continue
                c = C[k, j]
                if kind in ("fixed", "trail") and not math.isnan(c) and c <= level and k + 1 < b:
                    nxt = O[k + 1, j] if not math.isnan(O[k + 1, j]) else c
                    exit_px[t], out_day[t] = nxt, k + 1
                    continue
                if not math.isnan(c):
                    held[t] = (entry, max(peak, c))
            if kind == "port" and k + 1 < b:
                mtm = sum((exit_px[t] if t in exit_px else C[k, col[t]]) / O[a, col[t]] - 1 for t in names
                          if not math.isnan(C[k, col[t]]))
                if mtm * PER <= -pct * CAPITAL:
                    for t in names:
                        if t not in exit_px:
                            j = col[t]
                            exit_px[t] = O[k + 1, j] if not math.isnan(O[k + 1, j]) else C[k, j]
                    port_out = k + 1
                    break
        pnl = 0.0
        for t in names:
            j = col[t]
            p0 = O[a, j]
            if t in exit_px:
                p1 = exit_px[t]
                cost += sell_cost(PER)
                stops += 1
            else:
                p1 = O[b, j] if not math.isnan(O[b, j]) else close.iloc[:b + 1, j].dropna().iloc[-1]
            pnl += PER * (p1 / p0 - 1)
        for t in exit_px:                          # sold: no longer held at the next rebalance
            held.pop(t, None)
        rows.append({"month": day, "gross": pnl, "cost": cost, "net": pnl - cost,
                     "stopped": len(exit_px), "port_out": port_out is not None})
    return pd.DataFrame(rows).set_index("month"), stops


def main() -> None:
    close, raw_close, opn, low, value, nifty = load()
    days = rebalance_days(close)
    picks = picker(close, raw_close, value, days)
    nc = nifty["Close"]
    nifty_m = pd.Series({d: nc.loc[:days[i + 1]].iloc[-1] / nc.loc[:d].iloc[-1] - 1
                         for i, d in enumerate(days[:-1])})
    check = simulate(lambda d: picks[d], close, opn, days, PER)
    base, _ = run(picks, close, opn, low, days)
    print(f"check: original simulator CAGR {stats(check['net'], CAPITAL)['CAGR %']:.2f}%, "
          f"this one {stats(base['net'], CAPITAL)['CAGR %']:.2f}%")

    versions = [("no stop", "none", 0)]
    versions += [(f"fixed {p}% (close)", "fixed", p) for p in (8, 10, 15, 20, 25)]
    versions += [(f"trailing {p}% (close)", "trail", p) for p in (10, 15, 20, 25)]
    versions += [(f"fixed {p}% (intraday)", "ifixed", p) for p in (10, 15, 20)]
    versions += [(f"trailing {p}% (intraday)", "itrail", p) for p in (10, 15, 20)]
    versions += [(f"portfolio {p}% in a month", "port", p) for p in (5, 8, 10)]
    years = len(base) / 12
    rows, sims = [], {}
    for name, kind, p in versions:
        sim, stops = run(picks, close, opn, low, days, kind, p / 100) if kind != "none" else (base, 0)
        sims[name] = sim
        s = stats(sim["net"], CAPITAL, nifty_m)
        s1 = stats(sim["net"][sim.index < SPLIT], CAPITAL, nifty_m)
        s2 = stats(sim["net"][sim.index >= SPLIT], CAPITAL, nifty_m)
        diff = (sim["net"] - base["net"]) / CAPITAL
        t_diff = diff.mean() / (diff.std() / math.sqrt(len(diff))) if diff.std() > 0 else float("nan")
        yr = sim["net"].groupby(sim.index.year).sum() / CAPITAL * 100
        worst_m = sim["net"].min() / CAPITAL * 100
        rows.append({"version": name, "CAGR %": s["CAGR %"], "MaxDD %": s["MaxDD %"], "Sharpe": s["Sharpe"],
                     "worst month %": worst_m, "worst yr %": yr.min(),
                     "CAGR 1st": s1["CAGR %"], "CAGR 2nd": s2["CAGR %"],
                     "DD 1st": s1["MaxDD %"], "DD 2nd": s2["MaxDD %"],
                     "stops/yr": stops / years, "t vs no stop": t_diff})
        print(f"  {name}: CAGR {s['CAGR %']:.1f}%, max DD {s['MaxDD %']:.1f}%")

    table = pd.DataFrame(rows).set_index("version")
    out = [f"\n=== Momentum top200 n20, Rs 5 lakh, net of costs, {days[0].date()} to {days[-2].date()} "
           f"({len(base)} months) ===",
           table.round(2).to_string(),
           "\n1st = before 2020, 2nd = from 2020. t vs no stop = t-stat of the monthly difference "
           "against no stop (2+ needed, more with this many versions).",
           "(close) = sold at the next open after a close below the stop (what the evening job can do); "
           "(intraday) = a stop order filled at the stop."]
    out.append("\n=== Calendar years, net % ===")
    keep = ["no stop", "fixed 15% (close)", "trailing 20% (close)", "fixed 15% (intraday)", "portfolio 8% in a month"]
    yr = pd.DataFrame({n: sims[n]["net"].groupby(sims[n].index.year).sum() / CAPITAL * 100 for n in keep})
    out.append(yr.round(1).T.to_string())
    text = "\n".join(out)
    print(text)
    Path("logs").mkdir(exist_ok=True)
    Path("logs/momentum_stop.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
