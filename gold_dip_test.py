"""GOLDBEES top-up / sell rules: is buying dips (and selling rallies) better than just holding?

Start: Rs 1,27,142 in gold (as on the paper tab) + Rs 2,00,000 ready in cash. Cash earns
6 % a year (a liquid fund). Every month from 2006 is tried as a start; each start is followed
for 3 and 5 years. Gold = gold in rupees (COMEX x USD/INR), what GOLDBEES tracks.
Costs 0.1 % per buy or sell. Taxes are not included (selling within a year is taxed at your slab).

  hold + cash idle        today's plan: 1.27 L gold, 2 L stays in cash
  all in on day 1         3.27 L in gold at once
  monthly over 12 months  2 L added in 12 equal parts
  dip from 1-year high    4 x 50k: one each when gold is 5 / 10 / 15 / 20 % below its 1-year high
  dip 10/20/30            2 x 1 L at 10 % and 20 % below the 1-year high (deeper dips only)
  ladder X/Y              buy 50k when gold is X % below the last buy; sell that 50k when it is
                          Y % above its own buy price (cash goes back to the pot)
  band 60 % +/-10         keep gold at 60 % of the total; trade back to 60 % when it drifts
                          outside 50-70 % (buys after falls, sells after rises)

Run: python gold_dip_test.py   -> logs/gold_dip.txt
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gold_test import gold

G0, CASH0, COST, RATE = 127_142.0, 200_000.0, 0.001, 0.06


def sim(px: pd.Series, rule: str, arg=None) -> pd.Series:
    """Daily total value (gold + cash) for one start."""
    hi = px.rolling(250, min_periods=1).max()
    units = G0 / px.iloc[0] * (1 - COST)
    cash, out, day_rate = CASH0, [], (1 + RATE) ** (1 / 250) - 1
    used, lots, last_buy = set(), [], px.iloc[0]
    month0 = px.index[0].month
    if rule == "all in":
        units += cash / px.iloc[0] * (1 - COST)
        cash = 0.0
    for i, (d, p) in enumerate(px.items()):
        cash *= 1 + day_rate
        if rule == "monthly" and i and d.month != px.index[i - 1].month and len(used) < 12 and cash > 0:
            amt = min(CASH0 / 12, cash)
            units += amt / p * (1 - COST); cash -= amt; used.add(len(used))
        elif rule == "dip":
            for k, lvl in enumerate(arg[0]):
                if k not in used and p <= hi.iloc[i] * (1 - lvl) and cash > 0:
                    amt = min(arg[1], cash)
                    units += amt / p * (1 - COST); cash -= amt; used.add(k)
        elif rule == "ladder":
            x, y = arg
            if p <= last_buy * (1 - x) and cash >= 50_000:
                u = 50_000 / p * (1 - COST)
                lots.append((p, u)); units += u; cash -= 50_000; last_buy = p
            keep = []
            for bp, u in lots:
                if p >= bp * (1 + y):
                    cash += u * p * (1 - COST); units -= u
                else:
                    keep.append((bp, u))
            if len(keep) < len(lots):
                last_buy = keep[-1][0] if keep else p
            lots = keep
        elif rule == "band":
            w, band = arg
            tot = units * p + cash
            if abs(units * p / tot - w) > band:
                trade = w * tot - units * p
                units += trade / p
                cash -= trade + abs(trade) * COST
        out.append(units * p + cash)
    return pd.Series(out, index=px.index)


RULES = {"hold + cash idle": ("hold", None), "all in on day 1": ("all in", None),
         "monthly over 12 months": ("monthly", None),
         "dip from 1-year high 5/10/15/20": ("dip", ((.05, .10, .15, .20), 50_000)),
         "dip 10/20 (2 x 1 L)": ("dip", ((.10, .20), 100_000)),
         "ladder buy -5 % / sell +10 %": ("ladder", (.05, .10)),
         "ladder buy -10 % / sell +20 %": ("ladder", (.10, .20)),
         "band 60 % +/-10": ("band", (.6, .1)), "band 80 % +/-10": ("band", (.8, .1))}


def main() -> None:
    px = gold()["inr"]
    starts = [px.index[px.index >= s][0] for s in pd.date_range("2006-01-01", px.index[-1], freq="MS")]
    lines = [f"Gold in rupees {px.index[0].date()} to {px.index[-1].date()}. Start Rs 3.27 L "
             f"(1.27 L gold + 2 L cash at {RATE:.0%}). Costs {COST:.1%} per trade, no taxes."]
    for years in (3, 5):
        rows, finals = [], {}
        for name, (rule, arg) in RULES.items():
            f, dd = [], []
            for s in starts:
                seg = px[s:]
                if (seg.index[-1] - s).days < years * 365:
                    break
                seg = seg[:s + pd.DateOffset(years=years)]
                v = sim(seg, rule, arg)
                f.append(v.iloc[-1]); dd.append((v / v.cummax() - 1).min() * 100)
            finals[name] = np.array(f)
            rows.append({"rule": name, "avg end value L": np.mean(f) / 1e5, "median L": np.median(f) / 1e5,
                         "worst L": np.min(f) / 1e5, "avg worst fall %": np.mean(dd), "worst fall %": np.min(dd)})
        t = pd.DataFrame(rows).set_index("rule")
        base, lump = finals["hold + cash idle"], finals["all in on day 1"]
        t["beats hold+cash %"] = [(finals[n] > base).mean() * 100 for n in t.index]
        t["beats all-in %"] = [(finals[n] > lump).mean() * 100 for n in t.index]
        lines += [f"\n=== {years}-year holds, {len(base)} monthly starts ===", t.round(2).to_string()]

    dd = px / px.cummax() - 1
    lines.append("\n=== After gold in rupees was 20 %+ below its all-time high: return over the next 1 / 2 / 3 years ===")
    for yrs in (1, 2, 3):
        fwd = px.shift(-250 * yrs) / px - 1
        a, b = fwd[dd <= -0.20].dropna(), fwd.dropna()
        lines.append(f"  {yrs} yr: after a 20 % fall avg {a.mean() * 100:+.1f} % ({len(a)} days, "
                     f"{a.index.year.nunique()} distinct years), any day avg {b.mean() * 100:+.1f} %")
    lines.append("  Falls of 20 %+ so far: " + ", ".join(sorted({str(y) for y in dd[dd <= -0.20].index.year})))
    text = "\n".join(lines)
    print(text)
    open("logs/gold_dip.txt", "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    main()
