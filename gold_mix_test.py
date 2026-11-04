import sys, pandas as pd, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
from gold_test import gold, stats, tdiff
from momentum_stop_test import CAPITAL, N, PER, load, picker
from swing_screener_test import rebalance_days, simulate
px = gold()["inr"]
close, raw_close, opn, low, value, nifty = load()
days = rebalance_days(close)
picks = picker(close, raw_close, value, days)
sim = simulate(lambda d: picks[d], close, opn, days, PER)
mom = sim["net"] / CAPITAL
g = px.reindex(px.index.union(days)).ffill()
gm = pd.Series({d: g.loc[days[i + 1]] / g.loc[d] - 1 for i, d in enumerate(days[:-1])}).reindex(mom.index).dropna()
mom = mom.loc[gm.index]
pd.concat({"mom": mom, "gold": gm}, axis=1).to_csv(r"swing_data\mom_gold_monthly.csv")
COST = 0.001
def run(w, mode, band=0.05):
    wg = w; out = []
    for i, d in enumerate(mom.index):
        r = (1 - wg) * mom[d] + wg * gm[d]
        wg_new = wg * (1 + gm[d]) / (1 + r)
        reb = (mode == "monthly") or (mode == "yearly" and d.month == 1) or (mode == "band" and abs(wg_new - w) > band)
        if reb:
            r -= abs(wg_new - w) * COST * 2
            wg = w
        else:
            wg = wg_new
        out.append(r)
    return pd.Series(out, index=mom.index)
rows = []
for w in (0, .1, .15, .2, .25, .3, .4):
    for mode in (["monthly"] if w == 0 else ["monthly", "yearly", "band"]):
        r = run(w, mode)
        rows.append({"gold %": int(w*100), "rebalance": mode, **stats(r, 12), "t vs mom": tdiff(r, mom)})
print(f"{mom.index[0].date()} to {mom.index[-1].date()}, {len(mom)} months")
print(pd.DataFrame(rows).round(2).to_string(index=False))
