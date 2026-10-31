"""Parameter-robustness sweep for one strategy inside an .algtst file.

Each setting is varied on its own around the original (one-at-a-time), plus a
dip% x SL% grid. Every variant runs on the same days, loaded once. A strategy
whose profit only exists at the exact seminar settings is curve-fit; a real edge
shows a broad plateau of profitable neighbours.

  python sweep_algtst.py "D:\\algotest\\export_sensexbasket.algtst" --strategy "Sensex Directional"
"""
import argparse
import copy
import math
import os

import numpy as np
import pandas as pd

import algtst_backtest as ab


def variants(base):
    """(label, strategy) pairs: one-at-a-time changes plus a dip x SL grid."""
    def legs_set(st, **kw):
        for c in st["legs"]:
            c.update(kw)
        return st

    def make(label, fn):
        st = copy.deepcopy(base)
        fn(st)
        st["name"] = label
        return label, st

    c0 = base["legs"][0]
    out = [make("ORIGINAL", lambda s: None)]
    for p in (100, 150, 250, 300):
        out.append(make(f"premium {p}", lambda s, p=p: legs_set(s, param=float(p))))
    for hh, mm in ((9, 20), (9, 45), (10, 0), (10, 30), (11, 0)):
        out.append(make(f"entry {hh}:{mm:02d}", lambda s, m=hh * 60 + mm - ab.OPEN_MIN: s.update(entry=m)))
    for hh, mm in ((14, 30), (15, 15)):
        out.append(make(f"exit {hh}:{mm:02d}", lambda s, m=hh * 60 + mm - ab.OPEN_MIN: s.update(exit=m)))
    out.append(make("no trail", lambda s: legs_set(s, trail=None)))
    for tr in ((10.0, 5.0), (20.0, 20.0), (30.0, 10.0)):
        out.append(make(f"trail {tr[0]:g}/{tr[1]:g}", lambda s, tr=tr: legs_set(s, trail=tr)))
    out.append(make("no overall SL", lambda s: s.update(osl=None)))
    for v in (500, 1000, 1500):
        out.append(make(f"overall SL {v}", lambda s, v=v: s.update(osl=float(v))))
    for dip in (0, 1, 2, 3, 5):
        for sl in (10, 15, 20, 30):
            mom = None if dip == 0 else ("PercentageDown", float(dip))
            if dip == c0["mom"][1] and sl == c0["sl"]:
                continue        # that is ORIGINAL
            out.append(make(f"dip {dip}% SL {sl}%", lambda s, m=mom, sl=sl: legs_set(s, mom=m, sl=float(sl))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--start", default="2023-08-01")
    ap.add_argument("--end", default="2026-05-31")
    ap.add_argument("--slip-pct", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(ab.HERE, "logs", "sweep_daily.csv"))
    args = ap.parse_args()

    base = next(s for s in ab.load_file(args.file) if s["name"] == args.strategy)
    var = variants(base)
    lot = ab.LOT[base["ticker"]]
    rows = []
    for day, exp, d in ab.iter_days(base["ticker"], args.start, args.end):
        for label, st in var:
            net = sum(ab.add_costs(t, base["ticker"], args.slip_pct)["net"] for t in ab.run_day(st, d, lot))
            rows.append((label, day, net))
    daily = pd.DataFrame(rows, columns=["variant", "day", "net"])
    daily.to_csv(args.out, index=False)
    report(daily, [v[0] for v in var])


def report(daily, order):
    daily = daily.assign(year=daily["day"].str[:4])
    print(f"{'variant':<20}{'net Rs':>10}{'t':>6}{'win%':>6}{'maxDD':>9}   by year")
    for label in order:
        d = daily[daily["variant"] == label]
        x = d["net"].to_numpy()
        t = x.mean() / (x.std(ddof=1) / math.sqrt(len(x))) if x.std() > 0 else float("nan")
        eq = np.cumsum(x)
        dd = (eq - np.maximum.accumulate(eq)).min()
        yrs = d.groupby("year")["net"].sum()
        print(f"{label:<20}{x.sum():>10,.0f}{t:>6.2f}{100 * (x > 0).mean():>6.0f}{dd:>9,.0f}   "
              + "  ".join(f"{y}:{v / 1000:>5.0f}k" for y, v in yrs.items()))


if __name__ == "__main__":
    main()
