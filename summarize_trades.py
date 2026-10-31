"""One-line-per-run summary of algtst_backtest.py trade CSVs, with per-year net.

  python summarize_trades.py logs/batch/*.csv
"""
import math
import sys

import numpy as np
import pandas as pd


def main(paths):
    print(f"{'run':<34}{'days':>5}{'net Rs':>10}{'t':>6}{'win%':>5}{'maxDD':>9}{'best':>8}  by year")
    for p in paths:
        tr = pd.read_csv(p)
        for name, g in tr.groupby("strategy"):
            d = g.groupby("day")["net"].sum()
            x = d.to_numpy()
            t = x.mean() / (x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 and x.std() > 0 else float("nan")
            eq = np.cumsum(x)
            dd = (eq - np.maximum.accumulate(eq)).min()
            yrs = d.groupby(d.index.str[:4]).sum()
            label = p.replace("\\", "/").split("/")[-1].replace(".csv", "")
            if tr["strategy"].nunique() > 1:
                label += ":" + name[:10]
            print(f"{label:<34}{len(x):>5}{x.sum():>10,.0f}{t:>6.2f}{100 * (x > 0).mean():>5.0f}"
                  f"{dd:>9,.0f}{x.max():>8,.0f}  " + " ".join(f"{y}:{v / 1000:.0f}k" for y, v in yrs.items()))


if __name__ == "__main__":
    main(sys.argv[1:])
