"""Can the dashboard's 18 EMA + Elliott Wave signal improve Sensex Directional?

The signal (exactly app.py's analyze(): 18 EMA position+slope, Elliott impulse
score >= 55, RSI 50) is computed on DAILY Sensex candles up to the PREVIOUS
close, so it is known before the 09:35 entry. Merge rules tested:

  original     sell both CE and PE every day (as the seminar file)
  ema_only     18 EMA trend alone: up -> sell PE only, down -> sell CE only, flat -> both
  full_signal  app.py signal: CALL -> sell PE only, PUT -> sell CE only, none -> both
  ema_skip     18 EMA trend: up/down -> one side as above, flat -> both
               (same as ema_only; kept for symmetry) - see 'agree' table instead

Selling only the side AGAINST the trend means: in an uptrend, sell the put
(it decays if the market keeps rising) and do not sell the call (which would be
stopped out). Every variant is a real re-simulation with DaySim, including the
overall stop, on Aug 2023 - Jul 2026.

  python ema_merge_test.py
"""
import copy
import math
import os

import numpy as np
import pandas as pd

import algtst_backtest as ab

HERE = os.path.dirname(os.path.abspath(__file__))


def app_functions():
    """app.py's signal functions, without starting the Streamlit page."""
    src = open(os.path.join(HERE, "app.py"), encoding="utf-8").read()
    src = src[:src.index("st.set_page_config")]
    ns = {"__name__": "app_signals"}
    exec(compile(src, "app.py", "exec"), ns)
    return ns


def daily_sensex():
    s = pd.read_parquet(os.path.join(ab.DATA, "index", "SENSEX.parquet"))
    ts = s["timestamp"]
    m = ts.dt.hour * 60 + ts.dt.minute - ab.OPEN_MIN
    s = s[(m >= 0) & (m < ab.N_MIN)]
    g = s.groupby("trading_day")
    return pd.DataFrame({"Open": g["open"].first(), "High": g["high"].max(),
                         "Low": g["low"].min(), "Close": g["close"].last()})


def signals(daily, app):
    out = {}
    days = list(daily.index)
    for i, day in enumerate(days):
        if i < 60:
            continue
        hist = app["add_indicators"](daily.iloc[max(0, i - 250):i])      # up to yesterday
        trend = app["trend_18ema"](hist)
        wave = app["elliott_directional"](hist)
        rsi = float(hist.iloc[-1]["rsi14"])
        strong = int(wave["score"]) >= 55
        full = ("CALL" if trend == "up" and wave["bias"] == "Bullish" and strong and rsi >= 50 else
                "PUT" if trend == "down" and wave["bias"] == "Bearish" and strong and rsi <= 50 else None)
        out[day] = dict(trend=trend, full=full)
    return pd.DataFrame(out).T


def legs_for(kind_to_keep, st):
    s = copy.deepcopy(st)
    if kind_to_keep:
        s["legs"] = [c for c in s["legs"] if c["kind"] == kind_to_keep]
    return s


def main():
    app = app_functions()
    daily = daily_sensex()
    sig = signals(daily, app)
    base = next(s for s in ab.load_file(os.path.join(HERE, "strategies", "export_sensexbasket.algtst"))
                if "Directional" in s["name"])
    only_pe, only_ce = legs_for("PE", base), legs_for("CE", base)
    lot = ab.lot_size("SENSEX")

    rows = []
    for day, exp, d in ab.iter_days("SENSEX", "2023-08-01", "2026-07-31"):
        if day not in sig.index:
            continue
        t, f = sig.at[day, "trend"], sig.at[day, "full"]
        net = lambda st: sum(ab.add_costs(x, "SENSEX", 0.5)["net"] for x in ab.run_day(st, d, lot))
        both, pe, ce = net(base), net(only_pe), net(only_ce)
        rows.append(dict(day=day, trend=t, full=f, both=both, pe=pe, ce=ce,
                         move=(d.spot_open[345] / d.spot_open[20] - 1) * 100
                         if d.spot_open is not None else np.nan))
    r = pd.DataFrame(rows).set_index("day")
    r.to_csv(os.path.join(HERE, "logs", "ema_merge_daily.csv"))

    def pick(row, col):
        side = row[col]
        if side in ("up", "CALL"):
            return row["pe"]
        if side in ("down", "PUT"):
            return row["ce"]
        return row["both"]

    r["ema_only"] = r.apply(lambda x: pick(x, "trend"), axis=1)
    r["full_signal"] = r.apply(lambda x: pick(x, "full"), axis=1)

    print(f"{len(r)} days, {r.index.min()} .. {r.index.max()}\n")
    print("Does yesterday's daily signal predict today's 09:35 -> 15:00 Sensex move?")
    for col in ("trend", "full"):
        g = r.groupby(r[col].fillna("none"))["move"].agg(["count", "mean"])
        g["up_days%"] = r.groupby(r[col].fillna("none"))["move"].apply(lambda x: (x > 0).mean() * 100)
        print(f"  {col}:\n" + g.round(3).to_string() + "\n")

    print("Leg P&L by signal (Rs/day, selling only that leg):")
    for col in ("trend", "full"):
        print(f"  {col}:\n" + r.groupby(r[col].fillna("none"))[["pe", "ce", "both"]].mean().round(0).to_string() + "\n")

    def stats(x):
        eq = x.cumsum()
        t = x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))
        yrs = x.groupby(x.index.str[:4]).sum()
        return (f"net {x.sum():>9,.0f}  t {t:5.2f}  maxDD {(eq - eq.cummax()).min():>8,.0f}  "
                + " ".join(f"{y}:{v / 1e3:.0f}k" for y, v in yrs.items()))

    print("Merged strategies (Aug 2023 - Jul 2026, net after costs + 0.5% slippage):")
    for col in ("both", "ema_only", "full_signal"):
        print(f"  {col:<12} {stats(r[col])}")
    # control: on the same days the signal chose one side, choose a side at random
    rng = np.random.default_rng(17)
    for col, sigcol in (("ema_only", "trend"), ("full_signal", "full")):
        chosen = r[sigcol].isin(["up", "down", "CALL", "PUT"])
        sims = []
        for _ in range(2000):
            coin = rng.random(len(r)) < 0.5
            alt = np.where(chosen, np.where(coin, r["pe"], r["ce"]), r["both"])
            sims.append(alt.sum())
        pct = 100 * (np.array(sims) < r[col].sum()).mean()
        print(f"  {col}: one side on {chosen.sum()} days; beats random side choice in {pct:.0f}% of 2000 draws")


if __name__ == "__main__":
    main()
