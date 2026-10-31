"""Does WAITING for the base strategy's (app.py) signal help Sensex Directional?

The base strategy decides direction from 18 EMA trend, Elliott wave (score >= 55)
and RSI 50, with India VIX < 20 as a comfort check. Here it is run on INTRADAY
Sensex candles (5-min and 15-min, the last 250 bars up to the decision time), so
the signal sees the morning's move. Decision times 10:00 / 10:30 / 11:00.

  wait_both    skip 09:35; enter at the decision time, sell both sides (cost of waiting)
  wait_side    enter at the decision time; up -> sell PE only, down -> CE only, else both
  drop_leg     enter at 09:35 as normal; at the decision time close the leg that is
               AGAINST the signal (up -> close the CE, down -> close the PE), keep the other

Signals use only bars completed before the decision minute. Every variant is a
real DaySim re-simulation (overall stop included), Aug 2023 - Jul 2026,
net of costs + 0.5% slippage. Each side-choosing variant is compared with
choosing the side at random on the same days.

  python intraday_signal_test.py
"""
import math
import os

import numpy as np
import pandas as pd

import algtst_backtest as ab
from ema_merge_test import app_functions, legs_for

HERE = os.path.dirname(os.path.abspath(__file__))
TIMES = {"10:00": 45, "10:30": 75, "11:00": 105}       # minutes after 09:15
FRAMES = (5, 15)


class DropSim(ab.DaySim):
    """DaySim that, at minute drop_at, closes the drop_kind leg and stops it re-entering."""

    def __init__(self, st, day, lot, drop_kind, drop_at):
        super().__init__(st, day, lot)
        self.drop_kind, self.drop_at = drop_kind, drop_at

    def step(self, k):
        if not self.done and k == self.drop_at:
            for g in self.legs:
                if g["cfg"]["kind"] != self.drop_kind:
                    continue
                if g["state"] == "open":
                    self._close(g, k, self.day.O[g["c"], k], "signal_drop")
                g["state"] = "done"
        return super().step(k)


def run(sim):
    for k in range(sim.entry, sim.exit + 1):
        if sim.step(k):
            break
    return sum(ab.add_costs(t, "SENSEX", 0.5)["net"] for t in sim.trades)


def intraday_bars(frame):
    s = pd.read_parquet(os.path.join(ab.DATA, "index", "SENSEX.parquet"))
    ts = s["timestamp"]
    s = s.assign(m=(ts.dt.hour * 60 + ts.dt.minute - ab.OPEN_MIN).to_numpy())
    s = s[(s["m"] >= 0) & (s["m"] < ab.N_MIN)].groupby(["trading_day", "m"]).first().reset_index()
    s["b"] = s["m"] // frame
    g = s.groupby(["trading_day", "b"])
    bars = pd.DataFrame({"Open": g["open"].first(), "High": g["high"].max(),
                         "Low": g["low"].min(), "Close": g["close"].last()}).reset_index()
    bars["end"] = (bars["b"] + 1) * frame          # minute index at which the bar is complete
    return bars


def signal_at(bars_until, app):
    hist = app["add_indicators"](bars_until[["Open", "High", "Low", "Close"]].reset_index(drop=True))
    trend = app["trend_18ema"](hist)
    wave = app["elliott_directional"](hist)
    rsi = hist.iloc[-1]["rsi14"]
    rsi = 50.0 if pd.isna(rsi) else float(rsi)
    strong = int(wave["score"]) >= 55
    full = ("up" if trend == "up" and wave["bias"] == "Bullish" and strong and rsi >= 50 else
            "down" if trend == "down" and wave["bias"] == "Bearish" and strong and rsi <= 50 else "flat")
    return trend, full


def stats(x):
    eq = x.cumsum()
    t = x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))
    yrs = x.groupby(x.index.str[:4]).sum()
    return (f"net {x.sum():>9,.0f}  t {t:5.2f}  maxDD {(eq - eq.cummax()).min():>8,.0f}  "
            + " ".join(f"{y}:{v / 1e3:.0f}k" for y, v in yrs.items()))


def main():
    app = app_functions()
    base = next(s for s in ab.load_file(os.path.join(HERE, "strategies", "export_sensexbasket.algtst"))
                if "Directional" in s["name"])
    only = {"PE": legs_for("PE", base), "CE": legs_for("CE", base)}
    lot = ab.lot_size("SENSEX")
    bars = {f: intraday_bars(f) for f in FRAMES}
    vix = pd.read_csv(os.path.join(HERE, "vix_data", "nifty_vix_8y.csv"), index_col="Date")["VIX"]
    vix.index = pd.to_datetime(vix.index).strftime("%Y-%m-%d")

    rows = []
    for day, exp, d in ab.iter_days("SENSEX", "2023-08-01", "2026-07-31"):
        r = dict(day=day, orig=run(ab.DaySim(base, d, lot)))
        pv = vix[vix.index < day]
        r["vix_prev"] = float(pv.iloc[-1]) if len(pv) else np.nan
        for name, T in TIMES.items():
            r[f"both@{name}"] = run(ab.DaySim(base, d, lot, entry=T))
            for kind in ("PE", "CE"):
                r[f"{kind}@{name}"] = run(ab.DaySim(only[kind], d, lot, entry=T))
                r[f"drop{kind}@{name}"] = run(DropSim(base, d, lot, kind, T))
            for f in FRAMES:
                b = bars[f]
                until = b[(b["trading_day"] < day) | ((b["trading_day"] == day) & (b["end"] <= T))].tail(250)
                r[f"trend{f}@{name}"], r[f"full{f}@{name}"] = signal_at(until, app)
            r[f"move@{name}"] = ((d.spot_open[345] / d.spot_open[T] - 1) * 100
                                 if d.spot_open is not None else np.nan)
        rows.append(r)
        if len(rows) % 100 == 0:
            print(f"  {len(rows)} days done ({day})", flush=True)
    r = pd.DataFrame(rows).set_index("day")
    r.to_csv(os.path.join(HERE, "logs", "intraday_signal_daily.csv"))

    rng = np.random.default_rng(17)
    print(f"\n{len(r)} days, {r.index.min()} .. {r.index.max()}")
    print(f"\nORIGINAL (enter 09:35, both sides): {stats(r['orig'])}")
    for name in TIMES:
        print(f"\n######## decision time {name}")
        print(f"  wait_both (just enter later)       {stats(r[f'both@{name}'])}")
        for f in FRAMES:
            for kind in ("trend", "full"):
                sig = r[f"{kind}{f}@{name}"]
                mv = r[f"move@{name}"]
                pred = "  ".join(f"{s}: {int((sig == s).sum())}d avg {mv[sig == s].mean():+.3f}% up {100 * (mv[sig == s] > 0).mean():.0f}%"
                                 for s in ("up", "down", "flat") if (sig == s).any())
                print(f"  -- {kind} signal on {f}-min bars: rest-of-day move {pred}")
                chosen = sig.isin(["up", "down"]).to_numpy()
                for var, up_col, dn_col, else_col in (
                        ("wait_side", f"PE@{name}", f"CE@{name}", f"both@{name}"),
                        ("drop_leg", f"dropCE@{name}", f"dropPE@{name}", "orig")):
                    x = pd.Series(np.where(sig == "up", r[up_col], np.where(sig == "down", r[dn_col], r[else_col])),
                                  index=r.index)
                    sims = [np.where(chosen, np.where(rng.random(len(r)) < 0.5, r[up_col], r[dn_col]), r[else_col]).sum()
                            for _ in range(2000)]
                    pct = 100 * (np.array(sims) < x.sum()).mean()
                    print(f"     {var:<10} {stats(x)}  beats random side {pct:3.0f}%")
    # VIX comfort check from the base strategy, on the original 09:35 trade
    hi = r["vix_prev"] >= 20
    print(f"\nVIX check (previous close): VIX>=20 on {int(hi.sum())} days, avg Rs {r['orig'][hi].mean():,.0f}/day; "
          f"VIX<20 avg Rs {r['orig'][~hi].mean():,.0f}/day. Skip VIX>=20 -> {stats(r['orig'][~hi])}")


if __name__ == "__main__":
    main()
