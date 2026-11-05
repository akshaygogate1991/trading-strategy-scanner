"""Kronos forecast in the live market, traded at once: which time of day (if any) works?

The after-close daily forecast can't see overnight news, so here Kronos forecasts during the
session and the trade is taken right away:

  forecast times   09:45, 10:10, 11:00, 12:30, 13:30, 14:30
  candles          15-minute, 160 candles of history (the Kronos authors' setting)
  paths            10, temperature 0.6, top_p 0.9 (authors' direction setting)
  trade            index futures (index price as the futures price), entered at the forecast
                   time, exited after 30 or 60 minutes (14:30 -> 15:15 for 60). BUY if >= 60 %
                   of paths end up, SELL if <= 40 %, else no trade. 0.015 % per side.
  indices          NIFTY, BANKNIFTY, SENSEX, Jul 2024 - Jul 2026 (Kronos training data ends Jun 2024)
  checks           year 1 (Jul 2024 - Jun 2025) picks, year 2 (Jul 2025 on) judges;
                   against "follow the last hour's move" and against random BUY / SELL signs.

Step 1 is slow on a laptop CPU (all 6 times of a day are forecast in one batch); it saves every
forecast to logs/kronos_intraday_forecasts.csv and continues from there if stopped.

Run:  python kronos_intraday_test.py forecast [SENSEX NIFTY BANKNIFTY]   (step 1, resumable)
      python kronos_intraday_test.py report                             (step 2) -> logs/kronos_intraday.txt
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "Kronos"))

START, SPLIT = pd.Timestamp("2024-07-01"), pd.Timestamp("2025-07-01")
OUT = os.path.join(HERE, "logs", "kronos_intraday_forecasts.csv")
TIMES = ("09:45", "10:10", "11:00", "12:30", "13:30", "14:30")
LOOKBACK, STEPS, PATHS, T_DIR, TOP_P = 160, 4, 10, 0.6, 0.9
LAST_EXIT = "15:15"
COST = 0.00015                                           # per side, as the Kronos paper tab
INDICES = ("SENSEX", "NIFTY", "BANKNIFTY")
RNG = np.random.default_rng(5)


def minute_bars(sym: str) -> pd.DataFrame:
    s = pd.read_parquet(os.path.join(HERE, "hf_data", "index", f"{sym}.parquet"))
    s = s.set_index(s["timestamp"].dt.tz_localize(None))[["open", "high", "low", "close"]].sort_index()
    t = s.index.hour * 60 + s.index.minute
    return s[(t >= 555) & (t < 930)]


def bars15(m1: pd.DataFrame, shift: int = 0) -> pd.DataFrame:
    """15-minute candles; shift = minutes past the quarter hour at which candles end (10:10 -> 10)."""
    return m1.resample("15min", origin="start_day", offset=f"{shift}min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()


def at(day, hhmm) -> pd.Timestamp:
    return day + pd.Timedelta(hours=int(hhmm[:2]), minutes=int(hhmm[3:]))


def price_at(m1: pd.DataFrame, ts: pd.Timestamp) -> float | None:
    """Last 1-minute close before ts (the price when the clock reads ts)."""
    s = m1["close"][(m1.index < ts) & (m1.index >= ts.normalize())]
    return float(s.iloc[-1]) if len(s) else None


def predictor():
    import torch
    from model import Kronos, KronosPredictor, KronosTokenizer
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    mdl = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    tok.eval(), mdl.eval()
    print("device:", dev, flush=True)
    return KronosPredictor(mdl, tok, device=dev, max_context=512)


def forecast(sym: str, pred) -> None:
    m1 = minute_bars(sym)
    shifted = {sh: bars15(m1, sh) for sh in {int(t[3:]) % 15 for t in TIMES}}   # a candle ends at each forecast time
    b = shifted[0]
    done = set()
    if os.path.exists(OUT):
        old = pd.read_csv(OUT)
        done = set(old.loc[old["index"] == sym, "day"])
    days = [d for d in sorted(set(b.index.normalize())) if d >= START]
    t0, n = time.time(), 0
    for k, day in enumerate(days):
        ds = day.strftime("%Y-%m-%d")
        if ds in done:
            continue
        xs, xts, yts, meta = [], [], [], []
        for hhmm in TIMES:
            cut = at(day, hhmm)
            bb = shifted[int(hhmm[3:]) % 15]
            hist = bb[bb.index + pd.Timedelta("15min") <= cut].iloc[-LOOKBACK:]
            fut = pd.date_range(cut, periods=STEPS, freq="15min")
            if len(hist) < LOOKBACK or hist.index[-1].normalize() != day:
                continue
            for _ in range(PATHS):
                xs.append(hist.reset_index(drop=True))
                xts.append(pd.Series(hist.index))
                yts.append(pd.Series(fut))
            meta.append((hhmm, cut, float(hist["close"].iloc[-1])))
        if not meta:
            continue
        out = pred.predict_batch(xs, xts, yts, pred_len=STEPS, T=T_DIR, top_p=TOP_P, sample_count=1, verbose=False)
        rows = []
        for j, (hhmm, cut, last) in enumerate(meta):
            paths = out[j * PATHS:(j + 1) * PATHS]
            end60 = min(STEPS, int((at(day, LAST_EXIT) - cut) / pd.Timedelta("15min")))
            c30 = np.array([p["close"].iloc[1] for p in paths])
            c60 = np.array([p["close"].iloc[end60 - 1] for p in paths])
            entry = price_at(m1, cut)
            x30 = price_at(m1, cut + pd.Timedelta("30min"))
            x60 = price_at(m1, min(cut + pd.Timedelta("60min"), at(day, LAST_EXIT)))
            back = price_at(m1, max(cut - pd.Timedelta("60min"), at(day, "09:16")))   # before 10:16: move since the open
            if None in (entry, x30, x60):
                continue
            rows.append({"index": sym, "day": ds, "time": hhmm, "last_close": last, "entry": entry,
                         "p_up30": float((c30 > last).mean()), "p_up60": float((c60 > last).mean()),
                         "pred30": float(c30.mean() / last - 1) * 100, "pred60": float(c60.mean() / last - 1) * 100,
                         "real30": (x30 / entry - 1) * 100, "real60": (x60 / entry - 1) * 100,
                         "last60": (entry / back - 1) * 100 if back else np.nan})
        pd.DataFrame(rows).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
        n += 1
        print(f"{sym} {ds}  {k + 1}/{len(days)}  " + "  ".join(f"{r['time']} {r['p_up60']:.1f}" for r in rows)
              + f"  {(time.time() - t0) / n:.0f}s/day", flush=True)


# ---------------------------------------------------------------- step 2

def tstat(x) -> float:
    x = np.asarray(x, float)
    return x.mean() / (x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else float("nan")


def trades(g: pd.DataFrame, h: int, cut: float) -> pd.DataFrame:
    p = g[f"p_up{h}"]
    side = np.where(p >= cut, 1, np.where(p <= 1 - cut, -1, 0))
    t = g.assign(side=side)
    t = t[t.side != 0]
    return t.assign(net=t.side * t[f"real{h}"] - 2 * COST * 100)        # in %


def line(t: pd.DataFrame, lot: int, label: str) -> dict:
    if not len(t):
        return {"rule": label, "trades": 0}
    first = pd.to_datetime(t.day) < SPLIT
    rs = t.net / 100 * t.entry * lot
    move = (t.net + 2 * COST * 100).to_numpy() * t.side.to_numpy()   # the index move itself, in %
    rand = np.array([(RNG.choice([-1, 1], len(t)) * move).sum() for _ in range(1000)]) - 2 * COST * 100 * len(t)
    return {"rule": label, "trades": len(t), "right %": round((t.net + 2 * COST * 100 > 0).mean() * 100, 1),
            "avg net %": round(t.net.mean(), 4), "t": round(tstat(t.net), 2),
            "yr1 Rs/lot": round(rs[first].sum()), "yr2 Rs/lot": round(rs[~first].sum()),
            "random as good %": round((rand >= t.net.sum()).mean() * 100)}


def report() -> None:
    lots = json.load(open(os.path.join(HERE, "hf_data", "lots.json")))
    f = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(HERE, "logs", "kronos_intraday_forecasts*.csv"))])
    f = f.drop_duplicates(["index", "day", "time"])
    lines = [f"Kronos intraday, 15-minute candles, {len(f)} forecasts. BUY if >= cut of 10 paths up, SELL if <= 1 - cut. "
             f"Net of {COST * 100:.3f} % per side. yr1 = to Jun 2025, yr2 = Jul 2025 on (1 lot, index price)."]
    best = []
    for sym, g in f.groupby("index"):
        lot = lots.get(sym, 1)
        lines.append(f"\n=== {sym}: {g.day.nunique()} days, {g.day.min()} to {g.day.max()}, lot {lot}")
        rows = []
        for hhmm, gt in g.groupby("time"):
            for h in (30, 60):
                for cut in (0.6, 0.7, 0.8):
                    rows.append({"time": hhmm, "exit": f"{h} min", "cut": cut, **line(trades(gt, h, cut), lot, "kronos")})
                mom = gt.dropna(subset=["last60"]).assign(side=lambda x: np.sign(x.last60))
                mom = mom[mom.side != 0].assign(net=lambda x: x.side * x[f"real{h}"] - 2 * COST * 100)
                rows.append({"time": hhmm, "exit": f"{h} min", "cut": "last hour", **line(mom, lot, "follow last hour")})
        t = pd.DataFrame(rows).drop(columns="rule")
        lines.append(t.to_string(index=False))
        k = t[t.cut != "last hour"]
        b = k.loc[k["yr1 Rs/lot"].idxmax()]
        best.append(f"  {sym}: best on yr1 = {b.time} exit {b.exit} cut {b.cut}: yr1 Rs {b['yr1 Rs/lot']:+,.0f}, "
                    f"yr2 Rs {b['yr2 Rs/lot']:+,.0f} ({b.trades} trades, right {b['right %']} %)")
        dir_hit = [((gt.p_up60 > .5) == (gt.real60 > 0))[gt.p_up60 != .5].mean() * 100 for _, gt in g.groupby("time")]
        lines.append("  direction right (majority of paths, 60 min) by time: "
                     + ", ".join(f"{tm} {h:.1f}%" for tm, h in zip(sorted(g.time.unique()), dir_hit))
                     + f"  (50 +/- {100 / math.sqrt(g.day.nunique()):.1f} is luck)")
    lines += ["\n=== Picked on year 1, judged on year 2 ==="] + best
    lines += ["", "right % = trades that moved the predicted way before costs. random as good = share of random "
              "BUY/SELL signs on the same trades with at least the same total; under 5 % = better than luck. "
              "Many rows are tried, so a few will look good by luck: only a rule that also holds in yr2 counts."]
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(HERE, "logs", "kronos_intraday.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "report"
    if mode == "forecast":
        pred = predictor()
        for s in (sys.argv[2:] or INDICES):
            forecast(s.upper(), pred)
    else:
        report()
