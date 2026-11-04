"""Kronos as a helper for the two option strategies (Sensex Directional, Chota Packet Sensex).

Uses the Kronos authors' own settings (paper arXiv 2508.02739, Tables 6 and 8):
  5-minute candles : 480 candles of history      15-minute candles: 160 candles of history
  direction        : 10 sampled paths, temperature 0.6, top_p 0.9
  volatility       : 1 path, temperature 0.9, top_p 0.9; realized volatility = sum of squared
                     log returns of the predicted closes (the paper's definition)
The forecast runs to 15:00 (the strategies' exit), which is inside the authors' horizons
(96 x 5-minute, 32 x 15-minute candles). Kronos's training data ends in June 2024, so only
July 2024 on is used.

  forecasts at 09:35 (every day)   for Sensex Directional (enters 09:35): 5m and 15m
  forecasts at 09:20 (expiry days) for Chota Packet (most entries 09:21+): 5m only

What is tested (step 2, cheap, uses the saved forecasts):
  1. Does Kronos predict the rest-of-day volatility better than simple facts known at the
     same time (the morning's range, yesterday's range)?  Rank correlation with the real one.
  2. Direction: share of days the 10 paths' majority side was right (09:35 -> 15:00).
  3. Directional, cut one leg: when Kronos is confident (>= 70 % or <= 30 % of paths up),
     drop the sell leg on the side Kronos expects (keep only the other); also: skip the day
     when Kronos expects a big move. Against random cutting of the same number of legs.
  4. Chota (option buyer, expiry days): trade only on days Kronos expects a big move, or only
     the trades whose side (CE / PE) agrees with Kronos. Against random skipping.
  Thresholds are chosen on Jul 2024 - Jun 2025 and judged on Jul 2025 - Jul 2026.

Step 1 is slow on a laptop CPU (minutes per forecast). It saves every forecast to
logs/kronos_options_forecasts.csv and continues from there if stopped. A free GPU (Google
Colab) makes it about 20x faster: upload this file, ../Kronos, hf_data/index/SENSEX.parquet.

Run:  python kronos_options_test.py forecast [5m|15m|chota]   (step 1, resumable)
      python kronos_options_test.py report                      (step 2)
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "Kronos"))

START = pd.Timestamp("2024-07-01")
SPLIT = pd.Timestamp("2025-07-01")
OUT = os.path.join(HERE, "logs", "kronos_options_forecasts.csv")
SETUPS = {  # name: (bar minutes, lookback, forecast time, expiry days only)
    "5m": (5, 480, "09:35", False),
    "15m": (15, 160, "09:30", False),   # last full 15-minute candle before the 09:35 entry ends 09:30
    "chota": (5, 480, "09:20", True),
}
EXIT = "15:00"
N_DIR, T_DIR, T_VOL, TOP_P = 10, 0.6, 0.9, 0.9
RNG = np.random.default_rng(11)


def minute_bars() -> pd.DataFrame:
    s = pd.read_parquet(os.path.join(HERE, "hf_data", "index", "SENSEX.parquet"))
    s = s.set_index(s["timestamp"].dt.tz_localize(None))[["open", "high", "low", "close"]].sort_index()
    t = s.index.hour * 60 + s.index.minute
    return s[(t >= 555) & (t < 930)]


def bars(m1: pd.DataFrame, minutes: int) -> pd.DataFrame:
    return m1.resample(f"{minutes}min", origin="start_day", offset="15min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()


def expiry_days() -> set:
    t = pd.read_csv(os.path.join(HERE, "logs", "batch", "chotasx_expiry.csv"))
    days = set(t["day"])
    for f in ("export_sensexbasket_trades.csv", "logs/oos_export_sensexbasket_trades.csv"):
        x = pd.read_csv(os.path.join(HERE, f))
        days |= set(x.loc[x["is_expiry"], "day"])
    return days


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


def realized_vol(closes: np.ndarray) -> float:
    r = np.diff(np.log(closes))
    return float(np.sqrt((r ** 2).sum()) * 100)


def forecast(setup: str) -> None:
    minutes, lookback, at, expiry_only = SETUPS[setup]
    m1 = minute_bars()
    b = bars(m1, minutes)
    done = set()
    if os.path.exists(OUT):
        old = pd.read_csv(OUT)
        done = set(old.loc[old["setup"] == setup, "day"])
    keep = expiry_days() if expiry_only else None
    days = [d for d in sorted(set(b.index.normalize())) if d >= START]
    pred = predictor()
    t0 = time.time()
    for k, day in enumerate(days):
        ds = day.strftime("%Y-%m-%d")
        if ds in done or (keep is not None and ds not in keep):
            continue
        cut = day + pd.Timedelta(hours=int(at[:2]), minutes=int(at[3:]))
        end = day + pd.Timedelta(hours=int(EXIT[:2]), minutes=int(EXIT[3:]))
        hist = b[b.index < cut].iloc[-lookback:]          # candles that have closed by the forecast time
        fut = b[(b.index >= cut) & (b.index < end)]
        if len(hist) < lookback or len(fut) < 5:
            continue
        x = hist.reset_index(drop=True)
        xt = pd.Series(hist.index)
        yt = pd.Series(fut.index)
        n = len(fut)
        out = pred.predict_batch([x] * N_DIR, [xt] * N_DIR, [yt] * N_DIR, pred_len=n, T=T_DIR, top_p=TOP_P,
                                 sample_count=1, verbose=False)
        vol = pred.predict_batch([x], [xt], [yt], pred_len=n, T=T_VOL, top_p=TOP_P, sample_count=1,
                                 verbose=False)[0]
        last = hist["close"].iloc[-1]
        ends = np.array([o["close"].iloc[-1] for o in out])
        hi = np.array([o["high"].max() for o in out])
        lo = np.array([o["low"].min() for o in out])
        today = m1[(m1.index >= day) & (m1.index < cut)]
        prev = m1[(m1.index < day)].tail(375)
        real = m1[(m1.index >= cut) & (m1.index < end)]
        row = {"setup": setup, "day": ds,
               "p_up": float((ends > last).mean()), "pred_ret": float(ends.mean() / last - 1) * 100,
               "pred_range": float(np.median((hi - lo) / last)) * 100,
               "pred_vol": realized_vol(np.r_[last, vol["close"].to_numpy()]),
               "real_ret": float(real["close"].iloc[-1] / last - 1) * 100,
               "real_range": float((real["high"].max() - real["low"].min()) / last) * 100,
               "real_vol": realized_vol(np.r_[last, fut["close"].to_numpy()]),
               "morning_range": float((today["high"].max() - today["low"].min()) / last) * 100 if len(today) else np.nan,
               "prev_range": float((prev["high"].max() - prev["low"].min()) / last) * 100}
        pd.DataFrame([row]).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
        print(f"{setup} {ds}  {k + 1}/{len(days)}  p_up {row['p_up']:.1f}  pred_range {row['pred_range']:.2f}% "
              f"real {row['real_range']:.2f}%  {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- step 2

def tstat(x) -> float:
    x = np.asarray(x, float)
    return x.mean() / (x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else float("nan")


def leg_pnl() -> pd.DataFrame:
    """Sensex Directional net P&L per day and leg side (CE / PE)."""
    t = pd.concat([pd.read_csv(os.path.join(HERE, f)) for f in
                   ("logs/oos_export_sensexbasket_trades.csv", "export_sensexbasket_trades.csv")])
    t = t[t["strategy"] == "Sensex Directional"]
    return t.pivot_table(index="day", columns="kind", values="net", aggfunc="sum").fillna(0)


def skill(f: pd.DataFrame, label: str) -> list[str]:
    out = [f"\n=== {label}: {len(f)} days, {f.day.min()} to {f.day.max()}"]
    for col in ("pred_vol", "pred_range", "morning_range", "prev_range"):
        out.append(f"  {col:<14} vs real range: rank corr {f[col].corr(f.real_range, method='spearman'):+.3f}   "
                   f"vs real vol: {f[col].corr(f.real_vol, method='spearman'):+.3f}")
    up = f.real_ret > 0
    hit = ((f.p_up > .5) == up)[f.p_up != .5].mean() * 100
    conf = f[(f.p_up >= .7) | (f.p_up <= .3)]
    chit = ((conf.p_up > .5) == (conf.real_ret > 0)).mean() * 100 if len(conf) else float("nan")
    out.append(f"  direction 09:xx -> 15:00: majority of paths right {hit:.1f}% "
               f"(50 +/- {100 / math.sqrt(len(f)):.1f} is luck); confident days ({len(conf)}) right {chit:.1f}%; "
               f"real up days {up.mean() * 100:.1f}%")
    return out


def directional(f: pd.DataFrame, label: str) -> list[str]:
    legs = leg_pnl()
    d = f.set_index("day").join(legs, how="inner")
    base = d["CE"] + d["PE"]
    out = [f"\n=== Sensex Directional with Kronos ({label}): {len(d)} days, base net Rs {base.sum():,.0f} "
           f"(1st yr {base[d.index < str(SPLIT.date())].sum():,.0f}, 2nd yr {base[d.index >= str(SPLIT.date())].sum():,.0f})"]
    for cut in (0.7, 0.8, 0.9):
        up, dn = d.p_up >= cut, d.p_up <= 1 - cut
        # expecting up: the sold call is the risk -> keep only the put leg (and the other way)
        pnl = np.where(up, d["PE"], np.where(dn, d["CE"], base))
        changed = (up | dn).to_numpy()
        diff = pnl - base.to_numpy()
        rand = []
        for _ in range(2000):
            side = RNG.integers(0, 2, changed.sum())
            alt = np.where(side == 0, d["PE"].to_numpy()[changed], d["CE"].to_numpy()[changed])
            rand.append((alt - base.to_numpy()[changed]).sum())
        beat = (np.array(rand) >= diff.sum()).mean() * 100
        out.append(f"  cut one leg when >= {cut:.0%} of paths agree: {changed.sum():>3} days changed, "
                   f"change Rs {diff.sum():+,.0f} (1st yr {diff[d.index < str(SPLIT.date())].sum():+,.0f}, "
                   f"2nd yr {diff[d.index >= str(SPLIT.date())].sum():+,.0f}), t {tstat(diff[changed]):+.2f}, "
                   f"random leg cuts as good {beat:.0f}%")
    out += skip_test(d, base, "pred_range", "Directional, skip days Kronos expects the biggest range")
    out += skip_test(d, base, "morning_range", "Directional, same skip by the morning range (simple rule)")
    return out


def skip_test(d, pnl, col, title, keep_high=False) -> list[str]:
    """Pick the cut-off on the 1st year, judge on the 2nd. Skips the top (or bottom) share of `col`."""
    out = [f"  {title}:"]
    first = d.index < str(SPLIT.date())
    for share in (0.2, 0.33, 0.5):
        q = d.loc[first, col].quantile(share if keep_high else 1 - share)
        skip = (d[col] < q) if keep_high else (d[col] > q)
        res = []
        for part, name in ((first, "1st yr"), (~first, "2nd yr")):
            s, p = skip[part], pnl[part]
            rand = [p.sample(int(s.sum()), random_state=int(RNG.integers(1e9))).sum() for _ in range(1000)] \
                if s.sum() else [0]
            worse = (np.array(rand) >= p[s].sum()).mean() * 100   # random skips that removed more profit
            res.append(f"{name} skips {int(s.sum())} days worth Rs {p[s].sum():+,.0f} (random as good {100 - worse:.0f}%)")
        out.append(f"    {'keep only top' if keep_high else 'skip top'} {share:.0%}: " + "; ".join(res))
    return out


def chota(f: pd.DataFrame) -> list[str]:
    t = pd.read_csv(os.path.join(HERE, "logs", "batch", "chotasx_expiry.csv"))
    t = t[t["day"] >= str(START.date())]
    day_pnl = t.groupby("day")["net"].sum()
    d = f.set_index("day").join(day_pnl.rename("net"), how="inner")
    out = [f"\n=== Chota Packet Sensex (expiry days) with Kronos at 09:20: {len(d)} days, base net Rs {d.net.sum():,.0f}"]
    out += skip_test(d, d.net, "pred_range", "Chota, trade only days Kronos expects a big range", keep_high=True)
    out += skip_test(d, d.net, "morning_range", "Chota, same by the morning range (simple rule)", keep_high=True)
    tt = t.merge(f[["day", "p_up"]], on="day")
    for cut in (0.6, 0.7):
        agree = ((tt.kind == "CE") & (tt.p_up >= cut)) | ((tt.kind == "PE") & (tt.p_up <= 1 - cut))
        against = ((tt.kind == "CE") & (tt.p_up <= 1 - cut)) | ((tt.kind == "PE") & (tt.p_up >= cut))
        out.append(f"  trades agreeing with Kronos (>= {cut:.0%}): {agree.sum()} trades Rs {tt.net[agree].sum():+,.0f} "
                   f"(avg {tt.net[agree].mean():+,.0f}); against: {against.sum()} trades Rs {tt.net[against].sum():+,.0f} "
                   f"(avg {tt.net[against].mean():+,.0f}); all trades avg {tt.net.mean():+,.0f}; "
                   f"t of 'against' {tstat(tt.net[against]):+.2f}")
    return out


def report() -> None:
    # the laptop and a Colab run may each leave a file: logs/kronos_options_forecasts*.csv
    import glob
    files = sorted(glob.glob(os.path.join(HERE, "logs", "kronos_options_forecasts*.csv")))
    f = pd.concat([pd.read_csv(p) for p in files]).drop_duplicates(["setup", "day"], keep="last")
    lines = []
    for setup in ("5m", "15m"):
        g = f[f.setup == setup]
        if len(g):
            lines += skill(g, f"{setup} candles, forecast at {SETUPS[setup][2]} to 15:00")
            lines += directional(g, setup)
    g = f[f.setup == "chota"]
    if len(g):
        lines += skill(g, "5m candles, forecast at 09:20 (expiry days)")
        lines += chota(g)
    lines += ["", "1st yr = Jul 2024 - Jun 2025 (cut-offs chosen here), 2nd yr = Jul 2025 - Jul 2026.",
              "'random as good' = share of random choices (same number of days / legs) at least as good; "
              "below 5 % = better than luck."]
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(HERE, "logs", "kronos_options.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "report"
    if mode == "forecast":
        for s in (sys.argv[2:] or ["5m", "15m", "chota"]):
            forecast(s)
    else:
        report()
