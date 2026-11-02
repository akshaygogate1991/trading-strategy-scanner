"""Bollinger Band 1.4 "alert candle" reversal (YouTube, Tarun) - tested properly.

THE RULES AS GIVEN
  * Bollinger Bands: length 20, SMA, source close, std-dev 1.4
  * 5-minute candles (15-minute also allowed)
  * ALERT candle: a candle entirely outside the band - neither wick nor body
    touches it.  Above the upper band -> short setup, below the lower -> long.
  * ENTRY: only the very next candle.  Short when it breaks the alert candle's
    low, long when it breaks the alert candle's high.
  * STOP-LOSS: the other end of the alert candle (its high for a short).
  * TARGET: minimum 1:3 reward:risk, then trail.
  * Daily Fibonacci pivots: "additional confirmation" only (vague) - tested
    here as an optional filter (long only above the pivot P, short only below).

HOW IT IS SIMULATED
  * Candles are built from 1-minute data; fills and exits are found on the
    1-minute bars, so there is no guessing about which of stop/target came
    first inside a candle.  Stop orders fill at the trigger, or at the
    minute's open if price gapped through.
  * Intraday only: no entries after 15:00, anything open is closed at 15:20.
  * Bands run continuously across days (as on TradingView).
  * PROXY (not given in the video): the trail after 1:3.  Tested as
    "after +3R, stop follows the previous candle's low/high".
  * Costs per side, as a fraction of price: index futures 0.015 %, stocks
    0.03 % (STT, charges and slippage).
  * Control: same mechanics on random candles (random side), many runs.
    If the alert candle carries no information, the real rule will look
    like the random runs.

Run:  python bb_alert_test.py            (indices + stocks, 5m and 15m)
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BB_LEN, BB_STD = 20, 1.4
RR = 3.0
LAST_ENTRY, FLAT = "15:00", "15:20"
COST = {"index": 0.00015, "stock": 0.0003}
RANDOM_RUNS = 40
RNG = np.random.default_rng(7)


# ---------------------------------------------------------------- data
def load_index(sym):
    d = pd.read_parquet(os.path.join(HERE, "hf_data", "index", f"{sym}.parquet"))
    d["ts"] = pd.to_datetime(d["timestamp"]).dt.tz_localize(None)
    d = d.rename(columns=str.capitalize)[["Ts", "Open", "High", "Low", "Close"]]
    return d.rename(columns={"Ts": "ts"})


def load_stock(path):
    d = pd.read_csv(path, parse_dates=["ts"])
    return d[["ts", "Open", "High", "Low", "Close"]]


def clean(m):
    m = m.drop_duplicates("ts").sort_values("ts")
    t = m["ts"].dt.strftime("%H:%M")
    return m[(t >= "09:15") & (t <= "15:29")].set_index("ts")


def candles(m, minutes):
    """Resample 1-min to N-min candles anchored at 09:15 each day."""
    out = m.resample(f"{minutes}min", origin="start_day", offset="15min",
                     label="left", closed="left").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    mid = out["Close"].rolling(BB_LEN).mean()
    sd = out["Close"].rolling(BB_LEN).std(ddof=0)      # TradingView uses population sd
    out["up"], out["lo"] = mid + BB_STD * sd, mid - BB_STD * sd
    day = out.index.normalize()
    daily = m.groupby(m.index.normalize()).agg(H=("High", "max"), L=("Low", "min"), C=("Close", "last"))
    piv = ((daily.H + daily.L + daily.C) / 3).shift(1)  # previous day's pivot P
    out["pivot"] = piv.reindex(day).to_numpy()
    out["first_of_day"] = ~pd.Series(day, index=out.index).duplicated().to_numpy()
    return out


# ---------------------------------------------------------------- trading
class Minutes:
    """1-minute bars as numpy arrays, plus which N-min candle each minute is in."""
    def __init__(self, m1, c):
        self.t = m1.index.to_numpy()
        self.o, self.h, self.l, self.cl = (m1[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
        self.cid = np.searchsorted(c.index.to_numpy(), self.t, "right") - 1
        day = m1.index.normalize()
        flat = day + pd.Timedelta(hours=int(FLAT[:2]), minutes=int(FLAT[3:]))
        self.last = np.searchsorted(self.t, flat.to_numpy(), "right") - 1   # last minute to hold
        self.c_start = np.searchsorted(self.t, c.index.to_numpy(), "left")
        self.c_day = c.index.normalize().to_numpy()
        self.c_hhmm = c.index.strftime("%H:%M").to_numpy()
        self.ch, self.cl_ = c["High"].to_numpy(), c["Low"].to_numpy()


def trade(M, i, side, trail):
    """Alert candle at row i, entry window = candle i+1.  Returns a dict or None."""
    j = i + 1
    if j >= len(M.c_day) or M.c_day[j] != M.c_day[i] or M.c_hhmm[j] > LAST_ENTRY:
        return None
    trig, stop = (M.ch[i], M.cl_[i]) if side == 1 else (M.cl_[i], M.ch[i])
    if trig == stop:
        return None
    k, end = M.c_start[j], (M.c_start[j + 1] if j + 1 < len(M.c_start) else len(M.t))
    entry = None
    while k < end:
        if side == 1 and M.h[k] >= trig:
            entry = max(trig, M.o[k]); break
        if side == -1 and M.l[k] <= trig:
            entry = min(trig, M.o[k]); break
        k += 1
    if entry is None:
        return None
    risk = abs(entry - stop)
    target = entry + side * RR * risk
    last = M.last[k]
    sl, reached, exit_px = stop, False, None
    # entry minute: the order of moves inside it is unknown -> only a stop at its close counts
    if side * (M.cl[k] - sl) <= 0:
        exit_px = sl
    cur = M.cid[k]
    q = k + 1
    while exit_px is None and q <= last:
        if trail and reached and M.cid[q] != cur:      # new candle: trail to previous candle's extreme
            prev = M.cid[q] - 1
            sl = max(sl, M.cl_[prev]) if side == 1 else min(sl, M.ch[prev])
        cur = M.cid[q]
        if side == 1:
            if M.l[q] <= sl:
                exit_px = min(sl, M.o[q]); break
            if not reached and M.h[q] >= target:
                if not trail:
                    exit_px = max(target, M.o[q]) if M.o[q] > target else target; break
                reached, sl = True, max(sl, entry + (RR - 1) * risk)
        else:
            if M.h[q] >= sl:
                exit_px = max(sl, M.o[q]); break
            if not reached and M.l[q] <= target:
                if not trail:
                    exit_px = target; break
                reached, sl = True, min(sl, entry - (RR - 1) * risk)
        q += 1
    if exit_px is None:
        exit_px = M.cl[min(last, len(M.cl) - 1)]
    pts = side * (exit_px - entry)
    return dict(ts=M.t[k], side=side, R=pts / risk, pct=pts / entry, risk_pct=risk / entry)


def alerts(c, skip_open, pivot):
    longs = c["High"] < c["lo"]
    shorts = c["Low"] > c["up"]
    if skip_open:
        longs &= ~c["first_of_day"]
        shorts &= ~c["first_of_day"]
    if pivot:
        longs &= c["Close"] > c["pivot"]
        shorts &= c["Close"] < c["pivot"]
    sig = pd.Series(0, index=c.index)
    sig[longs], sig[shorts] = 1, -1
    return sig


def run(M, sig, trail, cost):
    rows = []
    s = sig.to_numpy()
    for i in np.flatnonzero(s != 0):
        t = trade(M, i, int(s[i]), trail)
        if t:
            t["net"] = t["pct"] - 2 * cost
            rows.append(t)
    return pd.DataFrame(rows)


def random_runs(M, c, n_alerts, trail, cost, runs):
    ok = np.flatnonzero(c["up"].notna().to_numpy())
    means = []
    for _ in range(runs):
        idx = RNG.choice(ok, size=n_alerts, replace=False)
        sig = pd.Series(0, index=c.index)
        sig.iloc[idx] = RNG.choice([-1, 1], size=n_alerts)
        r = run(M, sig, trail, cost)
        means.append(r["net"].mean() if len(r) else np.nan)
    return np.array(means)


def summary(name, r, rand=None):
    if r.empty:
        return f"{name:<44} no trades"
    net = r["net"] * 100
    t = net.mean() / (net.std(ddof=1) / np.sqrt(len(net))) if len(net) > 1 else np.nan
    half = len(r) // 2
    h1, h2 = net.iloc[:half].mean(), net.iloc[half:].mean()
    s = (f"{name:<44} n={len(r):>4}  win%={(r.R > 0).mean()*100:5.1f}  hit3R%={(r.R >= RR - 0.01).mean()*100:5.1f}  "
         f"avgR={r.R.mean():+.2f}  net/trade={net.mean():+.3f}%  t={t:+.2f}  halves {h1:+.3f}/{h2:+.3f}  "
         f"sum={net.sum():+.1f}%  risk={r.risk_pct.mean()*100:.2f}%")
    if rand is not None and len(rand):
        pr = (np.nanmean(rand) * 100)
        beat = (rand * 100 >= net.mean()).mean() * 100
        s += f"\n{'':<44} random same-mechanics avg net/trade={pr:+.3f}%  share of random runs >= real: {beat:.0f}%"
    return s


def main():
    lines = []
    out = lambda s="": (print(s), lines.append(s))
    markets = [("NIFTY", "index", load_index("NIFTY")),
               ("BANKNIFTY", "index", load_index("BANKNIFTY")),
               ("SENSEX", "index", load_index("SENSEX"))]
    for p in sorted(glob.glob(os.path.join(HERE, "nifty_fut_data", "EQ_*_NS_180d.csv"))):
        markets.append((os.path.basename(p).split("_")[1], "stock", load_stock(p)))

    pooled = {}
    for name, kind, raw in markets:
        m1 = clean(raw)
        out(f"\n=== {name} ({kind})  {m1.index[0].date()} .. {m1.index[-1].date()}")
        for minutes in (5, 15):
            c = candles(m1, minutes)
            M = Minutes(m1, c)
            for label, skip, piv, trail in [("rules as given, fixed 1:3", False, False, False),
                                            ("rules, 1:3 then trail (PROXY)", False, False, True),
                                            ("skip 1st candle of day, 1:3", True, False, False),
                                            ("+ pivot filter, 1:3", False, True, False)]:
                sig = alerts(c, skip, piv)
                r = run(M, sig, trail, COST[kind])
                rand = None
                if label.startswith("rules as given") and len(r) >= 10:
                    rand = random_runs(M, c, int((sig != 0).sum()), trail, COST[kind], RANDOM_RUNS)
                out(summary(f"{minutes}m  {label}", r, rand))
                pooled.setdefault((kind, minutes, label), []).append(r)

    out("\n=== POOLED")
    for (kind, minutes, label), rs in pooled.items():
        r = pd.concat([x for x in rs if not x.empty], ignore_index=True) if any(not x.empty for x in rs) else pd.DataFrame()
        out(summary(f"{kind:<5} {minutes}m  {label}", r))
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    with open(os.path.join(HERE, "logs", "bb_alert.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
