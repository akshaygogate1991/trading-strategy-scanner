"""EMA 9 + VWAP "wick" strategy (Instagram reel, "EMA VWAP cross") - tested.

THE RULES AS GIVEN
  * EMA 9 and session VWAP, 3-minute candles
  * Enter when a candle wicks through the VWAP (and the EMA 9):
      wicks through the BOTTOM of VWAP and EMA 9 -> buy puts  (short)
      wicks the TOP of VWAP and pushes through EMA 9 -> buy calls (long)
  * Exit when a candle breaks through the EMA 9 (against the trade)

WHAT IS AMBIGUOUS, SO BOTH READINGS ARE TESTED
  break    the candle crosses both lines and CLOSES beyond them
           (opens above VWAP and EMA, closes below both -> short)
  reject   the candle's wick pokes through VWAP but it closes back on the
           other side and beyond the EMA (high above VWAP, close below VWAP
           and EMA -> short: a rejection from above)
  "Breaks through EMA" for the exit = a candle CLOSES on the other side of
  the EMA.  Entry at the next candle's open; entries until 15:00, flat 15:20.

PERMUTATIONS (the reel's settings first, then neighbours)
  candles 3m (given), 1m, 5m;  EMA 9 (given), 20;  VWAP source hlc3.
  A setting only counts if it works on every market and in both halves.

MARKETS
  * Real VWAP (has volume): 8 large stocks Apr-Oct 2026 (nifty_fut_data/EQ_*),
    Bank Nifty Oct-26 futures Jun-Oct 2026 (bn_fut_data).
  * Indices 2021-2026 (hf_data/index) have NO volume, so VWAP there is a
    time-weighted stand-in (PROXY).  Shown separately.
  Results are on the underlying.  The reel buys options: that adds the
  option spread and time decay, so option buyers would do WORSE than this.
COSTS     0.015 % per side for index/futures, 0.03 % per side for stocks.
CONTROL   random entry candles (random side), same EMA exit, many runs.

Run:  python ema_vwap_cross_test.py
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

import bb_alert_test as B

HERE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(3)
RANDOM_RUNS = 50


def load_csv(path):
    d = pd.read_csv(path, parse_dates=["ts"])
    d = d.drop_duplicates("ts").set_index("ts").sort_index()
    t = d.index.strftime("%H:%M")
    return d[(t >= "09:15") & (t <= "15:29")][["Open", "High", "Low", "Close", "Volume"]]


def load_index(sym):
    m = B.clean(B.load_index(sym))
    m["Volume"] = 1.0                                   # PROXY: time-weighted "VWAP"
    return m


def bars(m, minutes, ema_len):
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    c = m.resample(f"{minutes}min", origin="start_day", offset="15min",
                   label="left", closed="left").agg(agg).dropna(subset=["Open"])
    c["Volume"] = c["Volume"].clip(lower=1e-9)
    tp = (c.High + c.Low + c.Close) / 3
    day = c.index.normalize()
    c["vwap"] = (tp * c.Volume).groupby(day).cumsum() / c.Volume.groupby(day).cumsum()
    c["ema"] = c.Close.ewm(span=ema_len, adjust=False).mean()
    c["day"] = day
    c["hhmm"] = c.index.strftime("%H:%M")
    return c


def signals(c, reading):
    o, h, l, cl, v, e = (c[k].to_numpy() for k in ("Open", "High", "Low", "Close", "vwap", "ema"))
    first = ~pd.Series(c.day).duplicated().to_numpy()
    if reading == "break":
        sh = (o > v) & (o > e) & (cl < v) & (cl < e)
        lg = (o < v) & (o < e) & (cl > v) & (cl > e)
    else:  # reject
        sh = (h > v) & (cl < v) & (cl < e)   # wick above VWAP, closes below both
        lg = (l < v) & (cl > v) & (cl > e)
    s = lg.astype(int) - sh.astype(int)
    s[first] = 0                                         # VWAP is one bar old at the open: skip
    return s


def simulate(c, sig, cost):
    o, cl, e = c.Open.to_numpy(), c.Close.to_numpy(), c.ema.to_numpy()
    day, hhmm = c.day.to_numpy(), c.hhmm.to_numpy()
    n, rows, i = len(c), [], 0
    idx = np.flatnonzero(sig)
    busy = -1
    for s_i in idx:
        e_i = s_i + 1
        if e_i >= n or e_i <= busy or day[e_i] != day[s_i] or hhmm[e_i] > B.LAST_ENTRY:
            continue
        side, entry = sig[s_i], o[e_i]
        x = e_i
        while True:
            if side * (cl[x] - e[x]) < 0:                # closed back through the EMA
                px = cl[x]
                break
            if x + 1 >= n or day[x + 1] != day[e_i] or hhmm[x] >= B.FLAT:
                px = cl[x]
                break
            x += 1
        busy = x
        g = side * (px - entry) / entry
        rows.append((day[e_i], g - 2 * cost, g, x - e_i + 1))
    return pd.DataFrame(rows, columns=["day", "net", "gross", "bars"])


def summarise(name, r, rand=None):
    if r.empty:
        return f"{name:<38} no trades"
    net = r.net * 100
    daily = net.groupby(r.day).sum()
    t = daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily))) if len(daily) > 1 else np.nan
    h = len(r) // 2
    s = (f"{name:<38} trades {len(r):>5} ({len(r) / r.day.nunique():.1f}/day)  win {(r.net > 0).mean() * 100:4.1f}%  "
         f"gross/trade {r.gross.mean() * 100:+.3f}%  net/trade {net.mean():+.3f}%  total {net.sum():+7.1f}%  t {t:+.2f}  "
         f"halves {net.iloc[:h].sum():+.1f}/{net.iloc[h:].sum():+.1f}  avg hold {r.bars.mean():.1f} bars")
    if rand is not None:
        s += f"\n{'':<38} random entries, same exit: avg total {rand.mean():+.1f}%  share of random runs >= real {(rand >= net.sum()).mean() * 100:.0f}%"
    return s


def main():
    lines = []
    out = lambda s="": (print(s, flush=True), lines.append(s))
    markets = []
    for p in sorted(glob.glob(os.path.join(HERE, "nifty_fut_data", "EQ_*_NS_180d.csv"))):
        markets.append((os.path.basename(p).split("_")[1], "stock", load_csv(p), 0.0003))
    markets.append(("BANKNIFTY FUT", "real VWAP fut", load_csv(os.path.join(HERE, "bn_fut_data", "BANKNIFTY27OCT26FUT.csv")), 0.00015))
    for s in ("NIFTY", "BANKNIFTY", "SENSEX"):
        markets.append((s, "index PROXY", load_index(s), 0.00015))

    pooled = {}
    for name, kind, m, cost in markets:
        out(f"\n=== {name} ({kind})  {m.index[0].date()} .. {m.index[-1].date()}")
        for minutes in (3, 1, 5):
            for ema_len in (9, 20):
                c = bars(m, minutes, ema_len)
                for reading in ("break", "reject"):
                    sig = signals(c, reading)
                    r = simulate(c, sig, cost)
                    rand = None
                    if minutes == 3 and ema_len == 9 and not r.empty:
                        n_sig = int((sig != 0).sum())
                        tots = []
                        for _ in range(RANDOM_RUNS):
                            rs = np.zeros(len(c), int)
                            pos = RNG.choice(len(c) - 1, size=min(n_sig, len(c) - 1), replace=False)
                            rs[pos] = RNG.choice([-1, 1], size=len(pos))
                            rr = simulate(c, rs, cost)
                            tots.append(rr.net.sum() * 100 if len(rr) else 0)
                        rand = np.array(tots)
                    label = f"{minutes}m EMA{ema_len} {reading}" + ("  <- as given" if minutes == 3 and ema_len == 9 else "")
                    out(summarise(label, r, rand))
                    pooled.setdefault((kind.split()[0], label), []).append(r)
    out("\n=== POOLED")
    for (kind, label), rs in pooled.items():
        rs = [x for x in rs if not x.empty]
        if rs:
            out(summarise(f"{kind:<6} {label}", pd.concat(rs, ignore_index=True)))
    with open(os.path.join(HERE, "logs", "ema_vwap_cross.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
