"""EMA / RSI / Elliott rule search with a locked test period.

WHY THIS DESIGN
  Trying hundreds of settings always finds some that look great by luck.
  So the data is split BEFORE searching:
    search period   start of data .. 2025-06-30   (all choices made here)
    hidden period   2025-07-01 .. end of data     (opened once, for the top 3)
  The rule list below is fixed in advance and is never edited after a run.

MARKETS   Nifty, Bank Nifty, Sensex index, from 1-minute data (hf_data/index),
          traded as if with index futures.  One rule = all three indices.
CANDLES   5m, 15m, 60m (intraday: entries until 15:00, flat at the close)
          and daily (swing: positions carried overnight).
SIGNALS   decided at a candle's close, entered at the next candle's open.
  ema_x      fast EMA crosses slow EMA            (9/21, 9/50, 21/50, 50/200)
  pull       trend EMA + RSI pullback             (EMA 50/200 x RSI2 10/90, 5/95, RSI14 30/70, 40/60)
  rsi_rev    RSI crosses back out of an extreme   (RSI14 30/70, RSI2 10/90)
  app18      18 EMA rising + RSI14 crosses 50 (or 60/40)  - app.py style
  elliott    app.py's full daily signal (18 EMA + Elliott score>=55 + RSI 50)
FILTERS   none | daily 18 EMA trend agrees | app.py daily signal agrees
          (both from YESTERDAY's daily close, so nothing from the future)
EXITS     time: hold 3/6/12 candles (daily 5/10/20 days)
          atr:  stop 1 ATR & target 2 ATR, or stop 2 ATR & target 4 ATR
                (stop assumed first if both inside one candle)
SIDES     long only | long and short
COSTS     0.015 % of price per side (futures charges + slippage)

SCORING   P&L per trading day (all three indices summed), t-stat over days.
          A rule must also be positive on each index and in both halves.
          With ~2,000 rules tried, luck alone produces t of about 3.5-4, so
          that is the bar, not 2.

Run:  python ema_rsi_search.py            (search, then hidden test of top 3)
"""
from __future__ import annotations

import itertools
import os
import pickle
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT = pd.Timestamp("2025-07-01")
COST = 0.00015
INDICES = ["NIFTY", "BANKNIFTY", "SENSEX"]
TFS = ["5m", "15m", "60m", "D"]
LAST_ENTRY = "15:00"
APP_CACHE = os.path.join(HERE, "swing_data", "app_daily_signal.pkl")
RNG = np.random.default_rng(11)


# ---------------------------------------------------------------- data
def minute_data(sym):
    d = pd.read_parquet(os.path.join(HERE, "hf_data", "index", f"{sym}.parquet"))
    d["ts"] = pd.to_datetime(d["timestamp"]).dt.tz_localize(None)
    d = d.drop_duplicates("ts").set_index("ts").sort_index()
    t = d.index.strftime("%H:%M")
    d = d[(t >= "09:15") & (t <= "15:29")]
    return d.rename(columns=str.capitalize)[["Open", "High", "Low", "Close"]]


def resample(m, tf):
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if tf == "D":
        return m.groupby(m.index.normalize()).agg(agg)
    n = int(tf[:-1])
    return m.resample(f"{n}min", origin="start_day", offset="15min",
                      label="left", closed="left").agg(agg).dropna()


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(c, n=14):
    pc = c["Close"].shift()
    tr = pd.concat([c["High"] - c["Low"], (c["High"] - pc).abs(), (c["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def app_daily(sym, daily):
    """app.py full signal per day (+1 CALL, -1 PUT, 0 none) and 18 EMA trend,
    computed from data up to the PREVIOUS close.  Cached - it is slow."""
    cache = pickle.load(open(APP_CACHE, "rb")) if os.path.exists(APP_CACHE) else {}
    key = (sym, str(daily.index[-1].date()), len(daily))
    if key in cache:
        return cache[key]
    import ema_merge_test
    app = ema_merge_test.app_functions()
    full, trend = {}, {}
    for i, day in enumerate(daily.index):
        if i < 60:
            continue
        hist = app["add_indicators"](daily.iloc[max(0, i - 250):i])
        tr = app["trend_18ema"](hist)
        wave = app["elliott_directional"](hist)
        r = float(hist.iloc[-1]["rsi14"])
        strong = int(wave["score"]) >= 55
        full[day] = (1 if tr == "up" and wave["bias"] == "Bullish" and strong and r >= 50 else
                     -1 if tr == "down" and wave["bias"] == "Bearish" and strong and r <= 50 else 0)
        trend[day] = 1 if tr == "up" else -1 if tr == "down" else 0
    out = pd.DataFrame({"app": pd.Series(full), "t18": pd.Series(trend)}).reindex(daily.index).fillna(0)
    cache[key] = out
    os.makedirs(os.path.dirname(APP_CACHE), exist_ok=True)
    pickle.dump(cache, open(APP_CACHE, "wb"))
    return out


# ---------------------------------------------------------------- rules
def cross_up(a, b):
    return (a > b) & (a.shift() <= b.shift())


def base_signals(c):
    """Every base rule -> Series of +1 / -1 / 0 at candle close."""
    cl = c["Close"]
    e = {n: ema(cl, n) for n in (9, 18, 21, 50, 200)}
    r2, r14 = rsi(cl, 2), rsi(cl, 14)
    out = {}
    for f, s in [(9, 21), (9, 50), (21, 50), (50, 200)]:
        out[f"ema_x {f}/{s}"] = cross_up(e[f], e[s]).astype(int) - cross_up(e[s], e[f]).astype(int)
    for t in (50, 200):
        for name, r, lo, hi in [("rsi2 10/90", r2, 10, 90), ("rsi2 5/95", r2, 5, 95),
                                ("rsi14 30/70", r14, 30, 70), ("rsi14 40/60", r14, 40, 60)]:
            lg = (cl > e[t]) & (r < lo) & (r.shift() >= lo)
            sh = (cl < e[t]) & (r > hi) & (r.shift() <= hi)
            out[f"pull ema{t} {name}"] = lg.astype(int) - sh.astype(int)
    for name, r, lo, hi in [("rsi14 30/70", r14, 30, 70), ("rsi2 10/90", r2, 10, 90)]:
        out[f"rsi_rev {name}"] = cross_up(r, pd.Series(lo, index=r.index)).astype(int) - \
            cross_up(pd.Series(hi, index=r.index), r).astype(int)
    rising, falling = e[18] > e[18].shift(), e[18] < e[18].shift()
    for lo, hi in [(50, 50), (60, 40)]:
        lg = rising & (cl > e[18]) & cross_up(r14, pd.Series(lo, index=r14.index))
        sh = falling & (cl < e[18]) & cross_up(pd.Series(hi, index=r14.index), r14)
        out[f"app18 rsi{lo}/{hi}"] = lg.astype(int) - sh.astype(int)
    return {k: v.fillna(0).astype(int).to_numpy() for k, v in out.items()}


# ---------------------------------------------------------------- simulation
class Market:
    def __init__(self, sym, m1, tf, daily_sig):
        c = resample(m1, tf)
        self.tf, self.idx = tf, c.index
        self.o, self.h, self.l, self.c = (c[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
        self.atr = atr(c).to_numpy()
        day = c.index.normalize()
        self.day = day.to_numpy()
        self.hhmm = c.index.strftime("%H:%M").to_numpy()
        # last candle of each day (intraday exits)
        last = pd.Series(np.arange(len(c)), index=c.index).groupby(day).transform("max").to_numpy()
        self.day_last = last
        # yesterday's daily signals mapped onto each candle (daily tf: the candle's own day,
        # because app_daily already uses data only up to the previous close)
        ds = daily_sig.reindex(day).fillna(0)
        self.f_t18, self.f_app = ds["t18"].to_numpy(), ds["app"].to_numpy()
        self.sigs = base_signals(c)
        if tf == "D":
            self.sigs["elliott app"] = (ds["app"].to_numpy() *
                                        (ds["app"].to_numpy() != np.r_[0, ds["app"].to_numpy()[:-1]])).astype(int)
        self.entry_day = None


def simulate(M, sig, exit_kind, hold, k_stop, k_tgt, long_only):
    """Non-overlapping trades.  Returns arrays (entry_day, net_return)."""
    n = len(M.o)
    days, rets = [], []
    i, busy_until = 0, -1
    for s_i in np.flatnonzero(sig):
        side = sig[s_i]
        if long_only and side < 0:
            continue
        e_i = s_i + 1
        if e_i >= n or e_i <= busy_until:
            continue
        intraday = M.tf != "D"
        if intraday and (M.day[e_i] != M.day[s_i] or M.hhmm[e_i] > LAST_ENTRY):
            continue
        entry = M.o[e_i]
        cap = M.day_last[e_i] if intraday else n - 1
        if exit_kind == "time":
            x_i = min(e_i + hold - 1, cap)
            px = M.c[x_i]
        else:
            a = M.atr[s_i]
            if not np.isfinite(a) or a <= 0:
                continue
            stop, tgt = entry - side * k_stop * a, entry + side * k_tgt * a
            last = min(cap, e_i + (hold - 1 if not intraday else 10 ** 9))
            px, x_i = None, last
            for j in range(e_i, last + 1):
                if side == 1:
                    if M.l[j] <= stop:
                        px = min(stop, M.o[j]) if j > e_i else stop
                    elif M.h[j] >= tgt:
                        px = max(tgt, M.o[j]) if j > e_i else tgt
                else:
                    if M.h[j] >= stop:
                        px = max(stop, M.o[j]) if j > e_i else stop
                    elif M.l[j] <= tgt:
                        px = min(tgt, M.o[j]) if j > e_i else tgt
                if px is not None:
                    x_i = j
                    break
            if px is None:
                px = M.c[last]
        rets.append(side * (px - entry) / entry - 2 * COST)
        days.append(M.day[e_i])
        busy_until = x_i
    return np.array(days, dtype="datetime64[ns]"), np.array(rets)


def apply_filter(M, base, filt):
    if filt == "none":
        return base
    f = M.f_t18 if filt == "t18" else M.f_app
    return np.where(np.sign(base) == np.sign(f), base, 0)


EXITS = {"time": [("time", 3, 0, 0), ("time", 6, 0, 0), ("time", 12, 0, 0)],
         "D": [("time", 5, 0, 0), ("time", 10, 0, 0), ("time", 20, 0, 0)]}
ATR_EXITS = [("atr", 20, 1, 2), ("atr", 20, 2, 4)]     # hold cap only used on daily


def rules(tf, sig_names):
    exits = (EXITS["D"] if tf == "D" else EXITS["time"]) + ATR_EXITS
    for name, filt, ex, lo in itertools.product(sig_names, ["none", "t18", "app"], exits, [True, False]):
        if name == "elliott app" and filt != "none":
            continue
        yield (tf, name, filt, ex, lo)


def score(trades, mask_fn):
    """trades: {sym: (days, rets)} -> per-day P&L stats inside the period."""
    per_sym, alld = {}, []
    for sym, (d, r) in trades.items():
        keep = mask_fn(d)
        per_sym[sym] = r[keep].sum() * 100
        alld.append(pd.Series(r[keep] * 100, index=pd.to_datetime(d[keep])))
    s = pd.concat(alld) if alld else pd.Series(dtype=float)
    if len(s) < 2:
        return None
    daily = s.groupby(level=0).sum()
    t = daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily))) if daily.std(ddof=1) > 0 else 0
    half = len(daily) // 2
    return dict(trades=len(s), days=len(daily), avg_trade=s.mean(), total=s.sum(), t=t,
                h1=daily.iloc[:half].sum(), h2=daily.iloc[half:].sum(),
                win=(s > 0).mean() * 100, **{f"sum_{k}": v for k, v in per_sym.items()})


def label(rule):
    tf, name, filt, ex, lo = rule
    kind, hold, ks, kt = ex
    exs = f"hold {hold}" if kind == "time" else f"ATR stop {ks} tgt {kt}"
    return f"{tf:>3} | {name:<22} | filter {filt:<4} | {exs:<16} | {'long only' if lo else 'long+short'}"


def main():
    t0 = time.time()
    lines = []
    out = lambda s="": (print(s, flush=True), lines.append(s))
    mins = {s: minute_data(s) for s in INDICES}
    dsig = {}
    for s in INDICES:
        daily = resample(mins[s], "D")
        dsig[s] = app_daily(s, daily)
    out(f"data ready {time.time() - t0:.0f}s")

    search = lambda d: d < np.datetime64(SPLIT)
    hidden = lambda d: d >= np.datetime64(SPLIT)
    results, trades_by_rule = [], {}
    for tf in TFS:
        mk = {s: Market(s, mins[s], tf, dsig[s]) for s in INDICES}
        names = list(mk[INDICES[0]].sigs)
        for rule in rules(tf, names):
            _, name, filt, ex, lo = rule
            tr = {}
            for s, M in mk.items():
                sig = apply_filter(M, M.sigs[name], filt)
                tr[s] = simulate(M, sig, ex[0], ex[1], ex[2], ex[3], lo)
            sc = score(tr, search)
            if sc:
                results.append(dict(rule=rule, **sc))
                trades_by_rule[rule] = tr
        out(f"{tf} done, {len(results)} rules so far, {time.time() - t0:.0f}s")

    df = pd.DataFrame(results)
    n_rules = len(df)
    good = df[(df.trades >= 100) & (df.h1 > 0) & (df.h2 > 0) &
              (df[[f"sum_{s}" for s in INDICES]] > 0).all(axis=1)]
    out(f"\nRules tried: {n_rules}.  Positive on all 3 indices and both halves, >=100 trades: {len(good)}")
    out(f"Share of all rules with t > 2: {(df.t > 2).mean() * 100:.1f}%   (pure luck would give ~2.5%)")
    out("Luck bar for this many tries: t about 3.5-4.\n")

    fmt = lambda r: (f"{label(r['rule'])}\n      trades {r['trades']:>5}  days {r['days']:>4}  win {r['win']:4.1f}%  "
                     f"avg/trade {r['avg_trade']:+.3f}%  total {r['total']:+7.1f}%  t {r['t']:+.2f}  "
                     f"halves {r['h1']:+.1f}/{r['h2']:+.1f}  N/BN/S {r['sum_NIFTY']:+.1f}/{r['sum_BANKNIFTY']:+.1f}/{r['sum_SENSEX']:+.1f}")
    out("TOP 15 IN THE SEARCH PERIOD (by t, after the consistency checks)")
    top = good.sort_values("t", ascending=False)
    for _, r in top.head(15).iterrows():
        out(fmt(r))
    out("\nBEST BY FAMILY (search period)")
    df["family"] = df.rule.map(lambda r: r[1].split()[0])
    for fam, g in df.groupby("family"):
        b = g.sort_values("t", ascending=False).iloc[0]
        out(f"  {fam:<8} best t {b.t:+.2f}   median t {g.t.median():+.2f}   rules {len(g)}")

    # pick top 3 that are not near-duplicates (different signal name)
    picks, seen = [], set()
    for _, r in top.iterrows():
        key = (r.rule[0], r.rule[1])
        if key in seen:
            continue
        seen.add(key)
        picks.append(r)
        if len(picks) == 3:
            break
    out(f"\n=== HIDDEN PERIOD {SPLIT.date()} .. end  (opened once, top 3 only)")
    for r in picks:
        h = score(trades_by_rule[r.rule], hidden)
        out(label(r.rule))
        out(f"   search: t {r.t:+.2f}, avg/trade {r.avg_trade:+.3f}%, total {r.total:+.1f}%")
        if h is None:
            out("   hidden: no trades")
            continue
        out(f"   hidden: trades {h['trades']}, win {h['win']:.1f}%, avg/trade {h['avg_trade']:+.3f}%, "
            f"total {h['total']:+.1f}%, t {h['t']:+.2f}, N/BN/S {h['sum_NIFTY']:+.1f}/{h['sum_BANKNIFTY']:+.1f}/{h['sum_SENSEX']:+.1f}")
        # random control in the hidden period: same timeframe, exit and side rule, random entry candles
        tf, name, filt, ex, lo = r.rule
        mk = {s: Market(s, mins[s], tf, dsig[s]) for s in INDICES}
        rand_tot = []
        for _ in range(100):
            tr = {}
            for s, M in mk.items():
                real = apply_filter(M, M.sigs[name], filt)
                k = int((real != 0).sum())
                sig = np.zeros(len(M.o), dtype=int)
                pos = RNG.choice(len(M.o) - 1, size=k, replace=False)
                sig[pos] = 1 if lo else RNG.choice([-1, 1], size=k)
                tr[s] = simulate(M, sig, ex[0], ex[1], ex[2], ex[3], lo)
            hs = score(tr, hidden)
            rand_tot.append(hs["total"] if hs else 0)
        rand_tot = np.array(rand_tot)
        out(f"   random entries, same exit (hidden): avg total {rand_tot.mean():+.1f}%  "
            f"share of random runs >= real: {(rand_tot >= h['total']).mean() * 100:.0f}%")

    with open(os.path.join(HERE, "logs", "ema_rsi_search.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    df.drop(columns=["family"]).assign(rule=df.rule.map(label)).to_csv(
        os.path.join(HERE, "logs", "ema_rsi_search_all.csv"), index=False)
    out(f"\ndone {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
