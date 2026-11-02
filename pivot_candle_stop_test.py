"""Reel #3 variant of the 9 EMA + Fibonacci pivot breakout: stop = signal candle's low/high."""
import numpy as np, pandas as pd
import bb_alert_test as B
import pivot_ema_intraday_test as P

RNG = np.random.default_rng(9)
lines = []
out = lambda s="": (print(s, flush=True), lines.append(s))
pooled = {}
for sym in ("NIFTY", "BANKNIFTY", "SENSEX"):
    m1 = B.clean(B.load_index(sym))
    out(f"\n=== {sym}  {m1.index[0].date()} .. {m1.index[-1].date()}")
    for minutes in (5, 15):
        c, lv = P.setup(m1, minutes)
        M = B.Minutes(m1, c.assign(up=np.nan, lo=np.nan))
        for big in (True, False):
            side, _ = P.signals(c, lv, big)
            stop = np.where(side == 1, c["Low"].to_numpy(), np.where(side == -1, c["High"].to_numpy(), np.nan))
            for rr in (2.0, 3.0):
                r = P.run(M, c, side, stop, rr)
                rand = None
                if big and minutes == 5 and len(r):
                    n = int((side != 0).sum()); tots = []
                    for _ in range(50):
                        rs = np.zeros(len(c), int); pos = RNG.choice(len(c) - 1, size=n, replace=False)
                        rs[pos] = RNG.choice([-1, 1], size=n)
                        rp = np.zeros(len(c)); rp[pos] = RNG.choice(r.risk_pct.to_numpy(), size=n)
                        x = P.run(M, c, rs, None, rr, rp); tots.append(x.net.sum() * 100 if len(x) else 0)
                    rand = np.array(tots)
                name = f"{minutes}m {'strong candle' if big else 'any candle  '} 1:{rr:.0f}" + ("  <- as given" if minutes == 5 and big else "")
                out(P.line(name, r, rand))
                pooled.setdefault(name, []).append(r)
out("\n=== POOLED")
for k, rs in pooled.items():
    out(P.line(k, pd.concat(rs, ignore_index=True)))
open("logs/pivot_candle_stop.txt", "w", encoding="utf-8").write("\n".join(lines))
