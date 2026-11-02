"""Inna Rosputnia's 18-day moving-average system (daily chart) - tested.

THE RULES AS PUBLISHED (via her site / MoneyShow, as summarised by PickMyTrade)
  * 18-day simple moving average of closes, daily candles
  * Long setup: two consecutive days whose LOWS are both above the MA,
    neither of them an inside day (high <= previous high and low >= previous low)
  * Entry: buy-stop at the higher high of those two days
  * Short setup: mirror (two highs below the MA, sell-stop at the lower low)
  * Exit: the opposite signal (stop-and-reverse); no fixed stop or target
  * No RSI or other filter.

WHAT IS UNCLEAR, SO TESTED BOTH WAYS
  * How long a setup stays valid: 1 day or 5 days.
  * Long+short (her version) or long only.
  * Her site says the MA length "is not important": 10/18/30/50 also shown
    as a robustness check, never to pick the best one.

MARKETS
  * Nifty, Bank Nifty, Sensex daily (Yahoo, 2007 onwards), as index futures:
    cost 0.015 % per side, overnight shorts allowed.
  * 195 Nifty-200 stocks (swing_data/prices.pkl, 2019-2026), long only
    (cash shares cannot be shorted overnight), cost 0.15 % per side.
    Note: today's Nifty 200 list, so the stock part flatters any long rule.
COMPARISON  buy-and-hold over the same days; and the return per day
            actually invested, vs the market's average day.

Run:  python rosputnia18_test.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_COST, STOCK_COST = 0.00015, 0.0015


def system(d, ma_len=18, valid=1, shorts=True):
    """Daily positions (+1/-1/0) held from each day's open-or-fill, and trade list."""
    o, h, l, c = (d[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    ma = pd.Series(c).rolling(ma_len).mean().to_numpy()
    n = len(c)
    inside = np.r_[False, (h[1:] <= h[:-1]) & (l[1:] >= l[:-1])]
    ret = np.zeros(n)                       # strategy return earned on day i
    pos = 0
    entry_px, trades, turns = None, [], 0
    long_trig = short_trig = None
    long_left = short_left = 0
    for i in range(1, n):
        day_ret = 0.0
        prev_c = c[i - 1]
        # fills today from triggers set at yesterday's close
        hit_l = long_trig is not None and long_left > 0 and h[i] >= long_trig and pos != 1
        hit_s = short_trig is not None and short_left > 0 and l[i] <= short_trig and pos != -1
        if hit_l and hit_s:                  # both inside one day: take the one nearer the open
            if abs(o[i] - long_trig) <= abs(o[i] - short_trig):
                hit_s = False
            else:
                hit_l = False
        new = 1 if hit_l else (-1 if shorts else 0) if hit_s else None
        if hit_s and not shorts and pos == 0:
            new = None                       # long-only and flat: a short signal does nothing
        if new is not None and new != pos:
            px = max(long_trig, o[i]) if hit_l else min(short_trig, o[i])
            # held position earns up to the fill, new one from the fill to the close
            day_ret += pos * (px / prev_c - 1) + new * (c[i] / px - 1)
            if pos != 0:
                trades.append(pos * (px / entry_px - 1))
            turns += abs(new - pos)
            pos, entry_px = new, px
            if hit_l:
                long_trig = None
            else:
                short_trig = None
        else:
            day_ret += pos * (c[i] / prev_c - 1)
        ret[i] = day_ret
        long_left -= 1
        short_left -= 1
        # new setups at today's close
        if np.isfinite(ma[i]) and np.isfinite(ma[i - 1]) and not inside[i] and not inside[i - 1]:
            if l[i] > ma[i] and l[i - 1] > ma[i - 1]:
                long_trig, long_left = max(h[i], h[i - 1]), valid
            if h[i] < ma[i] and h[i - 1] < ma[i - 1]:
                short_trig, short_left = min(l[i], l[i - 1]), valid
    return pd.Series(ret, index=d.index), np.array(trades), turns


def stats(ret, cost_turns, cost, bh):
    net = ret.copy()
    years = len(ret) / 250
    cagr = (1 + net).prod() ** (1 / years) - 1 - cost_turns * cost / years
    eq = (1 + net).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    bh_cagr = (1 + bh).prod() ** (1 / years) - 1
    ex = net - bh
    t_ex = ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex)))
    t_abs = net.mean() / (net.std(ddof=1) / np.sqrt(len(net)))
    return cagr * 100, dd * 100, bh_cagr * 100, t_abs, t_ex


def index_data():
    import yfinance as yf
    out = {}
    for name, tk in (("NIFTY", "^NSEI"), ("BANKNIFTY", "^NSEBANK"), ("SENSEX", "^BSESN")):
        d = yf.download(tk, start="2007-01-01", progress=False, auto_adjust=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d = d[["Open", "High", "Low", "Close"]].dropna()
        d = d[(d.High >= d.Low) & (d.Open > 0)]
        out[name] = d
    return out


def main():
    lines = []
    out = lambda s="": (print(s, flush=True), lines.append(s))
    idx = index_data()
    out("INDICES (futures, cost 0.015%/side).  CAGR / max DD / buy&hold CAGR / t vs 0 / t vs buy&hold")
    for name, d in idx.items():
        out(f"\n=== {name} {d.index[0].date()} .. {d.index[-1].date()}")
        bh = d["Close"].pct_change().fillna(0)
        for shorts in (True, False):
            for valid in (1, 5):
                for ma_len in (18, 10, 30, 50):
                    r, tr, turns = system(d, ma_len, valid, shorts)
                    cagr, dd, bhc, ta, te = stats(r, turns, INDEX_COST, bh)
                    tag = f"MA{ma_len:<2} valid {valid}d {'long+short' if shorts else 'long only '}"
                    out(f"  {tag}  CAGR {cagr:+6.1f}%  maxDD {dd:6.1f}%  B&H {bhc:+5.1f}%  "
                        f"trades {len(tr):>3}  win {np.mean(tr > 0) * 100 if len(tr) else 0:4.0f}%  t {ta:+.2f}  t_vs_B&H {te:+.2f}")
                # halves for the published version
                r, tr, turns = system(d, 18, valid, shorts)
                half = len(r) // 2
                for lab, sl in (("1st half", slice(0, half)), ("2nd half", slice(half, None))):
                    rr = r.iloc[sl]
                    out(f"      MA18 {lab} {rr.index[0].year}-{rr.index[-1].year}: avg/yr {rr.mean() * 250 * 100:+.1f}%  "
                        f"B&H avg/yr {bh.iloc[sl].mean() * 250 * 100:+.1f}%")

    out("\n\nSTOCKS: 195 Nifty-200 names, long only, cost 0.15%/side (2019-2026)")
    px = pd.read_pickle(os.path.join(HERE, "swing_data", "prices.pkl"))
    for valid in (1, 5):
        rows = []
        for tk, d in px.items():
            if tk.startswith("^") or len(d) < 300:
                continue
            d = d[["Open", "High", "Low", "Close"]].dropna()
            d = d[(d.High >= d.Low) & (d.Open > 0)]
            r, tr, turns = system(d, 18, valid, False)
            bh = d["Close"].pct_change().fillna(0)
            years = len(r) / 250
            strat = ((1 + r).prod()) ** (1 / years) - 1 - turns * STOCK_COST / years
            bhc = ((1 + bh).prod()) ** (1 / years) - 1
            invested = (r != 0).mean()
            rows.append(dict(tk=tk, strat=strat * 100, bh=bhc * 100, invested=invested * 100,
                             trades=len(tr), win=np.mean(tr > 0) * 100 if len(tr) else np.nan))
        s = pd.DataFrame(rows)
        diff = s.strat - s.bh
        t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
        out(f"  valid {valid}d: stocks {len(s)}  median CAGR system {s.strat.median():+.1f}% vs buy&hold {s.bh.median():+.1f}%  "
            f"system beats B&H in {(diff > 0).mean() * 100:.0f}% of stocks  avg diff {diff.mean():+.1f}%/yr (t {t:+.2f})  "
            f"time invested {s.invested.median():.0f}%  trades/stock {s.trades.median():.0f}  win {s.win.median():.0f}%")
    with open(os.path.join(HERE, "logs", "rosputnia18.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
