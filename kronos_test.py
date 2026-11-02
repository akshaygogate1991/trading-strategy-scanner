"""Kronos (AAAI 2026 candlestick foundation model) - can it predict Nifty/BankNifty/Sensex?

THE CLAIM (Instagram reel)
  Feed it recent candles, it gives the probability of price going up next;
  plug that into a strategy.

FAIR TEST
  * Model: NeoQuasar/Kronos-small (24.7M) + Kronos-Tokenizer-base, zero-shot,
    code from github.com/shiyu-coder/Kronos (cloned to ../Kronos).
  * Only data AFTER its release (paper Aug 2025), so it cannot have seen the
    test period:  daily from 2025-09-01, intraday from 2025-09-01.
  * Each forecast sees only the 400 candles before it.
  * N sampled future paths per forecast:
        P(up)        = share of paths whose next close is above the last close
        pred_return  = mean of the paths' next close / last close - 1
  * Tokenising can bias every forecast the same way, so the trading signal
    is pred_return minus the median of the PREVIOUS 30 forecasts (no future data).
  * Measures: direction hit rate (vs always-up and vs 50 %), rank correlation
    of pred_return with the real move (IC), and a long/short rule after costs
    (0.015 % per side, index futures).

  daily     forecast after day t-1's close; trade day t open -> close
            (also reports t-1 close -> t close, the move the model predicts)
  intraday  5-minute candles; forecasts at 10:00, 11:00, 12:00, 13:00, 14:00
            for the next 12 candles (one hour); trade entry at the forecast
            minute's next candle open, exit one hour later.

Run:  python kronos_test.py daily
      python kronos_test.py intraday
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "Kronos"))
from model import Kronos, KronosTokenizer, KronosPredictor   # noqa: E402

import torch   # noqa: E402

torch.manual_seed(0)
START = pd.Timestamp("2025-09-01")
LOOKBACK = 400
SAMPLES = 20
COST = 0.00015
INDICES = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN"}


def predictor():
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    mdl = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    tok.eval(), mdl.eval()
    return KronosPredictor(mdl, tok, device="cpu", max_context=512)


def forecast(pred, frames, ts_next, pred_len):
    """frames: list of (df[open,high,low,close], timestamps).  Returns sampled closes [n, SAMPLES, pred_len]."""
    dfs, xts, yts = [], [], []
    for (df, xt), yt in zip(frames, ts_next):
        for _ in range(SAMPLES):
            dfs.append(df)
            xts.append(xt)
            yts.append(yt)
    out = pred.predict_batch(dfs, xts, yts, pred_len=pred_len, T=1.0, top_p=0.9, sample_count=1, verbose=False)
    closes = np.array([o["close"].to_numpy() for o in out])
    return closes.reshape(len(frames), SAMPLES, pred_len)


def evaluate(df, label):
    """df columns: day, sym, p_up, pred_ret, real_ret (what the model predicts), trade_ret (tradable)."""
    lines = [f"\n=== {label}: {len(df)} forecasts, {df.day.nunique()} days"]
    df = df.sort_values(["sym", "ts"]).copy()
    df["bias"] = df.groupby("sym").pred_ret.transform(lambda s: s.shift(1).rolling(30, min_periods=10).median())
    df["sig"] = df.pred_ret - df.bias
    up = df.real_ret > 0
    lines.append(f"  real moves up: {up.mean() * 100:.1f}%   model says P(up)>50%: {(df.p_up > .5).mean() * 100:.1f}%")
    lines.append(f"  hit rate, raw P(up)>50% vs real direction: {((df.p_up > .5) == up).mean() * 100:.1f}%   "
                 f"(always guessing 'up' gets {max(up.mean(), 1 - up.mean()) * 100:.1f}%)")
    d = df.dropna(subset=["sig"])
    hit = (np.sign(d.sig) == np.sign(d.real_ret)).mean() * 100
    ic = d[["sig", "real_ret"]].corr(method="spearman").iloc[0, 1]
    n = len(d)
    lines.append(f"  hit rate, de-biased signal: {hit:.1f}%  (n={n}, 50% +/- {100 / np.sqrt(n):.1f} is luck)   "
                 f"rank IC {ic:+.3f} (t {ic * np.sqrt(n - 2) / np.sqrt(1 - ic ** 2):+.2f})")
    for name, cond in [("all forecasts", np.ones(n, bool)),
                       ("confident (P(up)>=0.65 or <=0.35)", ((d.p_up >= .65) | (d.p_up <= .35)).to_numpy()),
                       ("top/bottom third of signal", (d.sig.abs() >= d.sig.abs().quantile(2 / 3)).to_numpy())]:
        x = d[cond]
        if len(x) < 5:
            lines.append(f"  {name:<36} too few ({len(x)})")
            continue
        side = np.sign(x.sig) if "P(up)" not in name else np.where(x.p_up >= .5, 1, -1)
        pnl = side * x.trade_ret - 2 * COST
        t = pnl.mean() / (pnl.std(ddof=1) / np.sqrt(len(pnl)))
        lines.append(f"  trade {name:<36} n={len(x):>4}  hit {(side * x.trade_ret > 0).mean() * 100:4.1f}%  "
                     f"avg {pnl.mean() * 100:+.3f}%/trade  total {pnl.sum() * 100:+6.1f}%  t {t:+.2f}")
    return lines


def daily():
    import yfinance as yf
    pred = predictor()
    rows = []
    for sym, tk in INDICES.items():
        d = yf.download(tk, start="2023-01-01", progress=False, auto_adjust=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d = d[["Open", "High", "Low", "Close"]].dropna()
        d.columns = ["open", "high", "low", "close"]
        d = d[d.high >= d.low]
        idx = [i for i in range(LOOKBACK, len(d)) if d.index[i] >= START]
        t0 = time.time()
        for k in range(0, len(idx), 10):
            chunk = idx[k:k + 10]
            frames = [(d.iloc[i - LOOKBACK:i].reset_index(drop=True),
                       pd.Series(d.index[i - LOOKBACK:i])) for i in chunk]
            nxt = [pd.Series([d.index[i]]) for i in chunk]
            closes = forecast(pred, frames, nxt, 1)[:, :, 0]
            for j, i in enumerate(chunk):
                last = d.close.iloc[i - 1]
                rows.append(dict(sym=sym, ts=d.index[i], day=d.index[i].normalize(),
                                 p_up=(closes[j] > last).mean(), pred_ret=closes[j].mean() / last - 1,
                                 real_ret=d.close.iloc[i] / last - 1,
                                 trade_ret=d.close.iloc[i] / d.open.iloc[i] - 1))
            print(f"{sym} {k + len(chunk)}/{len(idx)}  {time.time() - t0:.0f}s", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(HERE, "logs", "kronos_daily.csv"), index=False)
    lines = evaluate(res, "DAILY (predicts close-to-close; trades open-to-close)")
    # also the close-to-close version (not exactly tradable: needs entry at the previous close)
    lines += evaluate(res.assign(trade_ret=res.real_ret), "DAILY, close-to-close trade (upper bound)")
    return lines


def intraday():
    import bb_alert_test as B
    pred = predictor()
    rows = []
    for sym in INDICES:
        m1 = B.clean(B.load_index(sym))
        m1 = m1[m1.index >= START - pd.Timedelta(days=30)]
        c = m1.resample("5min", origin="start_day", offset="15min", label="left", closed="left").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
        c.columns = ["open", "high", "low", "close"]
        hhmm = c.index.strftime("%H:%M")
        idx = [i for i in range(LOOKBACK, len(c) - 12)
               if c.index[i] >= START and hhmm[i] in ("10:00", "11:00", "12:00", "13:00", "14:00")]
        t0 = time.time()
        for k in range(0, len(idx), 8):
            chunk = idx[k:k + 8]
            frames = [(c.iloc[i - LOOKBACK:i].reset_index(drop=True), pd.Series(c.index[i - LOOKBACK:i])) for i in chunk]
            nxt = [pd.Series(c.index[i:i + 12]) for i in chunk]
            closes = forecast(pred, frames, nxt, 12)[:, :, -1]
            for j, i in enumerate(chunk):
                last = c.close.iloc[i - 1]
                rows.append(dict(sym=sym, ts=c.index[i], day=c.index[i].normalize(),
                                 p_up=(closes[j] > last).mean(), pred_ret=closes[j].mean() / last - 1,
                                 real_ret=c.close.iloc[i + 11] / last - 1,
                                 trade_ret=c.close.iloc[i + 11] / c.open.iloc[i] - 1))
            print(f"{sym} {k + len(chunk)}/{len(idx)}  {time.time() - t0:.0f}s", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(HERE, "logs", "kronos_intraday.csv"), index=False)
    return evaluate(res, "INTRADAY 5m, next 1 hour")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "daily"
    lines = daily() if mode == "daily" else intraday()
    print("\n".join(lines))
    with open(os.path.join(HERE, "logs", f"kronos_{mode}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
