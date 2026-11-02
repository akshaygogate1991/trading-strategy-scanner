"""Kronos paper trades (no orders - paper only).

Kronos (github.com/shiyu-coder/Kronos, AAAI 2026) is a model trained on
12 billion candles. Every weekday evening, after the close, this script:

  1. Settles yesterday's signals: a paper trade of 1 lot of index futures in
     Nifty, Bank Nifty and Sensex, bought or sold at today's open and closed
     at today's close (the index price stands in for the futures price).
  2. Asks Kronos for tomorrow: it samples SAMPLES possible next daily candles
     from the last LOOKBACK candles.
        P(up) = (paths closing higher + half of the unchanged ones) / paths
        BUY if P(up) >= LONG_AT, SELL if P(up) <= SHORT_AT, else no trade.
     The signal is stored and traded at the next session's open.

The dashboard also uses the Sensex signal to show "Sensex Directional with
Kronos picking the side" (computed in dashboard_api.py from trades.csv).

Files in paper_data/:
  kronos_signals.csv   one row per index per signal (made after a close)
  kronos_trades.csv    one row per settled paper trade
  kronos.json          snapshot for the dashboard (latest signals, settings)

Costs per side: 0.015 % of the contract value (charges + slippage) + Rs 5
brokerage + GST, the same as the backtest (kronos_test.py).

  python kronos_paper.py              # normal evening run (deploy/kronos-paper.timer)
  python kronos_paper.py --dry-run    # print tomorrow's signals, save nothing
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
KRONOS_DIR = os.environ.get("KRONOS_DIR", os.path.join(os.path.dirname(HERE), "Kronos"))
IST = timezone(timedelta(hours=5, minutes=30))
INDICES = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN"}
LOTS = {"NIFTY": 65, "BANKNIFTY": 30, "SENSEX": 20}
MODEL, TOKENIZER = "NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base"
LOOKBACK, SAMPLES = 400, 20
LONG_AT, SHORT_AT = 0.60, 0.40
COST_PCT, BROKERAGE = 0.00015, 5 * 1.18
SIG_COLS = ["made", "sym", "last_close", "n_up", "n_flat", "n_down", "p_up", "pred_ret", "side", "settled"]
TRADE_COLS = ["day", "sym", "side", "qty", "entry", "exit", "points", "gross", "charges", "net", "p_up", "made"]


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_rows(path, cols, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def daily_bars():
    import yfinance as yf
    out = {}
    for sym, tk in INDICES.items():
        d = yf.download(tk, period="3y", interval="1d", progress=False, auto_adjust=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d = d[["Open", "High", "Low", "Close"]].dropna()
        d = d[(d.High >= d.Low) & (d.Open > 0)]
        d.columns = ["open", "high", "low", "close"]
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        out[sym] = d
    return out


def load_predictor():
    sys.path.insert(0, KRONOS_DIR)
    import torch
    from model import Kronos, KronosTokenizer, KronosPredictor
    torch.manual_seed(0)
    tok = KronosTokenizer.from_pretrained(TOKENIZER)
    mdl = Kronos.from_pretrained(MODEL)
    tok.eval(), mdl.eval()
    return KronosPredictor(mdl, tok, device="cpu", max_context=512)


def next_session(day):
    nxt = day + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt


def signal(pred, d):
    """Kronos view of the session after d's last bar."""
    hist = d.iloc[-LOOKBACK:]
    x_ts = pd.Series(hist.index)
    y_ts = pd.Series([pd.Timestamp(next_session(hist.index[-1].date()))])
    frames = [hist.reset_index(drop=True)] * SAMPLES
    out = pred.predict_batch(frames, [x_ts] * SAMPLES, [y_ts] * SAMPLES, pred_len=1,
                             T=1.0, top_p=0.9, sample_count=1, verbose=False)
    closes = np.array([o["close"].iloc[0] for o in out])
    last = float(hist.close.iloc[-1])
    closes = closes.astype(float)
    n_up, n_down = int((closes > last).sum()), int((closes < last).sum())
    n_flat = SAMPLES - n_up - n_down
    p_up = (n_up + 0.5 * n_flat) / SAMPLES
    side = "BUY" if p_up >= LONG_AT else "SELL" if p_up <= SHORT_AT else "NONE"
    return dict(last_close=round(last, 2), n_up=n_up, n_flat=n_flat, n_down=n_down, p_up=round(p_up, 3),
                pred_ret=round(float(closes.mean() / last - 1) * 100, 3), side=side)


def settle(sig_rows, trade_rows, bars):
    """Turn signals into trades once the session after them has a daily bar."""
    done = 0
    for s in sig_rows:
        if s["settled"]:
            continue
        d = bars.get(s["sym"])
        if d is None:
            continue
        after = d[d.index > pd.Timestamp(s["made"])]
        if after.empty:
            continue
        day, bar = after.index[0], after.iloc[0]
        s["settled"] = day.strftime("%Y-%m-%d")
        done += 1
        if s["side"] == "NONE":
            continue
        sign = 1 if s["side"] == "BUY" else -1
        qty = LOTS[s["sym"]]
        pts = sign * (bar.close - bar.open)
        gross = pts * qty
        chg = (bar.open + bar.close) * qty * COST_PCT + 2 * BROKERAGE
        trade_rows.append(dict(day=s["settled"], sym=s["sym"], side=s["side"], qty=qty,
                               entry=round(bar.open, 2), exit=round(bar.close, 2), points=round(pts, 2),
                               gross=round(gross, 2), charges=round(chg, 2), net=round(gross - chg, 2),
                               p_up=s["p_up"], made=s["made"]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "paper_data"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.data, exist_ok=True)
    sig_path, tr_path = os.path.join(a.data, "kronos_signals.csv"), os.path.join(a.data, "kronos_trades.csv")
    sig_rows, trade_rows = read_rows(sig_path), read_rows(tr_path)
    bars = daily_bars()

    settled = settle(sig_rows, trade_rows, bars)
    today = datetime.now(IST).date()
    pred, new = None, []
    for sym, d in bars.items():
        made = d.index[-1].strftime("%Y-%m-%d")
        if len(d) < LOOKBACK or any(r["made"] == made and r["sym"] == sym for r in sig_rows):
            continue
        if d.index[-1].date() > today:
            continue
        pred = pred or load_predictor()
        s = signal(pred, d)
        new.append(dict(made=made, sym=sym, settled="", **s))
        print(f"{sym}: after {made} close {s['last_close']}  P(up) {s['p_up']:.2f} "
              f"({s['n_up']} up / {s['n_flat']} flat / {s['n_down']} down)  -> {s['side']}", flush=True)
    if a.dry_run:
        print("dry run: nothing saved")
        return
    sig_rows += new
    write_rows(sig_path, SIG_COLS, sig_rows)
    trade_rows.sort(key=lambda r: (r["day"], r["sym"]))
    write_rows(tr_path, TRADE_COLS, trade_rows)
    latest = {}
    for r in sig_rows:
        latest[r["sym"]] = r
    write_json(os.path.join(a.data, "kronos.json"), dict(
        updated=datetime.now(IST).strftime("%Y-%m-%d %H:%M"),
        start=sig_rows[0]["made"] if sig_rows else None,
        latest=[latest[s] for s in INDICES if s in latest],
        model=MODEL, samples=SAMPLES, lookback=LOOKBACK, lots=LOTS,
        rule=f"BUY 1 lot if Kronos P(up) >= {LONG_AT:.0%}, SELL if <= {SHORT_AT:.0%}, else no trade. "
             f"Enter at the open, exit at the close of the next session."))
    print(f"settled {settled} signal(s), {len(new)} new signal(s), {len(trade_rows)} trades in total")


if __name__ == "__main__":
    main()
