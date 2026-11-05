"""Kronos intraday paper trades: forecast in the live market, trade at once (no orders - paper only).

The same rule as kronos_intraday_test.py, run live on the VPS for Nifty, Bank Nifty and Sensex:

  at 09:45, 10:10, 11:00, 12:30, 13:30, 14:30 IST (timer fires one minute later)
    Kronos looks at the last 160 fifteen-minute candles (candles end exactly at that time, built
    from Yahoo 5-minute candles) and samples 10 paths of the next 4 candles.
    P(up) = share of paths ending above the last close.
    BUY 1 lot if P(up) >= 60 %, SELL if <= 40 %, else no trade - separately for two exits:
      30 min: P(up) of the 2nd candle, exit 30 minutes later
      60 min: P(up) of the 4th candle, exit 60 minutes later (14:30 -> 15:15)
    Entry = the index price at the forecast time (the close of the 5-minute candle ending then);
    real orders would fill a minute or two later.
  15:45 run: settles every trade whose exit time has passed, at the 5-minute close at the exit time.

The index price stands in for the futures price. Costs per side: 0.015 % of the contract value +
Rs 5 brokerage + GST (as kronos_paper.py). Prices from Yahoo (free, no key, no login).

Files in paper_data/: kronos_intraday_signals.csv (every forecast), kronos_intraday_trades.csv,
kronos_intraday.json (dashboard snapshot).

  python kronos_intraday_paper.py              # timer run: settle, then forecast the current slot
  python kronos_intraday_paper.py --dry-run    # forecast the latest slot now, save nothing
  python kronos_intraday_paper.py --now "2026-11-04 10:11" --data some_test_dir   # replay a run time
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
TIMES = ("09:45", "10:10", "11:00", "12:30", "13:30", "14:30")
LOOKBACK, STEPS, PATHS, T_DIR, TOP_P = 160, 4, 10, 0.6, 0.9
LONG_AT, SHORT_AT = 0.60, 0.40
LAST_EXIT = "15:15"
LATE = timedelta(minutes=10)               # a slot is forecast only within 10 minutes of its time
COST_PCT, BROKERAGE = 0.00015, 5 * 1.18
SIG_COLS = ["day", "time", "sym", "last", "p_up30", "p_up60", "pred30", "pred60", "made_at"]
TRADE_COLS = ["day", "time", "sym", "exit_after", "side", "p_up", "qty", "entry", "exit_time", "exit",
              "points", "gross", "charges", "net", "status"]


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


def at(day, hhmm) -> pd.Timestamp:
    return pd.Timestamp(day) + pd.Timedelta(hours=int(hhmm[:2]), minutes=int(hhmm[3:]))


def five_min_bars(now: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Completed 5-minute candles (IST, start-labelled) for the last month."""
    import yfinance as yf
    out = {}
    for sym, tk in INDICES.items():
        d = yf.download(tk, period="1mo", interval="5m", progress=False, auto_adjust=False)
        if d is None or not len(d):
            continue
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d = d[["Open", "High", "Low", "Close"]].dropna()
        d.columns = ["open", "high", "low", "close"]
        idx = pd.to_datetime(d.index)
        idx = idx.tz_convert("Asia/Kolkata") if idx.tz is not None else idx.tz_localize("UTC").tz_convert("Asia/Kolkata")
        d.index = idx.tz_localize(None)
        t = d.index.hour * 60 + d.index.minute
        d = d[(t >= 555) & (t < 930) & (d.index + pd.Timedelta("5min") <= now)]   # drop the candle still forming
        out[sym] = d
    return out


def bars15(b5: pd.DataFrame, shift: int) -> pd.DataFrame:
    """15-minute candles ending at :shift, :shift+15, ... (10:10 -> shift 10)."""
    return b5.resample("15min", origin="start_day", offset=f"{shift}min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()


def price_at(b5: pd.DataFrame, ts: pd.Timestamp) -> float | None:
    """Close of the 5-minute candle ending at ts (the price when the clock reads ts)."""
    row = b5[b5.index == ts - pd.Timedelta("5min")]
    return float(row["close"].iloc[0]) if len(row) else None


def load_predictor():
    sys.path.insert(0, KRONOS_DIR)
    import torch
    from model import Kronos, KronosPredictor, KronosTokenizer
    torch.manual_seed(0)
    tok = KronosTokenizer.from_pretrained(TOKENIZER)
    mdl = Kronos.from_pretrained(MODEL)
    tok.eval(), mdl.eval()
    return KronosPredictor(mdl, tok, device="cpu", max_context=512)


def forecast(pred, bars, day, hhmm) -> list[dict]:
    """One batch for all indices: 10 paths each of the next 4 fifteen-minute candles."""
    cut = at(day, hhmm)
    xs, xts, yts, meta = [], [], [], []
    for sym, b5 in bars.items():
        b = bars15(b5[b5.index < cut], int(hhmm[3:]) % 15)
        hist = b[b.index + pd.Timedelta("15min") <= cut].iloc[-LOOKBACK:]
        if len(hist) < LOOKBACK or hist.index[-1] + pd.Timedelta("15min") != cut:
            print(f"{sym} {hhmm}: not enough candles up to {cut} ({len(hist)}); skipped", flush=True)
            continue
        fut = pd.Series(pd.date_range(cut, periods=STEPS, freq="15min"))
        for _ in range(PATHS):
            xs.append(hist.reset_index(drop=True))
            xts.append(pd.Series(hist.index))
            yts.append(fut)
        meta.append((sym, float(hist["close"].iloc[-1])))
    if not meta:
        return []
    out = pred.predict_batch(xs, xts, yts, pred_len=STEPS, T=T_DIR, top_p=TOP_P, sample_count=1, verbose=False)
    end60 = min(STEPS, int((at(day, LAST_EXIT) - cut) / pd.Timedelta("15min")))
    rows = []
    for j, (sym, last) in enumerate(meta):
        paths = out[j * PATHS:(j + 1) * PATHS]
        c30 = np.array([p["close"].iloc[1] for p in paths], float)
        c60 = np.array([p["close"].iloc[end60 - 1] for p in paths], float)
        rows.append(dict(day=str(day), time=hhmm, sym=sym, last=round(last, 2),
                         p_up30=round(float((c30 > last).mean()), 2), p_up60=round(float((c60 > last).mean()), 2),
                         pred30=round(float(c30.mean() / last - 1) * 100, 3),
                         pred60=round(float(c60.mean() / last - 1) * 100, 3),
                         made_at=datetime.now(IST).strftime("%H:%M:%S")))
    return rows


def open_trades(sig) -> list[dict]:
    cut = at(sig["day"], sig["time"])
    out = []
    for h in (30, 60):
        p = float(sig[f"p_up{h}"])
        side = "BUY" if p >= LONG_AT else "SELL" if p <= SHORT_AT else None
        if side is None:
            continue
        exit_ts = min(cut + pd.Timedelta(minutes=h), at(sig["day"], LAST_EXIT))
        out.append(dict(day=sig["day"], time=sig["time"], sym=sig["sym"], exit_after=h, side=side, p_up=p,
                        qty=LOTS[sig["sym"]], entry=sig["last"], exit_time=exit_ts.strftime("%H:%M"), exit="",
                        points="", gross="", charges="", net="", status="open"))
    return out


def settle(trades, bars, now) -> int:
    n = 0
    for t in trades:
        if t["status"] != "open":
            continue
        ts = at(t["day"], t["exit_time"])
        if ts > now:
            continue
        b5 = bars.get(t["sym"])
        px = price_at(b5, ts) if b5 is not None else None
        if px is None:
            if now.normalize() > ts.normalize() + pd.Timedelta(days=3):
                t["status"] = "no data"           # Yahoo never gave the exit candle; leave it out
            continue
        sign = 1 if t["side"] == "BUY" else -1
        entry, qty = float(t["entry"]), int(t["qty"])
        pts = sign * (px - entry)
        chg = (entry + px) * qty * COST_PCT + 2 * BROKERAGE
        t.update(exit=round(px, 2), points=round(pts, 2), gross=round(pts * qty, 2), charges=round(chg, 2),
                 net=round(pts * qty - chg, 2), status="closed")
        n += 1
    return n


def snapshot(sigs, trades) -> dict:
    closed = [t for t in trades if t["status"] == "closed"]
    table = []
    for sym in INDICES:
        for hhmm in TIMES:
            for h in (30, 60):
                ts = [t for t in closed if t["sym"] == sym and t["time"] == hhmm and int(t["exit_after"]) == h]
                table.append(dict(sym=sym, time=hhmm, exit_after=h, trades=len(ts),
                                  right=sum(float(t["points"]) > 0 for t in ts),
                                  net=round(sum(float(t["net"]) for t in ts), 2)))
    last_day = max((s["day"] for s in sigs), default=None)
    return dict(updated=datetime.now(IST).strftime("%Y-%m-%d %H:%M"),
                start=min((s["day"] for s in sigs), default=None),
                today=[s for s in sigs if s["day"] == last_day], table=table,
                trades=[t for t in trades if t["status"] in ("closed", "open")][-400:],
                total=round(sum(float(t["net"]) for t in closed), 2), times=list(TIMES), lots=LOTS,
                rule=f"At {', '.join(TIMES)} Kronos forecasts the next 4 fifteen-minute candles (10 paths, "
                     f"{LOOKBACK} candles of history). BUY 1 lot if P(up) >= {LONG_AT:.0%}, SELL if <= {SHORT_AT:.0%}; "
                     f"exit after 30 or 60 minutes (14:30 -> {LAST_EXIT}). Index price as the futures price.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "paper_data"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--now", help="testing: act as if the clock read this IST time, e.g. '2026-11-04 10:11'")
    a = ap.parse_args()
    os.makedirs(a.data, exist_ok=True)
    sig_path = os.path.join(a.data, "kronos_intraday_signals.csv")
    tr_path = os.path.join(a.data, "kronos_intraday_trades.csv")
    sigs, trades = read_rows(sig_path), read_rows(tr_path)
    now = pd.Timestamp(a.now) if a.now else pd.Timestamp(datetime.now(IST).replace(tzinfo=None))
    bars = five_min_bars(now)
    if not bars:
        print("no 5-minute data from Yahoo; nothing done")
        return
    day = now.date()
    traded_today = any(b.index[-1].date() == day for b in bars.values() if len(b))
    due = [t for t in TIMES if timedelta(0) <= now - at(day, t) <= LATE]
    slot = due[-1] if due else None
    if a.dry_run:                                         # the latest slot that has passed today
        past = [t for t in TIMES if at(day, t) <= now]
        if not past or not traded_today:
            print("no slot has passed in today's session yet")
            return
        for r in forecast(load_predictor(), bars, day, past[-1]):
            print(r)
        print("dry run: nothing saved")
        return

    settled = settle(trades, bars, now)
    new = []
    if slot and traded_today and not any(s["day"] == str(day) and s["time"] == slot for s in sigs):
        new = forecast(load_predictor(), bars, day, slot)
        for r in new:
            ot = open_trades(r)
            trades += ot
            print(f"{r['sym']} {slot}: P(up) 30 min {r['p_up30']:.0%}, 60 min {r['p_up60']:.0%} -> "
                  + (", ".join(f"{t['side']} exit {t['exit_time']}" for t in ot) or "no trade"), flush=True)
        sigs += new
        settled += settle(trades, bars, now)
    elif not traded_today:
        print(f"{day}: no market data for today (holiday or weekend)")
    write_rows(sig_path, SIG_COLS, sigs)
    write_rows(tr_path, TRADE_COLS, trades)
    write_json(os.path.join(a.data, "kronos_intraday.json"), snapshot(sigs, trades))
    print(f"slot {slot or '-'}: {len(new)} forecast(s), settled {settled} trade(s), "
          f"{sum(t['status'] == 'open' for t in trades)} open")


if __name__ == "__main__":
    main()
