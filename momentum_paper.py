"""Monthly momentum paper tracker (no orders - paper only).

The version that held up best in swing_robust_test.py: every month, the 20
stocks with the best 12-month return (skipping the last month) among the 200
most traded NSE stocks, each above its 200-day average. Rs 5 lakh, equal money
per stock, delivery, no market filter.

Run once every weekday evening (deploy/momentum-paper.timer):
  * First trading day of a new month: picks from the previous month-end close,
    sells what dropped out and buys what came in at today's open (the same fill
    as the backtest). Continuing stocks are left as they are.
  * Every run: marks the holdings at today's close and writes
    paper_data/momentum.json for the dashboard.

Files in paper_data/: momentum_state.json (holdings and cash),
momentum_trades.csv (every paper buy and sell), momentum.json (snapshot).

Prices come from Yahoo Finance (free, no key). The stock list is NSE's
EQUITY_L.csv, refreshed each month, with nse_equity_list.csv as the fallback.

  python momentum_paper.py              # normal evening run
  python momentum_paper.py --dry-run    # print what it would pick now, save nothing
  python momentum_paper.py --replay 2026-09-01 2026-10-30 --offline swing_data/prices_all.pkl --data paper_data_momentum_replay
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import pickle
from datetime import date, datetime, timedelta, timezone

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAPITAL = 500_000
N_PICKS = 20
UNIVERSE = 200
MIN_PRICE = 20
INDEX = "^NSEI"
IST = timezone(timedelta(hours=5, minutes=30))
EQUITY_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
SLIP = 0.001                       # same as the backtest: 0.1% each side


def charges(value: float, side: str) -> float:
    """Delivery charges + slippage for one order (as in swing_screener_test.py)."""
    exch = value * 0.0000297
    c = value * (0.001 + SLIP) + exch * 1.18 + 5 * 1.18
    return c + value * 0.00015 if side == "BUY" else c + 15.93 * 1.18


# ---------------------------------------------------------------- data
class Source:
    """Daily bars from Yahoo, or from a saved pickle when replaying."""

    def __init__(self, offline: str | None = None):
        self.offline = None
        if offline:
            with open(offline, "rb") as fh:
                self.offline = pickle.load(fh)
        self.today = None                 # replay: hide bars after this day

    def bars(self, tickers: list[str], period: str) -> dict[str, pd.DataFrame]:
        if self.offline is not None:
            out = {}
            for t in tickers:
                d = self.offline.get(t)
                if d is not None:
                    d = d.loc[:pd.Timestamp(self.today)]
                    if len(d):
                        out[t] = d
            return out
        import yfinance as yf
        out = {}
        for i in range(0, len(tickers), 200):
            part = tickers[i:i + 200]
            raw = yf.download(part, period=period, auto_adjust=True, group_by="ticker",
                              progress=False, threads=True)
            for t in part:
                if t in raw.columns.get_level_values(0):
                    d = raw[t].dropna(subset=["Close"])
                    if len(d):
                        out[t] = d[["Open", "High", "Low", "Close", "Volume"]]
        return out

    def symbols(self) -> list[str]:
        if self.offline is not None:
            return [t for t in self.offline if t != INDEX]
        text = None
        try:
            import requests
            r = requests.get(EQUITY_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            if r.ok and "SYMBOL" in r.text[:200]:
                text = r.text
        except Exception as e:                       # NSE often blocks servers; use the saved copy
            print(f"NSE list download failed ({e}); using nse_equity_list.csv")
        if text is None:
            with open(os.path.join(HERE, "nse_equity_list.csv"), encoding="utf-8") as fh:
                text = fh.read()
        lst = pd.read_csv(io.StringIO(text), skipinitialspace=True)
        return [f"{s.strip()}.NS" for s in lst.loc[lst["SERIES"].str.strip() == "EQ", "SYMBOL"]]


def select(frames: dict[str, pd.DataFrame], signal_day: pd.Timestamp) -> list[tuple[str, float]]:
    """The 20 picks at a month-end close, same rules as swing_pit_test.py."""
    raw_close = pd.DataFrame({t: d["Close"] for t, d in frames.items()}).sort_index().loc[:signal_day]
    close = raw_close.ffill(limit=5)
    value = pd.DataFrame({t: d["Close"] * d["Volume"] for t, d in frames.items()}).reindex(close.index)
    last = close.index[-1]
    bars = raw_close.notna().sum()
    recent = raw_close.tail(5).notna().any()
    liq = value.tail(126)
    liq = liq.median().where(liq.notna().sum() >= 100)
    ok = (bars >= 260) & recent & (close.loc[last] >= MIN_PRICE)
    uni = liq[ok].dropna().nlargest(UNIVERSE).index
    sma200 = close[uni].tail(200).mean().where(close[uni].tail(200).notna().sum() >= 200)
    if len(close) < 253:
        return []
    ret = close[uni].iloc[-22] / close[uni].iloc[-253] - 1
    ret = ret[close.loc[last, uni] > sma200]
    return [(t, float(r)) for t, r in ret.dropna().nlargest(N_PICKS).items()]


# ---------------------------------------------------------------- state
def load_state(data_dir: str) -> dict:
    p = os.path.join(data_dir, "momentum_state.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    return {"cash": CAPITAL, "holdings": {}, "months": [], "history": [], "start": None,
            "nifty_start": None, "picks": [], "signal_day": None}


def write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def log_trades(data_dir: str, rows: list[dict]) -> None:
    if not rows:
        return
    p = os.path.join(data_dir, "momentum_trades.csv")
    new = not os.path.exists(p)
    with open(p, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


def rebalance(state, today, picks, opens) -> list[dict]:
    """Sell dropped stocks and buy new ones at today's open."""
    trades, keep = [], {t for t, _ in picks}
    for t in list(state["holdings"]):
        if t in keep:
            continue
        h = state["holdings"].pop(t)
        px = opens.get(t) or h["last_px"]
        val = h["qty"] * px
        c = charges(val, "SELL")
        state["cash"] += val - c
        gross = (px - h["entry_px"]) * h["qty"]
        trades.append(dict(day=today, ticker=t[:-3], side="SELL", qty=h["qty"], price=round(px, 2),
                           value=round(val, 2), charges=round(c, 2), pnl=round(gross - c - h["buy_charges"], 2),
                           entry_day=h["entry_day"], entry_px=h["entry_px"]))
    held_val = sum(h["qty"] * (opens.get(t) or h["last_px"]) for t, h in state["holdings"].items())
    target = (state["cash"] + held_val) / N_PICKS
    for t, _ in picks:
        if t in state["holdings"] or not opens.get(t):
            continue
        px = opens[t]
        qty = math.floor(min(target, state["cash"]) / (px * (1 + 0.003)))
        if qty < 1 and px <= 1.5 * target and px * 1.003 <= state["cash"]:
            qty = 1                          # one costly share (e.g. Rs 32k) rather than skipping the pick
        if qty < 1:
            continue
        val = qty * px
        c = charges(val, "BUY")
        state["cash"] -= val + c
        state["holdings"][t] = dict(qty=qty, entry_day=today, entry_px=round(px, 2), buy_charges=round(c, 2),
                                    last_px=px)
        trades.append(dict(day=today, ticker=t[:-3], side="BUY", qty=qty, price=round(px, 2),
                           value=round(val, 2), charges=round(c, 2), pnl="", entry_day=today,
                           entry_px=round(px, 2)))
    return trades


def snapshot(state, today, nifty_close) -> dict:
    hold = []
    for t, h in sorted(state["holdings"].items()):
        val = h["qty"] * h["last_px"]
        cost = h["qty"] * h["entry_px"]
        pnl = val - cost - h["buy_charges"]
        hold.append(dict(ticker=t[:-3], qty=h["qty"], entry_day=h["entry_day"], entry_px=h["entry_px"],
                         last_px=round(h["last_px"], 2), value=round(val, 2), pnl=round(pnl, 2),
                         pnl_pct=round(pnl / cost * 100, 2) if cost else None))
    value = state["cash"] + sum(x["value"] for x in hold)
    months = {}
    for row in state["history"]:
        months[row["day"][:7]] = row                      # last run of each month
    out_months, prev_v, prev_n = [], CAPITAL, state["nifty_start"]
    for m, row in sorted(months.items()):
        out_months.append(dict(month=m, value=round(row["value"], 2),
                               ret_pct=round((row["value"] / prev_v - 1) * 100, 2),
                               nifty_pct=round((row["nifty"] / prev_n - 1) * 100, 2) if prev_n else None))
        prev_v, prev_n = row["value"], row["nifty"]
    return dict(as_of=today, updated=datetime.now(IST).strftime("%Y-%m-%d %H:%M"),
                capital=CAPITAL, value=round(value, 2), cash=round(state["cash"], 2),
                total_pnl=round(value - CAPITAL, 2), total_pct=round((value / CAPITAL - 1) * 100, 2),
                nifty_pct=round((nifty_close / state["nifty_start"] - 1) * 100, 2) if state["nifty_start"] else None,
                start=state["start"], signal_day=state["signal_day"],
                picks=[dict(ticker=t[:-3], ret12_1=round(r * 100, 1)) for t, r in state["picks"]],
                holdings=sorted(hold, key=lambda x: -(x["pnl_pct"] or 0)), months=out_months,
                rule=f"Top {N_PICKS} by 12-1 month return among the {UNIVERSE} most traded NSE stocks, "
                     f"above the 200-day average. Rebalanced on the first trading day of each month.")


# ---------------------------------------------------------------- run
def run_day(src: Source, data_dir: str, today: date, dry_run: bool = False) -> str:
    state = load_state(data_dir)
    month = today.strftime("%Y-%m")
    held = list(state["holdings"])
    marks = src.bars(held + [INDEX], "1mo")
    idx = marks.get(INDEX)
    traded_today = idx is not None and idx.index[-1].date() == today
    if not traded_today and not dry_run:
        return f"{today}: no NSE bar for today (holiday, weekend or Yahoo not updated yet); nothing done"

    msg = []
    if dry_run or month not in state["months"]:
        frames = src.bars(src.symbols(), "2y")
        # the previous month-end close; a dry run just uses the latest close
        cut = pd.Timestamp(today) if dry_run else pd.Timestamp(today.replace(day=1)) - pd.Timedelta(days=1)
        signal_day = max(d for d in idx.index if d <= cut)
        picks = select(frames, signal_day)
        if dry_run:
            return (f"picks from the close of {signal_day.date()} ({len(frames)} stocks with data): "
                    + ", ".join(f"{t[:-3]} {r * 100:+.0f}%" for t, r in picks))
        opens = {t: float(d["Open"].iloc[-1]) for t, d in frames.items()
                 if d.index[-1].date() == today and d["Open"].iloc[-1] > 0}
        trades = rebalance(state, today.isoformat(), picks, opens)
        log_trades(data_dir, trades)
        state["months"].append(month)
        state["picks"], state["signal_day"] = picks, str(signal_day.date())
        if state["start"] is None:
            state["start"] = today.isoformat()
            state["nifty_start"] = float(idx["Open"].iloc[-1])
        marks.update({t: frames[t] for t in state["holdings"] if t in frames})
        msg.append(f"rebalanced: {sum(t['side'] == 'BUY' for t in trades)} buys, "
                   f"{sum(t['side'] == 'SELL' for t in trades)} sells (signal {signal_day.date()})")

    for t, h in state["holdings"].items():
        d = marks.get(t)
        if d is not None and len(d):
            h["last_px"] = float(d["Close"].iloc[-1])
    nifty_close = float(idx["Close"].iloc[-1])
    value = state["cash"] + sum(h["qty"] * h["last_px"] for h in state["holdings"].values())
    state["history"] = [r for r in state["history"] if r["day"] != today.isoformat()]
    state["history"].append(dict(day=today.isoformat(), value=round(value, 2), nifty=nifty_close))
    write_json(os.path.join(data_dir, "momentum_state.json"), state)
    snap = snapshot(state, today.isoformat(), nifty_close)
    write_json(os.path.join(data_dir, "momentum.json"), snap)
    msg.append(f"value Rs {value:,.0f} ({snap['total_pct']:+.2f}%), Nifty {snap['nifty_pct']:+.2f}%")
    return f"{today}: " + "; ".join(msg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="paper_data")
    ap.add_argument("--offline", help="pickle of daily bars (replay/testing)")
    ap.add_argument("--replay", nargs=2, metavar=("START", "END"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    data_dir = a.data if os.path.isabs(a.data) else os.path.join(HERE, a.data)
    os.makedirs(data_dir, exist_ok=True)
    src = Source(a.offline)
    if a.replay:
        days = src.offline[INDEX].loc[a.replay[0]:a.replay[1]].index
        for d in days:
            src.today = d
            print(run_day(src, data_dir, d.date()))
        return
    today = datetime.now(IST).date()
    src.today = today
    print(run_day(src, data_dir, today, a.dry_run))


if __name__ == "__main__":
    main()
