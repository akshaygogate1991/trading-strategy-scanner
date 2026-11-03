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

--live (deploy/momentum-live.timer, every 15 minutes in market hours) only re-prices
the holdings at the latest 5-minute price and rewrites momentum.json with live=true.
It never buys, sells or touches momentum_state.json; the evening run stays the record.

Files in paper_data/: momentum_state.json (holdings and cash),
momentum_trades.csv (every paper buy and sell), momentum.json (snapshot).

Prices come from Yahoo Finance (free, no key). The stock list is NSE's
EQUITY_L.csv, refreshed each month, with nse_equity_list.csv as the fallback.

  python momentum_paper.py              # normal evening run
  python momentum_paper.py --dry-run    # print what it would pick now, save nothing
  python momentum_paper.py --live       # re-price the holdings now, for the dashboard
  python momentum_paper.py --replay 2026-09-01 2026-10-30 --offline swing_data/prices_all.pkl --data paper_data_momentum_replay
"""
from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import math
import os
import pickle
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAPITAL = 500_000
N_PICKS = 20
UNIVERSE = 200
MIN_PRICE = 20
MIN_STOCKS = 1700                  # a full download has ~2,000 stocks with a year of prices
# Yahoo blocks much of a 2,400-stock download from a server, so the VPS downloads only
# the 700 most traded stocks. In the backtest a list up to 3 months old gave exactly the
# same picks as using all stocks. Refresh it on the laptop every 3 months.
CANDIDATES = os.path.join(HERE, "momentum_candidates.csv")
N_CANDIDATES = 700
MAX_MISSING = 0.03                 # rebalance only if at most 3% of the candidates are missing
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

    def bars(self, tickers: list[str], period: str, interval: str = "1d") -> dict[str, pd.DataFrame]:
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
        out, todo = {}, list(tickers)
        # Yahoo silently drops part of a big batch when it throttles (seen on the VPS:
        # 1,382 of ~2,000 stocks), so retry the missing ones in smaller batches
        for attempt, size in enumerate((200, 50, 20)):
            if attempt:
                time.sleep(20)
            failed = []
            for i in range(0, len(todo), size):
                part = todo[i:i + size]
                try:
                    raw = yf.download(part, period=period, interval=interval, auto_adjust=True, group_by="ticker",
                                      progress=False, threads=True)
                except Exception as e:
                    print(f"  download error ({e}); will retry")
                    failed += part
                    continue
                for t in part:
                    d = raw[t].dropna(subset=["Close"]) if t in raw.columns.get_level_values(0) else []
                    if len(d):
                        out[t] = d[["Open", "High", "Low", "Close", "Volume"]]
                    else:
                        failed.append(t)
            todo = failed
            if len(tickers) > 50:
                print(f"  pass {attempt + 1}: {len(out)} of {len(tickers)} have data, {len(todo)} missing")
            if not todo:
                break
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


# ---------------------------------------------------------------- candidates
def load_candidates() -> tuple[list[str] | None, str | None]:
    if not os.path.exists(CANDIDATES):
        return None, None
    c = pd.read_csv(CANDIDATES)
    return c["symbol"].tolist(), str(c["made"].iloc[0])


def refresh_candidates(src: Source, today: date) -> str:
    """Save the N_CANDIDATES most traded stocks (laptop: Yahoo blocks much of a full
    download from servers). Any top-200 stock of the next few months is in this list."""
    symbols = src.symbols()
    frames = src.bars(symbols, "1y")
    frames.pop(INDEX, None)
    if len(frames) < 0.95 * len(symbols):
        return (f"only {len(frames)} of {len(symbols)} stocks came back from Yahoo; "
                f"list not saved, try again later")
    raw_close = pd.DataFrame({t: d["Close"] for t, d in frames.items()}).sort_index()
    value = pd.DataFrame({t: d["Close"] * d["Volume"] for t, d in frames.items()}).reindex(raw_close.index)
    liq = value.tail(126)
    liq = liq.median().where(liq.notna().sum() >= 100)
    recent = raw_close.tail(5).notna().any()
    top = liq[recent].dropna().nlargest(N_CANDIDATES)
    pd.DataFrame({"symbol": top.index, "median_traded_value_cr": (top.values / 1e7).round(2),
                  "made": today.isoformat()}).to_csv(CANDIDATES, index=False)
    return (f"saved {len(top)} candidates to {os.path.basename(CANDIDATES)} "
            f"(from {len(frames)} stocks). Push it so the VPS uses it.")


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
    cands, made = load_candidates() if src.offline is None else (None, None)
    if dry_run or month not in state["months"]:
        frames = src.bars(cands or src.symbols(), "2y")
        enough = sum(len(d) >= 260 for d in frames.values())
        if cands:
            missing = [t for t in cands if t not in frames]
            bad = len(missing) > MAX_MISSING * len(cands)
            note = (f"{len(missing)} of the {len(cands)} candidate stocks came back without prices "
                    f"(allowed {int(MAX_MISSING * len(cands))}); picks would be wrong")
        else:
            bad = enough < MIN_STOCKS
            note = (f"only {enough} stocks came back with a year of prices (need {MIN_STOCKS}); "
                    f"picks would be wrong")
        if bad:
            if not dry_run:
                return f"{today}: {note}. Not rebalancing; will try again at the next run"
            print("WARNING:", note)
        # the previous month-end close; a dry run just uses the latest close
        cut = pd.Timestamp(today) if dry_run else pd.Timestamp(today.replace(day=1)) - pd.Timedelta(days=1)
        signal_day = max(d for d in idx.index if d <= cut)
        picks = select(frames, signal_day)
        if dry_run:
            source = f"candidate list of {made}, {len(cands)} stocks" if cands else "all NSE stocks"
            return (f"picks from the close of {signal_day.date()} ({source}; {len(frames)} with data, "
                    f"{enough} with a year of prices): "
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
    snap["candidates_made"] = made
    write_json(os.path.join(data_dir, "momentum.json"), snap)
    msg.append(f"value Rs {value:,.0f} ({snap['total_pct']:+.2f}%), Nifty {snap['nifty_pct']:+.2f}%")
    return f"{today}: " + "; ".join(msg)


def live_mark(src: Source, data_dir: str) -> str:
    """Re-price the holdings at the latest intraday price; writes only momentum.json."""
    state = load_state(data_dir)
    if not state["holdings"]:
        return "no holdings yet; nothing to price"
    held = list(state["holdings"])
    marks = src.bars(held + [INDEX], "1d", "5m")
    idx = marks.get(INDEX)
    if idx is None or not len(idx):
        return "no live Nifty price from Yahoo; momentum.json left as it was"
    view = copy.deepcopy(state)                   # the saved state stays the evening record
    priced = 0
    for t, h in view["holdings"].items():
        d = marks.get(t)
        if d is not None and len(d):
            h["last_px"] = float(d["Close"].iloc[-1])
            priced += 1
    nifty = float(idx["Close"].iloc[-1])
    day = idx.index[-1].strftime("%Y-%m-%d")
    value = view["cash"] + sum(h["qty"] * h["last_px"] for h in view["holdings"].values())
    view["history"] = [r for r in view["history"] if r["day"] != day]
    view["history"].append(dict(day=day, value=round(value, 2), nifty=nifty))
    snap = snapshot(view, day, nifty)
    old = os.path.join(data_dir, "momentum.json")
    if os.path.exists(old):
        with open(old, encoding="utf-8") as fh:
            snap["candidates_made"] = json.load(fh).get("candidates_made")
    snap["live"] = True
    snap["priced"] = f"{priced} of {len(held)}"
    write_json(old, snap)
    return f"live {day}: {priced} of {len(held)} priced, value Rs {value:,.0f} ({snap['total_pct']:+.2f}%)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="paper_data")
    ap.add_argument("--offline", help="pickle of daily bars (replay/testing)")
    ap.add_argument("--replay", nargs=2, metavar=("START", "END"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--live", action="store_true", help="re-price the holdings now (market hours)")
    ap.add_argument("--refresh-candidates", action="store_true",
                    help="laptop, every 3 months: rebuild momentum_candidates.csv, then push it")
    a = ap.parse_args()
    data_dir = a.data if os.path.isabs(a.data) else os.path.join(HERE, a.data)
    os.makedirs(data_dir, exist_ok=True)
    src = Source(a.offline)
    if a.live:
        print(live_mark(src, data_dir))
        return
    if a.refresh_candidates:
        print(refresh_candidates(src, datetime.now(IST).date()))
        return
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
