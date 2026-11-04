"""Gold paper holding: GOLDBEES, bought once and kept (no orders - paper only).

gold_test.py: holding gold in rupees made about 14.8 % a year (2004-2026, worst fall -29 %),
and every rule that timed gold in and out did worse than holding. So this buys GOLDBEES once
for Rs 1,27,142 (the amount bought on 4 Nov 2026) and keeps it - no selling, no top-ups.
It is separate from the momentum portfolio.

  python gold_paper.py           # evening: mark at today's close (buys on the very first run)
  python gold_paper.py --live    # market hours: re-price at the latest 5-minute price

Files in paper_data/: gold_state.json (units, cash, history), gold_trades.csv, gold.json.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from datetime import datetime

from momentum_paper import IST, Source, charges, write_json

HERE = os.path.dirname(os.path.abspath(__file__))
TICKER = "GOLDBEES.NS"
CAPITAL = 127_142


def read(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def gold_state(data_dir):
    return read(os.path.join(data_dir, "gold_state.json")) or {
        "units": 0, "cash": 0.0, "capital": None, "start": None, "start_px": None, "history": []}


def log_trades(data_dir, rows) -> None:
    if not rows:
        return
    p = os.path.join(data_dir, "gold_trades.csv")
    new = not os.path.exists(p)
    with open(p, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


def buy(state, day, px) -> dict:
    units = int(state["cash"] / (px * 1.003))           # leave room for the charges
    val = units * px
    c = charges(val, "BUY")
    state["units"] += units
    state["cash"] -= val + c
    return dict(day=day, ticker=TICKER[:-3], side="BUY", qty=units, price=round(px, 2),
                value=round(val, 2), charges=round(c, 2))


def snapshot(state, day, px, prev_px, live=False) -> dict:
    if day == state["start"]:                            # bought today: today's move counts from the buy
        prev_px = state["start_px"]
    value = state["units"] * px + state["cash"]
    cost = state["capital"] - state["cash"]              # money spent on units, incl. charges
    months, prev_px_m = {}, state["start_px"]
    for r in state["history"]:
        months[r["day"][:7]] = r
    out_months = []
    for m, r in sorted(months.items()):
        v = state["units"] * r["gold_px"] + state["cash"]
        out_months.append(dict(month=m, pct=round((r["gold_px"] / prev_px_m - 1) * 100, 2), value=round(v, 2)))
        prev_px_m = r["gold_px"]
    peak = max([state["start_px"]] + [r["gold_px"] for r in state["history"]] + [px])
    return dict(as_of=day, updated=datetime.now(IST).strftime("%Y-%m-%d %H:%M"), live=live,
                start=state["start"], capital=state["capital"], units=state["units"],
                buy_px=round(cost / state["units"], 2) if state["units"] else None,
                price=round(px, 2), cash=round(state["cash"], 2), value=round(value, 2),
                holding_value=round(state["units"] * px, 2),
                pnl=round(value - state["capital"], 2), pnl_pct=round((value / state["capital"] - 1) * 100, 2),
                day_pnl=round(state["units"] * (px - prev_px), 2) if prev_px else None,
                day_pct=round((px / prev_px - 1) * 100, 2) if prev_px else None,
                dd=round((px / peak - 1) * 100, 2), months=out_months,
                rule=f"GOLDBEES bought once with Rs {state['capital']:,.0f} on {state['start']} and kept. "
                     f"No selling, no top-ups, no timing.")


def run_day(src, data_dir) -> str:
    today = datetime.now(IST).date().isoformat()
    bars = src.bars([TICKER], "1mo").get(TICKER)
    if bars is None or bars.index[-1].date().isoformat() != today:
        return f"{today}: no GOLDBEES bar for today (holiday or Yahoo late); nothing done"
    px = float(bars["Close"].iloc[-1])
    prev_px = float(bars["Close"].iloc[-2]) if len(bars) > 1 else None
    state = gold_state(data_dir)
    msg = []
    if state["start"] is None:
        state.update(capital=CAPITAL, cash=float(CAPITAL), start=today, start_px=round(px, 4))
        t = buy(state, today, px)
        log_trades(data_dir, [t])
        msg.append(f"bought {t['qty']} GOLDBEES for Rs {t['value']:,.0f}")
    state["history"] = [r for r in state["history"] if r["day"] != today]
    state["history"].append(dict(day=today, gold_px=round(px, 4)))
    write_json(os.path.join(data_dir, "gold_state.json"), state)
    snap = snapshot(state, today, px, prev_px)
    write_json(os.path.join(data_dir, "gold.json"), snap)
    msg.append(f"GOLDBEES {px:.2f}, value Rs {snap['value']:,.0f} ({snap['pnl_pct']:+.2f}%)")
    return f"{today}: " + "; ".join(msg)


def live_mark(src, data_dir) -> str:
    """Re-price at the latest 5-minute price; writes only gold.json."""
    state = gold_state(data_dir)
    if state["start"] is None:
        return "gold holding not started yet; nothing to price"
    bars = src.bars([TICKER], "1d", "5m").get(TICKER)
    if bars is None or not len(bars):
        return "no live GOLDBEES price from Yahoo; gold.json left as it was"
    px = float(bars["Close"].iloc[-1])
    day = bars.index[-1].strftime("%Y-%m-%d")
    daily = src.bars([TICKER], "5d").get(TICKER)
    before = daily[daily.index.strftime("%Y-%m-%d") < day] if daily is not None else []
    prev_px = float(before["Close"].iloc[-1]) if len(before) else None
    view = copy.deepcopy(state)                          # the saved state stays the evening record
    view["history"] = [r for r in view["history"] if r["day"] != day]
    view["history"].append(dict(day=day, gold_px=px))
    snap = snapshot(view, day, px, prev_px, live=True)
    write_json(os.path.join(data_dir, "gold.json"), snap)
    return f"live {day}: GOLDBEES {px:.2f}, value Rs {snap['value']:,.0f} ({snap['pnl_pct']:+.2f}%)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="paper_data")
    ap.add_argument("--live", action="store_true", help="re-price now (market hours)")
    a = ap.parse_args()
    data_dir = a.data if os.path.isabs(a.data) else os.path.join(HERE, a.data)
    os.makedirs(data_dir, exist_ok=True)
    src = Source()
    print(live_mark(src, data_dir) if a.live else run_day(src, data_dir))


if __name__ == "__main__":
    main()
