"""Gold paper holding: GOLDBEES kept for good, topped up on dips (no orders - paper only).

gold_test.py: holding gold in rupees made about 14.8 % a year (2004-2026), and every rule that
timed gold in and out did worse than holding. gold_dip_test.py: with Rs 2 lakh kept ready, adding
on dips beat leaving that cash idle in about 70 % of 5-year periods; selling on rises lost.

  start     Rs 1,27,142 of GOLDBEES (bought 4 Nov 2026)
  reserve   Rs 2,00,000 kept ready, in 4 parts of Rs 50,000
  top-up    one part is bought at the evening close when GOLDBEES is 5 / 10 / 15 / 20 % below
            its highest close of the last year (about 250 trading days). Each part is used once.
  never     sells, and never refills a used part
The reserve earns nothing here (the backtest gave it 6 % a year in a liquid fund).
It is separate from the momentum portfolio.

  python gold_paper.py           # evening: mark at today's close, buy any dip part that is due
  python gold_paper.py --live    # market hours: re-price at the latest 5-minute price (never buys)

Files in paper_data/: gold_state.json (units, cash, reserve, buys, history), gold_trades.csv, gold.json.
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
RESERVE = 200_000
DIPS = (0.05, 0.10, 0.15, 0.20)                          # below the 1-year high
PART = RESERVE / len(DIPS)
YEAR = 250                                               # trading days


def read(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def gold_state(data_dir):
    state = read(os.path.join(data_dir, "gold_state.json")) or {
        "units": 0, "cash": 0.0, "capital": None, "start": None, "start_px": None, "history": []}
    # older state files (before the dip rule) have no reserve or buy list
    state.setdefault("reserve", float(RESERVE))
    state.setdefault("dips_used", [])
    state.setdefault("high_1y", None)
    if "buys" not in state:
        state["buys"] = []
        if state["start"]:
            state["buys"].append(dict(day=state["start"], qty=state["units"], price=state["start_px"],
                                      cost=round(state["capital"] - state["cash"], 2)))
    return state


def log_trades(data_dir, rows) -> None:
    if not rows:
        return
    p = os.path.join(data_dir, "gold_trades.csv")
    old = []
    if os.path.exists(p):
        with open(p, newline="", encoding="utf-8") as fh:
            old = list(csv.DictReader(fh))
    with open(p, "w", newline="", encoding="utf-8") as fh:   # rewritten: older files lack the "why" column
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), restval="")
        w.writeheader()
        w.writerows(old + rows)


def buy(state, day, px, amount, pot, why) -> dict:
    """Spend about `amount` from state[pot] ('cash' or 'reserve') on GOLDBEES at px."""
    units = int(min(amount, state[pot]) / (px * 1.003))   # leave room for the charges
    val = units * px
    c = charges(val, "BUY")
    state["units"] += units
    state[pot] -= val + c
    state["buys"].append(dict(day=day, qty=units, price=round(px, 4), cost=round(val + c, 2)))
    return dict(day=day, ticker=TICKER[:-3], side="BUY", qty=units, price=round(px, 2),
                value=round(val, 2), charges=round(c, 2), why=why)


def snapshot(state, day, px, prev_px, live=False) -> dict:
    invested = state["capital"] + RESERVE
    gold_val = state["units"] * px
    value = gold_val + state["cash"] + state["reserve"]
    cost = sum(b["cost"] for b in state["buys"])
    # today's P&L: units held from before today move from the previous close, today's buys from their price
    today_buys = [b for b in state["buys"] if b["day"] == day]
    old_units = state["units"] - sum(b["qty"] for b in today_buys)
    day_pnl = None
    if prev_px or today_buys:
        day_pnl = (old_units * (px - prev_px) if prev_px and old_units else 0.0) + \
                  sum(b["qty"] * px - b["cost"] for b in today_buys)
    months = {}
    for r in state["history"]:
        months[r["day"][:7]] = r
    out_months, prev_v, prev_px_m = [], invested, state["start_px"]
    for m, r in sorted(months.items()):
        v = r.get("value", state["units"] * r["gold_px"] + state["cash"] + state["reserve"])
        out_months.append(dict(month=m, gold_pct=round((r["gold_px"] / prev_px_m - 1) * 100, 2),
                               ret_pct=round((v / prev_v - 1) * 100, 2), value=round(v, 2)))
        prev_v, prev_px_m = v, r["gold_px"]
    high = max(state["high_1y"] or px, px)
    dips = [dict(level=round(lvl * 100), trigger=round(high * (1 - lvl), 2), amount=PART,
                 done=next((b["day"] for b in state["buys"] if b.get("dip") == lvl), None))
            for lvl in DIPS]
    return dict(as_of=day, updated=datetime.now(IST).strftime("%Y-%m-%d %H:%M"), live=live,
                start=state["start"], capital=state["capital"], reserve_start=RESERVE, invested=invested,
                units=state["units"], buy_px=round(cost / state["units"], 2) if state["units"] else None,
                gold_cost=round(cost, 2), price=round(px, 2), cash=round(state["cash"], 2),
                reserve=round(state["reserve"], 2), value=round(value, 2), holding_value=round(gold_val, 2),
                gold_pnl=round(gold_val - cost, 2), gold_pnl_pct=round((gold_val / cost - 1) * 100, 2) if cost else None,
                pnl=round(value - invested, 2), pnl_pct=round((value / invested - 1) * 100, 2),
                day_pnl=None if day_pnl is None else round(day_pnl, 2),
                day_pct=round((px / prev_px - 1) * 100, 2) if prev_px else None,
                high_1y=round(high, 2), off_high=round((px / high - 1) * 100, 2), dips=dips,
                buys=[dict(b, cost=round(b["cost"], 2)) for b in state["buys"]], months=out_months,
                rule=f"Rs {state['capital']:,.0f} of GOLDBEES on {state['start']}, plus Rs {RESERVE:,} kept ready in "
                     f"{len(DIPS)} parts of Rs {PART:,.0f}: one part is bought at the evening close when GOLDBEES is "
                     + " / ".join(f"{round(l * 100)}" for l in DIPS) +
                     " % below its 1-year high close. Each part is used once. Never sells.")


def run_day(src, data_dir) -> str:
    today = datetime.now(IST).date().isoformat()
    bars = src.bars([TICKER], "2y").get(TICKER)
    if bars is None or bars.index[-1].date().isoformat() != today:
        return f"{today}: no GOLDBEES bar for today (holiday or Yahoo late); nothing done"
    px = float(bars["Close"].iloc[-1])
    prev_px = float(bars["Close"].iloc[-2]) if len(bars) > 1 else None
    high = float(bars["Close"].tail(YEAR).max())
    state = gold_state(data_dir)
    msg, trades = [], []
    if state["start"] is None:
        state.update(capital=CAPITAL, cash=float(CAPITAL), start=today, start_px=round(px, 4))
        trades.append(buy(state, today, px, CAPITAL, "cash", "start"))
        msg.append(f"bought {trades[-1]['qty']} GOLDBEES for Rs {trades[-1]['value']:,.0f}")
    state["high_1y"] = high
    for lvl in DIPS:
        if lvl not in state["dips_used"] and px <= high * (1 - lvl) and state["reserve"] > px * 1.003:
            t = buy(state, today, px, PART, "reserve", f"dip {round(lvl * 100)}% below 1-year high {high:.2f}")
            state["buys"][-1]["dip"] = lvl
            state["dips_used"].append(lvl)
            trades.append(t)
            msg.append(f"dip {round(lvl * 100)}%: bought {t['qty']} for Rs {t['value']:,.0f}")
    log_trades(data_dir, trades)
    value = state["units"] * px + state["cash"] + state["reserve"]
    state["history"] = [r for r in state["history"] if r["day"] != today]
    state["history"].append(dict(day=today, gold_px=round(px, 4), value=round(value, 2)))
    write_json(os.path.join(data_dir, "gold_state.json"), state)
    snap = snapshot(state, today, px, prev_px)
    write_json(os.path.join(data_dir, "gold.json"), snap)
    msg.append(f"GOLDBEES {px:.2f} ({snap['off_high']:+.1f}% vs 1-year high {high:.2f}), value Rs {value:,.0f} "
               f"({snap['pnl_pct']:+.2f}%), reserve left Rs {state['reserve']:,.0f}")
    return f"{today}: " + "; ".join(msg)


def live_mark(src, data_dir) -> str:
    """Re-price at the latest 5-minute price; writes only gold.json. Never buys."""
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
    view["history"].append(dict(day=day, gold_px=px, value=view["units"] * px + view["cash"] + view["reserve"]))
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
