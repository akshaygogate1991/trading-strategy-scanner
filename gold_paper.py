"""Gold next to the momentum paper portfolio (no orders - paper only).

The one idea that held up in gold_test.py / gold_mix_test.py: 80 % momentum + 20 % gold,
rebalanced once a year. Backtest 2014-2026: 25.1 % a year with the worst fall -22 %,
against 26.1 % and -34 % for momentum alone.

The momentum portfolio (momentum_paper.py) is left exactly as it is. This keeps a separate
GOLDBEES paper holding worth 25 % of the momentum value on the start day, so the two
together are 80 / 20. On the first trading day of each new year the gold holding is set
back to 20 % of the combined value; the money for that comes from (or goes to) the
momentum side, so it is booked as "transfer" (gold cash can go below zero). Momentum's
own stock sizes do not change - paper only, the combined value is still right.

Run after momentum_paper.py (same services):
  python gold_paper.py           # evening: mark at today's close, start / yearly rebalance
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
WEIGHT = 0.20


def read(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def gold_state(data_dir):
    return read(os.path.join(data_dir, "gold_state.json")) or {
        "units": 0, "cash": 0.0, "capital": None, "start": None, "start_px": None,
        "mom_start": None, "years": [], "history": []}


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


def trade(state, day, px, target_value) -> dict | None:
    """Buy or sell GOLDBEES units so the holding is worth about target_value."""
    units = int(target_value / (px * 1.003)) - state["units"]     # leave room for the charges
    if units == 0:
        return None
    side = "BUY" if units > 0 else "SELL"
    val = abs(units) * px
    c = charges(val, side)
    state["units"] += units
    state["cash"] -= units * px + c
    return dict(day=day, ticker=TICKER[:-3], side=side, qty=abs(units), price=round(px, 2),
                value=round(val, 2), charges=round(c, 2))


def snapshot(state, day, px, prev_px, mom, live=False) -> dict:
    gold_val = state["units"] * px
    gold_net = gold_val + state["cash"]                  # gold side incl. its cash / transfers
    combined = mom["value"] + gold_net
    start_total = state["mom_start"] + state["capital"]
    months, prev = {}, None
    for r in state["history"]:
        months[r["day"][:7]] = r
    out_months = []
    first = dict(gold_px=state["start_px"], mom=state["mom_start"], combined=start_total)
    for m, r in sorted(months.items()):
        p = prev or first
        out_months.append(dict(month=m, gold_pct=round((r["gold_px"] / p["gold_px"] - 1) * 100, 2),
                               mom_pct=round((r["mom"] / p["mom"] - 1) * 100, 2),
                               combined_pct=round((r["combined"] / p["combined"] - 1) * 100, 2),
                               combined=round(r["combined"], 2)))
        prev = r
    peak = max([start_total] + [r["combined"] for r in state["history"]] + [combined])
    mpeak = max([state["mom_start"]] + [r["mom"] for r in state["history"]] + [mom["value"]])
    year = int(day[:4])
    return dict(as_of=day, updated=datetime.now(IST).strftime("%Y-%m-%d %H:%M"), live=live,
                start=state["start"], capital=state["capital"], units=state["units"],
                price=round(px, 2), start_px=state["start_px"],
                day_pct=round((px / prev_px - 1) * 100, 2) if prev_px else None,
                gold_value=round(gold_val, 2), gold_cash=round(state["cash"], 2),
                gold_pct=round((px / state["start_px"] - 1) * 100, 2),
                weight_pct=round(gold_val / combined * 100, 2) if combined else None,
                mom_value=round(mom["value"], 2), mom_as_of=mom.get("as_of"),
                mom_pct=round((mom["value"] / state["mom_start"] - 1) * 100, 2),
                combined_value=round(combined, 2), combined_start=round(start_total, 2),
                combined_pct=round((combined / start_total - 1) * 100, 2),
                combined_dd=round((combined / peak - 1) * 100, 2),
                mom_dd=round((mom["value"] / mpeak - 1) * 100, 2),
                months=out_months, next_rebalance=f"first trading day of {year + 1}",
                rule=f"GOLDBEES worth {WEIGHT:.0%} of momentum + gold together, reset to {WEIGHT:.0%} "
                     f"on the first trading day of each year. Momentum is unchanged.")


def run_day(src, data_dir) -> str:
    today = datetime.now(IST).date().isoformat()
    mom = read(os.path.join(data_dir, "momentum.json"))
    if not mom:
        return "momentum portfolio not started yet; nothing done"
    bars = src.bars([TICKER], "1mo").get(TICKER)
    if bars is None or bars.index[-1].date().isoformat() != today:
        return f"{today}: no GOLDBEES bar for today (holiday or Yahoo late); nothing done"
    px = float(bars["Close"].iloc[-1])
    prev_px = float(bars["Close"].iloc[-2]) if len(bars) > 1 else None
    state = gold_state(data_dir)
    trades, msg = [], []
    if state["start"] is None:                           # 80 / 20: gold = momentum / 4
        state.update(capital=round(mom["value"] * WEIGHT / (1 - WEIGHT), 2), cash=0.0, start=today,
                     start_px=round(px, 4), mom_start=mom["value"], years=[today[:4]])
        state["cash"] = state["capital"]
        trades.append(trade(state, today, px, state["capital"]))
        msg.append(f"started: Rs {state['capital']:,.0f} in GOLDBEES")
    elif today[:4] not in state["years"]:
        combined = mom["value"] + state["units"] * px + state["cash"]
        t = trade(state, today, px, WEIGHT * combined)
        trades.append(t)
        state["years"].append(today[:4])
        msg.append(f"yearly rebalance to {WEIGHT:.0%}: {t['side'] + ' ' + str(t['qty']) if t else 'no change'}")
    log_trades(data_dir, [t for t in trades if t])
    combined = mom["value"] + state["units"] * px + state["cash"]
    state["history"] = [r for r in state["history"] if r["day"] != today]
    state["history"].append(dict(day=today, gold_px=round(px, 4), mom=round(mom["value"], 2),
                                 combined=round(combined, 2)))
    write_json(os.path.join(data_dir, "gold_state.json"), state)
    snap = snapshot(state, today, px, prev_px, mom)
    write_json(os.path.join(data_dir, "gold.json"), snap)
    msg.append(f"GOLDBEES {px:.2f} ({snap['gold_pct']:+.2f}%), combined Rs {combined:,.0f} "
               f"({snap['combined_pct']:+.2f}%), momentum alone {snap['mom_pct']:+.2f}%")
    return f"{today}: " + "; ".join(msg)


def live_mark(src, data_dir) -> str:
    """Re-price at the latest 5-minute price; writes only gold.json."""
    state = gold_state(data_dir)
    mom = read(os.path.join(data_dir, "momentum.json"))
    if state["start"] is None or not mom:
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
    view["history"].append(dict(day=day, gold_px=px, mom=mom["value"],
                                combined=mom["value"] + view["units"] * px + view["cash"]))
    snap = snapshot(view, day, px, prev_px, mom, live=True)
    write_json(os.path.join(data_dir, "gold.json"), snap)
    return f"live {day}: GOLDBEES {px:.2f}, combined Rs {snap['combined_value']:,.0f} ({snap['combined_pct']:+.2f}%)"


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
