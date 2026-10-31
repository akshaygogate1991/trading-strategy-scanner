"""What target does a 10% return on invested value actually require?

Two separate questions, answered separately:

  1. ARITHMETIC — given the verified charges, how far must price move?
  2. REALITY     — does price ever move that far, intraday, in our data?

Question 1 is a calculation and always has an answer. Question 2 is a
measurement, and it is the one that decides whether the target is a plan
or a wish.

CHARGES — verified against https://zerodha.com/charges on 14 Oct 2026
  Equity intraday
    brokerage    0.03% or Rs 20 per executed order, WHICHEVER IS LOWER
    STT          0.025%, SELL side only
    transaction  NSE 0.00307%, both sides
    SEBI         Rs 10 per crore (0.0001%), both sides
    stamp        0.003%, BUY side only
    GST          18% on (brokerage + SEBI + transaction) only
                 -- NOT on STT, NOT on stamp duty
  Nothing here is charged on profit. Every line is charged on TURNOVER,
  so a losing trade pays the same as a winning one.
  DP charges (Rs 15.34/scrip) do NOT apply to intraday - only delivery sells.

Run:  python target_calc.py
      python target_calc.py --return-pct 10 --capital 100000
"""
from __future__ import annotations

import argparse

import numpy as np

# --- official equity-intraday rates, Oct 2026 ---
BROKERAGE_RATE = 0.0003      # 0.03%
BROKERAGE_CAP = 20.0         # Rs 20, whichever is LOWER
STT_SELL = 0.00025           # 0.025% sell side
EXCH_TXN = 0.0000307         # NSE 0.00307% each side
SEBI_FEE = 0.000001          # Rs 10/crore each side
STAMP_BUY = 0.00003          # 0.003% buy side
GST_RATE = 0.18
SLIPPAGE = 0.0004            # 0.04% - not a tax, but it is real money


def breakdown(notional: float, slippage: bool = True) -> dict:
    """Every charge on one intraday round trip of `notional` rupees.

    Buy and sell legs are the same size here. That slightly understates the
    sell-side charges on a winning trade and overstates them on a loser; the
    difference is a rounding error next to the totals.
    """
    brokerage = 2 * min(BROKERAGE_RATE * notional, BROKERAGE_CAP)
    stt = STT_SELL * notional
    txn = EXCH_TXN * notional * 2
    sebi = SEBI_FEE * notional * 2
    stamp = STAMP_BUY * notional
    gst = GST_RATE * (brokerage + sebi + txn)
    slip = SLIPPAGE * notional if slippage else 0.0
    total = brokerage + stt + txn + sebi + stamp + gst + slip
    return {"brokerage": brokerage, "STT": stt, "transaction": txn,
            "SEBI": sebi, "stamp": stamp, "GST": gst, "slippage": slip,
            "TOTAL": total, "pct": 100 * total / notional}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--return-pct", type=float, default=10.0,
                    help="desired return on INVESTED value, per trade")
    ap.add_argument("--capital", type=float, default=100000.0)
    ap.add_argument("--atr-pct", type=float, default=0.090,
                    help="measured 1-min ATR as %% of price (from our data)")
    ap.add_argument("--day-range-pct", type=float, default=1.94,
                    help="measured average full-day range, %% (from our data)")
    ap.add_argument("--best-day-pct", type=float, default=7.72)
    args = ap.parse_args()

    want = args.return_pct / 100.0
    cap = args.capital

    print("=" * 68)
    print("  PART 1 - THE CHARGES, CONFIRMED")
    print("=" * 68)
    b = breakdown(cap)
    print(f"\n  One intraday round trip on Rs {cap:,.0f} of stock:\n")
    for k in ("brokerage", "STT", "transaction", "SEBI", "stamp", "GST"):
        print(f"    {k:14s} Rs {b[k]:>9,.2f}")
    print(f"    {'-' * 28}")
    stat = b["TOTAL"] - b["slippage"]
    print(f"    {'taxes+fees':14s} Rs {stat:>9,.2f}   "
          f"({100 * stat / cap:.3f}% of turnover)")
    print(f"    {'slippage':14s} Rs {b['slippage']:>9,.2f}   "
          f"(not a tax, but you still pay it)")
    print(f"    {'TOTAL':14s} Rs {b['TOTAL']:>9,.2f}   ({b['pct']:.3f}%)")
    print("\n  Every line above is charged on TURNOVER, not on profit.")
    print("  A losing trade pays exactly the same as a winning one.")

    c = b["pct"] / 100.0

    print("\n" + "=" * 68)
    print(f"  PART 2 - TARGET FOR {args.return_pct:.0f}% ON INVESTED VALUE")
    print("=" * 68)
    print(f"\n  Formula:  move needed = {args.return_pct:.0f}%/leverage + "
          f"{b['pct']:.3f}% costs\n")
    print(f"  {'leverage':>10s}{'notional':>13s}{'price move':>13s}"
          f"{'= x ATR':>10s}{'= x day range':>15s}")
    print("  " + "-" * 61)
    rows = []
    for lev, label in [(1.0, "1x (cash)"), (2.0, "2x"), (5.0, "5x (MIS)")]:
        move = want / lev + c
        xatr = 100 * move / args.atr_pct
        xday = 100 * move / args.day_range_pct
        rows.append((lev, label, move, xatr, xday))
        print(f"  {label:>10s}{lev * cap:>13,.0f}{100 * move:>12.3f}%"
              f"{xatr:>10.0f}{xday:>14.1f}")

    print("\n" + "=" * 68)
    print("  PART 3 - IS THAT REACHABLE? (measured, not assumed)")
    print("=" * 68)
    print(f"\n  From 366,000 one-minute candles across 8 liquid NSE stocks:")
    print(f"    average 1-minute ATR      {args.atr_pct:.3f}% of price")
    print(f"    average FULL-DAY range    {args.day_range_pct:.2f}%")
    print(f"    best single day observed  {args.best_day_pct:.2f}%")
    print()
    for lev, label, move, xatr, xday in rows:
        pm = 100 * move
        if pm > args.best_day_pct:
            verdict = (f"IMPOSSIBLE - bigger than the best day "
                       f"({args.best_day_pct:.2f}%) in 180 days")
        elif pm > args.day_range_pct:
            verdict = (f"needs {pm / args.day_range_pct:.1f}x a typical day's "
                       f"ENTIRE range")
        else:
            verdict = f"{100 * pm / args.day_range_pct:.0f}% of a typical day's range"
        print(f"    {label:>10s}  need {pm:>6.2f}%  ->  {verdict}")

    print("\n" + "=" * 68)
    print("  PART 4 - THE STOP THAT COMES WITH IT")
    print("=" * 68)
    stop_pct = 1.5 * args.atr_pct
    print(f"\n  Our stop sits 1.5 x ATR = {stop_pct:.3f}% away. Pairing it with")
    print(f"  these targets gives:\n")
    print(f"  {'leverage':>10s}{'target%':>10s}{'stop%':>9s}{'R:R':>8s}"
          f"{'need win%':>12s}{'measured':>11s}")
    print("  " + "-" * 60)
    for lev, label, move, xatr, xday in rows:
        pm = 100 * move
        rr = pm / stop_pct
        need = 100 * stop_pct / (stop_pct + pm)
        print(f"  {label:>10s}{pm:>9.2f}%{stop_pct:>8.3f}%{rr:>8.0f}"
              f"{need:>11.2f}%{'~0%':>11s}")
    print("\n  'need win%' is what break-even requires at that reward:risk.")
    print("  It looks easy - under 1% - and that is exactly the trap: you")
    print("  need the price to travel 100x further than the stop, on a")
    print("  1-minute chart, before 15:30. Our grid measured the win rate")
    print("  falling to 31% by 4x ATR. At 100x+ it is effectively zero.")

    print("\n" + "=" * 68)
    print("  PART 5 - WHAT 10% IS ACTUALLY A REASONABLE ASK OF")
    print("=" * 68)
    n_trades = [1, 20, 100, 250]
    print(f"\n  Rs {cap:,.0f} capital, wanting Rs {want * cap:,.0f} profit.")
    print(f"  If spread over N trades instead of demanded from one:\n")
    print(f"  {'trades':>8s}{'need net/trade':>17s}{'need gross/trade':>19s}")
    print("  " + "-" * 44)
    for n in n_trades:
        per_net = want / n
        per_gross = per_net + c
        print(f"  {n:>8d}{100 * per_net:>16.3f}%{100 * per_gross:>18.3f}%")
    print(f"\n  Note the last column never drops below {b['pct']:.3f}% - the cost")
    print("  floor. Spreading the goal over more trades lowers what each")
    print("  trade must EARN, but every extra trade adds another full toll.")
    print("  Our measured gross edge was -0.010%. Against a cost floor of")
    print(f"  {b['pct']:.3f}%, no number of trades reaches +{args.return_pct:.0f}%.")
    print()


if __name__ == "__main__":
    main()
