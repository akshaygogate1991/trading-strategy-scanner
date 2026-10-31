"""Futures vs equity: which instrument has the lower cost wall?

Written because "futures cost ~16 Nifty points per round trip" sounds enormous
and "equity costs 0.123%" sounds small, so the two were never compared on the
same scale. They must be, because the only number that matters is cost as a
FRACTION of the move you are trying to capture.

All rates verified against https://zerodha.com/charges on 14 Oct 2026.

Run:  python cost_compare.py
"""
from __future__ import annotations

GST = 0.18
SEBI = 0.000001          # Rs 10/crore per side
BROK_RATE, BROK_CAP = 0.0003, 20.0


def futures_cost(notional: float, tick_slip_pts: float, price: float) -> dict:
    """F&O futures: STT 0.05% sell, NSE txn 0.00183%, stamp 0.002% buy."""
    brokerage = 2 * min(BROK_RATE * notional, BROK_CAP)
    stt = 0.0005 * notional
    txn = 0.0000183 * notional * 2
    sebi = SEBI * notional * 2
    stamp = 0.00002 * notional
    gst = GST * (brokerage + sebi + txn)
    stat = brokerage + stt + txn + sebi + stamp + gst
    slip = (tick_slip_pts / price) * notional
    return {"stat": stat, "slip": slip, "total": stat + slip,
            "pct": 100 * (stat + slip) / notional,
            "stat_pct": 100 * stat / notional}


def equity_cost(notional: float, slip_pct: float) -> dict:
    """Equity intraday: STT 0.025% sell, NSE txn 0.00307%, stamp 0.003% buy."""
    brokerage = 2 * min(BROK_RATE * notional, BROK_CAP)
    stt = 0.00025 * notional
    txn = 0.0000307 * notional * 2
    sebi = SEBI * notional * 2
    stamp = 0.00003 * notional
    gst = GST * (brokerage + sebi + txn)
    stat = brokerage + stt + txn + sebi + stamp + gst
    slip = slip_pct * notional
    return {"stat": stat, "slip": slip, "total": stat + slip,
            "pct": 100 * (stat + slip) / notional,
            "stat_pct": 100 * stat / notional}


def main():
    # Approximate Oct 2026 levels and lots - adjust if the live values differ.
    # The CONCLUSION is driven by the rate structure, not these exact numbers.
    cases = [
        ("NIFTY fut",      26000.0, 65,  1.0),   # 1 point of slippage
        ("BANKNIFTY fut",  58000.0, 30,  3.0),   # wider, so assume 3 points
    ]

    print("=" * 72)
    print("  COST PER ROUND TRIP, AS A FRACTION OF NOTIONAL")
    print("=" * 72)
    print(f"\n  {'instrument':16s}{'notional':>12s}{'taxes+fees':>12s}"
          f"{'slippage':>11s}{'TOTAL':>10s}{'as %':>9s}")
    print("  " + "-" * 70)

    rows = []
    for name, price, lot, slip_pts in cases:
        n = price * lot
        c = futures_cost(n, slip_pts, price)
        rows.append((name, price, lot, n, c))
        print(f"  {name:16s}{n:>12,.0f}{c['stat']:>12,.0f}"
              f"{c['slip']:>11,.0f}{c['total']:>10,.0f}{c['pct']:>8.3f}%")

    eq = equity_cost(100000.0, 0.0004)
    print(f"  {'equity intraday':16s}{100000:>12,.0f}{eq['stat']:>12,.0f}"
          f"{eq['slip']:>11,.0f}{eq['total']:>10,.0f}{eq['pct']:>8.3f}%")

    print("\n  Why futures come out cheaper despite STT being 2x higher")
    print("  (0.05% vs 0.025%): on equity the Rs 20 brokerage is 0.04% of a")
    print("  Rs 1 lakh trade, and slippage on a single stock is ~0.04%. On a")
    print("  Rs 17 lakh futures lot the same Rs 20 is 0.002%, and the bid-ask")
    print("  is a couple of points on a 58,000 index. Size dilutes the fixed")
    print("  costs, and index liquidity crushes the spread.")

    print("\n" + "=" * 72)
    print("  WHAT GROSS EDGE EACH INSTRUMENT NEEDS")
    print("=" * 72)
    # measured 1-minute ATR as % of price
    atr = {"NIFTY fut": 0.055, "BANKNIFTY fut": 0.075, "equity intraday": 0.090}
    print("\n  'cost in ATRs' is the real number: how much of a typical")
    print("  one-minute move is eaten by one round trip.\n")
    print(f"  {'instrument':16s}{'cost %':>9s}{'1-min ATR %':>14s}"
          f"{'cost in ATRs':>15s}")
    print("  " + "-" * 54)
    for name, price, lot, n, c in rows:
        a = atr[name]
        print(f"  {name:16s}{c['pct']:>8.3f}%{a:>13.3f}%{c['pct'] / a:>15.2f}")
    a = atr["equity intraday"]
    print(f"  {'equity intraday':16s}{eq['pct']:>8.3f}%{a:>13.3f}%"
          f"{eq['pct'] / a:>15.2f}")

    print("\n  NOTE: the Nifty and Bank Nifty ATR figures above are estimates.")
    print("  Nifty's is anchored to our 12 months of 1-minute data; Bank")
    print("  Nifty's is scaled from its usual ~1.4x relative volatility and")
    print("  has NOT been measured. That is exactly what needs testing.")
    print()


if __name__ == "__main__":
    main()
