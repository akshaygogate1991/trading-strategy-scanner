"""Three questions that decide whether the 18 SMA setup belongs in the app.

  Q1 HOLDING PERIOD - how long do these trades actually last?
     Decides whether a month-end option can survive the trade. If the typical
     hold runs into the last two weeks before expiry, theta eats the move even
     when the direction is right.

  Q2 REDUNDANCY - is 18 SMA telling us anything 18 EMA doesn't?
     Both are 18-period averages of the same closes. If they agree ~all the
     time, "both agree" is one signal counted twice, not two confirmations,
     and using it to raise conviction is self-deception.

  Q3 WEEK-OF-MONTH - does entering in week 1-2 actually help?
     Tests the proposed rule against doing nothing.

Run:  python sma18_analysis.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import sma18_backtest as B


def ema(series: pd.Series, span: int = 18) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def main() -> None:
    rows, agree, disagree = [], 0, 0
    ema_says_up_when_sma_fires = 0

    for t in B.ALL_TICKERS:
        df = B.load(t, 4.0)
        if df is None or len(df) < B.SMA_LEN + 40:
            continue
        trades = B.signals(df)
        d = df.copy()
        d["sma"] = d["Close"].rolling(B.SMA_LEN).mean()
        d["ema"] = ema(d["Close"], 18)

        for x in trades:
            x["ticker"] = t
            x["kind"] = "index" if t in B.INDEXES else "stock"
            ts = pd.Timestamp(x["date"])
            if ts in d.index:
                row = d.loc[ts]
                # what did the 18 EMA say at the same moment?
                ema_up = bool(row["Close"] > row["ema"])
                x["ema_agrees"] = ema_up
                ema_says_up_when_sma_fires += ema_up
            x["week_of_month"] = (ts.day - 1) // 7 + 1
            rows.append(x)

    if not rows:
        print("No trades — run python sma18_backtest.py first to build the cache.")
        return
    d = pd.DataFrame(rows)

    # ---------------------------------------------------------------- Q1
    print("=" * 68)
    print("Q1  HOLDING PERIOD — can a month-end option survive these trades?")
    print("=" * 68)
    h = d["days_held"]
    print(f"  trades {len(h)} | mean {h.mean():.1f} days | median {h.median():.0f} days")
    print(f"  25th/75th percentile: {h.quantile(.25):.0f} / {h.quantile(.75):.0f} days")
    for thresh in (10, 15, 20, 30):
        print(f"  longer than {thresh:2d} trading days: {100 * (h > thresh).mean():5.1f}%")
    print(f"\n  Winners  (target hit): median {d[d.outcome == 'target']['days_held'].median():.0f} days")
    print(f"  Losers (SMA exit)    : median {d[d.outcome == 'sma_exit']['days_held'].median():.0f} days")
    long_share = 100 * (h > 15).mean()
    print(f"\n  Verdict: {long_share:.0f}% of trades run beyond 15 trading days (~3 weeks).")
    if long_share > 30:
        print("  A month-end option bought in week 1-2 has ~3-4 weeks of life. A large")
        print("  share of these trades outlive the low-theta part of that window, so")
        print("  the option decays through the worst period while the trade is open.")

    # ---------------------------------------------------------------- Q2
    print("\n" + "=" * 68)
    print("Q2  REDUNDANCY — does 18 SMA add anything to 18 EMA?")
    print("=" * 68)
    if "ema_agrees" in d.columns:
        agree_pct = 100 * d["ema_agrees"].mean()
        print(f"  When the 18 SMA setup fires, the 18 EMA already says 'up' "
              f"{agree_pct:.1f}% of the time.")
        n_new = int((~d["ema_agrees"]).sum())
        print(f"  Genuinely new information: {n_new} of {len(d)} trades "
              f"({100 * n_new / len(d):.1f}%)")
        if agree_pct > 85:
            print("\n  These are effectively the SAME signal. Treating agreement between")
            print("  them as extra conviction counts one piece of information twice.")
            print("  It would make you size up on confirmation that does not exist.")
        # and does the rare disagreement perform differently?
        for flag, g in d.groupby("ema_agrees"):
            s = B.summarise(g["r"].tolist())
            label = "EMA agrees" if flag else "EMA disagrees"
            print(f"    {label:14s} n={s['n']:4d}  exp {s.get('expectancy', 0):+.3f}R  "
                  f"t={s.get('t', 0):+.2f}")

    # ---------------------------------------------------------------- Q3
    print("\n" + "=" * 68)
    print("Q3  WEEK OF MONTH — does entering in week 1-2 help?")
    print("=" * 68)
    for w, g in d.groupby("week_of_month"):
        s = B.summarise(g["r"].tolist())
        print(f"  week {int(w)}   n={s['n']:4d} | win {s.get('win_rate', 0):5.1f}% "
              f"| exp {s.get('expectancy', 0):+.3f}R | t={s.get('t', 0):+.2f}")
    early = d[d["week_of_month"] <= 2]
    late = d[d["week_of_month"] > 2]
    se, sl = B.summarise(early["r"].tolist()), B.summarise(late["r"].tolist())
    print(f"\n  week 1-2   n={se['n']:4d} exp {se.get('expectancy', 0):+.3f}R "
          f"(t={se.get('t', 0):+.2f})")
    print(f"  week 3+    n={sl['n']:4d} exp {sl.get('expectancy', 0):+.3f}R "
          f"(t={sl.get('t', 0):+.2f})")
    diff = se.get("expectancy", 0) - sl.get("expectancy", 0)
    print(f"  difference {diff:+.3f}R per trade")
    if abs(diff) < 0.1 or abs(se.get("t", 0)) < 2:
        print("  No reliable week-of-month effect. The timing rule is sound options")
        print("  practice for theta, but it is not itself an edge in the signal.")

    # ---------------------------------------------------------------- index only
    print("\n" + "=" * 68)
    print("INDEX-ONLY (what you actually want to trade), week 1-2 entries")
    print("=" * 68)
    idx = d[(d["kind"] == "index") & (d["week_of_month"] <= 2)]
    s = B.summarise(idx["r"].tolist())
    if s["n"]:
        print(f"  n={s['n']} | win {s.get('win_rate', 0):.1f}% | "
              f"exp {s.get('expectancy', 0):+.3f}R | t={s.get('t', 0):+.2f} | "
              f"total {s.get('total_R', 0):+.1f}R")
        if s["n"] < 50 or abs(s.get("t", 0)) < 2:
            print("  Sample far too small to conclude anything. This is the exact")
            print("  slice you would trade, and it is the thinnest evidence we have.")
    d.to_csv("sma18_analysis.csv", index=False)
    print("\nDetail: sma18_analysis.csv")


if __name__ == "__main__":
    main()
