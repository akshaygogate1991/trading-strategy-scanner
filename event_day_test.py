"""Big scheduled news days vs Sensex Directional and Chota Packet Sensex.

The idea from Forex Factory's economic calendar: do the strategies do better or worse on
days with a big scheduled event? The dates are the official ones (RBI, US Fed, US Bureau of
Labor Statistics, Union Budget, election results), typed in below - nothing is scraped.

Which Indian trading day an event hits:
  RBI policy        same day (decision about 10:00 IST, after the 09:35 entry)
  Union Budget      same day (speech from 11:00 IST)
  Election results  same day (India 4 Jun 2024; US 6 Nov 2024 IST); Indian exit polls -> 3 Jun 2024
  US Fed (FOMC)     the next Indian session (decision about 23:30 / 00:30 IST)
  US CPI            the next Indian session (release 18:00 / 19:00 IST)

For each kind of day: average net P&L on those days vs all other days, t of the difference,
and how often the same number of random days does at least as well (or as badly). There are
only 10-35 days of each kind, so only a big, consistent difference means anything.

Run: python event_day_test.py   -> logs/event_day.txt
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(23)

RBI = ["2023-08-10", "2023-10-06", "2023-12-08", "2024-02-08", "2024-04-05", "2024-06-07", "2024-08-08",
       "2024-10-09", "2024-12-06", "2025-02-07", "2025-04-09", "2025-06-06", "2025-08-06", "2025-10-01",
       "2025-12-05", "2026-02-06", "2026-04-08", "2026-06-05"]
BUDGET = ["2024-02-01", "2024-07-23", "2025-02-01", "2026-02-01"]
ELECTION = ["2024-06-03", "2024-06-04", "2024-11-06"]
FOMC_US = ["2023-09-20", "2023-11-01", "2023-12-13", "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
           "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18", "2025-01-29", "2025-03-19", "2025-05-07",
           "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10", "2026-01-28", "2026-03-18",
           "2026-04-29", "2026-06-17"]
CPI_US = ["2023-08-10", "2023-09-13", "2023-10-12", "2023-11-14", "2023-12-12", "2024-01-11", "2024-02-13",
          "2024-03-12", "2024-04-10", "2024-05-15", "2024-06-12", "2024-07-11", "2024-08-14", "2024-09-11",
          "2024-10-10", "2024-11-13", "2024-12-11", "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10",
          "2025-05-13", "2025-06-11", "2025-07-15", "2025-08-12", "2025-09-11", "2025-10-24", "2025-12-18",
          "2026-01-13", "2026-02-13", "2026-03-11", "2026-04-10", "2026-05-12", "2026-06-10"]


def trading_days() -> pd.DatetimeIndex:
    s = pd.read_parquet(os.path.join(HERE, "hf_data", "index", "SENSEX.parquet"), columns=["trading_day"])
    return pd.DatetimeIndex(sorted(pd.to_datetime(s["trading_day"].unique())))


def next_session(dates, days) -> list[str]:
    out = []
    for d in pd.to_datetime(dates):
        later = days[days > d]
        if len(later):
            out.append(later[0].strftime("%Y-%m-%d"))
    return out


def sensex_range() -> pd.Series:
    s = pd.read_parquet(os.path.join(HERE, "hf_data", "index", "SENSEX.parquet"))
    g = s.groupby("trading_day")
    return (g["high"].max() - g["low"].min()) / g["open"].first() * 100


def pnl(files, strategy, by_leg=False):
    t = pd.concat([pd.read_csv(os.path.join(HERE, f)) for f in files])
    t = t[t["strategy"] == strategy]
    if by_leg:
        return t.pivot_table(index="day", columns="kind", values="net", aggfunc="sum").fillna(0)
    return t.groupby("day")["net"].sum()


def compare(series: pd.Series, days: list[str], name: str, pct: bool = False) -> str:
    on = series.index.isin(days)
    a, b = series[on], series[~on]
    if len(a) < 2:
        return f"  {name:<28} {len(a)} days - too few"
    diff = a.mean() - b.mean()
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    rand = np.array([series.sample(len(a), random_state=int(RNG.integers(1e9))).mean() for _ in range(3000)])
    p_hi, p_lo = (rand >= a.mean()).mean() * 100, (rand <= a.mean()).mean() * 100
    if pct:
        return (f"  {name:<28} {len(a):>3} days  avg {a.mean():.2f}%  (other days {b.mean():.2f}%)  t {diff / se:+.2f}  "
                f"random as high {p_hi:3.0f}% / as low {p_lo:3.0f}%")
    return (f"  {name:<28} {len(a):>3} days  avg {a.mean():>+8,.0f}  (other days {b.mean():>+7,.0f})  "
            f"total {a.sum():>+9,.0f}  win {(a > 0).mean() * 100:3.0f}%  t {diff / se:+.2f}  "
            f"random as high {p_hi:3.0f}% / as low {p_lo:3.0f}%")


def main() -> None:
    days = trading_days()
    events = {"RBI policy": RBI, "Union Budget": BUDGET, "Election results": ELECTION,
              "US Fed (next session)": next_session(FOMC_US, days), "US CPI (next session)": next_session(CPI_US, days)}
    events["any of these"] = sorted(set(sum(events.values(), [])))

    rng = sensex_range()
    lines = ["=== Sensex day range (high - low, % of open) ==="]
    lines += [compare(rng, d, n, pct=True) for n, d in events.items()]

    direct = pnl(["logs/oos_export_sensexbasket_trades.csv", "export_sensexbasket_trades.csv"], "Sensex Directional")
    legs = pnl(["logs/oos_export_sensexbasket_trades.csv", "export_sensexbasket_trades.csv"], "Sensex Directional",
               by_leg=True)
    chota_x = pnl(["logs/batch/chotasx_expiry.csv"], "Chota Packet Bada Dhamaka Sensex")
    chota_all = pnl(["logs/batch/chotasx_expiry.csv", "logs/batch/chotasx_other.csv"],
                    "Chota Packet Bada Dhamaka Sensex")

    for title, s in ((f"Sensex Directional, net Rs per day ({direct.index.min()} to {direct.index.max()})", direct),
                     ("  call leg (CE sold) only", legs["CE"]), ("  put leg (PE sold) only", legs["PE"]),
                     (f"Chota Packet Sensex, all days ({chota_all.index.min()} to {chota_all.index.max()})", chota_all),
                     ("Chota Packet Sensex, expiry days only (the live plan)", chota_x)):
        lines += ["", f"=== {title} ==="] + [compare(s, d, n) for n, d in events.items()]

    lines += ["", "=== Each event day: Sensex range %, Directional Rs, Chota (all days) Rs ==="]
    for n, d in events.items():
        if n == "any of these" or n.startswith("US CPI"):
            continue
        for day in d:
            lines.append(f"  {n:<22} {day}  range {rng.get(day, np.nan):5.2f}%  Directional "
                         f"{direct.get(day, np.nan):>+8,.0f}  Chota {chota_all.get(day, np.nan):>+8,.0f}")
    lines += ["", "t = event days vs other days (Welch). 'random as high / as low' = share of random sets of the "
              "same number of days with an average at least as high / as low: under 5 % = unusual, not luck."]
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(HERE, "logs", "event_day.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
