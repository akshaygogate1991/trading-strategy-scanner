"""Swing highs / lows from the colour-coded Nifty sheet, as a testable rule.

The "Nifty D" sheet of index chart.xlsx (daily Nifty, 2000 on) marks closes by hand:
blue = swing high, orange = swing low, and they alternate 95 % of the time - a
zigzag. Marking a top needs the fall that follows it, so the marks cannot be
traded as they are. Here the same idea is made mechanical, known at each close:

  zigzag X%   a swing high is confirmed once the close is X% below the highest close
              since the last swing low (and the other way round). Up-leg = long.
  Dow trend   on the confirmed zigzag points: long after a higher swing low, out
              (or short) after a lower swing high.
  day marks   the sheet's 2023+ marks on single days: a higher low (low above
              yesterday's) = long the next day, a lower high = out / short.
  hand marks  the sheet's own blue / orange marks acted on 1 or 2 days late - a
              reference only (the marks themselves use hindsight).

Each signal is known at a close and traded at the next open; 0.015 % per side.
Checks: before / from 2020, t-stat of the monthly difference vs buy and hold, and
a shift test (the same positions slid in time over the returns; how often does
that do as well?). Many versions are tested, so one good number can be luck.

Nifty comes from the sheet (2000 on, with the hand marks). BankNifty (2007 on) and
Sensex (1997 on) come from Yahoo daily data - the sheet's own tabs for them start in
2018 and Dec 2023, too short for this.

Run: python zigzag_swing_test.py [nifty] [banknifty] [sensex] ["path/to/index chart.xlsx"]
     (no index named = all three)  -> logs/zigzag_swing.txt
The sheet is read once into swing_data/nifty_sheet.csv (needs openpyxl that time);
Yahoo data is cached in swing_data/<index>_daily.csv.
"""
from __future__ import annotations

import io
import math
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
CACHE = HERE / "swing_data" / "nifty_sheet.csv"
DEFAULT_XLSX = Path.home() / "Downloads" / "index chart.xlsx"
YAHOO = {"banknifty": "^NSEBANK", "sensex": "^BSESN"}
PERIODS = [("2000-07", "2000", "2007"), ("2008-13", "2008", "2013"), ("2014-19", "2014", "2019"),
           ("2020-26", "2020", "2026")]
COST = 0.00015
SPLIT = pd.Timestamp("2020-01-01")
RNG = np.random.default_rng(7)


def read_sheet(path: Path) -> pd.DataFrame:
    """Daily OHLC + the close / high / low fill colours from the 'Nifty D' sheet."""
    import openpyxl
    # the workbook's table parts have a broken range that openpyxl refuses: drop them
    src = zipfile.ZipFile(path)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as out:
        for item in src.infolist():
            if item.filename.startswith("xl/tables/"):
                continue
            data = src.read(item.filename)
            if item.filename.startswith("xl/worksheets/") and item.filename.endswith(".xml"):
                data = re.sub(rb"<tableParts.*?</tableParts>|<tableParts[^>]*/>", b"", data, flags=re.S)
            if item.filename.endswith(".rels"):
                data = re.sub(rb"<Relationship[^>]*tables/[^>]*/>", b"", data)
            out.writestr(item, data)
    ws = openpyxl.load_workbook(buf, data_only=True)["Nifty D"]

    def fill(c):
        f = c.fill
        if not f or not f.fill_type:
            return ""
        fc = f.fgColor
        k = fc.rgb if fc.type == "rgb" else f"t{fc.theme}"
        return "" if k in ("t0", "t1", "00000000", "FFFFFFFF") else k

    rows = []
    for r in range(1, ws.max_row + 1):
        d = ws.cell(r, 1).value
        v = [ws.cell(r, k).value for k in range(4, 8)]
        if not hasattr(d, "year") or not all(isinstance(x, (int, float)) for x in v):
            continue
        rows.append({"Date": pd.Timestamp(d).normalize(), "Open": v[0], "High": v[1], "Low": v[2],
                     "Close": v[3], "close_mark": fill(ws.cell(r, 7)),
                     "high_mark": fill(ws.cell(r, 5)), "low_mark": fill(ws.cell(r, 6))})
    df = pd.DataFrame(rows).drop_duplicates("Date").set_index("Date").sort_index()
    return df


def read_yahoo(ticker: str) -> pd.DataFrame:
    import yfinance as yf
    d = yf.download(ticker, period="max", interval="1d", auto_adjust=False, progress=False)
    d.columns = d.columns.get_level_values(0)
    d = d[["Open", "High", "Low", "Close"]].dropna()
    d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize().rename("Date")
    return d.assign(close_mark="", high_mark="", low_mark="")


def load(name: str = "nifty", xlsx: Path = DEFAULT_XLSX) -> pd.DataFrame:
    cache = CACHE if name == "nifty" else HERE / "swing_data" / f"{name}_daily.csv"
    if not cache.exists():
        cache.parent.mkdir(exist_ok=True)
        (read_sheet(xlsx) if name == "nifty" else read_yahoo(YAHOO[name])).to_csv(cache)
    df = pd.read_csv(cache, parse_dates=["Date"], index_col="Date").fillna("")
    # old rows with a missing or broken open: use the previous close
    bad = (df.Open <= 0) | (df.Open > df.High * 1.001) | (df.Open < df.Low * 0.999)
    df.loc[bad, "Open"] = df.Close.shift()[bad]
    return df.dropna(subset=["Open"])


def zigzag(close: np.ndarray, x: float):
    """Confirmed swings. Returns (leg, pivots): leg[t] = +1 up-leg / -1 down-leg as known
    at close t; pivots = list of (confirmed_at, pivot_index, +1 high / -1 low)."""
    n = len(close)
    leg = np.zeros(n, int)
    pivots = []
    direction, ext_i = 0, 0
    lo_i = hi_i = 0
    for t in range(n):
        c = close[t]
        if direction == 0:                       # wait for the first X% move either way
            if c >= close[lo_i] * (1 + x):
                direction, ext_i = 1, t
                pivots.append((t, lo_i, -1))
            elif c <= close[hi_i] * (1 - x):
                direction, ext_i = -1, t
                pivots.append((t, hi_i, 1))
            else:
                lo_i = t if c < close[lo_i] else lo_i
                hi_i = t if c > close[hi_i] else hi_i
        elif direction == 1:
            if c > close[ext_i]:
                ext_i = t
            elif c <= close[ext_i] * (1 - x):
                pivots.append((t, ext_i, 1))
                direction, ext_i = -1, t
        else:
            if c < close[ext_i]:
                ext_i = t
            elif c >= close[ext_i] * (1 + x):
                pivots.append((t, ext_i, -1))
                direction, ext_i = 1, t
        leg[t] = direction
    return leg, pivots


def dow(close: np.ndarray, pivots, n: int) -> np.ndarray:
    """+1 after a higher swing low, -1 after a lower swing high, else keep the last state."""
    out = np.zeros(n, int)
    last = {1: None, -1: None}
    state, k = 0, 0
    piv = sorted(pivots)
    for t in range(n):
        while k < len(piv) and piv[k][0] == t:
            _, i, kind = piv[k]
            prev = last[kind]
            if prev is not None:
                if kind == -1 and close[i] > close[prev]:
                    state = 1
                elif kind == 1 and close[i] < close[prev]:
                    state = -1
            last[kind] = i
            k += 1
        out[t] = state
    return out


def daily_returns(df: pd.DataFrame, signal: np.ndarray) -> pd.Series:
    """signal[t] known at close t -> held from the open of t+1."""
    o, c = df.Open.to_numpy(), df.Close.to_numpy()
    pos = np.r_[0, signal[:-1]].astype(float)
    prev = np.r_[0, pos[:-1]]
    pc = np.r_[c[0], c[:-1]]
    same = pos == prev
    r = np.where(same, pos * (c / pc - 1), prev * (o / pc - 1) + pos * (c / o - 1))
    r -= np.abs(pos - prev) * COST
    return pd.Series(r, index=df.index)


def summary(r: pd.Series, bench: pd.Series, pos: np.ndarray) -> dict:
    years = len(r) / 252
    eq = (1 + r).cumprod()
    m = (1 + r).groupby(r.index.to_period("M")).prod() - 1
    mb = (1 + bench).groupby(bench.index.to_period("M")).prod() - 1
    diff = m - mb
    halves = {}
    for name, part in (("1st", r[r.index < SPLIT]), ("2nd", r[r.index >= SPLIT])):
        halves[name] = ((1 + part).prod() ** (252 / len(part)) - 1) * 100
    p = pd.Series(pos)
    return {"CAGR %": (eq.iloc[-1] ** (1 / years) - 1) * 100,
            "MaxDD %": (eq / eq.cummax() - 1).min() * 100,
            "Sharpe": r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else float("nan"),
            "CAGR 1st": halves["1st"], "CAGR 2nd": halves["2nd"],
            "in market %": (p != 0).mean() * 100,
            "trades/yr": (p.diff().abs() > 0).sum() / years,
            "t vs hold": diff.mean() / (diff.std() / math.sqrt(len(diff))) if diff.std() > 0 else float("nan")}


def shift_test(df, signal, sharpe, tries=500) -> float:
    """Slide the same positions to a random point in time: % of slides with a Sharpe as good."""
    n, good = len(signal), 0
    for _ in range(tries):
        k = RNG.integers(252, n - 252)
        r = daily_returns(df, np.roll(signal, k))
        good += r.mean() / r.std() * math.sqrt(252) >= sharpe
    return good / tries * 100


def match_marks(df, pivots, tol=1):
    """How well do mechanical swing points line up with the hand marks (within tol days)?"""
    hand_hi = np.flatnonzero(df.close_mark.to_numpy() == "t4")
    hand_lo = np.flatnonzero(df.close_mark.to_numpy() == "t9")
    mech_hi = np.array([i for _, i, k in pivots if k == 1])
    mech_lo = np.array([i for _, i, k in pivots if k == -1])

    def hit(a, b):
        if len(a) == 0 or len(b) == 0:
            return 0.0
        return np.mean([np.min(np.abs(b - i)) <= tol for i in a]) * 100
    return {"hand found %": (hit(hand_hi, mech_hi) * len(hand_hi) + hit(hand_lo, mech_lo) * len(hand_lo))
            / (len(hand_hi) + len(hand_lo)),
            "mech on a mark %": (hit(mech_hi, hand_hi) * len(mech_hi) + hit(mech_lo, hand_lo) * len(mech_lo))
            / max(len(mech_hi) + len(mech_lo), 1),
            "swings/yr": len(pivots) / (len(df) / 252)}


def hand_signal(df, lag):
    """Up-leg after an orange (low) mark, down-leg after a blue (high) mark, acted on `lag` days late."""
    mark = df.close_mark.to_numpy()
    s = np.zeros(len(df), int)
    state = 0
    for t in range(len(df)):
        if t - lag >= 0:
            if mark[t - lag] == "t9":
                state = 1
            elif mark[t - lag] == "t4":
                state = -1
        s[t] = state
    return s


def run(name: str, xlsx: Path) -> list[str]:
    df = load(name, xlsx)
    close = df.Close.to_numpy()
    n = len(df)
    hold = np.ones(n, int)
    bench = daily_returns(df, hold)
    marked = (df.close_mark != "").any()

    versions = [("buy and hold", hold)]
    lines = [f"\n##### {name.upper()} daily ({'the sheet' if name == 'nifty' else 'Yahoo'}): "
             f"{df.index[0].date()} to {df.index[-1].date()}, {n} days"]
    if marked:
        lines += [f"Hand marks: {(df.close_mark == 't4').sum()} swing highs (blue), "
                  f"{(df.close_mark == 't9').sum()} swing lows (orange)", "",
                  "=== How close is a mechanical zigzag to the hand marks? (within 1 day) ==="]
    for x in (0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05):
        leg, piv = zigzag(close, x)
        if marked:
            mm = match_marks(df, piv)
            lines.append(f"  zigzag {x * 100:>4.2g}%: {mm['swings/yr']:5.1f} swings/yr, "
                         f"hand marks found {mm['hand found %']:4.1f}%, mechanical points on a hand mark "
                         f"{mm['mech on a mark %']:4.1f}%")
        if x in (0.01, 0.02, 0.03, 0.05):
            versions.append((f"zigzag {x * 100:g}% long / flat", np.where(leg > 0, 1, 0)))
            versions.append((f"zigzag {x * 100:g}% long / short", leg))
            d = dow(close, piv, n)
            versions.append((f"Dow {x * 100:g}% long / flat", np.where(d > 0, 1, 0)))
            versions.append((f"Dow {x * 100:g}% long / short", d))

    hl = (df.Low > df.Low.shift()).to_numpy()
    lh = (df.High < df.High.shift()).to_numpy()
    day = np.where(hl & ~lh, 1, np.where(lh & ~hl, -1, 0))
    versions.append(("day marks: higher low long, lower high short", day))
    versions.append(("day marks: higher low long, else flat", np.where(day > 0, 1, 0)))
    for lag in ((1, 2) if marked else ()):
        s = hand_signal(df, lag)
        versions.append((f"[ref] hand marks {lag} day late, long / short", s))
        versions.append((f"[ref] hand marks {lag} day late, long / flat", np.where(s > 0, 1, 0)))

    rows, per = [], {}
    for vname, sig in versions:
        r = daily_returns(df, sig)
        s = summary(r, bench, np.r_[0, sig[:-1]])
        s["shift as good %"] = shift_test(df, sig, s["Sharpe"]) if vname != "buy and hold" else float("nan")
        rows.append({"version": vname, **s})
        if "long / flat" in vname or vname == "buy and hold":
            # the same rule one day later too: a real edge should not vanish with a day's delay
            late = daily_returns(df, np.r_[0, sig[:-1]])
            for label, rr in ((vname, r), (vname + ", 1 day late", late)):
                per[label] = {p: ((1 + rr.loc[a:b]).prod() ** (252 / len(rr.loc[a:b])) - 1) * 100
                              for p, a, b in PERIODS if len(rr.loc[a:b]) > 120}
        print(f"  {name} {vname}: CAGR {s['CAGR %']:.1f}%, Sharpe {s['Sharpe']:.2f}, "
              f"t vs hold {s['t vs hold']:+.2f}", flush=True)

    table = pd.DataFrame(rows).set_index("version")
    lines += ["", f"=== Strategies, net of {COST * 100:.3f}% per side, signal at the close, "
              f"trade at the next open ===", table.round(2).to_string(), "",
              "=== CAGR % by period (long / flat versions, on time and one day late) ===",
              pd.DataFrame(per).T.round(1).to_string()]
    return lines


def main() -> None:
    args = sys.argv[1:]
    xlsx = next((Path(a) for a in args if a.lower().endswith(".xlsx")), DEFAULT_XLSX)
    names = [a.lower() for a in args if a.lower() in ("nifty", "banknifty", "sensex")]
    lines = []
    for name in names or ["nifty", "banknifty", "sensex"]:
        lines += run(name, xlsx)
    lines += ["", "1st = before 2020, 2nd = from 2020. t vs hold = t-stat of the monthly difference vs buy and hold.",
              "shift as good % = share of 500 time-shifted copies of the same positions with a Sharpe at least "
              "as high (low = the timing matters, 50 = luck).",
              "[ref] rows use the hand marks, which are drawn with hindsight: not a tradable rule."]
    text = "\n".join(lines)
    print(text)
    (HERE / "logs").mkdir(exist_ok=True)
    (HERE / "logs" / "zigzag_swing.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
