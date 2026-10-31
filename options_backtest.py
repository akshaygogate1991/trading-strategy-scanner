"""Intraday Nifty WEEKLY OPTIONS backtest — buy an ATM call or put on a signal.

Why options: every strategy so far made between -2 and +2 Nifty points gross,
and futures cost ~16 points a round trip. An option's costs are charged on the
PREMIUM, not the notional, so one round trip costs ~1-2 premium points, i.e.
roughly 2-4 Nifty points at delta 0.5. That is the lowest bar of any instrument.

What options do NOT change: they only earn delta x the underlying move, minus
time decay. A signal with zero gross on the index still earns ~zero gross on
the option, then pays theta on top. Options lower the cost wall; they do not
create an edge. This script tells you whether a given signal clears the LOWER
wall.

THE MODEL (read before trusting any number):
  Angel One deletes expired contracts, so real historical option prices do not
  exist. Each option is PRICED with Black-Scholes from:
    * the Nifty index 1-minute candles in nifty_months/ (12 months, verified)
    * IV = the PREVIOUS day's India VIX close (no lookahead), x --iv-mult
    * the nearest weekly expiry (Tuesday, moved to the previous trading day
      when Tuesday is a holiday), never the same-day expiry unless asked
    * strike = nearest 50, i.e. at-the-money at entry
  VIX is a 30-day implied vol; a weekly ATM option's IV differs from it, and
  intraday IV moves are ignored. Run option_model_check.py to measure how far
  this model is from real option prices and set --iv-mult from that.

Rules carried over from every earlier test:
  * random-entry control with identical exits and the same CALL/PUT mix
  * gross and net reported separately; t is computed on GROSS only
  * pessimistic tie-break: if one bar touches both stop and target, stop wins
  * intraday only: no entries after --last-entry, everything closed 15:20

Signals:
  rsi    dual RSI (1-min + 5-min, OB 80 / OS 20) from rsi_month_test.py
  ema    18 EMA trend on 5-min, the live app's position+slope rule, on flips
  file   your own CSV with columns ts,side (side = CALL/PUT) — any strategy
         you can express as timestamps can be tested without editing this file

Run:  python options_backtest.py --signal ema
      python options_backtest.py --signal rsi --grid
      python options_backtest.py --signal file --file my_signals.csv --stop 40 --target 20
      python options_backtest.py --signal ema --iv-mult 1.1 --slip 1.0
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import os

import numpy as np
import pandas as pd

import rsi_month_test as R

HERE = os.path.dirname(os.path.abspath(__file__))
VIX_FILES = [os.path.join(HERE, "vix_data", f)
             for f in ("nifty_vix_8y.csv", "nifty_vix_17y.csv")]

STRIKE_STEP = 50
RISK_FREE = 0.065
LOT_FALLBACK = 65
EXPIRY_WEEKDAY = 1               # Tuesday (NSE moved Nifty weeklies there, Sep 2025)
MIN_PER_YEAR = 365 * 24 * 60     # calendar time, so weekends decay too

# Option charges (zerodha.com/charges, checked 14 Oct 2026 — re-verify).
# All on PREMIUM turnover. GST is on fees only, not on STT or stamp.
BROKERAGE_PER_LEG = 20.0
STT_SELL = 0.0015
EXCH_TXN = 0.0003553
STAMP_BUY = 0.00003
SEBI_FEE = 0.000001
GST_RATE = 0.18

STOP_GRID = [0.20, 0.40, 0.60]
TARGET_GRID = [0.10, 0.20, 0.30, 0.50, 0.80, None]


# ---------------------------------------------------------------- pricing
def ncdf(x):
    """Standard normal CDF, vectorised (Abramowitz & Stegun 7.1.26, err < 1.5e-7).
    Avoids a scipy dependency on the machine that runs this."""
    x = np.asarray(x, dtype=float)
    z = np.abs(x) / math.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * z)
    poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
             - 0.284496736) * t + 0.254829592) * t
    erf = 1.0 - poly * np.exp(-z * z)
    return 0.5 * (1.0 + np.sign(x) * erf)


def bs_price(S, K, T, iv, kind):
    """kind +1 = call, -1 = put. Arrays broadcast."""
    S = np.asarray(S, dtype=float)
    T = np.maximum(np.asarray(T, dtype=float), 1e-9)
    sd = iv * np.sqrt(T)
    d1 = (np.log(S / K) + (RISK_FREE + 0.5 * iv * iv) * T) / sd
    d2 = d1 - sd
    disc = K * np.exp(-RISK_FREE * T)
    call = S * ncdf(d1) - disc * ncdf(d2)
    put = disc * ncdf(-d2) - S * ncdf(-d1)
    return np.where(kind > 0, call, put)


def bs_delta(S, K, T, iv, kind):
    T = np.maximum(np.asarray(T, dtype=float), 1e-9)
    d1 = (np.log(S / K) + (RISK_FREE + 0.5 * iv * iv) * T) / (iv * np.sqrt(T))
    return np.where(kind > 0, ncdf(d1), ncdf(d1) - 1.0)


def charges_rs(buy_prem, sell_prem, lot):
    """Statutory + broker charges in rupees for one lot, round trip."""
    buy, sell = buy_prem * lot, sell_prem * lot
    brok = 2 * BROKERAGE_PER_LEG
    txn = EXCH_TXN * (buy + sell)
    sebi = SEBI_FEE * (buy + sell)
    gst = GST_RATE * (brok + txn + sebi)
    return brok + STT_SELL * sell + txn + STAMP_BUY * buy + sebi + gst


# ---------------------------------------------------------------- data
def load_vix() -> pd.Series:
    for path in VIX_FILES:
        if os.path.exists(path):
            v = pd.read_csv(path, index_col=0, parse_dates=True)["VIX"].dropna()
            return v.sort_index()
    raise SystemExit("No vix_data/*.csv. Run:  python vix_rank_test.py")


def weekly_expiries(days: list[dt.date]) -> list[dt.date]:
    """Tuesday of every week; if Tuesday did not trade (holiday), the last
    trading day before it in that week. Weeks past the data keep the plain
    Tuesday, since there is no calendar to check them against."""
    traded = set(days)
    last = max(days)
    out = []
    week = days[0] - dt.timedelta(days=days[0].weekday())
    while week <= last + dt.timedelta(days=14):
        e = week + dt.timedelta(days=EXPIRY_WEEKDAY)
        if e <= last:
            while e not in traded and e >= week:
                e -= dt.timedelta(days=1)
            if e < week:              # whole week missing from the data
                week += dt.timedelta(days=7)
                continue
        out.append(e)
        week += dt.timedelta(days=7)
    return out


def overnight_weight(m1: pd.DataFrame) -> float:
    """How much variance one overnight gap carries, in trading-minute units.

    Measured from the data itself: var(open / previous close) against
    var(close / open) for the same days, scaled to a 375-minute session.
    """
    g = m1.groupby(m1.index.date).agg(o=("Open", "first"), c=("Close", "last"))
    on = np.log(g["o"] / g["c"].shift(1)).dropna()
    intra = np.log(g["c"] / g["o"])
    return float(375.0 * on.var() / intra.var())


def prepare(m1: pd.DataFrame, vix: pd.Series, iv_mult: float,
            allow_expiry_day: bool, gap_units: float) -> pd.DataFrame:
    """Per-bar inputs for pricing.

    TIME TO EXPIRY — the bug the first version had. Calendar time charges a
    2-hour intraday hold only 2/24 of a day's decay, while all the price
    movement an option buyer collects happens in market hours. That made random
    intraday option buying look PROFITABLE (+1.7 pts gross). Real premiums
    decay with the variance still to come, not with the clock.

    So time is counted in VARIANCE units: each trading minute is 1 unit, each
    overnight gap is `gap_units` units. The total variance to expiry is matched
    to VIX at entry (VIX is quoted on calendar time), then runs down in
    proportion to units remaining: T(t) = T_cal(entry) * U(t) / U(entry).
    """
    d = m1[["Open", "High", "Low", "Close"]].copy()
    d["day"] = d.index.date
    d["mod"] = d.index.hour * 60 + d.index.minute           # minute of day
    d["tmin"] = d.index.values.astype("datetime64[m]").astype(np.int64)
    d["ym"] = d.index.to_period("M").astype(str)

    days = sorted(set(d["day"]))
    # IV for a day = the last VIX close strictly BEFORE that day
    vdates = np.array([x.date() for x in vix.index])
    iv_day = {}
    for day in days:
        j = np.searchsorted(vdates, day) - 1               # strictly earlier
        iv_day[day] = float(vix.iloc[j]) / 100.0 * iv_mult if j >= 0 else np.nan
    exps = weekly_expiries(days)
    exp_day = {}
    for day in days:
        nxt = [e for e in exps if (e >= day if allow_expiry_day else e > day)]
        exp_day[day] = nxt[0] if nxt else None
    # trading calendar: the data's days, then plain weekdays past its end
    cal = list(days)
    x = days[-1]
    while x < days[-1] + dt.timedelta(days=21):
        x += dt.timedelta(days=1)
        if x.weekday() < 5:
            cal.append(x)
    cal_ix = {c: i for i, c in enumerate(cal)}
    n_after = {day: (cal_ix[exp_day[day]] - cal_ix[day]) if exp_day[day] in cal_ix
               else np.nan for day in days}
    # variance units from this bar to 15:30 on expiry day
    d["units"] = (930 - d["mod"]) + d["day"].map(n_after) * (375 + gap_units)

    d["iv"] = d["day"].map(iv_day)
    d["expiry"] = d["day"].map(exp_day)
    d["exp_min"] = [np.int64(np.datetime64(dt.datetime.combine(e, dt.time(15, 30)), "m")
                             .astype(np.int64)) if e else -1
                    for e in d["expiry"]]
    return d


# ---------------------------------------------------------------- signals
def _five_min_state(m1, m5, fn):
    """Apply fn to 5-min candles, then shift(1): a 5-min value is only known
    once its candle has closed. Forward-fill onto the 1-minute index."""
    return fn(m5).shift(1).reindex(m1.index, method="ffill")


def sig_ema(m1, m5, args):
    def trend(m):
        c = m["Close"]
        e = c.ewm(span=18, adjust=False).mean()
        up = (c > e) & (e > e.shift(5))                     # app.trend_18ema
        dn = (c < e) & (e < e.shift(5))
        return pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=m.index)
    st = _five_min_state(m1, m5, trend).fillna(0).astype(int)
    flip = (st != st.shift(1)) & (st != 0)
    return [(ts, "CALL" if v > 0 else "PUT") for ts, v in st[flip].items()]


def sig_rsi(m1, m5, args):
    df = R.prepare(m1, m5, R.RSI_LEN)
    hi, lo = R.signals(df, 80, 20)
    a, b = ("PUT", "CALL") if args.reverse else ("CALL", "PUT")
    return ([(df.index[i], a) for i in hi] + [(df.index[i], b) for i in lo])


def sig_file(m1, m5, args):
    if not args.file:
        raise SystemExit("--signal file needs --file path.csv (columns ts,side)")
    f = pd.read_csv(args.file)
    f["ts"] = pd.to_datetime(f["ts"])
    side = f["side"].astype(str).str.upper().replace({"1": "CALL", "-1": "PUT",
                                                       "BUY": "CALL", "SELL": "PUT"})
    return list(zip(f["ts"], side))


SIGNALS = {"ema": sig_ema, "rsi": sig_rsi, "file": sig_file}


# ---------------------------------------------------------------- simulate
def eligible_mask(d, args) -> np.ndarray:
    first = _hhmm(args.first_entry)
    last = _hhmm(args.last_entry)
    nxt_same_day = np.r_[d["day"].values[1:] == d["day"].values[:-1], False]
    return ((d["mod"].values >= first) & (d["mod"].values <= last)
            & np.isfinite(d["iv"].values) & (d["exp_min"].values > 0)
            & np.isfinite(d["units"].values)
            & nxt_same_day)


def _hhmm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def paths(d, pos: np.ndarray, kind: np.ndarray, args):
    """Price every trade's option minute by minute, all trades at once.

    Returns a dict of 2-D arrays (trades x minutes held): mid premium at the
    close, the worst and best premium inside each bar, a validity mask.
    """
    H = args.max_hold
    n = len(d)
    close, high, low = (d[c].values for c in ("Close", "High", "Low"))
    day, mod, tmin = d["day"].values, d["mod"].values, d["tmin"].values
    iv = d["iv"].values[pos]
    exp_min = d["exp_min"].values[pos]

    S0 = close[pos]
    K = np.round(S0 / STRIKE_STEP) * STRIKE_STEP
    T0 = (exp_min - tmin[pos]) / MIN_PER_YEAR
    p0 = bs_price(S0, K, T0, iv, kind)
    delta0 = bs_delta(S0, K, T0, iv, kind)

    idx = pos[:, None] + 1 + np.arange(H)[None, :]
    idx_c = np.minimum(idx, n - 1)
    valid = ((idx < n) & (day[idx_c] == day[pos][:, None])
             & (mod[idx_c] <= _hhmm(args.square_off)))
    valid &= np.cumprod(valid, axis=1).astype(bool)          # prefix only

    units = d["units"].values
    T = T0[:, None] * units[idx_c] / units[pos][:, None]
    k_, iv_, kd = K[:, None], iv[:, None], kind[:, None]
    mid = bs_price(close[idx_c], k_, T, iv_, kd)
    p_hi = bs_price(high[idx_c], k_, T, iv_, kd)
    p_lo = bs_price(low[idx_c], k_, T, iv_, kd)
    best = np.where(kd > 0, p_hi, p_lo)
    worst = np.where(kd > 0, p_lo, p_hi)
    return dict(pos=pos, kind=kind, S0=S0, K=K, p0=p0, delta0=delta0,
                idx=idx_c, valid=valid, mid=mid, best=best, worst=worst)


def exits(P, stop, target, args, lot):
    """Apply one stop/target rule (fractions of entry premium) to priced paths."""
    p0, valid = P["p0"], P["valid"]
    H = valid.shape[1]
    L = valid.sum(axis=1)
    keep = (L > 0) & (p0 > 1.0)
    big = H + 1

    if stop is not None:
        hit = valid & (P["worst"] <= p0[:, None] * (1 - stop))
        ks = np.where(hit.any(1), hit.argmax(1), big)
    else:
        ks = np.full(len(p0), big)
    if target is not None:
        hit = valid & (P["best"] >= p0[:, None] * (1 + target))
        kt = np.where(hit.any(1), hit.argmax(1), big)
    else:
        kt = np.full(len(p0), big)

    last = np.maximum(L - 1, 0)
    rows = np.arange(len(p0))
    is_stop = (ks <= kt) & (ks < big)                         # tie -> stop
    is_tgt = ~is_stop & (kt < big)
    k_exit = np.where(is_stop, ks, np.where(is_tgt, kt, last))
    exit_mid = np.where(is_stop, p0 * (1 - (stop or 0)),
                        np.where(is_tgt, p0 * (1 + (target or 0)),
                                 P["mid"][rows, last]))
    reason = np.where(is_stop, "stop", np.where(is_tgt, "target",
                      np.where(L >= args.max_hold, "maxhold", "eod")))

    gross = exit_mid - p0
    buy_fill = p0 + args.slip
    sell_fill = np.maximum(exit_mid - args.slip, 0.05)
    chg = charges_rs(buy_fill, sell_fill, lot) / lot
    net = sell_fill - buy_fill - chg
    S_exit = P["S_path"][rows, k_exit]
    return dict(keep=keep, gross=gross, net=net, cost=2 * args.slip + chg,
                reason=reason, held=k_exit + 1, exit_mid=exit_mid, S_exit=S_exit)


def run(d, entries, args, lot, stop, target, P=None):
    if P is None:
        P = paths(d, entries[0], entries[1], args)
        P["S_path"] = d["Close"].values[P["idx"]]
    X = exits(P, stop, target, args, lot)
    k = X["keep"]
    rows = pd.DataFrame({
        "ts": d.index[P["pos"][k]],
        "side": np.where(P["kind"][k] > 0, "CALL", "PUT"),
        "ym": d["ym"].values[P["pos"][k]],
        "spot": P["S0"][k], "strike": P["K"][k],
        "premium": P["p0"][k], "delta": np.abs(P["delta0"][k]),
        "exit_prem": X["exit_mid"][k], "reason": X["reason"][k],
        "minutes": X["held"][k],
        "gross_pts": X["gross"][k], "net_pts": X["net"][k],
        "cost_pts": X["cost"][k],
        "und_pts": P["kind"][k] * (X["S_exit"][k] - P["S0"][k]),
    })
    rows["net_rs"] = rows["net_pts"] * lot
    return rows, P


def stats(r: pd.DataFrame) -> dict:
    if not len(r):
        return dict(n=0, win=0, gross=0, net=0, t=0, rs=0, gross_pct=0)
    g = r["gross_pts"].values
    sd_ = g.std(ddof=1) if len(g) > 1 else 0.0
    return dict(n=len(r), win=100 * float((r["net_pts"] > 0).mean()),
                gross=float(g.mean()), net=float(r["net_pts"].mean()),
                gross_pct=100 * float((r["gross_pts"] / r["premium"]).mean()),
                t=float(g.mean() / (sd_ / math.sqrt(len(g)))) if sd_ else 0.0,
                rs=float(r["net_rs"].sum()))


def random_entries(d, elig_pos, sides, rng):
    pos = np.sort(rng.choice(elig_pos, size=len(sides), replace=False))
    return pos, rng.permutation(sides)


def control(d, elig_pos, kinds, args, lot, rules, draws):
    """Gross per trade for random entries, for every rule at once."""
    rng = np.random.default_rng(17)
    out = {r: [] for r in rules}
    for _ in range(draws):
        pos, kd = random_entries(d, elig_pos, kinds, rng)
        P = paths(d, pos, kd, args)
        P["S_path"] = d["Close"].values[P["idx"]]
        for r in rules:
            rows, _ = run(d, None, args, lot, r[0], r[1], P=P)
            out[r].append(rows["gross_pts"].mean())
    return {r: np.array(v) for r, v in out.items()}


def pct_vs(value, dist):
    return 100.0 * float((dist < value).mean())


def fmt_rule(stop, target):
    s = f"-{stop * 100:.0f}%" if stop is not None else "none"
    t = f"+{target * 100:.0f}%" if target is not None else "none"
    return f"{s:>6s} / {t:<6s}"


# ---------------------------------------------------------------- report
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--signal", choices=list(SIGNALS), default="ema")
    ap.add_argument("--file", default=None)
    ap.add_argument("--reverse", action="store_true", help="rsi: >80 PUT, <20 CALL")
    ap.add_argument("--stop", type=float, default=60.0,
                    help="premium stop %% (live app: 60)")
    ap.add_argument("--target", type=float, default=80.0,
                    help="premium target %%, 0 = none (live app shows 80)")
    ap.add_argument("--grid", action="store_true", help="sweep stops x targets")
    ap.add_argument("--max-hold", type=int, default=180, help="minutes")
    ap.add_argument("--first-entry", default="09:20")
    ap.add_argument("--last-entry", default="14:30")
    ap.add_argument("--square-off", default="15:20")
    ap.add_argument("--slip", type=float, default=0.5,
                    help="premium points lost to the spread, EACH side")
    ap.add_argument("--iv-mult", type=float, default=1.0,
                    help="scale VIX to weekly ATM IV (see option_model_check.py)")
    ap.add_argument("--gap-units", type=float, default=None,
                    help="variance of one overnight gap, in trading minutes "
                         "(default: measured from the data)")
    ap.add_argument("--allow-expiry-day", action="store_true",
                    help="trade the option that expires the same day")
    ap.add_argument("--lot", type=int, default=None)
    ap.add_argument("--draws", type=int, default=50)
    args = ap.parse_args()

    loaded = R.load_all()
    if loaded is None:
        return
    m1, m5 = loaded
    lot = args.lot or LOT_FALLBACK
    gap = args.gap_units if args.gap_units is not None else overnight_weight(m1)
    d = prepare(m1, load_vix(), args.iv_mult, args.allow_expiry_day, gap)

    raw = SIGNALS[args.signal](m1, m5, args)
    elig = eligible_mask(d, args)
    loc = d.index.get_indexer([ts for ts, _ in raw])
    pos, kind = [], []
    for p, (_, side) in zip(loc, raw):
        if p >= 0 and elig[p] and side in ("CALL", "PUT"):
            pos.append(p)
            kind.append(1 if side == "CALL" else -1)
    pos, kind = np.array(pos, dtype=np.int64), np.array(kind, dtype=np.int64)
    order = np.argsort(pos)
    pos, kind = pos[order], kind[order]
    elig_pos = np.flatnonzero(elig)

    months = sorted(d["ym"].unique())
    print(f"NIFTY WEEKLY OPTIONS — buy ATM {('call/put')} on signal '{args.signal}'"
          + (" (reversed)" if args.reverse else ""))
    print(f"  {len(d):,} one-minute bars, {months[0]} to {months[-1]} | lot {lot}")
    print(f"  IV = previous day's India VIX x {args.iv_mult:.2f} | slippage "
          f"{args.slip:.2f} pts/side | entries {args.first_entry}-{args.last_entry},"
          f" out by {args.square_off}, max {args.max_hold} min")
    print(f"  decay clock: trading minutes + {gap:.0f} minute-equivalents per "
          f"overnight gap (measured)")
    print(f"  {len(raw)} raw signals -> {len(pos)} inside the entry window "
          f"({int((kind > 0).sum())} CALL, {int((kind < 0).sum())} PUT)")
    if len(pos) < 20:
        print("\n  Too few trades to say anything.")
        return

    stop = args.stop / 100 if args.stop else None
    target = args.target / 100 if args.target else None
    rows, P = run(d, (pos, kind), args, lot, stop, target)

    # ---- the cost wall, in this instrument's units
    cost = rows["cost_pts"].mean()
    dlt = rows["delta"].mean()
    print(f"\n  Average entry premium {rows['premium'].mean():.1f} pts "
          f"(delta {dlt:.2f}, {rows['premium'].mean() * lot:,.0f} Rs/lot)")
    print(f"  Round-trip cost {cost:.2f} premium pts "
          f"(Rs {cost * lot:.0f}/lot) = ~{cost / dlt:.1f} Nifty pts at that delta")
    print("  ^ the signal's GROSS must beat this. Futures needed ~16-17 Nifty pts.")

    rules = ([(s, t) for s in STOP_GRID for t in TARGET_GRID] if args.grid
             else [(stop, target)])
    if (stop, target) not in rules:
        rules.append((stop, target))
    print(f"\n  running {args.draws} random-entry controls...", flush=True)
    ctrl = control(d, elig_pos, kind, args, lot, rules, args.draws)

    print(f"\n  {'stop / target':16s}{'trades':>7s}{'win%':>7s}{'gross':>8s}"
          f"{'gross%':>8s}{'net':>8s}{'Rs total':>12s}{'random':>8s}{'vs rnd':>8s}")
    print("  " + "-" * 82)
    for r in rules:
        rr = rows if r == (stop, target) else run(d, None, args, lot, r[0], r[1], P=P)[0]
        s = stats(rr)
        c = ctrl[r]
        print(f"  {fmt_rule(*r):16s}{s['n']:7d}{s['win']:7.1f}{s['gross']:+8.2f}"
              f"{s['gross_pct']:+8.1f}{s['net']:+8.2f}{s['rs']:+12,.0f}"
              f"{c.mean():+8.2f}{pct_vs(s['gross'], c):7.0f}%")
    print("  gross / net in premium points per trade; 'random' = mean gross of "
          "random entries.\n  'vs rnd' above 95% would be an edge; below 5% is "
          "worse than random.")

    # ---- detail for the chosen rule
    s = stats(rows)
    c = ctrl[(stop, target)]
    lo_, hi_ = np.percentile(c, [5, 95])
    print(f"\n  CHOSEN RULE {fmt_rule(stop, target).strip()}: {s['n']} trades | "
          f"win {s['win']:.1f}% | gross {s['gross']:+.2f} (t={s['t']:+.2f}) | "
          f"net {s['net']:+.2f} pts | Rs {s['rs']:+,.0f}")
    print(f"  underlying moved {rows['und_pts'].mean():+.2f} Nifty pts in the "
          f"trade's favour, on average")
    print(f"  control: signal gross {s['gross']:+.2f} vs random {c.mean():+.2f} "
          f"[5-95%: {lo_:+.2f}, {hi_:+.2f}] -> percentile {pct_vs(s['gross'], c):.0f}%")
    print("  " + ("BEATS random" if pct_vs(s['gross'], c) > 95 else
                  "WORSE than random" if pct_vs(s['gross'], c) < 5 else
                  "no evidence of edge"))

    print("\n  How trades ended:")
    for why, g in rows.groupby("reason"):
        print(f"    {why:8s}{len(g):6d} ({100 * len(g) / len(rows):3.0f}%)  avg "
              f"gross {g['gross_pts'].mean():+7.2f} pts, held {g['minutes'].mean():5.0f} min")

    print(f"\n  {'month':10s}{'trades':>8s}{'win%':>7s}{'gross':>8s}{'net':>8s}{'Rs':>11s}")
    print("  " + "-" * 52)
    wins = 0
    for ym in months:
        g = rows[rows["ym"] == ym]
        if not len(g):
            continue
        st = stats(g)
        wins += st["rs"] > 0
        print(f"  {ym:10s}{st['n']:8d}{st['win']:7.1f}{st['gross']:+8.2f}"
              f"{st['net']:+8.2f}{st['rs']:+11,.0f}")
    print(f"\n  {wins} of {len(set(rows['ym']))} months profitable (net)")

    out = os.path.join(HERE, "options_trades.csv")
    rows.to_csv(out, index=False)
    print(f"\n  trades saved to {os.path.basename(out)}")
    print("\n  MODEL CAVEAT: premiums are Black-Scholes from VIX, not real fills.")
    print("  Run option_model_check.py to measure the gap before believing a "
          "positive result.")


if __name__ == "__main__":
    main()
