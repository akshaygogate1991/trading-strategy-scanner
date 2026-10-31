"""Backtest AlgoTest (.algtst) strategy exports on real 1-minute index option data.

Data: hf_data/ from fetch_hf_data.py (Hugging Face thetrademarkk/india-index-options-1m,
one parquet per weekly expiry, plus hf_data/index/<TICKER>.parquet for spot).

Supported .algtst features (anything else stops the run with an error, so a
strategy is never silently tested with a setting ignored):
  - IntradaySameDay, time entry/exit, weekly expiry (nearest expiry >= today, so
    expiry day trades the expiring contract, as AlgoTest does)
  - EntryByPremium (closest premium) and EntryByStrikeType (ATM / ITMn / OTMn)
  - LegMomentum PercentageUp / PercentageDown, measured from the premium at selection
  - LegStopLoss / LegTarget Percentage, LegTrailSL Percentage (for every X% of
    entry price in favour, SL moves Y% of entry price)
  - LegReentrySL LikeOriginal (fresh strike selection + momentum after an SL hit)
  - OverallSL / OverallTgt MTM in rupees

Fill rules on 1-minute bars:
  - price "at" a minute = that minute's open (previous close if no trade that minute)
  - leg SL is checked against the bar's adverse extreme before the trail moves
    (pessimistic); a gap past the SL fills at the bar open
  - a momentum fill mid-bar starts SL checks on the next bar
  - overall MTM is checked on minute closes and exits at those closes (checking
    every leg at its worst extreme together would overstate straddle losses)

Costs (per executed order): brokerage Rs 5, STT 0.15% on sell premium, exchange
txn (NSE 0.03553%, BSE 0.0325%), stamp 0.003% on buy, SEBI Rs 10/cr, GST 18% on
brokerage+txn+SEBI. Slippage --slip-pct per side, against you.

Usage:
  python algtst_backtest.py "D:\\algotest\\chotapacketbadadhamakanifty.algtst"
  python algtst_backtest.py FILE --control 20      # + shifted-entry-time control
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "hf_data")

LOT = {"NIFTY": 65, "SENSEX": 20, "BANKNIFTY": 30}
STEP = {"NIFTY": 50, "SENSEX": 100, "BANKNIFTY": 100}   # others: inferred from the strikes
EXCH_TXN = {"NIFTY": 0.0003553, "SENSEX": 0.000325}
BROKERAGE_PER_ORDER = 5.0
STT_SELL = 0.0015
STAMP_BUY = 0.00003
SEBI_FEE = 0.000001
GST_RATE = 0.18

OPEN_MIN = 9 * 60 + 15
N_MIN = 375                      # 09:15 .. 15:29
LAST_MIN = N_MIN - 5             # latest exit the control may shift to (15:25)


# ---------------------------------------------------------------- strategy file

def lot_size(ticker):
    """Current lot size: built-in for indices, else hf_data/lots.json (Angel One master)."""
    if ticker in LOT:
        return LOT[ticker]
    p = os.path.join(DATA, "lots.json")
    lots = json.load(open(p)) if os.path.exists(p) else {}
    if ticker not in lots:
        raise SystemExit(f"Unknown lot size for {ticker}: pass --lot")
    return lots[ticker]


def _enum(v):
    return str(v).split(".")[-1] if v not in (None, "None") else None


def _time_of(tree):
    """Minute index of the TimeIndicator in an Entry/ExitIndicators tree."""
    found = []

    def walk(n):
        if isinstance(n, dict):
            v = n.get("Value")
            if isinstance(v, dict) and _enum(v.get("IndicatorName")) == "TimeIndicator":
                p = v["Parameters"]
                found.append(p["Hour"] * 60 + p["Minute"] - OPEN_MIN)
            elif isinstance(v, list):
                for c in v:
                    walk(c)
            elif _enum(n.get("Type")) == "DataNode":
                raise SystemExit(f"Unsupported indicator: {v}")
    walk(tree)
    if len(found) != 1:
        raise SystemExit(f"Expected exactly one time condition, got {found}")
    return found[0]


def _need_none(obj, keys, where):
    for k in keys:
        if _enum(obj.get(k, {}).get("Type")) is not None:
            raise SystemExit(f"{where}: {k}={obj[k]} is not supported yet")


def parse_strategy(name, s):
    if _enum(s["StrategyType"]) != "IntradaySameDay":
        raise SystemExit(f"{name}: only IntradaySameDay is supported")
    if str(s.get("TrailSLtoBreakeven")) == "True" or str(s.get("SquareOffAllLegs")) == "True":
        raise SystemExit(f"{name}: TrailSLtoBreakeven / SquareOffAllLegs not supported yet")
    _need_none(s, ["OverallReentrySL", "OverallReentryTgt", "OverallTrailSL", "LockAndTrail"], name)
    for k in ("OverallSL", "OverallTgt"):
        t = _enum(s[k]["Type"])
        if t not in (None, "MTM"):
            raise SystemExit(f"{name}: {k} type {t} not supported")
    legs = []
    for L in s["ListOfLegConfigs"]:
        if _enum(L["ExpiryKind"]) != "Weekly":
            raise SystemExit(f"{name}: only weekly expiry is supported")
        _need_none(L, ["LegReentryTP"], name)
        mom_t = _enum(L["LegMomentum"]["Type"])
        if mom_t not in (None, "PercentageUp", "PercentageDown"):
            raise SystemExit(f"{name}: momentum {mom_t} not supported")
        for k in ("LegStopLoss", "LegTarget"):
            if _enum(L[k]["Type"]) not in (None, "Percentage"):
                raise SystemExit(f"{name}: {k} {L[k]} not supported")
        tr_t = _enum(L["LegTrailSL"]["Type"])
        if tr_t not in (None, "Percentage"):
            raise SystemExit(f"{name}: trail {tr_t} not supported")
        re_t = _enum(L["LegReentrySL"]["Type"])
        if re_t not in (None, "LikeOriginal"):
            raise SystemExit(f"{name}: SL re-entry {re_t} not supported")
        et = _enum(L["EntryType"])
        if et == "EntryByPremium":
            param = float(L["StrikeParameter"])
        elif et == "EntryByStrikeType":
            param = _enum(L["StrikeParameter"])
        else:
            raise SystemExit(f"{name}: entry type {et} not supported")
        legs.append(dict(
            id=L["id"], kind=_enum(L["InstrumentKind"]),
            side=1 if _enum(L["PositionType"]) == "Buy" else -1,
            lots=int(L["Lots"]), entry=et, param=param,
            mom=(mom_t, float(L["LegMomentum"]["Value"])) if mom_t else None,
            sl=float(L["LegStopLoss"]["Value"]) if _enum(L["LegStopLoss"]["Type"]) else None,
            tgt=float(L["LegTarget"]["Value"]) if _enum(L["LegTarget"]["Type"]) else None,
            trail=(float(L["LegTrailSL"]["Value"]["InstrumentMove"]),
                   float(L["LegTrailSL"]["Value"]["StopLossMove"])) if tr_t else None,
            reentry=int(L["LegReentrySL"]["Value"]["ReentryCount"]) if re_t else 0,
        ))
    return dict(
        name=name, ticker=s["Ticker"],
        entry=_time_of(s["EntryIndicators"]), exit=_time_of(s["ExitIndicators"]),
        osl=float(s["OverallSL"]["Value"]) if _enum(s["OverallSL"]["Type"]) else None,
        otgt=float(s["OverallTgt"]["Value"]) if _enum(s["OverallTgt"]["Type"]) else None,
        legs=legs,
    )


def load_file(path):
    if not os.path.exists(path):
        folder = os.path.dirname(os.path.abspath(path))
        found = sorted(glob.glob(os.path.join(folder, "*.algtst")))
        raise SystemExit(f"File not found: {path}\n.algtst files in {folder}:\n  "
                         + ("\n  ".join(found) if found else "(none)"))
    d = json.load(open(path, encoding="utf-8"))
    if "strategy" in d:
        return [parse_strategy(d.get("name", os.path.basename(path)), d["strategy"])]
    strategies = d["data"]["strategies"]
    out = []
    for item in d["data"]["portfolio"]["items"]:
        if not item.get("checked", True):
            continue
        if not all(item.get("weekdays", {}).values()) or item.get("multiplier", 1) != 1:
            raise SystemExit("Portfolio weekday filters / multipliers not supported yet")
        st = strategies[item["strategy_id"]]
        out.append(parse_strategy(st.get("name", item["strategy_id"]), st["definition"]))
    return out


# ---------------------------------------------------------------- market data

class Day:
    """Minute bars of every contract of one expiry on one trading day."""

    def __init__(self, g, spot_open):
        ts = g["timestamp"]
        m = (ts.dt.hour * 60 + ts.dt.minute - OPEN_MIN).to_numpy()
        keep = (m >= 0) & (m < N_MIN)
        g, m = g[keep], m[keep]
        cents = np.round(g["strike"].to_numpy().astype(float) * 100).astype(np.int64)
        codes, uniq = pd.factorize(cents * 2 + (g["option_type"].to_numpy() == "PE"))
        uniq = [((u // 2) / 100, "PE" if u % 2 else "CE") for u in uniq]
        self.contracts = uniq
        self.index_of = {c: i for i, c in enumerate(uniq)}
        shape = (len(uniq), N_MIN)
        arr = {}
        for col in ("open", "high", "low", "close"):
            a = np.full(shape, np.nan)
            a[codes, m] = g[col].to_numpy()
            arr[col] = a
        close = pd.DataFrame(arr["close"]).ffill(axis=1).to_numpy()
        prev = np.concatenate([np.full((shape[0], 1), np.nan), close[:, :-1]], axis=1)
        self.C = close
        self.O = np.where(np.isnan(arr["open"]), prev, arr["open"])
        self.H = np.where(np.isnan(arr["high"]), prev, arr["high"])
        self.L = np.where(np.isnan(arr["low"]), prev, arr["low"])
        self.strike = np.array([c[0] for c in uniq], dtype=float)
        self.kind = np.array([c[1] for c in uniq])
        self.spot_open = spot_open     # array (N_MIN,) or None
        ks = np.unique(self.strike)
        self.step = float(np.median(np.diff(ks))) if len(ks) > 1 else 1.0

    def atm_price(self, kind, k, step):
        """Premium of the ATM option of this kind at minute k (None if not traded)."""
        a = self.atm(k, step)
        i = self.index_of.get((a, kind)) if a is not None else None
        return None if i is None or np.isnan(self.O[i, k]) else self.O[i, k]

    def pick(self, kind, entry, param, k, step):
        """Contract index for a leg at minute k, or None."""
        mask = (self.kind == kind) & (self.O[:, k] > 0)      # NaN or bad 0 prints excluded
        if not mask.any():
            return None
        idx = np.flatnonzero(mask)
        if entry == "EntryByPremium":
            return int(idx[np.argmin(np.abs(self.O[idx, k] - param))])
        atm = self.atm(k, step)
        if atm is None:
            return None
        step = step or self.step
        n = 0 if param == "ATM" else int(param[3:])
        sign = -1 if param.startswith("ITM") else 1
        if kind == "PE":
            sign = -sign
        target = atm + sign * n * step
        hit = idx[self.strike[idx] == target]
        return int(hit[0]) if len(hit) else None

    def atm(self, k, step):
        step = step or self.step
        if self.spot_open is not None and not np.isnan(self.spot_open[k]):
            return round(self.spot_open[k] / step) * step
        # fallback: strike where CE and PE premiums are closest (put-call parity)
        best, bd = None, np.inf
        ce = {s: i for i, (s, t) in enumerate(self.contracts) if t == "CE"}
        for i, (s, t) in enumerate(self.contracts):
            if t == "PE" and s in ce:
                d = abs(self.O[ce[s], k] - self.O[i, k])
                if d < bd:
                    best, bd = s, d
        return best


def load_spot(ticker):
    p = os.path.join(DATA, "index", f"{ticker}.parquet")
    if not os.path.exists(p):
        return {}
    s = pd.read_parquet(p, columns=["timestamp", "open", "trading_day"])
    m = (s["timestamp"].dt.hour * 60 + s["timestamp"].dt.minute - OPEN_MIN).to_numpy()
    s = s[(m >= 0) & (m < N_MIN)].assign(m=m[(m >= 0) & (m < N_MIN)])
    out = {}
    for day, g in s.groupby("trading_day"):
        a = np.full(N_MIN, np.nan)
        a[g["m"].to_numpy()] = g["open"].to_numpy()
        out[day] = pd.Series(a).ffill().to_numpy()
    return out


def iter_days(ticker, start, end, which="all"):
    """Yield (day, expiry, Day) using the nearest available expiry for each day.

    which: "all", "expiry" (expiry days only) or "other" (non-expiry days only).
    Weekly series: a day more than 7 days before its expiry means that week's own
    file is missing, so it is skipped. Monthly series (Bank Nifty, stocks) allow 40."""
    files = sorted(glob.glob(os.path.join(DATA, ticker, "*.parquet")))
    if not files:
        raise SystemExit(f"No data in {os.path.join(DATA, ticker)} - run fetch_hf_data.py")
    spot = load_spot(ticker)
    exps = [pd.Timestamp(os.path.basename(f)[:10]) for f in files]
    gaps = np.diff(exps).astype("timedelta64[D]").astype(int) if len(exps) > 1 else np.array([7])
    max_gap = 7 if np.median(gaps) <= 8 else 40
    prev = None
    cols = ["timestamp", "open", "high", "low", "close", "trading_day", "strike", "option_type"]
    for f in files:
        exp = os.path.basename(f)[:10]
        after, prev = prev or "0000", exp
        if exp < start or after >= end:
            continue
        df = pd.read_parquet(f, columns=cols)
        day_col = df["trading_day"]
        df = df[(day_col > after) & (day_col >= start) & (day_col <= min(exp, end))]
        for day, g in df.groupby("trading_day"):
            if (pd.Timestamp(exp) - pd.Timestamp(day)).days > max_gap:
                continue           # this day's own expiry file is missing
            if which != "all" and (day == exp) != (which == "expiry"):
                continue
            yield day, exp, Day(g, spot.get(day))


# ---------------------------------------------------------------- simulation

class DaySim:
    """One strategy on one day, advanced one completed minute bar at a time.

    The backtest calls step(k) for every minute; the live paper trader calls it
    as each bar closes, so both use exactly the same rules. `day` needs O/H/L/C
    arrays (contracts x minutes) filled at least up to column k, plus pick/atm.
    """

    def __init__(self, st, day, lot, entry=None, exit_=None):
        self.st, self.day, self.lot = st, day, lot
        self.entry = st["entry"] if entry is None else entry
        self.exit = st["exit"] if exit_ is None else exit_
        self.step_size = STEP.get(st["ticker"])
        self.legs = [dict(cfg=c, state="select", at=self.entry, re_left=c["reentry"]) for c in st["legs"]]
        self.trades, self.realized, self.done = [], 0.0, False
        # optional scaling for other underlyings: premium as a fraction of the ATM
        # premium, overall SL / target as a fraction of the premium of the first entries
        self.frac, self.osl, self.otgt = st.get("prem_frac"), st["osl"], st["otgt"]
        self.first_prem = 0.0
        if st.get("osl_frac") is not None:
            self.osl = None

    def _close(self, g, k, price, reason):
        c = g["cfg"]
        qty = c["lots"] * self.lot
        pnl = c["side"] * (price - g["px"]) * qty
        self.realized += pnl
        self.trades.append(dict(leg=c["id"], kind=c["kind"], side=c["side"], qty=qty,
                                strike=self.day.strike[g["c"]], entry_min=g["k0"], entry_px=g["px"],
                                exit_min=k, exit_px=price, reason=reason, gross=pnl))
        g["state"] = "done"

    def mtm(self, k):
        """Realized + open P&L at the close of minute k (gross, rupees)."""
        return self.realized + sum(g["cfg"]["side"] * (self.day.C[g["c"], k] - g["px"]) * g["cfg"]["lots"] * self.lot
                                   for g in self.legs if g["state"] == "open")

    def step(self, k):
        """Process completed minute bar k. Returns True once the day is finished."""
        if self.done or k < self.entry:
            return self.done
        st, day, lot, step = self.st, self.day, self.lot, self.step_size
        O, H, L, C = day.O, day.H, day.L, day.C
        legs, frac = self.legs, self.frac
        if k >= self.exit:
            for g in legs:
                if g["state"] == "open":
                    self._close(g, k, O[g["c"], k], "time")
            self.done = True
            return True
        for g in legs:
            c = g["cfg"]
            if g["state"] == "select" and k >= g["at"]:
                param = c["param"]
                if frac is not None and c["entry"] == "EntryByPremium":
                    a = day.atm_price(c["kind"], k, step)
                    param = None if a is None else frac * a
                ci = None if param is None else day.pick(c["kind"], c["entry"], param, k, step)
                if ci is None:
                    g["state"] = "done"
                    continue
                if st.get("osl_frac") is not None and not g.get("counted"):
                    g["counted"] = True
                    self.first_prem += O[ci, k] * c["lots"] * lot
                    self.osl = st["osl_frac"] * self.first_prem
                g["c"], ref = ci, O[ci, k]
                if c["mom"]:
                    up = c["mom"][0] == "PercentageUp"
                    g["trig"] = ref * (1 + c["mom"][1] / 100 * (1 if up else -1))
                    g["up"], g["state"] = up, "wait"
                else:
                    g.update(state="open", px=ref, k0=k, fresh=False)
            ci = g.get("c")
            if g["state"] == "wait":
                if g["up"] and H[ci, k] >= g["trig"]:
                    g.update(state="open", px=max(g["trig"], O[ci, k]), k0=k, fresh=True)
                elif not g["up"] and L[ci, k] <= g["trig"]:
                    g.update(state="open", px=min(g["trig"], O[ci, k]), k0=k, fresh=True)
                if g["state"] == "open":
                    g["best"] = g["px"]
                continue
            if g["state"] != "open":
                continue
            if g["fresh"]:              # filled mid-bar: start checks next bar
                g["fresh"] = False
                continue
            px, side = g["px"], c["side"]
            g.setdefault("best", px)
            sl = None
            if c["sl"] is not None:
                sl = px * (1 - side * c["sl"] / 100)
                if c["trail"]:
                    fav = (g["best"] - px) * side
                    steps = math.floor(fav / (px * c["trail"][0] / 100) + 1e-9) if fav > 0 else 0
                    sl += side * steps * px * c["trail"][1] / 100
            g["sl_level"] = sl
            adverse, favour = (L[ci, k], H[ci, k]) if side > 0 else (H[ci, k], L[ci, k])
            if sl is not None and (adverse - sl) * side <= 0:
                self._close(g, k, min(sl, O[ci, k]) if side > 0 else max(sl, O[ci, k]), "sl")
                if g["re_left"] > 0 and k + 1 < self.exit:
                    g.update(state="select", at=k + 1, re_left=g["re_left"] - 1)
                    g.pop("best")
                continue
            if c["tgt"] is not None:
                tp = px * (1 + side * c["tgt"] / 100)
                if (favour - tp) * side >= 0:
                    self._close(g, k, max(tp, O[ci, k]) if side > 0 else min(tp, O[ci, k]), "target")
                    continue
            if (favour - g["best"]) * side > 0:
                g["best"] = favour
        osl, otgt = self.osl, self.otgt
        if osl is not None or otgt is not None:
            # MTM at the bar's start and at its close, assuming each open leg moves
            # in a straight line between them; exit where MTM crosses the limit
            live = [g for g in legs if g["state"] == "open"]
            start = {id(g): g["px"] if g["k0"] == k else O[g["c"], k] for g in live}
            qs = {id(g): g["cfg"]["side"] * g["cfg"]["lots"] * lot for g in live}
            m0 = self.realized + sum(qs[id(g)] * (start[id(g)] - g["px"]) for g in live)
            m1 = self.realized + sum(qs[id(g)] * (C[g["c"], k] - g["px"]) for g in live)
            hit, f = None, 1.0
            for lim, lab, sgn in ((osl, "overall_sl", -1), (otgt, "overall_tgt", 1)):
                if lim is None or hit:
                    continue
                if (m0 - sgn * lim) * sgn >= 0:
                    hit, f = lab, 0.0
                elif (m1 - sgn * lim) * sgn >= 0:
                    hit, f = lab, (sgn * lim - m0) / (m1 - m0)
            if hit:
                for g in live:
                    s0 = start[id(g)]
                    self._close(g, k, s0 + f * (C[g["c"], k] - s0), hit)
                self.done = True
        return self.done


def run_day(st, day, lot, entry=None, exit_=None):
    """Simulate one strategy for one day. Returns list of leg-trade dicts."""
    sim = DaySim(st, day, lot, entry, exit_)
    for k in range(sim.entry, sim.exit + 1):
        if sim.step(k):
            break
    return sim.trades


def add_costs(t, ticker, slip_pct):
    """Charges in rupees and slippage-adjusted net for one leg trade."""
    buy_px, sell_px = (t["entry_px"], t["exit_px"]) if t["side"] > 0 else (t["exit_px"], t["entry_px"])
    q = t["qty"]
    buy, sell = buy_px * q, sell_px * q
    txn = EXCH_TXN.get(ticker, EXCH_TXN["NIFTY"]) * (buy + sell)
    sebi = SEBI_FEE * (buy + sell)
    brok = 2 * BROKERAGE_PER_ORDER
    t["charges"] = brok + STT_SELL * sell + txn + STAMP_BUY * buy + sebi + GST_RATE * (brok + txn + sebi)
    t["slippage"] = slip_pct / 100 * (buy + sell)
    t["net"] = t["gross"] - t["charges"] - t["slippage"]
    return t


# ---------------------------------------------------------------- reporting

def hhmm(k):
    return f"{(k + OPEN_MIN) // 60:02d}:{(k + OPEN_MIN) % 60:02d}"


def report(name, tr, ndays, lot):
    if tr.empty:
        print(f"\n=== {name}: no trades")
        return
    daily = tr.groupby("day")[["gross", "charges", "slippage", "net"]].sum()
    daily["net_noslip"] = daily["gross"] - daily["charges"]
    n = len(daily)
    eq = daily["net"].cumsum()
    dd = (eq - eq.cummax()).min()
    t_stat = daily["net"].mean() / (daily["net"].std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    print(f"\n=== {name}   (lot size {lot}, {n} trading days of {ndays} with data)")
    print(f"  Gross P&L        Rs {daily['gross'].sum():>12,.0f}")
    print(f"  Charges          Rs {daily['charges'].sum():>12,.0f}   ({len(tr)} leg trades, {2 * len(tr)} orders)")
    print(f"  Net (no slip)    Rs {daily['net_noslip'].sum():>12,.0f}   <- compare with AlgoTest")
    turnover = ((tr["entry_px"] + tr["exit_px"]) * tr["qty"]).sum()
    print(f"  Slippage         Rs {daily['slippage'].sum():>12,.0f}   "
          f"(break-even slippage {100 * daily['net_noslip'].sum() / turnover:.2f}% of premium per side)")
    print(f"  NET              Rs {daily['net'].sum():>12,.0f}   avg/day {daily['net'].mean():,.0f}   t={t_stat:.2f}")
    print(f"  Win days {100 * (daily['net'] > 0).mean():.0f}%   best day {daily['net'].max():,.0f}   "
          f"worst day {daily['net'].min():,.0f}   max drawdown {dd:,.0f}")
    print("  Exit reasons: " + ", ".join(f"{k} {v}" for k, v in tr["reason"].value_counts().items()))
    exp_day = tr.groupby("day")["is_expiry"].first()
    for lab, sel in (("expiry day", exp_day), ("other days", ~exp_day)):
        d = daily.loc[sel[sel].index, "net"]
        if len(d):
            print(f"  {lab:<11} {len(d):>3} days  net {d.sum():>10,.0f}  avg {d.mean():>7,.0f}")
    daily["ym"] = daily.index.str[:7]
    m = daily.groupby("ym").agg(days=("net", "size"), gross=("gross", "sum"), net=("net", "sum"))
    print("  Month     days      gross        net")
    for ym, r in m.iterrows():
        print(f"  {ym}  {r['days']:>4} {r['gross']:>10,.0f} {r['net']:>10,.0f}")
    print(f"  Months positive: {(m['net'] > 0).sum()} of {len(m)}")
    return daily


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--start", default="2025-06-01")
    ap.add_argument("--end", default="2026-07-31")
    ap.add_argument("--lot", type=int, help="lot size override (default: current NSE/BSE lot)")
    ap.add_argument("--slip-pct", type=float, default=0.5, help="slippage %% of premium per side")
    ap.add_argument("--control", type=int, default=0,
                    help="N draws of a control that shifts entry+exit by a random 0-90 min per day")
    ap.add_argument("--out", help="trades CSV (default <file name>_trades.csv)")
    ap.add_argument("--days", choices=["all", "expiry", "other"], default="all",
                    help="trade only expiry days, only non-expiry days, or all")
    ap.add_argument("--strategy", help="run only the strategy whose name contains this text")
    ap.add_argument("--ticker", help="run on another underlying (NIFTY, BANKNIFTY, a stock...)")
    ap.add_argument("--prem-frac", type=float,
                    help="premium-entry legs pick premium = this x the ATM premium (for other underlyings)")
    ap.add_argument("--osl-frac", type=float,
                    help="overall SL = this x premium of the first entries, instead of a fixed Rs amount")
    args = ap.parse_args()

    strategies = load_file(args.file)
    if args.strategy:
        strategies = [s for s in strategies if args.strategy.lower() in s["name"].lower()]
        if not strategies:
            raise SystemExit(f"No strategy named {args.strategy!r} in the file")
    for s in strategies:
        if args.ticker:
            s["ticker"] = args.ticker
        s["prem_frac"], s["osl_frac"] = args.prem_frac, args.osl_frac
    tickers = {s["ticker"] for s in strategies}
    if len(tickers) != 1:
        raise SystemExit("Mixed tickers in one file not supported")
    ticker = tickers.pop()
    lot = args.lot or lot_size(ticker)
    for s in strategies:
        print(f"{s['name']}: {ticker} entry {hhmm(s['entry'])} exit {hhmm(s['exit'])} "
              f"overall SL {s['osl']} target {s['otgt']}")
        for c in s["legs"]:
            print(f"   {'BUY ' if c['side'] > 0 else 'SELL'} {c['kind']} {c['param']} x{c['lots']} "
                  f"mom={c['mom']} sl={c['sl']} tgt={c['tgt']} trail={c['trail']} re={c['reentry']}")

    rng = np.random.default_rng(17)
    rows, ctrl, ndays = [], {s["name"]: np.zeros(args.control) for s in strategies}, 0
    for day, exp, d in iter_days(ticker, args.start, args.end, args.days):
        ndays += 1
        for s in strategies:
            for t in run_day(s, d, lot):
                rows.append(add_costs(dict(t, day=day, expiry=exp, is_expiry=day == exp,
                                           strategy=s["name"]), ticker, args.slip_pct))
            for i in range(args.control):
                off = int(rng.integers(0, 91))
                ex = min(s["exit"] + off, LAST_MIN)
                if s["entry"] + off >= ex:
                    continue
                ctrl[s["name"]][i] += sum(add_costs(t, ticker, args.slip_pct)["net"]
                                          for t in run_day(s, d, lot, s["entry"] + off, ex))
    if not rows:
        raise SystemExit("No trades - check the date range and data")
    tr = pd.DataFrame(rows)
    for col in ("entry_min", "exit_min"):
        tr[col.replace("_min", "_time")] = tr[col].map(hhmm)
    out = args.out or os.path.join(HERE, os.path.splitext(os.path.basename(args.file))[0] + "_trades.csv")
    tr.drop(columns=["entry_min", "exit_min"]).to_csv(out, index=False)
    print(f"\n{ndays} days with data, {tr['day'].min()} .. {tr['day'].max()}   trades -> {out}")
    print(f"Costs: brokerage Rs {BROKERAGE_PER_ORDER:g}/order, slippage {args.slip_pct}% per side, "
          f"days: {args.days}" + (f", premium {args.prem_frac} x ATM" if args.prem_frac else "")
          + (f", overall SL {args.osl_frac} x premium" if args.osl_frac else ""))

    for s in strategies:
        report(s["name"], tr[tr["strategy"] == s["name"]], ndays, lot)
        if args.control:
            real = tr.loc[tr["strategy"] == s["name"], "net"].sum()
            c = ctrl[s["name"]]
            print(f"  Control (entry/exit shifted 0-90 min at random, {args.control} draws): "
                  f"median {np.median(c):,.0f}, real is at percentile {100 * (c < real).mean():.0f}")
    if len(strategies) > 1:
        report("PORTFOLIO (all strategies together)", tr, ndays, lot)


if __name__ == "__main__":
    main()
