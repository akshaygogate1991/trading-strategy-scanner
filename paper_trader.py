"""Live PAPER trading of .algtst strategies on Sensex options. Places NO orders.

Every few seconds it polls Angel One for the last traded price of the Sensex
spot and of the nearest-weekly Sensex options around the money, builds 1-minute
OHLC bars from those prices, and as each minute closes feeds the bar to the
same DaySim engine the backtest uses (algtst_backtest.py). Trades are written to
paper_data/; nothing is ever sent to a broker.

Default jobs (the combination that backtested best, Aug 2023 - May 2026):
  - Sensex Directional        every trading day
  - Chota Packet Sensex       expiry days only

Run on the VPS from the project folder (credentials: .streamlit/secrets.toml,
read by smartapi_data.py - never printed or logged):
  python paper_trader.py                 # live, today
  python paper_trader.py --replay-hf 2026-05-14    # offline test on a past day
  python paper_trader.py --replay-saved paper_data/bars/2026-11-03.parquet

Outputs in paper_data/:
  events.log     every entry / exit as it happens
  trades.csv     one row per leg trade, with charges, slippage and net
  daily.csv      one row per strategy per day
  bars/DATE.parquet   the day's minute bars (replayable, and our own data set)
"""
import argparse
import datetime as dt
import glob
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

import algtst_backtest as ab

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "paper_data")
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
ALGO = os.environ.get("ALGTST_DIR", os.path.join(HERE, "strategies"))

JOBS = [  # (file, strategy-name filter, days: all / expiry / other)
    ("export_sensexbasket.algtst", "Directional", "all"),
    ("chotapacketbadadhamakasensex.algtst", "Chota", "expiry"),
]
TICKER = "SENSEX"
SPOT_TOKEN = ("BSE", "99919000")
STRIKE_BAND = 0.045         # watch strikes within +-4.5% of spot (99% of backtest trades were within 3.7%)
POLL_SEC = 5
SLIP_PCT = 0.5
BATCH = 50                  # tokens per getMarketData call


def log(msg):
    line = f"{dt.datetime.now(IST):%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "events.log"), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


class _HideRequestDetails(logging.Filter):
    """SmartApi logs the full request on failure - API key, client code, MPIN and
    TOTP included. Keep only the 'Response:' part, which is the actual error."""

    def filter(self, record):
        msg = record.getMessage()
        if "Headers:" in msg or "Request:" in msg:
            tail = msg.split("Response:", 1)[1].strip() if "Response:" in msg else ""
            record.msg, record.args = "SmartApi request failed (request details hidden). " + tail, ()
        return True


def hide_credentials_in_logs():
    import logzero                     # SmartApi's logger; installed with smartapi-python
    logzero.logger.addFilter(_HideRequestDetails())


def minute_index(t):
    return t.hour * 60 + t.minute - ab.OPEN_MIN


# ---------------------------------------------------------------- live bars

class LiveDay(ab.Day):
    """Same interface as ab.Day, filled one minute at a time from price ticks."""

    def __init__(self, contracts, spot=True):
        n = len(contracts)
        self.contracts = contracts
        self.index_of = {c: i for i, c in enumerate(contracts)}
        self.strike = np.array([c[0] for c in contracts], dtype=float)
        self.kind = np.array([c[1] for c in contracts])
        ks = np.unique(self.strike)
        self.step = float(np.median(np.diff(ks))) if len(ks) > 1 else 1.0
        self.O, self.H, self.L, self.C = (np.full((n, ab.N_MIN), np.nan) for _ in range(4))
        self.spot_open = np.full(ab.N_MIN, np.nan) if spot else None
        self._acc = {}          # contract index -> [first, high, low, last] for the open minute
        self._spot_first = None

    def tick(self, i, price):
        if price is None or not price > 0:
            return
        a = self._acc.get(i)
        if a is None:
            self._acc[i] = [price, price, price, price]
        else:
            a[1], a[2], a[3] = max(a[1], price), min(a[2], price), price

    def tick_spot(self, price):
        if self._spot_first is None and price and price > 0:
            self._spot_first = price

    def close_bar(self, m):
        """Finalise minute m. Contracts with no trade copy the previous close
        (exactly what ab.Day does with missing minutes)."""
        prev = self.C[:, m - 1] if m > 0 else np.full(len(self.contracts), np.nan)
        self.O[:, m] = self.H[:, m] = self.L[:, m] = self.C[:, m] = prev
        for i, (o, h, l, c) in self._acc.items():
            self.O[i, m], self.H[i, m], self.L[i, m], self.C[i, m] = o, h, l, c
        self._acc = {}
        if self.spot_open is not None:
            p = self._spot_first if self._spot_first is not None else (
                self.spot_open[m - 1] if m > 0 else np.nan)
            self.spot_open[m] = p
            self._spot_first = None

    def load_frame(self, df):
        """Refill bars saved earlier today (restart recovery). Returns last minute filled."""
        ts = df["timestamp"]
        m = (ts.dt.hour * 60 + ts.dt.minute - ab.OPEN_MIN).to_numpy()
        last = -1
        for r, mm in zip(df.itertuples(index=False), m):
            i = self.index_of.get((float(r.strike), r.option_type))
            if i is not None and 0 <= mm < ab.N_MIN:
                self.O[i, mm], self.H[i, mm], self.L[i, mm], self.C[i, mm] = r.open, r.high, r.low, r.close
                last = max(last, mm)
        if "spot" in df and self.spot_open is not None:
            sp = df.groupby(m)["spot"].first()
            self.spot_open[sp.index.to_numpy()] = sp.to_numpy()
        for k in range(last + 1):              # minutes with no trade copy the previous close
            for a in (self.O, self.H, self.L, self.C):
                miss = np.isnan(a[:, k])
                if k > 0:
                    a[miss, k] = self.C[miss, k - 1]
        return last

    def bars_frame(self, day, expiry):
        """The day's bars in the Hugging Face schema, so the backtester can replay them."""
        rows = []
        for i, (strike, kind) in enumerate(self.contracts):
            for m in np.flatnonzero(~np.isnan(self.C[i])):
                rows.append((m, self.O[i, m], self.H[i, m], self.L[i, m], self.C[i, m], strike, kind))
        df = pd.DataFrame(rows, columns=["m", "open", "high", "low", "close", "strike", "option_type"])
        base = pd.Timestamp(f"{day} 09:15", tz=IST)
        df.insert(0, "timestamp", base + pd.to_timedelta(df.pop("m"), unit="min"))
        df["trading_day"], df["expiry"], df["symbol"] = day, expiry, TICKER
        if self.spot_open is not None:
            mins = (df["timestamp"] - base).dt.total_seconds().floordiv(60).astype(int).to_numpy()
            df["spot"] = self.spot_open[mins]
        return df


# ---------------------------------------------------------------- strategies

def load_jobs():
    sims_cfg = []
    for fname, name_part, which in JOBS:
        path = os.path.join(ALGO, fname)
        for st in ab.load_file(path):
            if name_part.lower() in st["name"].lower():
                st["prem_frac"] = st["osl_frac"] = None
                sims_cfg.append((st, which))
    if not sims_cfg:
        raise SystemExit(f"No strategies loaded from {ALGO}")
    return sims_cfg


class Book:
    """Runs the sims for one day and reports entries / exits as they happen."""

    def __init__(self, day_obj, day, expiry, lot):
        self.day, self.expiry, self.lot = day, expiry, lot
        self.quiet = False
        self.sims = []
        for st, which in load_jobs():
            is_exp = day == expiry
            if which == "all" or (which == "expiry") == is_exp:
                self.sims.append(ab.DaySim(st, day_obj, lot))
                log(f"[{st['name']}] active today (entry {ab.hhmm(st['entry'])}, exit {ab.hhmm(st['exit'])})")
            else:
                log(f"[{st['name']}] not trading today ({which} days only)")
        self._seen = {id(s): (0, {}) for s in self.sims}

    def step(self, k):
        for s in self.sims:
            if s.done and self._seen[id(s)][0] == len(s.trades):
                continue
            s.step(k)
            self._report(s, k)

    def _report(self, s, k):
        n_seen, states = self._seen[id(s)]
        name = s.st["name"]
        if self.quiet:                          # restart recovery: rebuild state silently
            for g in s.legs:
                states[g["cfg"]["id"]] = g["state"]
            for t in s.trades[n_seen:]:
                ab.add_costs(t, TICKER, SLIP_PCT)
            self._seen[id(s)] = (len(s.trades), states)
            return
        for g in s.legs:
            c = g["cfg"]
            prev = states.get(c["id"])
            if g["state"] == "wait" and prev != "wait":
                log(f"[{name}] {c['kind']} {s.day.strike[g['c']]:.0f} picked, waiting for "
                    f"{'rise' if g['up'] else 'fall'} to {g['trig']:.2f}")
            if g["state"] == "open" and prev != "open":
                log(f"[{name}] ENTRY {'BUY' if c['side'] > 0 else 'SELL'} {c['kind']} "
                    f"{s.day.strike[g['c']]:.0f} x{c['lots'] * self.lot} @ {g['px']:.2f} ({ab.hhmm(k)})")
            states[c["id"]] = g["state"]
        for t in s.trades[n_seen:]:
            ab.add_costs(t, TICKER, SLIP_PCT)
            log(f"[{name}] EXIT {t['kind']} {t['strike']:.0f} @ {t['exit_px']:.2f} ({t['reason']}) "
                f"gross {t['gross']:+,.0f}  net {t['net']:+,.0f}")
        self._seen[id(s)] = (len(s.trades), states)
        if s.trades[n_seen:] or (k % 15 == 0 and not s.done):
            log(f"[{name}] MTM {s.mtm(k):+,.0f} at {ab.hhmm(k)}")

    def finish(self):
        os.makedirs(OUT, exist_ok=True)
        rows, daily = [], []
        for s in self.sims:
            for t in s.trades:
                if "net" not in t:
                    ab.add_costs(t, TICKER, SLIP_PCT)
                rows.append(dict(day=self.day, expiry=self.expiry, strategy=s.st["name"],
                                 entry_time=ab.hhmm(t["entry_min"]), exit_time=ab.hhmm(t["exit_min"]),
                                 **{k: v for k, v in t.items() if k not in ("entry_min", "exit_min")}))
            g = sum(t["gross"] for t in s.trades)
            n = sum(t["net"] for t in s.trades)
            daily.append(dict(day=self.day, strategy=s.st["name"], legs=len(s.trades), gross=round(g, 2),
                              charges=round(sum(t["charges"] for t in s.trades), 2),
                              slippage=round(sum(t["slippage"] for t in s.trades), 2), net=round(n, 2)))
            log(f"[{s.st['name']}] DAY DONE: {len(s.trades)} leg trades, gross {g:+,.0f}, net {n:+,.0f}")
        for name, data in (("trades.csv", rows), ("daily.csv", daily)):
            if data:
                p = os.path.join(OUT, name)
                pd.DataFrame(data).to_csv(p, mode="a", header=not os.path.exists(p), index=False)
        return daily


# ---------------------------------------------------------------- live mode

def contracts_today(client, today):
    """Nearest-expiry Sensex options within STRIKE_BAND of the current spot."""
    import smartapi_data as sd
    rows = []
    for r in sd._instrument_master():
        if r.get("exch_seg") == "BFO" and r.get("name") == TICKER and r.get("instrumenttype") == "OPTIDX":
            e = sd._parse_expiry(r.get("expiry", ""))
            if e and e >= today and r["symbol"][-2:] in ("CE", "PE"):
                rows.append((e, float(r["strike"]) / 100.0, r["symbol"][-2:], r["token"]))
    expiry = min(e for e, *_ in rows)
    spot = None
    for _ in range(60):
        spot = ltp_many(client, "BSE", [SPOT_TOKEN[1]]).get(SPOT_TOKEN[1])
        if spot:
            break
        time.sleep(5)
    if not spot:
        raise SystemExit("No Sensex spot price - market closed or API down")
    lo, hi = spot * (1 - STRIKE_BAND), spot * (1 + STRIKE_BAND)
    chosen = sorted((k, kind, tok) for e, k, kind, tok in rows if e == expiry and lo <= k <= hi)
    return expiry, spot, chosen


def market_open_today(client, today):
    """On a holiday Angel One still returns the last price, so check the
    exchange's own timestamp of the Sensex spot quote after 09:16."""
    now = dt.datetime.now(IST)
    wait = now.replace(hour=9, minute=16, second=30, microsecond=0)
    if now < wait:
        time.sleep((wait - now).total_seconds())
    for _ in range(10):
        try:
            resp = client.getMarketData("FULL", {SPOT_TOKEN[0]: [SPOT_TOKEN[1]]})
            row = resp["data"]["fetched"][0]
            stamp = row.get("exchFeedTime") or row.get("exchTradeTime") or ""
            return dt.datetime.strptime(stamp, "%d-%b-%Y %H:%M:%S").date() == today
        except (ValueError, TypeError):
            log("Could not read the exchange timestamp - assuming the market is open")
            return True
        except Exception:
            time.sleep(5)
    log("Spot quote unavailable - assuming the market is open")
    return True


def ltp_many(client, exch, tokens):
    """{token: ltp} for up to any number of tokens (batched)."""
    out = {}
    for i in range(0, len(tokens), BATCH):
        chunk = tokens[i:i + BATCH]
        for attempt in range(3):
            try:
                resp = client.getMarketData("LTP", {exch: chunk})
                for r in (resp or {}).get("data", {}).get("fetched", []) or []:
                    out[str(r["symbolToken"])] = float(r["ltp"])
                break
            except Exception as exc:          # network blip: retry, never crash the day
                if attempt == 2:
                    log(f"price fetch failed ({type(exc).__name__}); skipping this poll")
                time.sleep(1)
    return out


def run_live():
    import smartapi_data as sd
    hide_credentials_in_logs()
    now = dt.datetime.now(IST)
    if now.weekday() >= 5:
        log("Weekend - nothing to do")
        return
    open_at = now.replace(hour=9, minute=14, second=30, microsecond=0)
    if now < open_at:
        log(f"Waiting for market open ({open_at:%H:%M:%S} IST)")
        time.sleep((open_at - now).total_seconds())

    today = dt.datetime.now(IST).date()
    if os.path.exists(os.path.join(OUT, "bars", f"{today.isoformat()}.parquet")):
        log("Today's paper trading is already complete - exiting")
        return
    client = sd._session()
    expiry, spot, chosen = contracts_today(client, today)
    day, exp = today.isoformat(), expiry.isoformat()
    os.makedirs(os.path.join(OUT, "bars"), exist_ok=True)
    log(f"Sensex spot {spot:,.0f}; watching {len(chosen)} options of expiry {exp}"
        f"{' (EXPIRY DAY)' if day == exp else ''}")
    live = LiveDay([(k, kind) for k, kind, _ in chosen])
    tok_index = {tok: i for i, (_, _, tok) in enumerate(chosen)}
    tokens = list(tok_index)
    book = Book(live, day, exp, ab.lot_size(TICKER))
    partial = os.path.join(OUT, "bars", f"{day}.partial.parquet")
    cur = None
    if os.path.exists(partial):                 # restarted mid-day: restore the state
        last = live.load_frame(pd.read_parquet(partial))
        book.quiet = True
        for k in range(last + 1):
            book.step(k)
        book.quiet = False
        cur = last + 1
        for s in book.sims:
            open_legs = [f"{g['cfg']['kind']} {live.strike[g['c']]:.0f} @ {g['px']:.2f}"
                         for g in s.legs if g["state"] == "open"]
            log(f"[{s.st['name']}] RECOVERED to {ab.hhmm(last)}: {len(s.trades)} closed legs, "
                f"open: {', '.join(open_legs) or 'none'}, MTM {s.mtm(last):+,.0f}")

    else:
        now_m = minute_index(dt.datetime.now(IST))
        for s in book.sims:                     # started late: never enter after the entry time
            if now_m > s.entry:
                s.done = True
                log(f"[{s.st['name']}] started after its {ab.hhmm(s.entry)} entry time - skipping today")

    checked = cur is not None and cur >= 2
    last_login = time.time()
    while True:
        t = dt.datetime.now(IST)
        m = minute_index(t)
        if m >= ab.N_MIN:
            break
        if m >= 2 and not checked:                # 09:17: is the exchange actually trading?
            checked = True
            if not market_open_today(client, today):
                log("Exchange shows no trading today (holiday) - exiting")
                return
        if cur is not None and m > cur:           # minute(s) rolled over: close bars
            for k in range(cur, min(m, ab.N_MIN)):
                live.close_bar(k)
                book.step(k)
            live.bars_frame(day, exp).to_parquet(partial, index=False)   # for restart recovery
            if all(s.done for s in book.sims):
                log("All strategies finished for the day")
                break
        if m >= 0:
            cur = m if cur is None or m > cur else cur
            if time.time() - last_login > 90 * 60:   # refresh before the session ages out
                client, last_login = sd._session(force_new=True), time.time()
            prices = ltp_many(client, "BFO", tokens)
            for tok, p in prices.items():
                if tok in tok_index:
                    live.tick(tok_index[tok], p)
            s = ltp_many(client, "BSE", [SPOT_TOKEN[1]]).get(SPOT_TOKEN[1])
            live.tick_spot(s)
        time.sleep(max(0.5, POLL_SEC - (dt.datetime.now(IST) - t).total_seconds()))
    finish(book, live, day, exp)


def check():
    """Connectivity test for the VPS: no trading, works any day."""
    import smartapi_data as sd
    hide_credentials_in_logs()
    sims = load_jobs()
    print(f"Strategies loaded from {ALGO}: " + ", ".join(f"{st['name']} ({w} days)" for st, w in sims))
    client = sd._session()
    print("Angel One login: OK")
    expiry, spot, chosen = contracts_today(client, dt.datetime.now(IST).date())
    prices = ltp_many(client, "BFO", [tok for _, _, tok in chosen])
    print(f"Sensex spot {spot:,.2f}; nearest expiry {expiry}; {len(chosen)} contracts in the band, "
          f"{len(prices)} returned a price")
    near = sorted(chosen, key=lambda c: abs(c[0] - spot))[:4]
    for k, kind, tok in near:
        print(f"   {k:.0f} {kind}: {prices.get(tok)}")
    print(f"Server time now (IST): {dt.datetime.now(IST):%Y-%m-%d %H:%M:%S}")


def finish(book, live, day, exp):
    live.bars_frame(day, exp).to_parquet(os.path.join(OUT, "bars", f"{day}.parquet"), index=False)
    partial = os.path.join(OUT, "bars", f"{day}.partial.parquet")
    if os.path.exists(partial):
        os.remove(partial)
    book.finish()


# ---------------------------------------------------------------- replay (offline tests)

def replay(df, day, exp, spot=None):
    """Feed a day's stored minute bars through the live path: each bar becomes
    four ticks (open, high, low, close), then the minute is closed."""
    df = df[df["trading_day"] == day]
    contracts = sorted(set(zip(df["strike"].astype(float), df["option_type"])))
    live = LiveDay(contracts, spot is not None)
    idx = {c: i for i, c in enumerate(contracts)}
    ts = df["timestamp"]
    df = df.assign(m=(ts.dt.hour * 60 + ts.dt.minute - ab.OPEN_MIN).to_numpy())
    by_min = {m: g for m, g in df.groupby("m")}
    book = Book(live, day, exp, ab.lot_size(TICKER))
    for m in range(ab.N_MIN):
        g = by_min.get(m)
        if g is not None:
            for r in g.itertuples(index=False):
                i = idx[(float(r.strike), r.option_type)]
                for p in (r.open, r.high, r.low, r.close):
                    live.tick(i, p)
        if spot is not None and not np.isnan(spot[m]):
            live.tick_spot(spot[m])
        live.close_bar(m)
        book.step(m)
        if all(s.done for s in book.sims):
            break
    return book.finish()


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay-hf", metavar="DATE", help="offline test: replay a day from hf_data/")
    ap.add_argument("--replay-saved", metavar="FILE", help="replay a day saved by a live run")
    ap.add_argument("--check", action="store_true",
                    help="log in, show the Sensex price and the contracts it would watch, then exit")
    args = ap.parse_args()
    if args.check:
        check()
    elif args.replay_hf:
        OUT = os.path.join(HERE, "paper_data_replay")
        day = args.replay_hf
        f = next((p for p in sorted(glob.glob(os.path.join(ab.DATA, TICKER, "*.parquet")))
                  if os.path.basename(p)[:10] >= day), None)
        if f is None:
            raise SystemExit("No hf_data file covers that day")
        df = pd.read_parquet(f)
        spot = ab.load_spot(TICKER).get(day)
        replay(df, day, os.path.basename(f)[:10], spot)
    elif args.replay_saved:
        OUT = os.path.join(HERE, "paper_data_replay")
        df = pd.read_parquet(args.replay_saved)
        day = str(df["trading_day"].iloc[0])
        replay(df, day, str(df["expiry"].iloc[0]))
    else:
        run_live()


if __name__ == "__main__":
    sys.exit(main())
