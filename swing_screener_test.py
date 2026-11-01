"""Monthly swing screener on the Nifty 200: which stock-picking rule is worth using?

Every month-end, each rule picks up to 5 stocks from the close. They are bought at
the next day's open with equal money (Rs 1 lakh each, Rs 5 lakh capital) and
held until the next month's rebalance (about one month). Delivery trades, long only.

RULES
  app_signal    app.py's CALL signal: close above a rising 18 EMA, Elliott bullish
                impulse (score 55+), RSI 50+. Ranked by Elliott score, then RSI.
  momentum_12_1 best 12-month return skipping the last month, above the 200-day
                average (the classic momentum rule).
  momentum_high best 6-month return among stocks within 10% of their 52-week high.
  pullback      in an uptrend (above the 200-day average, top 30% 6-month return)
                but RSI(14) below 40: the 5 lowest RSI.
CONTROLS
  random        5 random stocks each month, 500 separate runs.
  all_200       all stocks with equal money (what the universe itself did).
  nifty50       buy and hold the Nifty 50 index (no costs).

COSTS (charged only on stocks that change at a rebalance)
  STT 0.1% buy and sell, stamp 0.015% buy, exchange 0.00297% + GST, Rs 5 per
  order + GST, DP Rs 15.93 + GST per sell, slippage 0.1% each side.

CAVEAT: the list is today's Nifty 200. Companies that fell out of the index are
missing, so every rule (and the random control) looks better than it would have.
Compare rules with the random control, not with zero.

Needs swing_data/prices.pkl from fetch_swing_data.py.
Run:  python swing_screener_test.py
"""
from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ema_merge_test import app_functions

DATA = Path(__file__).parent / "swing_data"
INDEX = "^NSEI"
N_PICKS = 5
PER_STOCK = 100_000
SLIP = 0.001
RF = 0.065
SPLIT = pd.Timestamp("2020-01-01")
RANDOM_RUNS = 500


def buy_cost(value: float) -> float:
    exch = value * 0.0000297
    return value * (0.001 + 0.00015 + SLIP) + exch * 1.18 + 5 * 1.18


def sell_cost(value: float) -> float:
    exch = value * 0.0000297
    return value * (0.001 + SLIP) + exch * 1.18 + 5 * 1.18 + 15.93 * 1.18


def load():
    with open(DATA / "prices.pkl", "rb") as f:
        prices = pickle.load(f)
    nifty = prices.pop(INDEX)
    close = pd.DataFrame({t: d["Close"] for t, d in prices.items()})
    opn = pd.DataFrame({t: d["Open"] for t, d in prices.items()})
    # Yahoo sometimes leaves a stock without a row on a trading day
    close = close.sort_index().ffill(limit=5)
    opn = opn.reindex(close.index)
    return prices, close, opn, nifty


def rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def rebalance_days(close: pd.DataFrame) -> list[pd.Timestamp]:
    """Last trading day of each month, from the first date with a year of history."""
    idx = close.index
    ends = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).last()
    first = idx[260]
    return [d for d in ends if d >= first]


def app_scores(prices, close, days) -> dict[pd.Timestamp, list[tuple[str, int, float]]]:
    """app.py CALL candidates at each rebalance day: (ticker, Elliott score, RSI)."""
    cache = DATA / "app_signal_cache.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            saved = pickle.load(f)
        if saved["days"] == days:
            return saved["picks"]
    picks = {}
    APP = app_functions()
    full = {t: APP["add_indicators"](d) for t, d in prices.items()}
    for i, day in enumerate(days):
        found = []
        for t, d in full.items():
            hist = d.loc[:day]
            if len(hist) < 200 or hist.index[-1] < day - pd.Timedelta(days=7):
                continue
            hist = hist.tail(200)
            if APP["trend_18ema"](hist) != "up":
                continue
            r = float(hist["rsi14"].iloc[-1])
            if not r >= 50:
                continue
            wave = APP["elliott_directional"](hist)
            if wave["bias"] == "Bullish" and int(wave["score"]) >= 55:
                found.append((t, int(wave["score"]), r))
        picks[day] = found
        if i % 20 == 0:
            print(f"  app.py signal: {day.date()} ({i + 1}/{len(days)}), {len(found)} stocks qualify")
    with open(cache, "wb") as f:
        pickle.dump({"days": days, "picks": picks}, f)
    return picks


def make_rules(prices, close, days):
    sma200 = close.rolling(200, min_periods=200).mean()
    high52 = close.rolling(252, min_periods=240).max()
    ret6 = close / close.shift(126) - 1
    ret12_1 = close.shift(21) / close.shift(252) - 1
    rsi14 = rsi(close)
    app = app_scores(prices, close, days)

    def live(day):
        c = close.loc[day]
        return c.notna() & sma200.loc[day].notna()

    def app_signal(day):
        cands = sorted(app[day], key=lambda x: (-x[1], -x[2]))
        return [t for t, _, _ in cands[:N_PICKS]]

    def momentum_12_1(day):
        ok = live(day) & (close.loc[day] > sma200.loc[day])
        return ret12_1.loc[day][ok].dropna().nlargest(N_PICKS).index.tolist()

    def momentum_high(day):
        ok = live(day) & (close.loc[day] >= 0.9 * high52.loc[day])
        return ret6.loc[day][ok].dropna().nlargest(N_PICKS).index.tolist()

    def pullback(day):
        ok = live(day) & (close.loc[day] > sma200.loc[day])
        r6 = ret6.loc[day][ok].dropna()
        strong = r6[r6 >= r6.quantile(0.7)].index
        r = rsi14.loc[day, strong]
        return r[r < 40].nsmallest(N_PICKS).index.tolist()

    return {"app_signal": app_signal, "momentum_12_1": momentum_12_1,
            "momentum_high": momentum_high, "pullback": pullback}, live


def simulate(pick_fn, close, opn, days, per_stock: float = PER_STOCK) -> pd.DataFrame:
    """Monthly P&L in rupees. Unused slots sit in cash at 0%."""
    idx = close.index
    rows, held = [], []
    for i, day in enumerate(days[:-1]):
        nxt = days[i + 1]
        t0 = idx[idx.get_loc(day) + 1]
        t1 = idx[idx.get_loc(nxt) + 1] if idx.get_loc(nxt) + 1 < len(idx) else nxt
        picks = [t for t in pick_fn(day) if pd.notna(opn.at[t0, t]) and opn.at[t0, t] > 0]
        pnl = cost = 0.0
        for t in picks:
            p0 = opn.at[t0, t]
            p1 = opn.at[t1, t] if pd.notna(opn.at[t1, t]) else close.loc[:t1, t].dropna().iloc[-1]
            pnl += per_stock * (p1 / p0 - 1)
        buys = set(picks) - set(held)
        cost += len(buys) * buy_cost(per_stock) + len(set(held) - set(picks)) * sell_cost(per_stock)
        held = picks
        rows.append({"month": day, "n": len(picks), "buys": len(buys), "gross": pnl, "cost": cost,
                     "net": pnl - cost, "picks": " ".join(s.replace(".NS", "") for s in picks)})
    return pd.DataFrame(rows).set_index("month")


def stats(monthly_net: pd.Series, capital: float, bench: pd.Series | None = None) -> dict:
    r = monthly_net / capital
    years = len(r) / 12
    equity = (1 + r).cumprod()
    out = {
        "CAGR %": (equity.iloc[-1] ** (1 / years) - 1) * 100,
        "Sharpe": (r.mean() * 12 - RF) / (r.std() * math.sqrt(12)) if r.std() > 0 else float("nan"),
        "MaxDD %": ((equity / equity.cummax()) - 1).min() * 100,
        "Win months %": (r > 0).mean() * 100,
    }
    if bench is not None:
        ex = r - bench.reindex(r.index)
        out["Beat Nifty %"] = (ex > 0).mean() * 100
        out["Excess/yr %"] = ex.mean() * 12 * 100
        out["t (excess)"] = ex.mean() / (ex.std() / math.sqrt(len(ex)))
    return out


def main() -> None:
    prices, close, opn, nifty = load()
    days = rebalance_days(close)
    print(f"{close.shape[1]} stocks, {len(days) - 1} monthly rebalances "
          f"{days[0].date()} to {days[-2].date()}")
    rules, live = make_rules(prices, close, days)

    nc = nifty["Close"]
    nifty_m = pd.Series({d: nc.loc[:days[i + 1]].iloc[-1] / nc.loc[:d].iloc[-1] - 1
                         for i, d in enumerate(days[:-1])})
    capital = N_PICKS * PER_STOCK

    tables = {name: simulate(fn, close, opn, days) for name, fn in rules.items()}

    # all_200: equal weight, monthly, costs ignored (it is a reference, not a strategy)
    idx = close.index
    eq = []
    for i, day in enumerate(days[:-1]):
        t0 = idx[idx.get_loc(day) + 1]
        t1 = idx[idx.get_loc(days[i + 1]) + 1] if idx.get_loc(days[i + 1]) + 1 < len(idx) else days[i + 1]
        ok = live(day)
        r = (opn.loc[t1, ok] / opn.loc[t0, ok] - 1).dropna()
        eq.append(r.mean())
    all200 = pd.Series(eq, index=days[:-1])

    rng = np.random.default_rng(7)
    eligible = {d: live(d)[live(d)].index.tolist() for d in days[:-1]}
    rand_stats = []
    for _ in range(RANDOM_RUNS):
        draws = {d: list(rng.choice(eligible[d], N_PICKS, replace=False)) for d in days[:-1]}
        sim = simulate(lambda d: draws[d], close, opn, days)
        rand_stats.append(stats(sim["net"], capital, nifty_m))
    rand_df = pd.DataFrame(rand_stats)

    def section(title, mask):
        lines = [f"\n=== {title} ===",
                 f"{'rule':<15}{'CAGR %':>8}{'Sharpe':>8}{'MaxDD %':>9}{'Win %':>7}"
                 f"{'Beat N %':>9}{'Excess/yr':>10}{'t':>6}{'avg n':>7}{'rand pct':>9}"]
        for name, sim in tables.items():
            s = sim[mask(sim.index)]
            st_ = stats(s["net"], capital, nifty_m)
            n = s["n"].mean()
            pct = (rand_df["CAGR %"] < st_["CAGR %"]).mean() * 100 if title.startswith("FULL") else float("nan")
            lines.append(f"{name:<15}{st_['CAGR %']:>8.1f}{st_['Sharpe']:>8.2f}{st_['MaxDD %']:>9.1f}"
                         f"{st_['Win months %']:>7.0f}{st_['Beat Nifty %']:>9.0f}{st_['Excess/yr %']:>10.1f}"
                         f"{st_['t (excess)']:>6.2f}{n:>7.1f}{pct:>9.0f}")
        a = all200[mask(all200.index)]
        n_ = nifty_m[mask(nifty_m.index)]
        sa = stats(a * capital, capital, nifty_m)
        sn = stats(n_ * capital, capital)
        lines.append(f"{'all_200 (gross)':<15}{sa['CAGR %']:>8.1f}{sa['Sharpe']:>8.2f}{sa['MaxDD %']:>9.1f}"
                     f"{sa['Win months %']:>7.0f}{sa['Beat Nifty %']:>9.0f}{sa['Excess/yr %']:>10.1f}{sa['t (excess)']:>6.2f}")
        lines.append(f"{'nifty50':<15}{sn['CAGR %']:>8.1f}{sn['Sharpe']:>8.2f}{sn['MaxDD %']:>9.1f}{sn['Win months %']:>7.0f}")
        return lines

    out = section("FULL PERIOD", lambda ix: np.ones(len(ix), bool))
    out += section(f"FIRST HALF (before {SPLIT.date()})", lambda ix: ix < SPLIT)
    out += section(f"SECOND HALF (from {SPLIT.date()})", lambda ix: ix >= SPLIT)
    q = rand_df.quantile([0.05, 0.5, 0.95])
    out.append(f"\nRandom 5 stocks, {RANDOM_RUNS} runs (full period, net): "
               f"CAGR 5%/50%/95% = {q.at[0.05, 'CAGR %']:.1f} / {q.at[0.5, 'CAGR %']:.1f} / {q.at[0.95, 'CAGR %']:.1f}, "
               f"Sharpe median {q.at[0.5, 'Sharpe']:.2f}")
    out.append("'rand pct' = share of random runs with a lower CAGR than the rule.")

    out.append("\n=== Costs (full period, Rs) ===")
    for name, sim in tables.items():
        out.append(f"{name:<15} gross {sim['gross'].sum():>12,.0f}  costs {sim['cost'].sum():>10,.0f}  "
                   f"net {sim['net'].sum():>12,.0f}  new buys/yr {sim['buys'].sum() / (len(sim) / 12):>5.1f}")

    out.append("\n=== Calendar years, net % on Rs 5 lakh ===")
    yr = pd.DataFrame({n: s["net"].groupby(s.index.year).sum() / capital * 100 for n, s in tables.items()})
    yr["nifty50"] = ((1 + nifty_m).groupby(nifty_m.index.year).prod() - 1) * 100
    out.append(yr.round(1).to_string())

    last = days[-1]
    out.append(f"\n=== Picks from the close of {last.date()} (for the coming month) ===")
    for name, fn in rules.items():
        out.append(f"{name:<15} " + ", ".join(t.replace(".NS", "") for t in fn(last)))

    text = "\n".join(out)
    print(text)
    Path("logs").mkdir(exist_ok=True)
    Path("logs/swing_screener.txt").write_text(text, encoding="utf-8")
    pd.concat(tables, names=["rule"]).to_csv("logs/swing_screener_months.csv")


if __name__ == "__main__":
    main()
