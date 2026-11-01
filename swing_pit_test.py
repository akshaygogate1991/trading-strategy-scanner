"""Momentum retest without today's-index survivorship bias.

swing_screener_test.py used today's Nifty 200 list for all 13 years, so stocks
that crashed and left the index were missing. Here the universe is rebuilt at
each month-end from every stock listed on NSE today: the 200 (or 100) stocks
with the highest median daily traded value over the previous 6 months, with at
least a year of history and a price of Rs 20 or more. Only data up to that
month-end is used, so a stock is in the universe only if it was big then.

Still missing: companies delisted before today (for example DHFL, Jet Airways,
Reliance Capital). Their losses are not in the test, so results remain a bit
optimistic. Traded value stands in for market cap, which NSE uses for its index.

Same trading as swing_screener_test.py: 5 picks at each month-end, bought at
the next day's open with Rs 1 lakh each, held a month, delivery costs included.

Needs swing_data/prices_all.pkl from `python fetch_swing_data.py --all`.
Run:  python swing_pit_test.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from swing_screener_test import (N_PICKS, PER_STOCK, SPLIT, rebalance_days,
                                 simulate, stats)

DATA = Path(__file__).parent / "swing_data"
INDEX = "^NSEI"
RANDOM_RUNS = 300
MIN_PRICE = 20


def load():
    with open(DATA / "prices_all.pkl", "rb") as f:
        prices = pickle.load(f)
    nifty = prices.pop(INDEX)
    close = pd.DataFrame({t: d["Close"] for t, d in prices.items()}).sort_index()
    raw_close = close.copy()
    close = close.ffill(limit=5)
    opn = pd.DataFrame({t: d["Open"] for t, d in prices.items()}).reindex(close.index)
    value = pd.DataFrame({t: d["Close"] * d["Volume"] for t, d in prices.items()}).reindex(close.index)
    return close, raw_close, opn, value, nifty


def main() -> None:
    close, raw_close, opn, value, nifty = load()
    days = rebalance_days(close)
    print(f"{close.shape[1]} stocks, {len(days) - 1} monthly rebalances "
          f"{days[0].date()} to {days[-2].date()}")

    bars = raw_close.notna().cumsum()
    traded_recently = raw_close.notna().rolling(5, min_periods=1).max().astype(bool)
    liq = value.rolling(126, min_periods=100).median()
    sma200 = close.rolling(200, min_periods=200).mean()
    high52 = close.rolling(252, min_periods=240).max()
    ret6 = close / close.shift(126) - 1
    ret12_1 = close.shift(21) / close.shift(252) - 1

    def universe(day, size):
        ok = (bars.loc[day] >= 260) & traded_recently.loc[day] & (close.loc[day] >= MIN_PRICE)
        return liq.loc[day][ok].dropna().nlargest(size).index

    out = []
    tables, extra = {}, {}
    nc = nifty["Close"]
    nifty_m = pd.Series({d: nc.loc[:days[i + 1]].iloc[-1] / nc.loc[:d].iloc[-1] - 1
                         for i, d in enumerate(days[:-1])})
    capital = N_PICKS * PER_STOCK
    idx = close.index

    for size in (200, 100):
        uni = {d: universe(d, size) for d in days}

        def momentum_12_1(day, uni=uni):
            u = uni[day]
            ok = close.loc[day, u] > sma200.loc[day, u]
            return ret12_1.loc[day, u][ok].dropna().nlargest(N_PICKS).index.tolist()

        def momentum_high(day, uni=uni):
            u = uni[day]
            ok = close.loc[day, u] >= 0.9 * high52.loc[day, u]
            return ret6.loc[day, u][ok].dropna().nlargest(N_PICKS).index.tolist()

        for name, fn in (("momentum_12_1", momentum_12_1), ("momentum_high", momentum_high)):
            tables[f"{name} top{size}"] = simulate(fn, close, opn, days)

        eq = []
        for i, day in enumerate(days[:-1]):
            t0 = idx[idx.get_loc(day) + 1]
            nxt = idx.get_loc(days[i + 1]) + 1
            t1 = idx[nxt] if nxt < len(idx) else days[i + 1]
            u = uni[day]
            p1 = opn.loc[t1, u].fillna(close.loc[t1, u])
            eq.append((p1 / opn.loc[t0, u] - 1).dropna().mean())
        extra[f"all top{size} (gross)"] = pd.Series(eq, index=days[:-1])

        rng = np.random.default_rng(7)
        rand = []
        for _ in range(RANDOM_RUNS):
            draws = {d: list(rng.choice(uni[d], N_PICKS, replace=False)) for d in days[:-1]}
            rand.append(stats(simulate(lambda d: draws[d], close, opn, days)["net"], capital, nifty_m))
        extra[f"random top{size}"] = pd.DataFrame(rand)
        if size == 200:
            uni200 = uni

    def section(title, mask):
        lines = [f"\n=== {title} ===",
                 f"{'rule':<24}{'CAGR %':>8}{'Sharpe':>8}{'MaxDD %':>9}{'Win %':>7}"
                 f"{'Beat N %':>9}{'Excess/yr':>10}{'t':>6}{'rand pct':>9}"]
        for name, sim in tables.items():
            s = sim[mask(sim.index)]
            st_ = stats(s["net"], capital, nifty_m)
            rnd = extra["random top" + name.split("top")[1]]
            pct = (rnd["CAGR %"] < st_["CAGR %"]).mean() * 100 if title.startswith("FULL") else float("nan")
            lines.append(f"{name:<24}{st_['CAGR %']:>8.1f}{st_['Sharpe']:>8.2f}{st_['MaxDD %']:>9.1f}"
                         f"{st_['Win months %']:>7.0f}{st_['Beat Nifty %']:>9.0f}{st_['Excess/yr %']:>10.1f}"
                         f"{st_['t (excess)']:>6.2f}{pct:>9.0f}")
        for name in ("all top200 (gross)", "all top100 (gross)"):
            a = extra[name][mask(extra[name].index)]
            sa = stats(a * capital, capital, nifty_m)
            lines.append(f"{name:<24}{sa['CAGR %']:>8.1f}{sa['Sharpe']:>8.2f}{sa['MaxDD %']:>9.1f}"
                         f"{sa['Win months %']:>7.0f}{sa['Beat Nifty %']:>9.0f}{sa['Excess/yr %']:>10.1f}{sa['t (excess)']:>6.2f}")
        sn = stats(nifty_m[mask(nifty_m.index)] * capital, capital)
        lines.append(f"{'nifty50':<24}{sn['CAGR %']:>8.1f}{sn['Sharpe']:>8.2f}{sn['MaxDD %']:>9.1f}{sn['Win months %']:>7.0f}")
        return lines

    out += section("FULL PERIOD", lambda ix: np.ones(len(ix), bool))
    out += section(f"FIRST HALF (before {SPLIT.date()})", lambda ix: ix < SPLIT)
    out += section(f"SECOND HALF (from {SPLIT.date()})", lambda ix: ix >= SPLIT)
    for size in (200, 100):
        q = extra[f"random top{size}"].quantile([0.05, 0.5, 0.95])
        out.append(f"Random 5 from top {size}, {RANDOM_RUNS} runs: CAGR 5%/50%/95% = "
                   f"{q.at[0.05, 'CAGR %']:.1f} / {q.at[0.5, 'CAGR %']:.1f} / {q.at[0.95, 'CAGR %']:.1f}")

    out.append("\n=== Calendar years, net % on Rs 5 lakh ===")
    yr = pd.DataFrame({n: s["net"].groupby(s.index.year).sum() / capital * 100 for n, s in tables.items()})
    yr["nifty50"] = ((1 + nifty_m).groupby(nifty_m.index.year).prod() - 1) * 100
    out.append(yr.round(1).to_string())

    # sanity check for bad Yahoo prices: the biggest single-stock months
    sim = tables["momentum_12_1 top200"]
    big = []
    for i, day in enumerate(days[:-1]):
        t0 = idx[idx.get_loc(day) + 1]
        nxt = idx.get_loc(days[i + 1]) + 1
        t1 = idx[nxt] if nxt < len(idx) else days[i + 1]
        for t in sim.at[day, "picks"].split():
            tk = t + ".NS"
            p1 = opn.at[t1, tk] if pd.notna(opn.at[t1, tk]) else close.loc[:t1, tk].dropna().iloc[-1]
            big.append((p1 / opn.at[t0, tk] - 1, day.date(), t))
    big.sort(reverse=True)
    out.append("\nBiggest single-stock months (momentum_12_1 top200): " +
               ", ".join(f"{t} {d} {r * 100:+.0f}%" for r, d, t in big[:6]))
    out.append("Worst: " + ", ".join(f"{t} {d} {r * 100:+.0f}%" for r, d, t in big[-6:]))
    without = sim["net"].sum() - sum(r * PER_STOCK for r, _, _ in big[:5])
    out.append(f"Net without the 5 best stock-months: Rs {without:,.0f} (with: Rs {sim['net'].sum():,.0f})")

    last = days[-1]
    u = uni200[last]
    pick = ret12_1.loc[last, u][close.loc[last, u] > sma200.loc[last, u]].dropna().nlargest(N_PICKS)
    out.append(f"\nPicks from the close of {last.date()}, momentum_12_1 top200: " +
               ", ".join(f"{t.replace('.NS', '')} ({r * 100:+.0f}%)" for t, r in pick.items()))
    old = set(pd.read_csv(DATA / "nifty200.csv")["Symbol"].str.strip() + ".NS")
    gone = sorted({t for d in days[:-1] for t in uni200[d]} - old)
    out.append(f"Stocks that were in the top 200 at some point but are not in today's Nifty 200: "
               f"{len(gone)} (e.g. {', '.join(t.replace('.NS', '') for t in gone[:12])})")

    text = "\n".join(out)
    print(text)
    Path("logs").mkdir(exist_ok=True)
    Path("logs/swing_pit.txt").write_text(text, encoding="utf-8")
    pd.concat(tables, names=["rule"]).to_csv("logs/swing_pit_months.csv")


if __name__ == "__main__":
    main()
