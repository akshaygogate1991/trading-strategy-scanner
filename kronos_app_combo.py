"""Kronos + the app's EMA/RSI/Elliott daily signal, together.

Uses the Kronos daily forecasts from kronos_test.py (logs/kronos_daily.csv, Sep 2025+,
all after Kronos's training data) and the app signal cached by ema_rsi_search.py
(swing_data/app_daily_signal.pkl, ends 2026-07-02 with the 1-minute data).

Both are known before the session they trade: app[day] is built from bars up to the
previous close, Kronos's row for `day` is the forecast of that day.

  app   = 18 EMA trend + Elliott bias (score >= 55) + RSI side, the full app CALL/PUT
  t18   = just the 18 EMA trend (up / down)
  kronos= P(up) >= 0.60 BUY, <= 0.40 SELL (the paper-tab rule)

Each rule trades the index at that day's open and exits at the close (and, as a second
view, the previous close to the close). Cost 0.015 % per side, as everywhere else.

  python kronos_app_combo.py      -> logs/kronos_app_combo.txt
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
COST = 0.00015
RNG = np.random.default_rng(5)


def load():
    k = pd.read_csv(os.path.join(HERE, "logs", "kronos_daily.csv"), parse_dates=["day"])
    app = pd.read_pickle(os.path.join(HERE, "swing_data", "app_daily_signal.pkl"))
    parts = []
    for (sym, _, _), df in app.items():
        x = k[k.sym == sym].set_index("day").join(df, how="inner")
        parts.append(x.assign(sym=sym))
    d = pd.concat(parts)
    d["kr"] = np.where(d.p_up >= .60, 1, np.where(d.p_up <= .40, -1, 0))
    return d


def line(name, side, ret):
    m = side != 0
    if m.sum() < 5:
        return f"  {name:<44} too few ({int(m.sum())})"
    pnl = side[m] * ret[m] - 2 * COST
    t = pnl.mean() / (pnl.std(ddof=1) / np.sqrt(len(pnl)))
    # random sides on the same days: how often does luck do as well?
    rnd = [(RNG.choice([-1, 1], len(pnl)) * ret[m] - 2 * COST).sum() for _ in range(2000)]
    beat = (np.array(rnd) >= pnl.sum()).mean() * 100
    return (f"  {name:<44} n={int(m.sum()):>4}  hit {(pnl > 0).mean() * 100:4.1f}%  avg {pnl.mean() * 100:+.3f}%  "
            f"total {pnl.sum() * 100:+6.1f}%  t {t:+.2f}  random as good {beat:4.1f}%")


def report(d, col, title):
    r = d[col].to_numpy()
    kr, app, t18 = d.kr.to_numpy(), d.app.to_numpy().astype(int), d.t18.to_numpy().astype(int)
    rules = [
        ("buy and hold (long every day)", np.ones(len(d), int)),
        ("Kronos alone (0.60 / 0.40)", kr),
        ("18 EMA trend alone", t18),
        ("app CALL/PUT alone", app),
        ("Kronos, only with the 18 EMA trend", np.where(kr == t18, kr, 0)),
        ("Kronos, only against the 18 EMA trend", np.where(kr == -t18, kr, 0)),
        ("app CALL/PUT, only if Kronos agrees", np.where(app == kr, app, 0)),
        ("app CALL/PUT, skip if Kronos disagrees", np.where(app * kr >= 0, app, 0)),
        ("18 EMA trend, skip if Kronos disagrees", np.where(t18 * kr >= 0, t18, 0)),
    ]
    out = [f"\n=== {title}  ({d.index.min().date()} to {d.index.max().date()}, {len(d)} index-days)"]
    out += [line(n, s, r) for n, s in rules]
    return out


def main():
    d = load()
    lines = [f"Days where Kronos and the app overlap: {len(d)}  "
             f"(app CALL/PUT on {int((d.app != 0).sum())} of them, Kronos traded on {int((d.kr != 0).sum())})",
             f"Kronos and 18 EMA trend agree on {(d.kr == d.t18)[d.kr != 0].mean() * 100:.0f}% of Kronos's trades"]
    lines += report(d, "trade_ret", "Open to close (tradable)")
    lines += report(d, "real_ret", "Previous close to close (upper bound, needs an overnight hold)")
    for sym, g in d.groupby("sym"):
        lines += report(g, "trade_ret", f"{sym}, open to close")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(HERE, "logs", "kronos_app_combo.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
