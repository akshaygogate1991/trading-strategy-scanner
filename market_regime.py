"""What kind of market was 2026, compared with 2023-2025?

Daily (yfinance + India VIX file, up to the latest day):
  return, realized volatility, India VIX, VIX minus realized (the option seller's
  edge), average daily range, gaps, big-move days, drawdown.
Intraday (hf_data 1-minute index bars, to Jul 2026):
  how often the day trends vs whipsaws after 09:35 - the thing Sensex
  Directional is most sensitive to.
Options (hf_data): ATM straddle at 09:35 as % of spot - how rich premiums were.
Stocks (yfinance daily, the 30 tested F&O names): median return / volatility.

  python market_regime.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "hf_data")
INDICES = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN"}
PERIODS = [("2023 H2", "2023-08-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
           ("2025", "2025-01-01", "2025-12-31"), ("2026 YTD", "2026-01-01", "2026-12-31")]


def yf_daily(symbols, start="2023-06-01"):
    import yfinance as yf
    d = yf.download(symbols, start=start, auto_adjust=False, progress=False, group_by="ticker", threads=True)
    return d


def period_of(dates):
    """Period label for each 'YYYY-MM-DD' string (numpy array, aligned by position)."""
    dates = np.asarray(dates, dtype=str)
    out = np.full(len(dates), None, dtype=object)
    for lab, a, b in PERIODS:
        out[(dates >= a) & (dates <= b)] = lab
    return out


def daily_table(df, vix=None):
    df = df.dropna(subset=["Close"]).copy()
    df["ret"] = df["Close"].pct_change()
    df["range"] = (df["High"] - df["Low"]) / df["Open"] * 100
    df["gap"] = (df["Open"] / df["Close"].shift() - 1).abs() * 100
    df["per"] = period_of(df.index.strftime("%Y-%m-%d"))
    rows = {}
    for lab, g in df.groupby("per"):
        eq = g["Close"]
        r = dict(
            ret=(eq.iloc[-1] / eq.iloc[0] - 1) * 100,
            rv=g["ret"].std() * np.sqrt(252) * 100,
            range=g["range"].mean(),
            gap=g["gap"].mean(),
            big=(g["ret"].abs() > 0.015).mean() * 100,
            down=(g["ret"] < 0).mean() * 100,
            dd=((eq / eq.cummax()) - 1).min() * 100,
        )
        if vix is not None:
            v = vix.reindex(g.index).dropna()
            r["vix"] = v.mean()
            r["vrp"] = r["vix"] - r["rv"]
        rows[lab] = r
    return pd.DataFrame(rows).T.reindex([p[0] for p in PERIODS])


def intraday_table(ticker):
    s = pd.read_parquet(os.path.join(DATA, "index", f"{ticker}.parquet"))
    s = s[s["trading_day"] >= "2023-08-01"]
    ts = s["timestamp"]
    s = s.assign(m=(ts.dt.hour * 60 + ts.dt.minute - 555).to_numpy())
    rows = []
    for day, g in s.groupby("trading_day"):
        g = g.set_index("m")
        if 20 not in g.index or 345 not in g.index:
            continue
        p0 = g.at[20, "open"]                          # 09:35
        w = g.loc[20:345]
        up = (w["high"].max() / p0 - 1) * 100
        dn = (1 - w["low"].min() / p0) * 100
        end = (g.at[345, "open"] / p0 - 1) * 100       # 15:00
        rows.append(dict(day=day, up=up, dn=dn, end=end))
    d = pd.DataFrame(rows)
    d["per"] = period_of(d["day"])
    out = {}
    for lab, g in d.groupby("per"):
        both = (g["up"] >= 0.5) & (g["dn"] >= 0.5)     # whipsaw: 0.5% each way from 09:35
        trend = g["end"].abs() >= 0.8                  # closed >= 0.8% away from 09:35
        quiet = (g["up"] < 0.5) & (g["dn"] < 0.5)
        out[lab] = dict(days=len(g), quiet=quiet.mean() * 100, trend=trend.mean() * 100,
                        whipsaw=both.mean() * 100, move=g["end"].abs().mean(),
                        excursion=np.maximum(g["up"], g["dn"]).mean())
    return pd.DataFrame(out).T.reindex([p[0] for p in PERIODS])


def straddle_table(ticker):
    """ATM CE+PE premium at 09:35 as % of spot, for the nearest expiry."""
    spot = pd.read_parquet(os.path.join(DATA, "index", f"{ticker}.parquet"))
    t = spot["timestamp"]
    spot = spot[(t.dt.hour == 9) & (t.dt.minute == 35)].groupby("trading_day")["open"].first()
    rows, prev = [], "0000"
    for f in sorted(glob.glob(os.path.join(DATA, ticker, "*.parquet"))):
        exp = os.path.basename(f)[:10]
        if exp < "2023-08-01":
            prev = exp
            continue
        o = pd.read_parquet(f, columns=["timestamp", "open", "trading_day", "strike", "option_type"])
        t = o["timestamp"]
        o = o[(o["trading_day"] > prev) & (t.dt.hour == 9) & (t.dt.minute == 35)]
        prev = exp
        for day, g in o.groupby("trading_day"):
            if day not in spot.index or (pd.Timestamp(exp) - pd.Timestamp(day)).days > 7:
                continue
            ks = g["strike"].unique()
            atm = ks[np.argmin(np.abs(ks - spot[day]))]
            a = g[g["strike"] == atm].groupby("option_type")["open"].first()
            if {"CE", "PE"} <= set(a.index):
                rows.append(dict(day=day, pct=(a["CE"] + a["PE"]) / spot[day] * 100,
                                 dte=(pd.Timestamp(exp) - pd.Timestamp(day)).days))
    d = pd.DataFrame(rows)
    d["per"] = period_of(d["day"])
    d["exp_pct"] = d["pct"].where(d["dte"] == 0)
    return d.groupby("per").agg(days=("pct", "size"), straddle_pct=("pct", "mean"),
                                expiry_day_pct=("exp_pct", "mean")).reindex([p[0] for p in PERIODS])


def main():
    pd.set_option("display.width", 160)
    vix = pd.read_csv(os.path.join(HERE, "vix_data", "nifty_vix_8y.csv"), index_col="Date", parse_dates=True)["VIX"]
    raw = yf_daily(list(INDICES.values()) + ["^INDIAVIX"])
    try:                                             # extend the VIX file with Yahoo's latest
        yv = raw["^INDIAVIX"]["Close"].dropna()
        vix = pd.concat([vix, yv[yv.index > vix.index.max()]])
    except KeyError:
        pass
    last = raw.index.max().date()
    print(f"Daily data to {last}; intraday and option data to Jul 2026\n")
    print("Columns: ret = return %, rv = realized volatility % (annualised), vix = India VIX,")
    print("vrp = VIX - rv (option sellers' edge; bigger is better for selling), range = avg daily")
    print("high-low %, gap = avg opening gap %, big = % of days moving >1.5%, down = % down days,")
    print("dd = worst fall from a peak %\n")
    for name, sym in INDICES.items():
        t = daily_table(raw[sym], vix if name != "SENSEX" else vix)
        print(f"== {name} (daily)\n{t.round(2).to_string()}\n")

    n = raw[INDICES["NIFTY"]].dropna(subset=["Close"])
    n26 = n[n.index >= "2025-12-31"]
    m = n26["Close"].resample("ME").last().pct_change().dropna() * 100
    mv = vix[vix.index >= "2026-01-01"].resample("ME").mean()
    mr = n26["Close"].pct_change().groupby(n26.index.to_period("M")).std() * np.sqrt(252) * 100
    mt = pd.DataFrame({"nifty_ret%": m.values}, index=m.index.strftime("%Y-%m"))
    mt["vix"] = mv.reindex(m.index).values
    mt["realized_vol"] = mr.reindex(m.index.to_period("M")).values
    print(f"== NIFTY month by month, 2026\n{mt.round(1).to_string()}\n")

    print("Intraday after 09:35 (what Sensex Directional lives on): quiet = stayed within 0.5%")
    print("both ways; trend = 15:00 close >= 0.8% from 09:35; whipsaw = went 0.5% BOTH ways;")
    print("move = avg |09:35 -> 15:00| %; excursion = avg biggest move away from 09:35 %\n")
    for name in INDICES:
        print(f"== {name} (intraday)\n{intraday_table(name).round(2).to_string()}\n")

    print("Option premium: ATM straddle (CE+PE) at 09:35 as % of spot, nearest weekly expiry\n")
    for name in ("NIFTY", "SENSEX"):
        print(f"== {name}\n{straddle_table(name).round(3).to_string()}\n")

    stocks = json.load(open(os.path.join(HERE, "logs", "stock_sizes.json")))[:30]
    syms = [n.replace("&", "%26") if False else n for n, _ in stocks]
    sraw = yf_daily([f"{s}.NS" for s in syms])
    rows = []
    for s in syms:
        try:
            t = daily_table(sraw[f"{s}.NS"])
        except KeyError:
            continue
        t["stock"] = s
        rows.append(t.reset_index(names="per"))
    st = pd.concat(rows)
    agg = st.groupby("per").agg(stocks=("stock", "nunique"), median_ret=("ret", "median"),
                                pct_down=("ret", lambda x: (x < 0).mean() * 100),
                                median_rv=("rv", "median"), median_range=("range", "median"),
                                median_dd=("dd", "median")).reindex([p[0] for p in PERIODS])
    print(f"== 30 F&O STOCKS (daily, medians across stocks)\n{agg.round(2).to_string()}\n")
    s26 = st[st["per"] == "2026 YTD"].set_index("stock")["ret"].sort_values()
    print("2026 YTD return by stock %:\n" + "  ".join(f"{k} {v:+.0f}" for k, v in s26.items()))


if __name__ == "__main__":
    main()
