"""Download daily prices for the Nifty 200 stocks plus the Nifty 50 index.

Free Yahoo Finance data, no key needed. Saved to swing_data/prices.pkl as
{ticker: DataFrame(Open, High, Low, Close, Volume)}. Re-running downloads again.

The stock list is today's Nifty 200 (swing_data/nifty200.csv, from NSE). Stocks
that were dropped from the index in the past are missing, so backtests on this
list look somewhat better than reality (survivorship bias).

With --all it downloads every EQ stock listed on NSE today
(swing_data/equity_list.csv, NSE's EQUITY_L.csv) into prices_all.pkl, so a test
can rebuild the top 200 at each past date. Delisted companies are still missing.

Run:  python fetch_swing_data.py
      python fetch_swing_data.py --all      (about 2,400 stocks, takes a while)
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA = Path(__file__).parent / "swing_data"
START = "2013-01-01"
INDEX = "^NSEI"


def download(tickers: list[str], chunk: int = 200) -> dict[str, pd.DataFrame]:
    out = {}
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        raw = yf.download(part, start=START, auto_adjust=True, group_by="ticker",
                          progress=False, threads=True)
        for t in part:
            if t not in raw.columns.get_level_values(0):
                continue
            df = raw[t].dropna(subset=["Close"])
            if len(df) > 250:
                out[t] = df[["Open", "High", "Low", "Close", "Volume"]]
        if len(tickers) > chunk:
            print(f"  {min(i + chunk, len(tickers))}/{len(tickers)} requested, {len(out)} usable")
    return out


def main() -> None:
    every = "--all" in sys.argv
    if every:
        lst = pd.read_csv(DATA / "equity_list.csv", skipinitialspace=True)
        symbols = lst.loc[lst["SERIES"].str.strip() == "EQ", "SYMBOL"].str.strip()
    else:
        symbols = pd.read_csv(DATA / "nifty200.csv")["Symbol"].str.strip()
    tickers = [f"{s}.NS" for s in symbols] + [INDEX]
    out = download(tickers)
    missing = sorted(set(tickers) - set(out))
    with open(DATA / ("prices_all.pkl" if every else "prices.pkl"), "wb") as f:
        pickle.dump(out, f)
    print(f"saved {len(out)} tickers, {min(d.index[0] for d in out.values()).date()} "
          f"to {max(d.index[-1] for d in out.values()).date()}")
    if missing and not every:
        print("missing:", ", ".join(missing))
    elif missing:
        print(f"{len(missing)} without enough history (mostly listed in the last year)")


if __name__ == "__main__":
    main()
