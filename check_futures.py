"""What index-futures contracts does Angel One actually expose?

Run:  python check_futures.py                  (NIFTY)
      python check_futures.py --name BANKNIFTY


Written because vwap_ma_test.py found zero futures contracts, and guessing
again would waste another round trip. This prints what is really in the
instrument master so we can see the field values rather than assume them.

Run:  python check_futures.py
"""
from __future__ import annotations

import datetime as dt
from collections import Counter

import smartapi_data as sd


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="NIFTY",
                    help="underlying name in the master, e.g. NIFTY, BANKNIFTY")
    args = ap.parse_args()
    NAME = args.name.upper()
    print(f"Looking for {NAME} futures.")
    print("Downloading Angel One instrument master...")
    master = sd._instrument_master()
    print(f"  {len(master):,} instruments total\n")

    nfo = [r for r in master if r.get("exch_seg") == "NFO"]
    print(f"NFO segment: {len(nfo):,} instruments")
    print("instrumenttype values present:")
    for k, v in Counter(r.get("instrumenttype", "") for r in nfo).most_common():
        print(f"    {k or '(blank)':12s} {v:,}")

    print("\nNames containing 'NIFTY' (top 12):")
    names = Counter(r.get("name", "") for r in nfo if "NIFTY" in r.get("name", "").upper())
    for k, v in names.most_common(12):
        print(f"    {k:18s} {v:,}")

    # every NIFTY future, whatever its type string
    futs = [r for r in nfo
            if r.get("name", "").upper() == NAME
            and "FUT" in str(r.get("instrumenttype", "")).upper()]
    print(f"\n{NAME} futures found: {len(futs)}")
    today = dt.date.today()
    rows = []
    for r in futs:
        d = sd._parse_expiry(r.get("expiry", ""))
        rows.append((d, r.get("symbol"), r.get("token"), r.get("lotsize"),
                     r.get("instrumenttype")))
    for d, s, t, l, it in sorted(rows, key=lambda x: (x[0] or dt.date(1900, 1, 1))):
        when = "EXPIRED" if (d and d <= today) else "live"
        print(f"    {str(d):12s} {when:8s} {s:22s} token={t} lot={l} type={it}")

    past = [x for x in rows if x[0] and x[0] <= today]
    print(f"\n  expired contracts still listed: {len(past)}")
    print(f"  live contracts: {len(rows) - len(past)}")
    if not past:
        print("\n  >>> The master lists only LIVE contracts. Angel One removes")
        print("      expired ones, so historical futures candles for past months")
        print("      cannot be fetched this way.")

    # can we pull candles from the nearest live contract at all?
    live = sorted([x for x in rows if x[0] and x[0] > today])
    if live:
        d, s, t, l, _ = live[0]
        print(f"\nTrying 1-minute candles from the nearest live contract {s}...")
        try:
            client = sd._session()
            end = dt.date.today()
            start = end - dt.timedelta(days=4)
            resp = client.getCandleData({
                "exchange": "NFO", "symboltoken": t, "interval": "ONE_MINUTE",
                "fromdate": start.strftime("%Y-%m-%d 09:15"),
                "todate": end.strftime("%Y-%m-%d 15:30"),
            })
            data = resp.get("data") if isinstance(resp, dict) else None
            if data:
                vols = [row[5] for row in data]
                nonzero = sum(1 for v in vols if v and float(v) > 0)
                print(f"  got {len(data)} candles, {nonzero} with volume > 0")
                print(f"  sample: {data[0]}")
                if nonzero:
                    print("\n  >>> Live futures candles DO carry real volume, so VWAP")
                    print("      works going forward — the limit is history, not data.")
            else:
                print("  no candles returned")
        except Exception as exc:
            print(f"  failed: {exc}")


if __name__ == "__main__":
    main()
