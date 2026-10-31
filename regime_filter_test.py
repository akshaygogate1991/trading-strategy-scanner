"""Does a market-regime filter improve Sensex Directional (or Chota Sensex expiry)?

Each filter skips a day using ONLY information known before the 09:35 entry:
  vrp        previous India VIX close - 20-day realized volatility of Sensex (%)
  vix        previous India VIX close
  vix_chg    previous VIX close - VIX close 5 sessions earlier
  prev_range previous day's high-low range (% of open)
  rv5        5-day realized volatility (%, annualised)
  gap        |09:15 open / previous close - 1| (%)
  open_rng   09:15-09:34 high-low range (% of 09:15 open) - known at the 09:35 entry

Honest test: every threshold is chosen on Aug 2023 - May 2025 only (in-sample),
then judged on Jun 2025 - Jul 2026 (out-of-sample, which includes 2026's bad
regime). Days are independent (intraday, flat every night), so a filter is
exactly "drop those days" from the strategy's real daily P&L. Each filter is
also compared with skipping the same number of days at random.

  python regime_filter_test.py
"""
import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT = "2025-06-01"
RNG = np.random.default_rng(17)


def daily_pnl(files, strategy):
    t = pd.concat([pd.read_csv(os.path.join(HERE, f)) for f in files])
    t = t[t["strategy"] == strategy]
    return t.groupby("day")["net"].sum()


def features():
    s = pd.read_parquet(os.path.join(HERE, "hf_data", "index", "SENSEX.parquet"))
    ts = s["timestamp"]
    s = s.assign(m=(ts.dt.hour * 60 + ts.dt.minute - 555).to_numpy())
    s = s[(s["m"] >= 0) & (s["m"] < 375)]
    g = s.groupby("trading_day")
    d = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                      "low": g["low"].min(), "close": g["close"].last()})
    first = s[s["m"] < 20].groupby("trading_day")
    d["open_rng"] = (first["high"].max() - first["low"].min()) / d["open"] * 100
    d["range"] = (d["high"] - d["low"]) / d["open"] * 100
    ret = d["close"].pct_change()
    d["prev_range"] = d["range"].shift()
    d["gap"] = (d["open"] / d["close"].shift() - 1).abs() * 100
    d["rv5"] = ret.rolling(5).std().shift() * math.sqrt(252) * 100
    rv20 = ret.rolling(20).std().shift() * math.sqrt(252) * 100
    vix = pd.read_csv(os.path.join(HERE, "vix_data", "nifty_vix_8y.csv"), index_col="Date")["VIX"]
    vix.index = pd.to_datetime(vix.index).strftime("%Y-%m-%d")
    v = vix.reindex(d.index).ffill()
    d["vix"] = v.shift()                    # yesterday's close: known before the open
    d["vix_chg"] = v.shift() - v.shift(6)
    d["vrp"] = d["vix"] - rv20
    return d


def tstat(x):
    return x.mean() / (x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else float("nan")


def evaluate(pnl, feat, label):
    df = pd.DataFrame({"pnl": pnl}).join(feat, how="left")
    ins, oos = df[df.index < SPLIT], df[df.index >= SPLIT]
    print(f"\n######## {label}: {len(ins)} in-sample days (net {ins.pnl.sum():,.0f}), "
          f"{len(oos)} out-of-sample days (net {oos.pnl.sum():,.0f})")

    print("\nIn-sample P&L by feature quintile (avg Rs/day, Q1 = lowest value):")
    print(f"{'feature':<11}" + "".join(f"{'Q' + str(i + 1):>8}" for i in range(5)))
    for f in ("vrp", "vix", "vix_chg", "prev_range", "rv5", "gap", "open_rng"):
        x = ins[[f, "pnl"]].dropna()
        q = pd.qcut(x[f], 5, labels=False, duplicates="drop")
        print(f"{f:<11}" + "".join(f"{v:>8,.0f}" for v in x.groupby(q)["pnl"].mean()))

    # rules: (name, feature, direction, threshold rule decided on in-sample)
    rules = []
    for f, side in (("vrp", "low"), ("vix", "high"), ("vix_chg", "high"), ("prev_range", "high"),
                    ("rv5", "high"), ("gap", "high"), ("open_rng", "high")):
        x = ins[f].dropna()
        for pct in (10, 20):
            thr = np.percentile(x, pct if side == "low" else 100 - pct)
            rules.append((f"skip {f} {'<' if side == 'low' else '>'} {thr:.2f} ({pct}% of days)", f, side, thr))
    rules.append(("skip vrp < 0 (fixed rule)", "vrp", "low", 0.0))

    print(f"\n{'rule':<40}{'IS skipped':>12}{'IS avg':>8}{'IS kept t':>10} |"
          f"{'OOS skip':>9}{'OOS skip Rs':>12}{'OOS net':>10}{'OOS t':>7}{'rand%':>7}")
    base_t = tstat(oos["pnl"])
    print(f"{'(no filter)':<40}{'':>12}{'':>8}{tstat(ins['pnl']):>10.2f} |{'':>9}{'':>12}"
          f"{oos['pnl'].sum():>10,.0f}{base_t:>7.2f}")
    for name, f, side, thr in rules:
        out = []
        for part in (ins, oos):
            skip = (part[f] < thr) if side == "low" else (part[f] > thr)
            skip = skip.fillna(False)
            out.append((skip, part["pnl"][skip], part["pnl"][~skip]))
        (s_i, sk_i, kp_i), (s_o, sk_o, kp_o) = out
        n = int(s_o.sum())
        # random control: skip the same number of OOS days at random 2000 times
        sums = np.array([oos["pnl"].drop(RNG.choice(oos.index, n, replace=False)).sum()
                         for _ in range(2000)]) if n else np.array([oos["pnl"].sum()])
        rand_pct = 100 * (sums < kp_o.sum()).mean()
        print(f"{name:<40}{int(s_i.sum()):>12}{sk_i.mean() if len(sk_i) else 0:>8,.0f}{tstat(kp_i):>10.2f} |"
              f"{n:>9}{sk_o.sum():>12,.0f}{kp_o.sum():>10,.0f}{tstat(kp_o):>7.2f}{rand_pct:>7.0f}")
    by_year = df.assign(y=df.index.str[:4])
    return by_year


def main():
    feat = features()
    dir_pnl = daily_pnl(["logs/oos_export_sensexbasket_trades.csv", "export_sensexbasket_trades.csv"],
                        "Sensex Directional")
    evaluate(dir_pnl, feat, "SENSEX DIRECTIONAL (all days)")
    chota = daily_pnl(["logs/batch/chotasx_expiry.csv"], "Chota Packet Bada Dhamaka Sensex")
    evaluate(chota, feat, "CHOTA PACKET SENSEX (expiry days only)")
    print("\nIS = Aug 2023 - May 2025 (thresholds chosen here). OOS = Jun 2025 - Jul 2026 (Chota: to May 2026).")
    print("'OOS skip Rs' = what the skipped days would have made: NEGATIVE means the filter helped.")
    print("'rand%' = share of random skips (same number of days) that did worse than the filter;")
    print("above 95 = the filter is better than chance, around 50 = no better than skipping at random.")


if __name__ == "__main__":
    main()
