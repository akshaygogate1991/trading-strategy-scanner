"""Is a candidate exit rule genuinely better, or did one random draw flatter it?

The problem with a single control: comparing one signal result against ONE
random sample tells you nothing when both are noisy. In the last run the signal
beat random at a +150% target and LOST to it at +200% - the ordering flipped on
noise, and either could be cherry-picked to support a conclusion.

This runs the random control many times to build a distribution, then asks where
the real signal falls in it. That is a permutation test, and it is the honest way
to separate "this rule has an edge" from "long calls in a rising market".

Reading the result:
  percentile > 95  ->  the signal genuinely beats random entry
  percentile 5-95  ->  indistinguishable from luck, whatever the average says
  percentile < 5   ->  the signal is actively worse than random

Run:  python rule_significance.py
      python rule_significance.py --draws 300
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import exit_matrix as E   # reuses its pricing, signals and cached Elliott mask

# The rules worth deciding between, all on the live EMA+Elliott signal.
CANDIDATES = {
    "-40% / +80%   (old app rule)": (-0.40, 0.80),
    "-60% / +80%": (-0.60, 0.80),
    "-60% / +150%": (-0.60, 1.50),
    "-60% / +200%": (-0.60, 2.00),
    "-60% / no target  (app now)": (-0.60, None),
    "-50% / +150%": (-0.50, 1.50),
    "none / no target": (None, None),
}


def signal_return(frames, entry_fn, stop, target) -> tuple[float, int]:
    vals = []
    for t, df in frames.items():
        vals += E.simulate(df, entry_fn(t, df), stop, target)
    return (100 * float(np.mean(vals)) if vals else 0.0), len(vals)


def random_distribution(frames, counts, stop, target, draws, rng) -> np.ndarray:
    """`draws` independent random-entry runs, each matching the real trade count."""
    out = np.empty(draws)
    for d in range(draws):
        vals = []
        for t, df in frames.items():
            n = counts.get(t, 0)
            if n <= 0 or len(df) < 200:
                continue
            idx = sorted(int(x) for x in rng.integers(121, len(df) - 35, size=n))
            vals += E.simulate(df, idx, stop, target)
        out[d] = 100 * float(np.mean(vals)) if vals else 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=200)
    args = ap.parse_args()

    E._load_mask_cache()
    frames = {}
    for t in E.B.ALL_TICKERS:
        df = E.B.load(t, 4.0)
        if df is not None and len(df) > 200:
            frames[t] = df

    counts = {t: len(E.entries_ema(t, df)) for t, df in frames.items()}
    E._save_mask_cache()
    total = sum(counts.values())
    print(f"Signal: EMA + Elliott (the live app), {total} trades over 4 years")
    print(f"Control: {args.draws} independent random-entry runs per rule, "
          f"each matched trade-for-trade\n")

    rng = np.random.default_rng(20260813)
    hdr = (f"{'exit rule':30s}{'signal':>9s}{'random avg':>12s}"
           f"{'random 5-95%':>18s}{'pctile':>8s}  verdict")
    print(hdr)
    print("-" * len(hdr))

    t0 = time.time()
    rows = []
    for name, (stop, target) in CANDIDATES.items():
        sig, n = signal_return(frames, E.entries_ema, stop, target)
        dist = random_distribution(frames, counts, stop, target, args.draws, rng)
        pct = 100.0 * float((dist < sig).mean())
        lo, hi = np.percentile(dist, [5, 95])
        if pct > 95:
            v = "BEATS random"
        elif pct < 5:
            v = "WORSE than random"
        else:
            v = "no evidence of edge"
        rows.append((name, sig, float(dist.mean()), lo, hi, pct, v))
        print(f"{name:30s}{sig:+9.2f}{dist.mean():+12.2f}"
              f"{f'[{lo:+.1f}, {hi:+.1f}]':>18s}{pct:7.0f}%  {v}")

    print(f"\n  ({time.time() - t0:.0f}s, n={n} trades per rule)")

    best = max(rows, key=lambda r: r[5])
    print("\n" + "=" * 78)
    if best[5] > 95:
        print(f"  Only rule that beats random entry: {best[0].strip()}")
        print(f"  Signal {best[1]:+.2f}% vs random median ~{best[2]:+.2f}% "
              f"({best[5]:.0f}th percentile).")
    else:
        print("  NO exit rule beats random entry at the 95% level.")
        print("  Every positive number in this table is explained by long calls in")
        print("  a rising market plus the exit geometry - not by the signal.")
        print()
        print("  That does NOT make the rules equal. Compare the RANDOM AVERAGE")
        print("  column: the exit geometry itself still decides how much of the")
        print("  market's move you keep. Pick the rule with the best random")
        print("  average, because that is the part that is real.")
    print("=" * 78)

    print("\n  Best exit geometry by random-entry average (the reliable part):")
    for name, _sig, rmean, _lo, _hi, _p, _v in sorted(rows, key=lambda r: -r[2])[:3]:
        print(f"    {name:30s} random avg {rmean:+.2f}%  "
              f"(net {rmean - E.COST_PCT * 100:+.2f}% after costs)")


if __name__ == "__main__":
    main()
