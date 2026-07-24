"""Tests the options-suggestion engine (everything above st.set_page_config)
on synthetic OHLCV data. No network, no Streamlit runtime.

Run:  python test_strategy.py
"""
import sys
import types
import numpy as np
import pandas as pd

src = open("app.py").read()
logic_src = src[: src.index("st.set_page_config")]
app = types.ModuleType("app_logic")
sys.modules["app_logic"] = app
exec(compile(logic_src, "app.py", "exec"), app.__dict__)
print("[1] logic block imports cleanly")


def make_series(n=300, trend=0.0, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-01", periods=n)
    close = 1000 * np.exp(np.cumsum(np.full(n, trend) + rng.normal(0, 0.008, n)))
    high = close * (1 + abs(rng.normal(0, 0.004, n)))
    low = close * (1 - abs(rng.normal(0, 0.004, n)))
    return pd.DataFrame(
        {"Open": close, "High": np.maximum(high, close),
         "Low": np.minimum(low, close), "Close": close,
         "Volume": rng.integers(4e5, 3e6, n).astype(float)}, index=idx)


def make_chop(n=300, seed=7):
    """Truly range-bound (mean-reverting) series - the honest 'no trade' case."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-01", periods=n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.9 * x[i - 1] + rng.normal(0, 0.01)
    close = 1000 * np.exp(x)
    high = close * (1 + abs(rng.normal(0, 0.004, n)))
    low = close * (1 - abs(rng.normal(0, 0.004, n)))
    return pd.DataFrame(
        {"Open": close, "High": np.maximum(high, close),
         "Low": np.minimum(low, close), "Close": close,
         "Volume": rng.integers(4e5, 3e6, n).astype(float)}, index=idx)


# --- 18 EMA trend detection, both directions ---
up = app.trend_18ema(app.add_indicators(make_series(trend=0.002, seed=1)))
dn = app.trend_18ema(app.add_indicators(make_series(trend=-0.002, seed=1)))
assert up == "up" and dn == "down", f"trend detection failed: {up}, {dn}"
print(f"[2] 18 EMA trend: uptrend->'{up}', downtrend->'{dn}'")

# --- Elliott directional: bullish, bearish, neutral ---
bull = app.elliott_directional(app.add_indicators(make_series(trend=0.0025, seed=3)))
bear = app.elliott_directional(app.add_indicators(make_series(trend=-0.0025, seed=3)))
assert bull["bias"] in ("Bullish", "Neutral")
assert bear["bias"] in ("Bearish", "Neutral")
found_bull = found_bear = wrong = 0
for seed in range(20):
    b = app.elliott_directional(app.add_indicators(make_series(trend=0.0025, seed=seed)))
    s = app.elliott_directional(app.add_indicators(make_series(trend=-0.0025, seed=seed)))
    found_bull += b["bias"] == "Bullish"
    found_bear += s["bias"] == "Bearish"
    # a noisy uptrend can end in a genuine correction, so an occasional opposite
    # read is legitimate — but it must be rare, and the EMA+wave AND gate in
    # analyze() (test [6]) guarantees it can never produce a wrong-direction trade
    wrong += (b["bias"] == "Bearish") + (s["bias"] == "Bullish")
assert found_bull > 5 and found_bear > 5, f"detection too weak: {found_bull}, {found_bear}"
assert wrong <= 6, f"too many wrong-direction wave reads: {wrong}/40"
print(f"[3] Elliott directional over 20 charts: bullish {found_bull}/20 in uptrends, "
      f"bearish {found_bear}/20 in downtrends, opposite reads {wrong}/40 (rare, gated out)")

# --- strike rounding ---
assert app.nearest_strike(23869, "^NSEI") == 23850
assert app.nearest_strike(51420, "^NSEBANK") == 51400
assert app.nearest_strike(1272, "X.NS") == 1275
print("[4] strike rounding: Nifty 50-step, BankNifty 100-step, stock 25-step OK")

# --- premium estimate sanity ---
p_idx = app.premium_estimate(24000, 14.0, is_stock=False)
p_stk = app.premium_estimate(24000, 14.0, is_stock=True)
assert 200 < p_idx < 700, f"index ATM premium implausible: {p_idx}"
assert p_stk > p_idx, "stock IV premium must exceed index"
assert app.premium_estimate(24000, None, False) > 0, "None VIX handled"
print(f"[5] premium model: Nifty ATM ~Rs.{p_idx:.0f} at VIX 14 (plausible), stock richer")

# --- analyze(): CALL in uptrend, PUT in downtrend, None in chop ---
calls = puts = flats = 0
for seed in range(25):
    c = app.analyze("^NSEI", "Nifty 50", make_series(trend=0.0025, seed=seed), 14.0)
    p = app.analyze("^NSEI", "Nifty 50", make_series(trend=-0.0025, seed=seed), 14.0)
    f = app.analyze("^NSEI", "Nifty 50", make_chop(seed=seed), 14.0)
    if c:
        assert c["direction"] == "CALL", "uptrend suggestion must be CALL"
        assert c["sl_premium"] == round(c["premium_est"] * 0.6, 2)
        assert c["target_premium"] == round(c["premium_est"] * 1.8, 2)
        assert c["capital_est"] == round(c["premium_est"] * 75, 0)
        calls += 1
    if p:
        assert p["direction"] == "PUT", "downtrend suggestion must be PUT"
        puts += 1
    flats += f is None
assert calls > 3 and puts > 3, f"too few suggestions: {calls} calls, {puts} puts"
assert flats >= 20, f"range-bound charts should stay silent, got {25 - flats} signals"
print(f"[6] analyze over 25 charts each: {calls} CALLs in uptrends, {puts} PUTs in "
      f"downtrends, {flats}/25 range-bound markets correctly silent")

# --- risk plan invariants ---
row = None
for seed in range(40):
    row = app.analyze("RELIANCE.NS", "RELIANCE", make_series(trend=0.0025, seed=seed), 15.0)
    if row:
        break
assert row, "no stock suggestion found in 40 uptrends"
assert row["sl_premium"] < row["premium_est"] < row["target_premium"]
# hedge economics: hedged max loss must be smaller than unhedged, profit capped positive
assert row["net_debit"] < row["premium_est"], "hedge must reduce max loss"
assert row["hedge_credit"] > 0 and row["spread_max_profit"] > 0
assert abs(row["net_debit"] + row["hedge_credit"] - row["premium_est"]) < 0.02
print(f"[7] risk plan: SL {row['sl_premium']} < premium {row['premium_est']} < "
      f"target {row['target_premium']}; hedged max loss {row['net_debit']} "
      f"(saves {row['hedge_credit']}), hedged max profit {row['spread_max_profit']}")

# --- tiny/garbage data never crashes ---
assert app.analyze("^NSEI", "Nifty 50", make_series(30), 14.0) is None
assert app.trend_18ema(app.add_indicators(make_series(20))) == "flat"
print("[8] tiny dataframe handled without crash")

print("\nAll checks passed.")
