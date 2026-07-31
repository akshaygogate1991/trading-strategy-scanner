"""Tests the options-suggestion engine (everything above st.set_page_config)
on synthetic OHLCV data. No network, no Streamlit runtime.

Run:  python test_strategy.py
"""
import sys
import types
import numpy as np
import pandas as pd

# app.py contains rupee signs, arrows and emoji. Windows defaults to cp1252,
# which cannot decode them, so the encoding must be explicit.
src = open("app.py", encoding="utf-8").read()
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
        # Assert the INVARIANT (capital = premium x lot), never a hardcoded lot.
        # This used to hardcode 75, but resolve_lot() reads the live NSE lot size
        # from Angel One's instrument master - and NSE changes lot sizes. A test
        # that pins today's number fails the day the exchange revises it, which
        # says nothing about whether our code is correct.
        if c["lot"]:
            assert c["capital_est"] == round(c["premium_est"] * c["lot"], 0), (
                f"capital {c['capital_est']} != premium {c['premium_est']} x lot {c['lot']}")
        else:
            assert c["capital_est"] is None, "no lot size means no capital figure"
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
# breakeven must sit between the two strikes (CALL: strike < BE < hedge strike)
lo, hi = sorted([row["strike"], row["hedge_strike"]])
assert lo < row["hedged_breakeven"] < hi, "breakeven must lie between the strikes"
print(f"[7] risk plan: SL {row['sl_premium']} < premium {row['premium_est']} < "
      f"target {row['target_premium']}; hedged max loss {row['net_debit']} "
      f"(saves {row['hedge_credit']}), hedged max profit {row['spread_max_profit']}")

# --- tiny/garbage data never crashes ---
assert app.analyze("^NSEI", "Nifty 50", make_series(30), 14.0) is None
assert app.trend_18ema(app.add_indicators(make_series(20))) == "flat"
print("[8] tiny dataframe handled without crash")


# --- real_option_info: exact expiry (no login needed) + live premium (needs login) ---
import datetime as _dt
import smartapi_data as sd

_orig_resolve, _orig_ltp = sd.resolve_option, sd.get_ltp

# This must pass BOTH with and without working Angel One credentials. It used to
# assert None, which only holds on a machine that cannot log in. On a machine
# with real secrets it resolves a real contract - and the test then failed for
# the best possible reason: everything worked.
r = app.real_option_info("TECHM.NS", "CALL", 1675)
if r is None:
    print("[15] no live Angel One session here -> returns None, no crash")
else:
    for key in ("strike", "expiry", "premium", "tradingsymbol"):
        assert key in r, f"missing '{key}' in real_option_info result"
    assert isinstance(r["expiry"], _dt.date), "expiry must be a date"
    assert r["strike"] > 0, "strike must be positive"
    assert r["expiry"] >= _dt.date.today(), "resolved contract must not be expired"
    prem = r["premium"]
    assert prem is None or prem > 0, f"premium must be positive or None, got {prem}"
    print(f"[15] LIVE Angel One session -> {r['tradingsymbol']} "
          f"strike {r['strike']:g}, expiry {r['expiry']}, "
          f"premium {'Rs.' + str(prem) if prem else 'unavailable (market closed?)'}")

sd.resolve_option = lambda ticker, opt, target: {
    "token": "1", "tradingsymbol": "TECHM_TEST_CE",
    "expiry": _dt.date(2026, 7, 31), "strike": 1675.0,
}
sd.get_ltp = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no session"))
r = app.real_option_info("TECHM.NS", "CALL", 1675)
assert r is not None and r["premium"] is None and r["strike"] == 1675.0
assert r["premium_error"] == "no session", "real error message must be captured for diagnosis"
print(f"[16] real_option_info: expiry resolved without login ({r['expiry']}), "
      f"premium None (needs login), error captured: '{r['premium_error']}'")

sd.get_ltp = lambda *a, **k: 42.5
r = app.real_option_info("TECHM.NS", "CALL", 1675)
assert r["premium"] == 42.5 and r["expiry"] == _dt.date(2026, 7, 31)
print(f"[17] real_option_info: full live path -> premium={r['premium']}, expiry={r['expiry']}")

sd.resolve_option, sd.get_ltp = _orig_resolve, _orig_ltp  # restore

# --- stale Angel One session must trigger a re-login, not a dead app ---
# Angel One tokens expire ~daily; Streamlit Cloud keeps the process alive for
# days. A session cached forever therefore works on day 1 and fails on day 2
# with "Invalid Token" - which is exactly what happened in production.
assert sd._looks_like_auth_error("Invalid Token")
assert sd._looks_like_auth_error("AB1010: token expired")
assert sd._looks_like_auth_error("Session Expired")
assert not sd._looks_like_auth_error("Invalid symboltoken 99999")
assert not sd._looks_like_auth_error("Rate limit exceeded")
print("[20] auth errors distinguished from ordinary request errors")


class _FakeClient:
    """Rejects the first session's calls the way an expired token does."""
    def __init__(self, generation):
        self.generation = generation

    def ltpData(self, exchange, tradingsymbol, token):
        if self.generation == 0:
            return {"status": False, "message": "Invalid Token"}
        return {"status": True, "data": {"ltp": 10.8}}


_gen = {"n": 0}
_orig_session = sd._session


def _fake_session(force_new=False):
    if force_new:
        _gen["n"] += 1
    return _FakeClient(_gen["n"])


sd._session = _fake_session
try:
    got = sd.get_ltp("NFO", "VEDL25AUG26270PE", "12345")
    assert got == 10.8, f"expected re-login to recover the quote, got {got}"
    assert _gen["n"] == 1, "should have logged in exactly once more"
    print(f"[21] stale session recovered: re-logged in and returned Rs.{got}")

    # a NON-auth failure must NOT trigger a pointless re-login
    _gen["n"] = 0
    sd._session = lambda force_new=False: type("C", (), {
        "ltpData": lambda self, *a: {"status": False, "message": "Invalid symboltoken"}
    })()
    try:
        sd.get_ltp("NFO", "BAD", "0")
        raise AssertionError("bad symbol should raise, not silently pass")
    except RuntimeError as exc:
        assert "symboltoken" in str(exc)
    print("[22] a bad symbol raises without a wasted re-login")
finally:
    sd._session = _orig_session

# --- build_plan is consistent whether fed the estimate or a real premium ---
plan_est = app.build_plan(1675.0, 33.0, "TECHM.NS", "CALL", 1669.0, True, 600)
plan_real = app.build_plan(1675.0, 42.5, "TECHM.NS", "CALL", 1669.0, True, 600)
for p, premium in ((plan_est, 33.0), (plan_real, 42.5)):
    assert p["sl_premium"] < premium < p["target_premium"]
    assert p["net_debit"] < premium
    assert p["hedge_strike"] > 1675.0  # OTM further out for a CALL hedge
print(f"[18] build_plan consistent for estimate ({plan_est['sl_premium']}-{plan_est['target_premium']}) "
      f"and real premium ({plan_real['sl_premium']}-{plan_real['target_premium']})")

# --- capital MUST track the premium actually shown (regression: it did not) ---
assert plan_est["capital_est"] == 33.0 * 600, "capital must match the estimate premium"
assert plan_real["capital_est"] == 42.5 * 600, "capital must be RECOMPUTED for a live premium"
assert plan_real["capital_est"] > plan_est["capital_est"], "richer premium => more capital"
assert app.build_plan(1675.0, 42.5, "TECHM.NS", "CALL", 1669.0, True, None)["capital_est"] is None
# and end-to-end: the row analyze() returns must be internally consistent
_row = None
for seed in range(40):
    _row = app.analyze("RELIANCE.NS", "RELIANCE", make_series(trend=0.0025, seed=seed), 15.0)
    if _row:
        break
if _row and _row["lot"]:
    assert _row["capital_est"] == round(_row["premium_est"] * _row["lot"], 0), \
        "analyze(): capital_est must equal premium x lot"
print(f"[19] capital tracks premium: Rs.{plan_est['capital_est']:,.0f} at premium 33 -> "
      f"Rs.{plan_real['capital_est']:,.0f} at premium 42.5 (no lot -> None)")

print("\nAll checks passed.")
