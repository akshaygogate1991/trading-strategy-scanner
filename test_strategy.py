"""Loads the pure-logic half of app.py (everything above st.set_page_config)
and exercises it on synthetic OHLCV data. No network, no Streamlit runtime.
"""
import sys
import types
import numpy as np
import pandas as pd

src = open("/home/claude/trading-strategy-scanner/app.py").read()
logic_src = src[: src.index("st.set_page_config")]
app = types.ModuleType("app_logic")
sys.modules["app_logic"] = app  # dataclass needs to resolve cls.__module__
exec(compile(logic_src, "app.py", "exec"), app.__dict__)
print("[1] logic block imports cleanly")


def make_series(n=420, trend=0.0006, seed=7, chop=False):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    drift = 0.0 if chop else trend
    ret = rng.normal(drift, 0.011, n)
    close = 1000 * np.exp(np.cumsum(ret))
    high = close * (1 + abs(rng.normal(0, 0.004, n)))
    low = close * (1 - abs(rng.normal(0, 0.004, n)))
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.002, n)),
            "High": np.maximum(high, close),
            "Low": np.minimum(low, close),
            "Close": close,
            "Volume": rng.integers(4e5, 3e6, n).astype(float),
        },
        index=idx,
    )


S = app.ScanSettings
base = dict(capital=100000, risk_percent=1.0, min_rr=2.0, data_period="2y",
            batch_size=1, use_elliott_filter=False, min_wave_score=45)

# --- moving_average ---
s = pd.Series(np.arange(1, 61, dtype=float))
ema = app.moving_average(s, 18, "EMA")
sma = app.moving_average(s, 18, "SMA")
assert not pd.isna(ema.iloc[-1]) and not pd.isna(sma.iloc[-1])
assert abs(sma.iloc[-1] - s.iloc[-18:].mean()) < 1e-9, "SMA must equal rolling mean"
assert ema.iloc[-1] > sma.iloc[-1], "EMA should lead SMA on a rising ramp"
assert pd.isna(sma.iloc[16]) and not pd.isna(sma.iloc[17]), "SMA seeds at length"
print("[2] moving_average: EMA/SMA both correct, EMA leads on uptrend")

# --- to_weekly ---
d = make_series(260)
w = app.to_weekly(d)
assert len(w) < len(d) / 4 + 3, "weekly bars should be ~1/5 of daily"
assert w["Volume"].sum() == d["Volume"].sum(), "weekly volume must conserve"
first = d.loc[w.index[1] - pd.Timedelta(days=6) : w.index[1]]
assert abs(w["High"].iloc[1] - first["High"].max()) < 1e-6, "weekly high = max of week"
assert abs(w["Close"].iloc[1] - first["Close"].iloc[-1]) < 1e-6, "weekly close = last close"
assert app.to_weekly(d.reset_index(drop=True)).empty, "non-datetime index handled"
print(f"[3] to_weekly: {len(d)} daily -> {len(w)} weekly, OHLCV aggregation correct")

# --- weekly_trend ---
def make_ramp(n=420, slope=0.0015, seed=3):
    """Low-noise monotonic ramp: the recent window is unambiguous by construction."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    close = 1000 * np.exp(np.cumsum(np.full(n, slope) + rng.normal(0, 0.001, n)))
    return pd.DataFrame(
        {"Open": close, "High": close * 1.004, "Low": close * 0.996,
         "Close": close, "Volume": np.full(n, 1e6)}, index=idx)


up = app.weekly_trend(make_ramp(slope=0.0015), S(**base, use_weekly_filter=True))
dn = app.weekly_trend(make_ramp(slope=-0.0015), S(**base, use_weekly_filter=True))
print(f"[4] weekly_trend uptrend  -> ok={up['ok']}  label={up['label']}")
print(f"    weekly_trend downtrend-> ok={dn['ok']}  label={dn['label']}")
assert up["ok"] is True and up["label"] == "Weekly up"
assert dn["ok"] is False and dn["label"] == "Weekly down"

short = app.weekly_trend(make_series(60), S(**base, use_weekly_filter=True))
assert short["ok"] is False, "strict mode must reject insufficient weekly history"
short_off = app.weekly_trend(make_series(60), S(**base, use_weekly_filter=False))
assert short_off["ok"] is True, "filter off must not block"
print(f"[5] short history: strict={short['ok']}, filter-off={short_off['ok']} -> {short['detail']}")

dn_off = app.weekly_trend(make_ramp(slope=-0.0015), S(**base, use_weekly_filter=False))
assert dn_off["ok"] is True and dn_off["label"] == "Weekly down", "label reports even when gate off"
print("[6] filter off: gate opens but label still reports the real weekly state")

# --- scan_symbol end-to-end ---
found = {"EMA": 0, "SMA": 0}
for ma in ("EMA", "SMA"):
    cfg = S(**base, ma_type=ma, ma_length=18, use_weekly_filter=True, weekly_ma_length=20)
    for seed in range(60):
        df = make_series(420, trend=0.0009, seed=seed)
        r = app.scan_symbol("TEST.NS", df, cfg, True)
        if r:
            found[ma] += 1
            assert r["Stop Loss"] < r["Entry"] < r["Target 1"] < r["Target 2"] < r["Target 3"]
            assert 0 <= r["Score"] <= 100, f"score out of range: {r['Score']}"
            assert r["Qty @ Risk"] >= 0
            assert r["Reward/Risk"] >= cfg.min_rr
            assert r["Weekly"] in ("Weekly up", "Weekly flat", "Weekly pullback", "Weekly down", "Weekly n/a")
print(f"[7] scan_symbol over 60 charts: EMA={found['EMA']} signals, SMA={found['SMA']} signals; "
      "all invariants (SL<Entry<T1<T2<T3, score 0-100, RR>=min) hold")

strict = 0
loose = 0
for seed in range(60):
    df = make_series(420, trend=0.0, seed=seed, chop=True)
    if app.scan_symbol("T.NS", df, S(**base, use_weekly_filter=True), True):
        strict += 1
    if app.scan_symbol("T.NS", df, S(**base, use_weekly_filter=False), True):
        loose += 1
print(f"[8] choppy market: weekly filter ON={strict} vs OFF={loose} "
      f"-> filter removed {loose - strict} of {loose} ({'redundant with daily gate' if strict == loose else 'binding'})")
assert strict <= loose, "weekly filter must never increase signal count"

# Where does the weekly gate actually bind? Count rejections directly.
blocked = 0
for seed in range(120):
    df = app.add_indicators(make_series(420, trend=0.0, seed=seed, chop=True))
    if not app.weekly_trend(df, S(**base, use_weekly_filter=True))["ok"]:
        blocked += 1
print(f"[8b] weekly gate alone rejects {blocked}/120 choppy charts; "
      "overlap with the daily close>ema50 gate is why net effect is small")

r = app.scan_symbol("TEST.NS", make_series(420, trend=0.0009, seed=1),
                    S(**base, ma_type="SMA", ma_length=21, use_weekly_filter=False,
                      use_mtf_wave=False), True)
if r:
    print(f"[9] setup label reflects selector: {r['Setup']!r}")

assert app.scan_symbol("X.NS", make_series(30), S(**base), True) is None
print("[10] tiny dataframe returns None, no crash")

# --- multi-timeframe Elliott Wave ---
# timeframe_wave returns a valid schema on a weekly frame, and degrades on tiny input
wk = app.to_weekly(make_series(520, trend=0.0012, seed=2))
w = app.timeframe_wave(wk)
assert set(("bias", "stage", "score")).issubset(w), "wave schema must be complete"
assert w["bias"] in ("Bullish", "Neutral") and 0 <= w["score"] <= 100
assert app.timeframe_wave(make_series(10))["bias"] == "Neutral", "tiny frame -> Neutral, no crash"
assert app.timeframe_wave(None)["stage"] == "Insufficient data", "None handled"
print(f"[11] timeframe_wave: weekly bias={w['bias']} score={w['score']}; tiny/None handled")

# MTF alignment gate must never *increase* signal count vs. off
mtf_on = mtf_off = 0
for seed in range(20):
    df = make_series(420, trend=0.0011, seed=seed)
    if app.scan_symbol("T.NS", df, S(**base, use_mtf_wave=True), True):
        mtf_on += 1
    if app.scan_symbol("T.NS", df, S(**base, use_mtf_wave=False), True):
        mtf_off += 1
assert mtf_on <= mtf_off, "MTF alignment gate must never add signals"
print(f"[12] MTF gate: aligned-only={mtf_on} vs off={mtf_off} "
      f"-> gate removed {mtf_off - mtf_on} of {mtf_off}")

# new output columns exist and carry sane values
row = None
for seed in range(40):
    row = app.scan_symbol("T.NS", make_series(420, trend=0.0011, seed=seed),
                          S(**base, use_mtf_wave=False), True)
    if row:
        break
assert row and "Weekly Wave" in row and "MTF Align" in row and "4H Wave" in row
assert row["MTF Align"] in ("Yes", "No")
assert row["Weekly Wave"] in ("Bullish", "Neutral")
print(f"[13] new columns present -> Weekly Wave={row['Weekly Wave']}, "
      f"MTF Align={row['MTF Align']}, 4H Wave={row['4H Wave']}")

# to_4h resamples hourly bars into 4-hour bars
_h = pd.DataFrame(
    {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0},
    index=pd.date_range("2025-01-01 09:00", periods=200, freq="1h"),
)
_4h = app.to_4h(_h)
assert 0 < len(_4h) <= len(_h) / 3, "4h bars should be far fewer than hourly"
assert app.to_4h(None).empty, "None handled"
assert app.to_4h(_h.reset_index(drop=True)).empty, "non-datetime index handled"
print(f"[14] to_4h: {len(_h)} hourly -> {len(_4h)} four-hour bars")

print("\nAll checks passed.")
