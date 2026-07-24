"""Nifty Options Suggestion Scanner — simple mode.

Strategy (both directions, decision-support only, NEVER places orders):
  BULLISH  = price above a rising 18 EMA  AND Elliott Wave structure bullish  -> BUY CALL idea
  BEARISH  = price below a falling 18 EMA AND Elliott Wave structure bearish -> BUY PUT idea
  Anything else -> no suggestion. Discipline over activity.

Conviction checklist per suggestion:
  Auto   : 18 EMA trend, Elliott Wave agreement, India VIX level
  Manual : FII/DII flows, market sentiment, NiftyBuddy conviction (you tick these)

Risk plan on every idea: stop-loss at -40% of premium, target +80% (2x reward).
Premium/capital figures are ESTIMATES - always verify the live premium on your
broker app before any decision. This tool is not financial advice.
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


INDICES = {
    "Nifty 50": "^NSEI",
    "Bank Nifty": "^NSEBANK",
}

# Liquid F&O stocks (options actually tradeable with tight spreads)
FO_STOCKS = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS",
    "KOTAKBANK.NS", "INFY.NS", "TCS.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
    "LT.NS", "BHARTIARTL.NS", "ITC.NS", "HINDUNILVR.NS", "MARUTI.NS",
    "TATAMOTORS.NS", "M&M.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS",
    "SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
    "TITAN.NS", "ULTRACEMCO.NS", "ADANIENT.NS", "ADANIPORTS.NS", "NTPC.NS",
    "POWERGRID.NS", "ONGC.NS", "COALINDIA.NS", "TRENT.NS", "BEL.NS", "DLF.NS",
    "VEDL.NS",
]

# Fallback lot sizes if the NSE instrument master is unreachable.
# Lot sizes are set by NSE (identical at every broker); NSE revises them periodically.
LOT_SIZES = {"^NSEI": 75, "^NSEBANK": 35}


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def resolve_lot(ticker: str) -> int | None:
    """Exact NSE lot size via the Angel One instrument master, with fallback."""
    try:
        import smartapi_data as sd

        lot = sd.get_lot_size(ticker)
        if lot:
            return lot
    except Exception:
        pass
    return LOT_SIZES.get(ticker)

DATA_PERIOD = "1y"
SL_PCT = 40      # stop-loss: exit if premium falls 40%
TARGET_PCT = 80  # target: +80% on premium (2x the risk)


# ---------------------------------------------------------------- data layer

@st.cache_data(ttl=60 * 20, show_spinner=False)
def fetch_yahoo(tickers: tuple[str, ...], period: str) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        frame = yf.download(ticker, period=period, interval="1d",
                            auto_adjust=True, progress=False, threads=False)
        if frame.empty:
            continue
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
        if len(frame) >= 80:
            data[ticker] = frame
    return data


@st.cache_data(ttl=60 * 20, show_spinner=False)
def fetch_angelone(tickers: tuple[str, ...], period: str) -> dict[str, pd.DataFrame]:
    import smartapi_data as sd

    return sd.fetch_market_data(tickers, period)


def get_market_data(tickers: tuple[str, ...]) -> tuple[dict[str, pd.DataFrame], str]:
    """Angel One (exact NSE) when configured, Yahoo otherwise."""
    try:
        data = fetch_angelone(tickers, DATA_PERIOD)
        if data:
            return data, "Angel One (exact NSE)"
    except Exception:
        pass
    return fetch_yahoo(tickers, DATA_PERIOD), "Yahoo Finance"


@st.cache_data(ttl=60 * 20, show_spinner=False)
def fetch_vix() -> float | None:
    try:
        frame = yf.download("^INDIAVIX", period="1mo", interval="1d",
                            auto_adjust=True, progress=False, threads=False)
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        if not frame.empty:
            return float(frame["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return None


# ------------------------------------------------------------------- engine

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema18"] = out["Close"].ewm(span=18, adjust=False).mean()
    delta = out["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi14"] = 100 - (100 / (1 + rs))
    prev_close = out["Close"].shift(1)
    tr = pd.concat([out["High"] - out["Low"],
                    (out["High"] - prev_close).abs(),
                    (out["Low"] - prev_close).abs()], axis=1).max(axis=1)
    out["atr14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return out


def detect_swing_pivots(df: pd.DataFrame, lookback: int = 3) -> list[dict]:
    pivots: list[dict] = []
    if len(df) < (lookback * 2) + 10:
        return pivots
    recent = df.tail(140)
    for pos in range(lookback, len(recent) - lookback):
        window = recent.iloc[pos - lookback: pos + lookback + 1]
        row = recent.iloc[pos]
        if row["High"] == window["High"].max():
            pivots.append({"type": "H", "price": float(row["High"]), "pos": pos})
        if row["Low"] == window["Low"].min():
            pivots.append({"type": "L", "price": float(row["Low"]), "pos": pos})
    filtered: list[dict] = []
    for pivot in pivots:
        if filtered and filtered[-1]["type"] == pivot["type"]:
            if pivot["type"] == "H" and pivot["price"] > filtered[-1]["price"]:
                filtered[-1] = pivot
            elif pivot["type"] == "L" and pivot["price"] < filtered[-1]["price"]:
                filtered[-1] = pivot
            continue
        filtered.append(pivot)
    return filtered[-9:]


def _bullish_wave(df: pd.DataFrame) -> dict:
    """Objective bullish impulse check (L-H-L-H-L with Elliott rules)."""
    pivots = detect_swing_pivots(df)
    default = {"bias": "Neutral", "stage": "No clean wave count", "score": 0}
    if len(pivots) < 4:
        return default
    close = float(df.iloc[-1]["Close"])
    atr = float(df.iloc[-1]["atr14"]) if not pd.isna(df.iloc[-1]["atr14"]) else close * 0.01
    tolerance = max(atr, close * 0.01)
    for start in range(max(0, len(pivots) - 6), len(pivots) - 3):
        candidate = pivots[start: start + 5]
        if [p["type"] for p in candidate] != ["L", "H", "L", "H", "L"]:
            continue
        w0, w1, w2, w3, w4 = candidate
        wave1 = w1["price"] - w0["price"]
        wave3 = w3["price"] - w2["price"]
        wave4 = w3["price"] - w4["price"]
        if wave1 <= 0 or wave3 <= 0:
            continue
        if w2["price"] <= w0["price"] or w4["price"] <= w1["price"]:
            continue
        if wave3 < min(wave1, max(wave4, tolerance)):
            continue
        score = 55
        if wave3 >= 1.2 * wave1:
            score += 15
        if 0.25 * wave3 <= wave4 <= 0.65 * wave3:
            score += 10
        if close > w3["price"]:
            score += 20
            stage = "Wave 5 continuation"
        elif abs(close - w4["price"]) <= 1.5 * tolerance or close > w4["price"]:
            score += 10
            stage = "Wave 4 support holding"
        else:
            stage = "Post Wave 4 watch"
        return {"bias": "Bullish", "stage": stage, "score": min(score, 100)}
    highs = [p["price"] for p in pivots if p["type"] == "H"][-2:]
    lows = [p["price"] for p in pivots if p["type"] == "L"][-2:]
    if len(highs) == 2 and len(lows) == 2 and highs[-1] > highs[-2] and lows[-1] > lows[-2]:
        return {"bias": "Bullish", "stage": "Higher highs / higher lows", "score": 45}
    return default


def _invert_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror prices so a downtrend looks like an uptrend (for bearish detection)."""
    out = df.copy()
    out["Close"], out["Open"] = -df["Close"], -df["Open"]
    out["High"], out["Low"] = -df["Low"], -df["High"]
    return out


def elliott_directional(df: pd.DataFrame) -> dict:
    """Elliott read in BOTH directions: Bullish, Bearish, or Neutral."""
    bull = _bullish_wave(df)
    if bull["bias"] == "Bullish":
        return bull
    bear = _bullish_wave(_invert_ohlc(df))
    if bear["bias"] == "Bullish":
        stage = bear["stage"].replace("Higher highs / higher lows",
                                      "Lower lows / lower highs")
        return {"bias": "Bearish", "stage": stage, "score": bear["score"]}
    return {"bias": "Neutral", "stage": "No clean structure either way", "score": 0}


def trend_18ema(df: pd.DataFrame) -> str:
    """'up' above a rising 18 EMA, 'down' below a falling one, else 'flat'."""
    latest = df.iloc[-1]
    if pd.isna(latest["ema18"]) or len(df) < 25:
        return "flat"
    close = float(latest["Close"])
    ema = float(latest["ema18"])
    ema_prev = float(df.iloc[-6]["ema18"])
    if close > ema and ema > ema_prev:
        return "up"
    if close < ema and ema < ema_prev:
        return "down"
    return "flat"


# ------------------------------------------------------- option suggestion

def strike_step(price: float, ticker: str) -> float:
    if ticker == "^NSEI":
        return 50.0
    if ticker == "^NSEBANK":
        return 100.0
    if price < 250:
        return 2.5
    if price < 500:
        return 5.0
    if price < 1000:
        return 10.0
    if price < 2500:
        return 25.0
    if price < 5000:
        return 50.0
    return 100.0


def nearest_strike(price: float, ticker: str) -> float:
    step = strike_step(price, ticker)
    return round(round(price / step) * step, 2)


def premium_estimate(spot: float, vix: float | None, is_stock: bool, days: int = 30) -> float:
    """Rough ATM premium: 0.4 * S * sigma * sqrt(T). Verify live premium on broker."""
    iv = (vix if vix else 14.0) / 100.0
    if is_stock:
        iv *= 1.4  # single stocks trade richer IV than the index
    return round(0.4 * spot * iv * math.sqrt(days / 365.0), 2)


def analyze(ticker: str, name: str, df: pd.DataFrame, vix: float | None) -> dict | None:
    df = add_indicators(df)
    trend = trend_18ema(df)
    wave = elliott_directional(df)
    rsi_now = float(df.iloc[-1]["rsi14"]) if not pd.isna(df.iloc[-1]["rsi14"]) else 50.0

    # require a proper impulse structure (score >= 55), not just higher-highs
    strong_wave = int(wave["score"]) >= 55

    if trend == "up" and wave["bias"] == "Bullish" and strong_wave and rsi_now >= 50:
        direction = "CALL"
    elif trend == "down" and wave["bias"] == "Bearish" and strong_wave and rsi_now <= 50:
        direction = "PUT"
    else:
        return None

    close = float(df.iloc[-1]["Close"])
    ema = float(df.iloc[-1]["ema18"])
    rsi = float(df.iloc[-1]["rsi14"]) if not pd.isna(df.iloc[-1]["rsi14"]) else 50.0
    is_stock = ticker not in LOT_SIZES
    strike = nearest_strike(close, ticker)
    premium = premium_estimate(close, vix, is_stock)
    lot = resolve_lot(ticker)
    capital = round(premium * lot, 0) if lot else None

    # optional hedge: sell a further-out option -> debit spread (lower cost & max loss)
    step = strike_step(close, ticker)
    hedge_strike = strike + 2 * step if direction == "CALL" else strike - 2 * step

    vix_ok = vix is not None and vix < 20

    return {
        "ticker": ticker,
        "name": name,
        "direction": direction,
        "strike": strike,
        "close": round(close, 2),
        "ema18": round(ema, 2),
        "rsi": round(rsi, 1),
        "wave_stage": wave["stage"],
        "wave_score": int(wave["score"]),
        "premium_est": premium,
        "lot": lot,
        "capital_est": capital,
        "sl_premium": round(premium * (1 - SL_PCT / 100), 2),
        "target_premium": round(premium * (1 + TARGET_PCT / 100), 2),
        "underlying_exit": round(ema, 2),
        "vix_ok": vix_ok,
        "hedge_strike": hedge_strike,
    }


def condor_idea(ticker: str, name: str, df: pd.DataFrame, vix: float | None) -> dict | None:
    """Hedged range idea (iron condor) for an index with NO directional signal.

    Info-only: the one structure with a (small) positive edge in our 2-year
    backtest, and only in range markets. Defined risk on both sides.
    """
    if ticker not in ("^NSEI", "^NSEBANK"):
        return None
    df = add_indicators(df)
    if trend_18ema(df) != "flat":
        return None
    close = float(df.iloc[-1]["Close"])
    step = strike_step(close, ticker)

    def snap(x: float) -> float:
        return round(round(x / step) * step, 2)

    return {
        "name": name,
        "spot": round(close, 2),
        "sell_call": snap(close * 1.015),
        "buy_call": snap(close * 1.030),
        "sell_put": snap(close * 0.985),
        "buy_put": snap(close * 0.970),
        "lot": resolve_lot(ticker),
    }


# ------------------------------------------------------------------------ UI

st.set_page_config(page_title="Options Suggestion Scanner", page_icon="🎯",
                   layout="centered")

st.title("Options Suggestion Scanner")
st.caption(
    "18 EMA trend + Elliott Wave agreement, both directions. Bullish alignment -> CALL idea, "
    "bearish alignment -> PUT idea. No alignment -> no trade. Decision support only - "
    "verify live premiums on your broker; this never places orders and is not financial advice."
)

vix = fetch_vix()
if vix is not None:
    vix_state = "calm - premiums reasonable" if vix < 14 else (
        "normal" if vix < 20 else "elevated - premiums expensive, extra caution")
    st.info(f"India VIX: **{vix:.1f}** ({vix_state})")
else:
    st.warning("India VIX unavailable right now - judge premium cost manually.")

st.markdown("**Your manual checks** (tick what you've verified today):")
c1, c2, c3 = st.columns(3)
with c1:
    fii_ok = st.checkbox("FII/DII flows supportive")
with c2:
    sentiment_ok = st.checkbox("Market sentiment supportive")
with c3:
    buddy_ok = st.checkbox("NiftyBuddy conviction agrees")
manual_score = sum([fii_ok, sentiment_ok, buddy_ok])
st.caption(
    "These ticks do NOT change which suggestions appear — signals come purely from "
    "price data (18 EMA + Elliott + RSI + VIX). The app cannot see FII/DII numbers, "
    "news sentiment, or NiftyBuddy's posts, so you verify those yourself and tick. "
    "Together they complete the conviction score (3 automatic + 3 yours = /6)."
)

if st.button("Scan Market", type="primary", use_container_width=True) or True:
    tickers = list(INDICES.values()) + FO_STOCKS
    names = {v: k for k, v in INDICES.items()}

    with st.spinner(f"Scanning {len(tickers)} instruments (18 EMA + Elliott, both directions)..."):
        data, source = get_market_data(tuple(tickers))
        suggestions = []
        for ticker, df in data.items():
            display = names.get(ticker, ticker.replace(".NS", ""))
            row = analyze(ticker, display, df, vix)
            if row:
                suggestions.append(row)

    suggestions.sort(key=lambda r: r["wave_score"], reverse=True)
    st.caption(f"Data source: {source}  |  Last checked: "
               f"{datetime.now().strftime('%d %b %Y, %I:%M %p')}  |  "
               f"{len(data)} instruments analysed")

    if not suggestions:
        st.warning(
            "No directional trade right now. 18 EMA trend and Elliott Wave do not agree "
            "on any instrument - staying out IS the strategy protecting your capital."
        )
        with st.expander("Market picture (indices)"):
            for label, tk in INDICES.items():
                if tk in data:
                    d = add_indicators(data[tk])
                    st.write(f"**{label}**: close {float(d.iloc[-1]['Close']):,.0f}, "
                             f"18 EMA trend = {trend_18ema(d)}, "
                             f"wave = {elliott_directional(d)['bias']}")
        # hedged range idea - only structure with a (small) positive edge in our backtest
        for label, tk in INDICES.items():
            if tk not in data:
                continue
            idea = condor_idea(tk, label, data[tk], vix)
            if idea:
                with st.container(border=True):
                    st.subheader(f"🛡️ Range market - hedged idea: {idea['name']} Iron Condor")
                    st.write(
                        f"Spot {idea['spot']:,.0f}. SELL {idea['sell_call']:g} CE + "
                        f"SELL {idea['sell_put']:g} PE, and BUY {idea['buy_call']:g} CE + "
                        f"BUY {idea['buy_put']:g} PE as protection (nearest monthly expiry"
                        + (f", lot {idea['lot']}" if idea["lot"] else "") + ")."
                    )
                    st.write(
                        "Profits if the index stays between the sold strikes till expiry; "
                        "losses capped by the bought wings on both sides. In our 2-year "
                        "backtest this was the only structure with a small positive edge, "
                        "and only in range markets - paper-trade it first, and get exact "
                        "premiums from your broker's option chain."
                    )
    else:
        st.success(f"{len(suggestions)} suggestion(s) found")
        for s in suggestions[:6]:
            arrow = "🟢 BUY CALL" if s["direction"] == "CALL" else "🔴 BUY PUT"
            auto_score = 2 + (1 if s["vix_ok"] else 0)
            with st.container(border=True):
                st.subheader(f"{arrow} - {s['name']}  {s['strike']:g} "
                             f"{'CE' if s['direction'] == 'CALL' else 'PE'} (nearest monthly expiry)")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Est. premium", f"₹{s['premium_est']:,.0f}")
                if s["capital_est"]:
                    m2.metric(f"Capital / lot ({s['lot']})", f"₹{s['capital_est']:,.0f}")
                else:
                    m2.metric("Capital / lot", "check broker")
                m3.metric(f"Stop-loss (-{SL_PCT}%)", f"₹{s['sl_premium']:,.0f}")
                m4.metric(f"Target (+{TARGET_PCT}%)", f"₹{s['target_premium']:,.0f}")
                st.write(
                    f"Spot **{s['close']:,.2f}** vs 18 EMA **{s['ema18']:,.2f}**  |  "
                    f"RSI {s['rsi']}  |  Elliott: *{s['wave_stage']}* (score {s['wave_score']})"
                )
                st.write(
                    f"Also exit if {'close falls below' if s['direction'] == 'CALL' else 'close rises above'} "
                    f"the 18 EMA (~{s['underlying_exit']:,.2f})."
                )
                opt = "CE" if s["direction"] == "CALL" else "PE"
                st.write(
                    f"🛡️ *Optional hedge:* also **SELL {s['hedge_strike']:g} {opt}** to make it a "
                    f"spread — cuts cost and max loss roughly 40%, in exchange for a capped profit."
                )
                if manual_score < 3:
                    st.warning("Complete your manual checks above before acting on this.", icon="☑️")
                checks = [
                    "18 EMA trend aligned ✓",
                    "Elliott Wave aligned ✓",
                    ("VIX acceptable ✓" if s["vix_ok"] else "VIX high ✗"),
                    (f"Your manual checks: {manual_score}/3 "
                     + ("✓" if manual_score == 3 else "— tick above")),
                ]
                st.caption("Conviction: " + "  |  ".join(checks)
                           + f"  →  {auto_score + manual_score}/6")
        st.caption(
            "Premiums are estimates from spot and VIX - ALWAYS check the live premium and "
            "lot size on your broker before deciding. Risk per trade = the premium you pay; "
            "planned exit at -40%, target +80% (2x reward). Never risk money you cannot lose."
        )
