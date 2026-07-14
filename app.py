from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


INDICES = {
    "Nifty 50": "^NSEI",
    "Bank Nifty": "^NSEBANK",
}

NIFTY_100 = [
    "ABB.NS",
    "ADANIENSOL.NS",
    "ADANIENT.NS",
    "ADANIGREEN.NS",
    "ADANIPORTS.NS",
    "AMBUJACEM.NS",
    "APOLLOHOSP.NS",
    "ASIANPAINT.NS",
    "AXISBANK.NS",
    "BAJAJ-AUTO.NS",
    "BAJAJFINSV.NS",
    "BAJFINANCE.NS",
    "BANKBARODA.NS",
    "BEL.NS",
    "BHARTIARTL.NS",
    "BOSCHLTD.NS",
    "BPCL.NS",
    "BRITANNIA.NS",
    "CANBK.NS",
    "CHOLAFIN.NS",
    "CIPLA.NS",
    "COALINDIA.NS",
    "DABUR.NS",
    "DIVISLAB.NS",
    "DLF.NS",
    "DMART.NS",
    "DRREDDY.NS",
    "EICHERMOT.NS",
    "GAIL.NS",
    "GODREJCP.NS",
    "GRASIM.NS",
    "HAVELLS.NS",
    "HCLTECH.NS",
    "HDFCBANK.NS",
    "HDFCLIFE.NS",
    "HEROMOTOCO.NS",
    "HINDALCO.NS",
    "HINDUNILVR.NS",
    "ICICIBANK.NS",
    "ICICIGI.NS",
    "ICICIPRULI.NS",
    "INDIGO.NS",
    "INDUSINDBK.NS",
    "INFY.NS",
    "IOC.NS",
    "IRCTC.NS",
    "ITC.NS",
    "JINDALSTEL.NS",
    "JSWSTEEL.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "LTIM.NS",
    "M&M.NS",
    "MARICO.NS",
    "MARUTI.NS",
    "MOTHERSON.NS",
    "NAUKRI.NS",
    "NESTLEIND.NS",
    "NTPC.NS",
    "ONGC.NS",
    "PIDILITIND.NS",
    "PNB.NS",
    "POWERGRID.NS",
    "RELIANCE.NS",
    "SBICARD.NS",
    "SBILIFE.NS",
    "SBIN.NS",
    "SHREECEM.NS",
    "SHRIRAMFIN.NS",
    "SIEMENS.NS",
    "SUNPHARMA.NS",
    "TATACONSUM.NS",
    "TATAMOTORS.NS",
    "TATASTEEL.NS",
    "TCS.NS",
    "TECHM.NS",
    "TITAN.NS",
    "TORNTPHARM.NS",
    "TRENT.NS",
    "TVSMOTOR.NS",
    "ULTRACEMCO.NS",
    "UNIONBANK.NS",
    "UNITDSPR.NS",
    "VBL.NS",
    "VEDL.NS",
    "WIPRO.NS",
    "ZYDUSLIFE.NS",
]


@dataclass(frozen=True)
class ScanSettings:
    capital: float
    risk_percent: float
    min_rr: float
    data_period: str
    batch_size: int
    use_elliott_filter: bool
    min_wave_score: int


def normalize_tickers(raw: Iterable[str]) -> list[str]:
    tickers = []
    seen = set()
    for ticker in raw:
        cleaned = ticker.strip().upper()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        tickers.append(cleaned)
    return tickers


@st.cache_data(ttl=60 * 20, show_spinner=False)
def fetch_market_data(tickers: tuple[str, ...], period: str) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        frame = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if frame.empty:
            continue
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
        if len(frame) >= 80:
            data[ticker] = frame
    return data


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema18"] = out["Close"].ewm(span=18, adjust=False).mean()
    out["ema50"] = out["Close"].ewm(span=50, adjust=False).mean()
    out["ema200"] = out["Close"].ewm(span=200, adjust=False).mean()
    out["vol20"] = out["Volume"].rolling(20).mean()
    out["high20_prev"] = out["High"].rolling(20).max().shift(1)

    delta = out["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi14"] = 100 - (100 / (1 + rs))

    prev_close = out["Close"].shift(1)
    tr = pd.concat(
        [
            out["High"] - out["Low"],
            (out["High"] - prev_close).abs(),
            (out["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return out


def detect_swing_pivots(df: pd.DataFrame, lookback: int = 3) -> list[dict]:
    pivots: list[dict] = []
    if len(df) < (lookback * 2) + 10:
        return pivots

    recent = df.tail(140)
    for pos in range(lookback, len(recent) - lookback):
        window = recent.iloc[pos - lookback : pos + lookback + 1]
        row = recent.iloc[pos]
        index_value = recent.index[pos]
        if row["High"] == window["High"].max():
            pivots.append({"type": "H", "date": index_value, "price": float(row["High"]), "pos": pos})
        if row["Low"] == window["Low"].min():
            pivots.append({"type": "L", "date": index_value, "price": float(row["Low"]), "pos": pos})

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


def elliott_wave_context(df: pd.DataFrame) -> dict:
    pivots = detect_swing_pivots(df)
    default = {
        "bias": "Neutral",
        "stage": "No clean wave count",
        "score": 0,
        "detail": "Not enough alternating swing pivots.",
    }
    if len(pivots) < 4:
        return default

    close = float(df.iloc[-1]["Close"])
    atr = float(df.iloc[-1]["atr14"])
    tolerance = max(atr, close * 0.01)

    for start in range(max(0, len(pivots) - 6), len(pivots) - 3):
        candidate = pivots[start : start + 5]
        if [p["type"] for p in candidate] != ["L", "H", "L", "H", "L"]:
            continue

        w0, w1, w2, w3, w4 = candidate
        wave1 = w1["price"] - w0["price"]
        wave3 = w3["price"] - w2["price"]
        wave4 = w3["price"] - w4["price"]

        if wave1 <= 0 or wave3 <= 0:
            continue
        if w2["price"] <= w0["price"]:
            continue
        if w4["price"] <= w1["price"]:
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
            detail = "Price has moved above the Wave 3 high after a valid Wave 4 pullback."
        elif abs(close - w4["price"]) <= 1.5 * tolerance or close > w4["price"]:
            score += 10
            stage = "Wave 4 pullback support"
            detail = "Price is near or above the recent Wave 4 pivot after a bullish impulse."
        else:
            stage = "Post Wave 4 watch"
            detail = "Bullish impulse exists, but price has not confirmed continuation."
        return {"bias": "Bullish", "stage": stage, "score": min(score, 100), "detail": detail}

    last_highs = [p["price"] for p in pivots if p["type"] == "H"][-2:]
    last_lows = [p["price"] for p in pivots if p["type"] == "L"][-2:]
    if len(last_highs) == 2 and len(last_lows) == 2 and last_highs[-1] > last_highs[-2] and last_lows[-1] > last_lows[-2]:
        return {
            "bias": "Bullish",
            "stage": "Higher high / higher low",
            "score": 45,
            "detail": "Trend structure is bullish, but a full Elliott impulse count is not confirmed.",
        }
    return default


def market_is_healthy(index_data: dict[str, pd.DataFrame]) -> tuple[bool, str]:
    nifty = index_data.get("^NSEI")
    if nifty is None or nifty.empty:
        return True, "Nifty data unavailable; market filter relaxed."
    latest = add_indicators(nifty).iloc[-1]
    if latest["Close"] > latest["ema50"] and latest["ema50"] > latest["ema200"]:
        return True, "Market filter bullish: Nifty above 50 EMA and 200 EMA."
    if latest["Close"] > latest["ema200"] and latest["ema50"] > add_indicators(nifty).iloc[-6]["ema50"]:
        return True, "Market filter acceptable: Nifty above 200 EMA and 50 EMA rising."
    return False, "Market filter weak: Nifty trend is not supportive for fresh long trades."


def scan_symbol(ticker: str, df: pd.DataFrame, settings: ScanSettings, allow_long: bool) -> dict | None:
    df = add_indicators(df)
    latest = df.iloc[-1]
    previous = df.iloc[-2]

    if any(pd.isna(latest[col]) for col in ["ema18", "ema50", "rsi14", "atr14", "high20_prev"]):
        return None

    close = float(latest["Close"])
    high = float(latest["High"])
    low = float(latest["Low"])
    atr = float(latest["atr14"])
    ema18 = float(latest["ema18"])
    ema50 = float(latest["ema50"])
    high20_prev = float(latest["high20_prev"])
    rsi = float(latest["rsi14"])
    volume = float(latest.get("Volume", 0))
    vol20 = float(latest.get("vol20", 0)) if not pd.isna(latest.get("vol20", np.nan)) else 0

    trend_ok = close > ema50 and ema50 >= float(df.iloc[-6]["ema50"])
    wave = elliott_wave_context(df)
    wave_ok = (not settings.use_elliott_filter) or (
        wave["bias"] == "Bullish" and int(wave["score"]) >= settings.min_wave_score
    )
    volume_ok = vol20 > 0 and volume >= 1.25 * vol20
    breakout = allow_long and trend_ok and wave_ok and close > high20_prev and volume_ok and rsi >= 55

    touched_ema18 = low <= ema18 * 1.01 and close >= ema18
    reclaimed_strength = close > float(previous["High"]) and rsi >= 48
    pullback = allow_long and trend_ok and wave_ok and touched_ema18 and reclaimed_strength

    if not breakout and not pullback:
        return None

    setup = "20-day breakout" if breakout else "18 EMA pullback"
    entry = round(max(close, high), 2)
    swing_low = float(df["Low"].tail(8).min())
    atr_stop = close - (1.5 * atr)
    stop_loss = round(max(swing_low, atr_stop), 2)

    if stop_loss >= entry:
        stop_loss = round(close - atr, 2)
    risk_per_share = round(entry - stop_loss, 2)
    if risk_per_share <= 0:
        return None

    target_1 = round(entry + (1.5 * risk_per_share), 2)
    target_2 = round(entry + (2.5 * risk_per_share), 2)
    target_3 = round(entry + (3.0 * risk_per_share), 2)
    rr = round((target_2 - entry) / risk_per_share, 2)
    if rr < settings.min_rr:
        return None

    risk_amount = settings.capital * settings.risk_percent / 100
    quantity = int(risk_amount // risk_per_share)
    score = 0
    score += 30 if breakout else 20
    score += min(25, max(0, int((rsi - 45) * 1.2)))
    score += 20 if volume_ok else 5
    score += 15 if close > ema18 > ema50 else 5
    score += min(20, int(wave["score"] * 0.2))
    score += 10 if ticker in INDICES.values() else 0

    return {
        "Symbol": ticker,
        "Setup": setup,
        "Score": score,
        "Last Close": round(close, 2),
        "Entry": entry,
        "Stop Loss": stop_loss,
        "Target 1": target_1,
        "Target 2": target_2,
        "Target 3": target_3,
        "Risk/Share": risk_per_share,
        "Qty @ Risk": quantity,
        "Wave Bias": wave["bias"],
        "Wave Stage": wave["stage"],
        "Wave Score": int(wave["score"]),
        "RSI 14": round(rsi, 1),
        "Reward/Risk": rr,
        "Volume x20": round(volume / vol20, 2) if vol20 else np.nan,
    }


def run_scan(tickers: list[str], settings: ScanSettings) -> tuple[pd.DataFrame, str]:
    index_tickers = tuple(INDICES.values())
    index_data = fetch_market_data(index_tickers, settings.data_period)
    allow_long, market_message = market_is_healthy(index_data)

    all_data = fetch_market_data(tuple(tickers), settings.data_period)
    rows = []
    for ticker, df in all_data.items():
        signal = scan_symbol(ticker, df, settings, allow_long or ticker in INDICES.values())
        if signal:
            rows.append(signal)

    signals = pd.DataFrame(rows)
    if not signals.empty:
        signals = signals.sort_values(["Score", "Reward/Risk"], ascending=False).reset_index(drop=True)
    return signals, market_message


st.set_page_config(page_title="Nifty Strategy Scanner", page_icon="📈", layout="wide")

st.title("Nifty Strategy Scanner")
st.caption("18 EMA pullback + 20-day breakout scanner for Nifty 100, Nifty 50, and Bank Nifty.")

with st.sidebar:
    st.header("Scan Settings")
    capital = st.number_input("Trading capital", min_value=1000, value=100000, step=5000)
    risk_percent = st.slider("Risk per trade (%)", min_value=0.25, max_value=2.0, value=1.0, step=0.25)
    min_rr = st.slider("Minimum reward:risk", min_value=1.5, max_value=3.0, value=2.0, step=0.25)
    data_period = st.selectbox("History", ["6mo", "1y", "2y"], index=1)
    use_elliott_filter = st.checkbox("Use Elliott Wave filter", value=True)
    min_wave_score = st.slider("Minimum Elliott score", min_value=0, max_value=100, value=45, step=5)
    scan_universe = st.multiselect(
        "Universe",
        ["Nifty 100 stocks", "Nifty 50 index", "Bank Nifty index"],
        default=["Nifty 100 stocks", "Nifty 50 index", "Bank Nifty index"],
    )
    extra = st.text_area("Extra tickers", placeholder="Example: TATAPOWER.NS, IRFC.NS")
    auto_scan = st.checkbox("Auto scan on page refresh", value=True)

tickers: list[str] = []
if "Nifty 100 stocks" in scan_universe:
    tickers.extend(NIFTY_100)
if "Nifty 50 index" in scan_universe:
    tickers.append(INDICES["Nifty 50"])
if "Bank Nifty index" in scan_universe:
    tickers.append(INDICES["Bank Nifty"])
if extra.strip():
    tickers.extend(extra.replace("\n", ",").split(","))
tickers = normalize_tickers(tickers)

settings = ScanSettings(
    capital=float(capital),
    risk_percent=float(risk_percent),
    min_rr=float(min_rr),
    data_period=data_period,
    batch_size=1,
    use_elliott_filter=bool(use_elliott_filter),
    min_wave_score=int(min_wave_score),
)

left, right = st.columns([2, 1])
with left:
    st.subheader("Trade Signals")
with right:
    st.write(f"Last checked: {datetime.now().strftime('%d %b %Y, %I:%M %p')}")

manual_scan = st.button("Scan Market", type="primary", use_container_width=True)

if auto_scan or manual_scan:
    with st.spinner(f"Scanning {len(tickers)} charts..."):
        signals, market_message = run_scan(tickers, settings)

    st.info(market_message)
    if signals.empty:
        st.warning("No trade available right now based on this strategy.")
    else:
        st.success(f"{len(signals)} trade setup(s) found.")
        st.dataframe(
            signals,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100),
                "Reward/Risk": st.column_config.NumberColumn("Reward/Risk", format="%.2fR"),
                "Volume x20": st.column_config.NumberColumn("Volume x20", format="%.2fx"),
            },
        )

        top = signals.iloc[0]
        st.subheader("Top Setup")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Symbol", top["Symbol"])
        c2.metric("Entry", top["Entry"])
        c3.metric("Stop Loss", top["Stop Loss"])
        c4.metric("Target 2", top["Target 2"])
        c5.metric("Qty", int(top["Qty @ Risk"]))

        st.caption(
            "Use this as a decision-support scanner, not automatic buy/sell advice. "
            "Confirm liquidity, news, gap risk, and your broker order rules before placing any trade."
        )
else:
    st.info("Click Scan Market whenever you open the app. If nothing matches, it will show no trade available.")
