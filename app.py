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

import trade_log as tl


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


@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_fii_dii() -> dict | None:
    """Latest FII/DII cash-market net flows (Rs. crore) from NSE's public API.

    Works best from a home/residential connection; NSE blocks many server IPs.
    Returns None on failure - the UI then falls back to a manual checkbox.
    """
    import requests

    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
    }
    def _parse(payload) -> dict | None:
        """Defensive parse: walk any JSON shape for FII/DII rows with a net value."""
        out = {"date": ""}
        stack = [payload]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                cat = str(node.get("category") or node.get("fii_dii_category") or "").upper()
                net = node.get("netValue") or node.get("net_value") or node.get("net")
                if net is not None and cat:
                    try:
                        val = float(str(net).replace(",", ""))
                    except ValueError:
                        val = None
                    if val is not None:
                        if "FII" in cat or "FPI" in cat:
                            out["fii_net"] = val
                            out["date"] = str(node.get("date") or node.get("trade_date") or "")
                        elif "DII" in cat:
                            out["dii_net"] = val
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
        return out if "fii_net" in out and "dii_net" in out else None

    # Source 1: NSE's own API (best from home connections; blocks many servers)
    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=8)
        resp = session.get("https://www.nseindia.com/api/fiidiiTradeReact",
                           headers=headers, timeout=8)
        parsed = _parse(resp.json())
        if parsed:
            return parsed
    except Exception:
        pass

    # Source 2: niftytrader public web API (often reachable from cloud servers)
    try:
        resp = requests.get(
            "https://webapi.niftytrader.in/webapi/Resource/fii-dii-activity-data",
            headers=headers, timeout=8,
        )
        parsed = _parse(resp.json())
        if parsed:
            return parsed
    except Exception:
        pass
    return None


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
    dist = close * (0.02 if not is_stock else 0.03)  # ~2% OTM indices, ~3% stocks
    n_steps = max(2, round(dist / step))
    hedge_strike = strike + n_steps * step if direction == "CALL" else strike - n_steps * step
    width = n_steps * step
    hedge_credit = round(0.4 * premium, 2)       # modeled OTM premium (~40% of ATM)
    net_debit = round(premium - hedge_credit, 2)  # hedged cost = hedged max loss
    spread_max_profit = round(width - net_debit, 2)
    # at expiry: below/above this underlying level the hedged trade flips loss<->profit
    hedged_breakeven = round(strike + net_debit if direction == "CALL" else strike - net_debit, 2)

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
        "hedge_credit": hedge_credit,
        "net_debit": net_debit,
        "spread_max_profit": spread_max_profit,
        "hedged_breakeven": hedged_breakeven,
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

tab_suggest, tab_log = st.tabs(["📊 Suggestions", "📒 Trade Log"])

with tab_suggest:
    vix = fetch_vix()
    if vix is not None:
        vix_state = "calm - premiums reasonable" if vix < 14 else (
            "normal" if vix < 20 else "elevated - premiums expensive, extra caution")
        st.info(f"India VIX: **{vix:.1f}** ({vix_state})")
    else:
        st.warning("India VIX unavailable right now - judge premium cost manually.")

    fii = fetch_fii_dii()
    if fii:
        total_flow = fii["fii_net"] + fii["dii_net"]
        st.info(
            f"Institutional flows ({fii['date']}): FII net ₹{fii['fii_net']:+,.0f} Cr, "
            f"DII net ₹{fii['dii_net']:+,.0f} Cr → combined **₹{total_flow:+,.0f} Cr** "
            f"({'buying' if total_flow > 0 else 'selling'})"
        )
        fii_manual = None
    else:
        st.warning("FII/DII data unreachable right now (NSE blocks some connections) - "
                   "check moneycontrol.com and tick manually below.")
        fii_manual = st.checkbox("FII/DII flows supportive (verified manually)")

    st.caption(
        "Conviction is 5 automatic checks per suggestion: 18 EMA trend, Elliott Wave, "
        "VIX level, FII/DII flow direction, and Nifty market alignment. "
        "All computed from data - nothing to tick. Note: market data is cached 20 min, "
        "so a refresh within that window reuses the last fetch on purpose."
    )

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
        # Nifty's own state, used as the market-alignment check for every card
        nifty_trend, nifty_wave = "flat", "Neutral"
        if "^NSEI" in data:
            nd = add_indicators(data["^NSEI"])
            nifty_trend = trend_18ema(nd)
            nifty_wave = elliott_directional(nd)["bias"]

        for i, s in enumerate(suggestions[:6]):
            arrow = "🟢 BUY CALL" if s["direction"] == "CALL" else "🔴 BUY PUT"

            if s["direction"] == "CALL":
                align_ok = nifty_trend != "down" and nifty_wave != "Bearish"
            else:
                align_ok = nifty_trend != "up" and nifty_wave != "Bullish"

            if fii:
                flow = fii["fii_net"] + fii["dii_net"]
                flows_ok = flow > 0 if s["direction"] == "CALL" else flow < 0
            else:
                flows_ok = bool(fii_manual)

            auto_score = 2 + sum([s["vix_ok"], flows_ok, align_ok])
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
                lot = s["lot"]
                per_lot = (lambda x: f" (₹{x * lot:,.0f}/lot)") if lot else (lambda x: "")
                st.markdown(
                    f"**🛡️ Hedged version (optional):** also SELL **{s['hedge_strike']:g} {opt}** "
                    f"for ~₹{s['hedge_credit']:,.0f} credit → net cost ₹{s['net_debit']:,.0f}/share"
                    f"{per_lot(s['net_debit'])}.\n\n"
                    f"- Max LOSS hedged: **₹{s['net_debit']:,.0f}/share**{per_lot(s['net_debit'])} "
                    f"vs ₹{s['premium_est']:,.0f} unhedged → risk cut ~"
                    f"{round(100 * s['hedge_credit'] / s['premium_est'])}%\n"
                    f"- Max PROFIT hedged: capped at **₹{s['spread_max_profit']:,.0f}/share**"
                    f"{per_lot(s['spread_max_profit'])} — no matter how far the move goes\n"
                    f"- The hedge does NOT profit when you're wrong — it only makes the loss smaller."
                )
                if s["direction"] == "CALL":
                    ladder = (
                        f"**📍 Hedged trade map** ({s['name']} share price at expiry):\n"
                        f"- Below **{s['strike']:g}** → MAX LOSS ₹{s['net_debit']:,.0f}/share{per_lot(s['net_debit'])}\n"
                        f"- At **{s['hedged_breakeven']:,.2f}** → no profit, no loss (breakeven)\n"
                        f"- Above {s['hedged_breakeven']:,.2f} → profit grows\n"
                        f"- At/above **{s['hedge_strike']:g}** → MAX PROFIT ₹{s['spread_max_profit']:,.0f}/share"
                        f"{per_lot(s['spread_max_profit'])} (fixed — hedged)"
                    )
                else:
                    ladder = (
                        f"**📍 Hedged trade map** ({s['name']} share price at expiry):\n"
                        f"- Above **{s['strike']:g}** → MAX LOSS ₹{s['net_debit']:,.0f}/share{per_lot(s['net_debit'])}\n"
                        f"- At **{s['hedged_breakeven']:,.2f}** → no profit, no loss (breakeven)\n"
                        f"- Below {s['hedged_breakeven']:,.2f} → profit grows\n"
                        f"- At/below **{s['hedge_strike']:g}** → MAX PROFIT ₹{s['spread_max_profit']:,.0f}/share"
                        f"{per_lot(s['spread_max_profit'])} (fixed — hedged)"
                    )
                st.markdown(ladder)
                st.caption(
                    f"Risk reality: the -{SL_PCT}% stop plans to lose only "
                    f"₹{round(s['premium_est'] * SL_PCT / 100):,.0f}/share, but if price gaps past it "
                    f"your true max loss is the full premium (₹{s['premium_est']:,.0f} unhedged, "
                    f"₹{s['net_debit']:,.0f} hedged)."
                )
                checks = [
                    "18 EMA ✓",
                    "Elliott ✓",
                    ("VIX ✓" if s["vix_ok"] else "VIX high ✗"),
                    ("FII/DII ✓" if flows_ok else "FII/DII against ✗"),
                    ("Nifty aligned ✓" if align_ok else "Nifty against ✗"),
                ]
                st.caption("Conviction: " + "  |  ".join(checks) + f"  →  **{auto_score}/5**")
                if auto_score < 4:
                    st.warning("Below 4/5 conviction — consider skipping or paper-trading this one.",
                               icon="⚖️")

                log_col1, log_col2 = st.columns([1, 2])
                with log_col1:
                    take_hedge = st.checkbox("Log the hedged version", key=f"hedge_{i}_{s['ticker']}")
                with log_col2:
                    if st.button("📝 Log this trade", key=f"log_{i}_{s['ticker']}",
                                use_container_width=True):
                        entry_premium = s["net_debit"] if take_hedge else s["premium_est"]
                        ok = tl.log_trade({
                            "ticker": s["ticker"],
                            "name": s["name"],
                            "direction": s["direction"],
                            "strike": s["strike"],
                            "entry_premium": entry_premium,
                            "lot": s["lot"],
                            "conviction": auto_score,
                            "hedged": bool(take_hedge),
                            "hedge_strike": s["hedge_strike"] if take_hedge else None,
                            "status": "OPEN",
                        })
                        if ok:
                            st.toast(f"Logged {s['name']} {s['direction']} — see Trade Log tab.",
                                     icon="✅")
                        elif not tl.is_configured():
                            st.error("Trade Log isn't set up yet — see the Trade Log tab for "
                                    "one-time Supabase setup steps.")
                        else:
                            st.error("Could not save right now — try again in a moment.")
        st.caption(
            "Premiums are estimates from spot and VIX - ALWAYS check the live premium and "
            "lot size on your broker before deciding. Risk per trade = the premium you pay; "
            "planned exit at -40%, target +80% (2x reward). Never risk money you cannot lose."
        )

with tab_log:
    st.subheader("Paper-trading log")
    if not tl.is_configured():
        st.warning(
            "Trade Log needs a free Supabase database to persist entries across app restarts "
            "(the cloud app's own files don't survive a restart)."
        )
        st.markdown(
            "**One-time setup (about 5 minutes):**\n\n"
            "1. In your Supabase project, open the **SQL Editor** and run:\n"
        )
        st.code(
            "create table trade_log (\n"
            "    id uuid primary key default gen_random_uuid(),\n"
            "    created_at timestamptz default now(),\n"
            "    ticker text, name text, direction text, strike numeric,\n"
            "    entry_premium numeric, lot integer, conviction integer,\n"
            "    hedged boolean default false, hedge_strike numeric,\n"
            "    status text default 'OPEN',\n"
            "    exit_premium numeric, closed_at timestamptz,\n"
            "    pnl_per_share numeric, pnl_total numeric\n"
            ");\n"
            "alter table trade_log enable row level security;\n"
            "create policy \"personal use - allow all\" on trade_log\n"
            "    for all using (true) with check (true);",
            language="sql",
        )
        st.markdown(
            "2. Go to **Settings -> API** in Supabase, copy the **Project URL** and the "
            "**anon public key**.\n"
            "3. Add them to your `.streamlit/secrets.toml` file (same file as your Angel One "
            "credentials):"
        )
        st.code(
            '[supabase]\nurl = "https://xxxxx.supabase.co"\nanon_key = "your-anon-key-here"',
            language="toml",
        )
        st.markdown("4. Restart the app. This tab will then show your logged trades.")
    else:
        open_trades = tl.fetch_trades("OPEN")
        closed_trades = tl.fetch_trades("CLOSED")

        st.markdown(f"**Open positions ({len(open_trades)})**")
        if not open_trades:
            st.caption("Nothing logged yet — use \"📝 Log this trade\" on a suggestion card.")
        for t in open_trades:
            with st.container(border=True):
                tag = "CE" if t["direction"] == "CALL" else "PE"
                hedge_note = " (hedged spread)" if t.get("hedged") else ""
                st.write(
                    f"**{t['name']} {t['strike']:g} {tag}**{hedge_note} — "
                    f"entry ₹{t['entry_premium']:,.2f}"
                    + (f", lot {t['lot']}" if t.get("lot") else "")
                    + f" — logged {str(t['created_at'])[:16].replace('T', ' ')}"
                )
                c1, c2 = st.columns([1, 1])
                exit_val = c1.number_input(
                    "Exit premium", min_value=0.0, step=0.5, key=f"exit_{t['id']}"
                )
                if c2.button("Close trade", key=f"close_{t['id']}", use_container_width=True):
                    pnl_ps = round(exit_val - t["entry_premium"], 2)
                    pnl_tot = round(pnl_ps * t["lot"], 2) if t.get("lot") else None
                    if tl.close_trade(t["id"], exit_val, pnl_ps, pnl_tot):
                        st.toast(f"Closed — P&L ₹{pnl_ps:+,.2f}/share", icon="✅")
                        st.rerun()
                    else:
                        st.error("Could not close — try again.")
                if st.button("🗑️ Delete (logged by mistake)", key=f"del_{t['id']}"):
                    if tl.delete_trade(t["id"]):
                        st.rerun()

        st.markdown("---")
        st.markdown(f"**Closed trades ({len(closed_trades)})**")
        if closed_trades:
            df = pd.DataFrame(closed_trades)[
                ["created_at", "name", "direction", "strike", "entry_premium",
                 "exit_premium", "pnl_per_share", "pnl_total", "conviction"]
            ]
            st.dataframe(df, use_container_width=True, hide_index=True)

            wins = sum(1 for t in closed_trades if (t.get("pnl_per_share") or 0) > 0)
            total_pnl = sum(t.get("pnl_total") or 0 for t in closed_trades)
            avg_pnl_ps = sum(t.get("pnl_per_share") or 0 for t in closed_trades) / len(closed_trades)
            c1, c2, c3 = st.columns(3)
            c1.metric("Win rate", f"{100 * wins / len(closed_trades):.0f}%")
            c2.metric("Total P&L", f"₹{total_pnl:+,.0f}")
            c3.metric("Avg P&L/share", f"₹{avg_pnl_ps:+.2f}")
        else:
            st.caption("No closed trades yet.")
