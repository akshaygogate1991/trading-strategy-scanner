"""Angel One SmartAPI data adapter (DATA ONLY — never places orders).

This module logs in to SmartAPI and returns historical candles in the SAME
shape the scanner already expects from Yahoo: a pandas DataFrame with columns
Open, High, Low, Close, Volume and a DatetimeIndex.

It is intentionally kept separate from app.py so the live Streamlit app keeps
working on free Yahoo data. Wire it in only when running locally with your
credentials in .streamlit/secrets.toml.

Credentials are read from st.secrets["angelone"]:
    api_key, client_id, mpin, totp_secret

Requires (install locally):  pip install smartapi-python pyotp
"""
from __future__ import annotations

import datetime as _dt
from functools import lru_cache

import pandas as pd
import requests

# yfinance-style ticker -> candidate names/symbols in the instrument master.
# Angel One lists indices inconsistently (name NIFTY vs symbol "Nifty 50"),
# so we try several known variants.
INDEX_MAP = {
    "^NSEI": (["Nifty 50", "NIFTY", "NIFTY 50"], "NSE"),
    "^NSEBANK": (["Nifty Bank", "BANKNIFTY", "NIFTY BANK"], "NSE"),
}

# SmartAPI candle intervals we use
INTERVAL_MAP = {
    "1d": "ONE_DAY",
    "1h": "ONE_HOUR",
    "60m": "ONE_HOUR",
    "15m": "FIFTEEN_MINUTE",
}

SCRIP_MASTER_URL = (
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
)


def _get_secrets():
    """Read credentials from Streamlit secrets, or fall back to the toml file
    directly (so this also works in a plain `python` script). Clear error if missing."""
    creds = None

    # 1) Streamlit runtime secrets (when running inside `streamlit run`)
    try:
        import streamlit as st

        if "angelone" in st.secrets:
            creds = dict(st.secrets["angelone"])
    except Exception:
        creds = None

    # 2) Fall back to reading .streamlit/secrets.toml directly
    if not creds:
        import pathlib

        try:
            import tomllib  # Python 3.11+
        except ModuleNotFoundError:  # pragma: no cover
            import tomli as tomllib  # type: ignore

        for path in (
            pathlib.Path(".streamlit/secrets.toml"),
            pathlib.Path(__file__).parent / ".streamlit" / "secrets.toml",
        ):
            if path.exists():
                with open(path, "rb") as fh:
                    creds = tomllib.load(fh).get("angelone")
                break

    if not creds:
        raise RuntimeError(
            "No [angelone] credentials found in .streamlit/secrets.toml "
            "(need api_key, client_id, mpin, totp_secret)."
        )
    for key in ("api_key", "client_id", "mpin", "totp_secret"):
        if not creds.get(key):
            raise RuntimeError(f"Missing '{key}' in [angelone] secrets.")
    return creds


@lru_cache(maxsize=1)
def _instrument_master() -> list:
    """Download the instrument master (symbol -> token) once, then cache it."""
    resp = requests.get(SCRIP_MASTER_URL, timeout=30)
    resp.raise_for_status()
    return resp.json()


def resolve_token(ticker: str) -> tuple[str, str, str]:
    """Map a yfinance-style ticker to (symboltoken, tradingsymbol, exchange).

    Stocks: "RELIANCE.NS" -> the NSE "RELIANCE-EQ" equity token.
    Indices: "^NSEI" / "^NSEBANK" -> the NSE index token.
    """
    master = _instrument_master()

    if ticker in INDEX_MAP:
        candidates, exch = INDEX_MAP[ticker]
        wanted = {c.upper() for c in candidates}
        for row in master:
            if row.get("exch_seg") != exch:
                continue
            # indices carry instrumenttype AMXIDX; match on symbol OR name
            if row.get("symbol", "").upper() in wanted or row.get("name", "").upper() in wanted:
                if row.get("instrumenttype", "AMXIDX") == "AMXIDX" or not row.get("expiry"):
                    return row["token"], row["symbol"], exch
        raise LookupError(f"Index not found in instrument master: {ticker}")

    base = ticker.replace(".NS", "").upper()
    want = f"{base}-EQ"
    for row in master:
        if row.get("exch_seg") == "NSE" and row.get("symbol", "").upper() == want:
            return row["token"], row["symbol"], "NSE"
    raise LookupError(f"Symbol not found on NSE: {ticker} (looked for {want})")


@lru_cache(maxsize=1)
def _session():
    """Log in once and cache the SmartConnect client. DATA ONLY."""
    try:
        from SmartApi import SmartConnect  # package: smartapi-python
        import pyotp
    except ImportError as exc:  # pragma: no cover - depends on local install
        raise RuntimeError(
            f"SmartAPI import failed: {exc!r}. "
            "If smartapi-python is installed, a dependency may be missing - "
            "try: pip install logzero websocket-client"
        ) from exc

    s = _get_secrets()
    client = SmartConnect(api_key=s["api_key"])
    otp = pyotp.TOTP(s["totp_secret"]).now()
    resp = client.generateSession(s["client_id"], s["mpin"], otp)
    if not resp.get("status"):
        raise RuntimeError(f"SmartAPI login failed: {resp.get('message', resp)}")
    return client


def get_history(ticker: str, interval: str = "1d", days: int = 400) -> pd.DataFrame | None:
    """Fetch candles for one ticker as an OHLCV DataFrame (yfinance-compatible)."""
    client = _session()
    token, tradingsymbol, exch = resolve_token(ticker)
    smart_interval = INTERVAL_MAP.get(interval, "ONE_DAY")

    to_dt = _dt.datetime.now()
    from_dt = to_dt - _dt.timedelta(days=days)
    params = {
        "exchange": exch,
        "symboltoken": token,
        "interval": smart_interval,
        "fromdate": from_dt.strftime("%Y-%m-%d 09:15"),
        "todate": to_dt.strftime("%Y-%m-%d 15:30"),
    }
    resp = client.getCandleData(params)
    candles = resp.get("data") if isinstance(resp, dict) else None
    if not candles:
        return None

    df = pd.DataFrame(candles, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts")
    df.index = df.index.tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)


_PERIOD_DAYS = {"6mo": 190, "1y": 380, "2y": 740}


def fetch_market_data(tickers, period: str) -> dict:
    """Drop-in replacement for app.fetch_market_data, backed by SmartAPI.

    Returns {ticker: OHLCV DataFrame} for tickers with enough daily history.
    """
    days = _PERIOD_DAYS.get(period, 380)
    out: dict = {}
    for ticker in tickers:
        try:
            frame = get_history(ticker, "1d", days)
        except Exception:
            continue
        if frame is not None and len(frame) >= 80:
            out[ticker] = frame
    return out
