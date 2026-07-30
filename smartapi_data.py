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


_master_failed = False


def get_lot_size(ticker: str) -> int | None:
    """Official NSE lot size from the instrument master (same at every broker).

    Looks up the F&O contract for the symbol and reads its lotsize field.
    Returns None if the master is unreachable or the symbol has no F&O.
    """
    global _master_failed
    if _master_failed:
        return None
    try:
        master = _instrument_master()
    except Exception:
        _master_failed = True  # don't retry on every call
        return None

    if ticker == "^NSEI":
        names = {"NIFTY"}
    elif ticker == "^NSEBANK":
        names = {"BANKNIFTY"}
    else:
        names = {ticker.replace(".NS", "").upper()}

    for row in master:
        if row.get("exch_seg") != "NFO":
            continue
        if row.get("name", "").upper() not in names:
            continue
        if row.get("instrumenttype") not in ("OPTSTK", "FUTSTK", "OPTIDX", "FUTIDX"):
            continue
        try:
            lot = int(float(row.get("lotsize", 0)))
        except (TypeError, ValueError):
            continue
        if lot > 0:
            return lot
    return None


def _parse_expiry(expiry_str: str) -> _dt.date | None:
    """Angel One's expiry field looks like '28JUL2026'."""
    try:
        return _dt.datetime.strptime(expiry_str, "%d%b%Y").date()
    except (ValueError, TypeError):
        return None


def resolve_option(ticker: str, option_type: str, target_strike: float) -> dict | None:
    """Find the nearest MONTHLY option contract and its closest listed strike.

    option_type: 'CE' or 'PE'. The "monthly" contract is identified purely from
    the exchange's own data (the last expiry date within a calendar month) -
    not by assuming a fixed weekday, since NSE has changed expiry weekdays
    more than once.

    Returns {"token", "tradingsymbol", "expiry" (date), "strike"} or None.
    """
    master = _instrument_master()
    if ticker == "^NSEI":
        name = "NIFTY"
    elif ticker == "^NSEBANK":
        name = "BANKNIFTY"
    else:
        name = ticker.replace(".NS", "").upper()

    rows = [
        r for r in master
        if r.get("exch_seg") == "NFO"
        and r.get("name", "").upper() == name
        and r.get("instrumenttype") in ("OPTSTK", "OPTIDX")
        and r.get("symbol", "").upper().endswith(option_type)
    ]
    if not rows:
        return None

    today = _dt.date.today()
    by_month: dict[tuple[int, int], list[tuple[_dt.date, dict]]] = {}
    for r in rows:
        d = _parse_expiry(r.get("expiry", ""))
        if d and d >= today:
            by_month.setdefault((d.year, d.month), []).append((d, r))
    if not by_month:
        return None

    # the monthly contract for each month = its LAST available expiry that month
    monthly_expiry_per_month = {
        ym: max(d for d, _ in items) for ym, items in by_month.items()
    }
    nearest_month = min(monthly_expiry_per_month, key=lambda ym: monthly_expiry_per_month[ym])
    expiry_date = monthly_expiry_per_month[nearest_month]
    candidates = [r for d, r in by_month[nearest_month] if d == expiry_date]

    def strike_of(row: dict) -> float | None:
        try:
            return float(row.get("strike", 0)) / 100.0
        except (TypeError, ValueError):
            return None

    priced = [(strike_of(r), r) for r in candidates]
    priced = [(s, r) for s, r in priced if s is not None and s > 0]
    if not priced:
        return None
    best_strike, best_row = min(priced, key=lambda pair: abs(pair[0] - target_strike))

    return {
        "token": best_row["token"],
        "tradingsymbol": best_row["symbol"],
        "expiry": expiry_date,
        "strike": best_strike,
    }


def get_ltp(exchange: str, tradingsymbol: str, token: str) -> float | None:
    """Live last-traded-price for one option contract. Requires an active session.

    Raises on any failure (login, network, bad response) instead of swallowing
    the error, so the caller can capture and surface the exact reason a live
    quote wasn't available - much easier to diagnose than a silent fallback.
    """
    client = _session()
    resp = client.ltpData(exchange, tradingsymbol, token)
    if not resp.get("status"):
        raise RuntimeError(f"ltpData rejected: {resp.get('message', resp)}")
    ltp = float(resp["data"]["ltp"])
    if ltp <= 0:
        raise RuntimeError(f"ltpData returned non-positive price: {ltp}")
    return ltp


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
