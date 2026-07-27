"""Paper-trading log, persisted in Supabase (Postgres + REST API).

Why Supabase: the live app runs on Streamlit Community Cloud, whose local
filesystem is wiped on restart/redeploy - a plain CSV would lose your log.
Supabase is a small hosted Postgres database with a free tier and a simple
REST API (PostgREST), so entries survive restarts.

Credentials read from st.secrets["supabase"]: url, anon_key.
If not configured, every function here degrades gracefully (returns False/[])
so the app still runs - the Trade Log tab just explains what to set up.

Table schema (run once in the Supabase SQL editor):

    create table trade_log (
        id uuid primary key default gen_random_uuid(),
        created_at timestamptz default now(),
        ticker text,
        name text,
        direction text,          -- CALL or PUT
        strike numeric,
        entry_premium numeric,
        lot integer,
        conviction integer,
        hedged boolean default false,
        hedge_strike numeric,
        status text default 'OPEN',   -- OPEN or CLOSED
        exit_premium numeric,
        closed_at timestamptz,
        pnl_per_share numeric,
        pnl_total numeric
    );
    alter table trade_log enable row level security;
    create policy "personal use - allow all" on trade_log
        for all using (true) with check (true);
"""
from __future__ import annotations

from datetime import datetime, timezone

import requests
import streamlit as st

TABLE = "trade_log"


def _config() -> tuple[str, str] | None:
    try:
        s = st.secrets["supabase"]
        url, key = s.get("url"), s.get("anon_key")
        if url and key:
            return url.rstrip("/"), key
    except Exception:
        pass
    return None


def is_configured() -> bool:
    return _config() is not None


def _headers(key: str, prefer: str = "return=representation") -> dict:
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": prefer,
    }


def log_trade(row: dict) -> bool:
    """Insert a new OPEN trade snapshot. Returns True on success."""
    cfg = _config()
    if not cfg:
        return False
    url, key = cfg
    try:
        resp = requests.post(f"{url}/rest/v1/{TABLE}", headers=_headers(key),
                             json=row, timeout=10)
        return resp.status_code in (200, 201)
    except Exception:
        return False


def fetch_trades(status: str | None = None) -> list[dict]:
    """Fetch trades, optionally filtered by status ('OPEN' or 'CLOSED')."""
    cfg = _config()
    if not cfg:
        return []
    url, key = cfg
    params = {"select": "*", "order": "created_at.desc"}
    if status:
        params["status"] = f"eq.{status}"
    try:
        resp = requests.get(f"{url}/rest/v1/{TABLE}", headers=_headers(key),
                            params=params, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


def close_trade(trade_id: str, exit_premium: float, pnl_per_share: float,
                pnl_total: float | None) -> bool:
    """Mark a trade CLOSED with the exit premium and computed P&L."""
    cfg = _config()
    if not cfg:
        return False
    url, key = cfg
    payload = {
        "status": "CLOSED",
        "exit_premium": exit_premium,
        "pnl_per_share": pnl_per_share,
        "pnl_total": pnl_total,
        "closed_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        resp = requests.patch(
            f"{url}/rest/v1/{TABLE}", headers=_headers(key, "return=minimal"),
            params={"id": f"eq.{trade_id}"}, json=payload, timeout=10,
        )
        return resp.status_code in (200, 204)
    except Exception:
        return False


def delete_trade(trade_id: str) -> bool:
    """Remove a logged trade entirely (e.g. logged by mistake)."""
    cfg = _config()
    if not cfg:
        return False
    url, key = cfg
    try:
        resp = requests.delete(
            f"{url}/rest/v1/{TABLE}", headers=_headers(key, "return=minimal"),
            params={"id": f"eq.{trade_id}"}, timeout=10,
        )
        return resp.status_code in (200, 204)
    except Exception:
        return False
