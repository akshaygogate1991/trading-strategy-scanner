"""Paper-trading log, persisted as a JSON file in a private GitHub repo.

Why GitHub instead of a database: it's free forever, needs no new signup
(you already have a GitHub account), and a private repo keeps your P&L
history out of public view - unlike committing it to the public scanner repo.

Setup (one-time, ~5 minutes):
  1. On github.com, create a NEW repo, e.g. "trading-journal-data".
     Set its visibility to PRIVATE. Do not put this file in your public
     trading-strategy-scanner repo - that one is public.
  2. Create a GitHub Personal Access Token:
     github.com -> Settings -> Developer settings -> Personal access tokens
     -> Fine-grained tokens -> Generate new token.
     Repository access: "Only select repositories" -> pick the new private repo.
     Permissions: Repository -> Contents -> Read and write.
  3. Add to .streamlit/secrets.toml (and to Streamlit Cloud's Secrets manager,
     since that's what your live app actually reads):

        [github_log]
        token = "github_pat_xxx..."
        repo  = "yourusername/trading-journal-data"

Credentials read from st.secrets["github_log"]: token, repo, and optionally
path (default "trade_log.json") and branch (default "main").

Every log/close/delete action reads the current file, edits it, and writes
it back as one commit - simple and safe for a single user.
"""
from __future__ import annotations

import base64
import json
import uuid
from datetime import datetime, timezone

import requests
import streamlit as st

API = "https://api.github.com"


def _config() -> dict | None:
    try:
        s = st.secrets["github_log"]
        token, repo = s.get("token"), s.get("repo")
        if token and repo:
            return {
                "token": token,
                "repo": repo,
                "path": s.get("path", "trade_log.json"),
                "branch": s.get("branch", "main"),
            }
    except Exception:
        pass
    return None


def is_configured() -> bool:
    return _config() is not None


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _get_file(cfg: dict) -> tuple[list, str | None]:
    """Return (trades, sha). sha is None if the file doesn't exist yet."""
    url = f"{API}/repos/{cfg['repo']}/contents/{cfg['path']}"
    resp = requests.get(url, headers=_headers(cfg["token"]),
                        params={"ref": cfg["branch"]}, timeout=10)
    if resp.status_code == 404:
        return [], None
    resp.raise_for_status()
    data = resp.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    trades = json.loads(content).get("trades", []) if content.strip() else []
    return trades, data["sha"]


def _put_file(cfg: dict, trades: list, sha: str | None, message: str) -> bool:
    url = f"{API}/repos/{cfg['repo']}/contents/{cfg['path']}"
    body = base64.b64encode(
        json.dumps({"trades": trades}, indent=2).encode("utf-8")
    ).decode("ascii")
    payload = {"message": message, "content": body, "branch": cfg["branch"]}
    if sha:
        payload["sha"] = sha
    resp = requests.put(url, headers=_headers(cfg["token"]), json=payload, timeout=10)
    return resp.status_code in (200, 201)


def log_trade(row: dict) -> bool:
    """Insert a new OPEN trade snapshot. Returns True on success."""
    cfg = _config()
    if not cfg:
        return False
    try:
        trades, sha = _get_file(cfg)
        row = dict(row)
        row["id"] = str(uuid.uuid4())
        row["created_at"] = datetime.now(timezone.utc).isoformat()
        trades.append(row)
        return _put_file(cfg, trades, sha, f"Log trade: {row.get('name')} {row.get('direction')}")
    except Exception:
        return False


def fetch_trades(status: str | None = None) -> list[dict]:
    """Fetch trades, optionally filtered by status ('OPEN' or 'CLOSED')."""
    cfg = _config()
    if not cfg:
        return []
    try:
        trades, _ = _get_file(cfg)
    except Exception:
        return []
    if status:
        trades = [t for t in trades if t.get("status") == status]
    return sorted(trades, key=lambda t: t.get("created_at", ""), reverse=True)


def close_trade(trade_id: str, exit_premium: float, pnl_per_share: float,
                pnl_total: float | None, extra: dict | None = None) -> bool:
    """Mark a trade CLOSED with the exit premium and computed P&L.

    `extra` carries the per-leg detail for a hedged spread (exit price of the leg
    bought and the leg sold), so the closed record shows how the net was reached
    rather than a single unexplained number.
    """
    cfg = _config()
    if not cfg:
        return False
    try:
        trades, sha = _get_file(cfg)
        found = False
        for t in trades:
            if t.get("id") == trade_id:
                t["status"] = "CLOSED"
                t["exit_premium"] = exit_premium
                t["pnl_per_share"] = pnl_per_share
                t["pnl_total"] = pnl_total
                t["closed_at"] = datetime.now(timezone.utc).isoformat()
                if extra:
                    t.update(extra)
                found = True
                break
        if not found:
            return False
        return _put_file(cfg, trades, sha, f"Close trade {trade_id[:8]}")
    except Exception:
        return False


def clear_all() -> bool:
    """Wipe every logged trade. Irreversible - the UI must confirm first."""
    cfg = _config()
    if not cfg:
        return False
    try:
        _, sha = _get_file(cfg)
        return _put_file(cfg, [], sha, "Clear trade log")
    except Exception:
        return False


def delete_trade(trade_id: str) -> bool:
    """Remove a logged trade entirely (e.g. logged by mistake)."""
    cfg = _config()
    if not cfg:
        return False
    try:
        trades, sha = _get_file(cfg)
        new_trades = [t for t in trades if t.get("id") != trade_id]
        if len(new_trades) == len(trades):
            return False
        return _put_file(cfg, new_trades, sha, f"Delete trade {trade_id[:8]}")
    except Exception:
        return False
