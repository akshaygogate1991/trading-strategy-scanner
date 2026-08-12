"""Correct a logged trade from the command line.

Written because a wrong exit premium can dominate the whole P&L: an exit of 84
entered instead of 31 on one AXISBANK put accounted for ~92% of a reported
+Rs.41,702, which would make the strategy look far better than it is.

This reads your GitHub token from .streamlit/secrets.toml on THIS machine.
Nothing is sent anywhere except your own private trade-log repo.

    python fix_trade.py --list
    python fix_trade.py --name AXISBANK --exit 31
    python fix_trade.py --name AXISBANK --exit 31 --entry 22.4
    python fix_trade.py --id 3f2a... --exit 31 --yes      (skip confirmation)
"""
from __future__ import annotations

import argparse

import trade_log as tl


def describe(t: dict) -> str:
    tag = "CE" if t.get("direction") == "CALL" else "PE"
    return (f"{t.get('name'):12s} {t.get('strike'):>8} {tag}  "
            f"status={t.get('status'):6s}  "
            f"entry={t.get('entry_premium')}  exit={t.get('exit_premium')}  "
            f"P&L={t.get('pnl_total')}  id={t.get('id', '')[:8]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="show every logged trade")
    ap.add_argument("--name", help="match by instrument name, e.g. AXISBANK")
    ap.add_argument("--id", help="match by trade id (or its first 8 characters)")
    ap.add_argument("--exit", type=float, help="corrected exit premium")
    ap.add_argument("--entry", type=float, help="corrected entry premium")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = ap.parse_args()

    if not tl.is_configured():
        print("Trade log is not configured on this machine.\n"
              "Expected a [github_log] section in .streamlit/secrets.toml.")
        return

    trades = tl.fetch_trades()
    if not trades:
        print("No trades found (or the repo could not be read).")
        return

    if args.list or not (args.name or args.id):
        print(f"{len(trades)} logged trades:\n")
        for t in trades:
            print("  " + describe(t))
        if not (args.name or args.id):
            print("\nTo correct one:  python fix_trade.py --name AXISBANK --exit 31")
        return

    matches = [
        t for t in trades
        if (args.name and str(t.get("name", "")).upper() == args.name.upper())
        or (args.id and str(t.get("id", "")).startswith(args.id))
    ]
    if not matches:
        print(f"No trade matched. Run --list to see what is there.")
        return
    if len(matches) > 1:
        print(f"{len(matches)} trades matched — narrow it with --id:\n")
        for t in matches:
            print("  " + describe(t))
        return

    t = matches[0]
    new_exit = args.exit if args.exit is not None else t.get("exit_premium")
    new_entry = args.entry if args.entry is not None else t.get("entry_premium")
    if new_exit is None or new_entry is None:
        print("Need both an entry and an exit premium to compute P&L.")
        return

    ps = round(float(new_exit) - float(new_entry), 2)
    tot = round(ps * t["lot"], 2) if t.get("lot") else None

    print("\nBEFORE: " + describe(t))
    after = f"AFTER : entry={new_entry}  exit={new_exit}  P&L/share={ps:+.2f}"
    if tot is not None:
        after += f"  total={tot:+,.2f}"
    print(after)

    old_tot = float(t.get("pnl_total") or 0)
    if tot is not None:
        print(f"\nChange in recorded P&L: {old_tot:+,.0f} -> {tot:+,.0f} "
              f"({tot - old_tot:+,.0f})")

    if not args.yes:
        if input("\nApply this correction? [y/N] ").strip().lower() != "y":
            print("Cancelled — nothing changed.")
            return

    ok = tl.update_trade(t["id"], {
        "entry_premium": float(new_entry),
        "exit_premium": float(new_exit),
        "pnl_per_share": ps,
        "pnl_total": tot,
    })
    print("Saved." if ok else "Could not save — check your token and try again.")


if __name__ == "__main__":
    main()
