"""Read-only JSON feed of the paper trades, for the Vercel dashboard.

Includes the Sensex paper trades (trades.csv, daily.csv, live.json) and the
monthly momentum paper portfolio (momentum.json, momentum_trades.csv) and the gold
holding next to it (gold.json, gold_trades.csv).

Serves ONE endpoint, GET /api/dashboard, and only with the right token:
    Authorization: Bearer <token>
The token is read from the file .dashboard_token next to this script (git-ignored,
chmod 600) or from the DASH_TOKEN environment variable. Nothing here can trade or
change any file; it only reads paper_data/.

It listens on 127.0.0.1 only. Caddy (deploy/Caddyfile) puts HTTPS in front of it,
and the Vercel function is the only caller.

  python dashboard_api.py                    # 127.0.0.1:8787, data in paper_data/
  python dashboard_api.py --data paper_data_replay --port 8787
"""
import argparse
import csv
import hmac
import json
import os
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))


def load_token():
    tok = os.environ.get("DASH_TOKEN", "").strip()
    path = os.path.join(HERE, ".dashboard_token")
    if not tok and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            tok = fh.read().strip()
    if len(tok) < 24:
        raise SystemExit("Set a token of 24+ characters in .dashboard_token (see deploy/DASHBOARD_SETUP.md)")
    return tok


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def num(x):
    try:
        return round(float(x), 2)
    except (TypeError, ValueError):
        return None


def build(data_dir):
    trades = [dict(day=r["day"], strategy=r["strategy"], kind=r["kind"],
                   side="BUY" if num(r["side"]) and num(r["side"]) > 0 else "SELL",
                   strike=num(r["strike"]), qty=int(float(r["qty"])),
                   entry_time=r["entry_time"], entry_px=num(r["entry_px"]),
                   exit_time=r["exit_time"], exit_px=num(r["exit_px"]), reason=r["reason"],
                   premium=num(float(r["entry_px"]) * float(r["qty"])),
                   gross=num(r["gross"]), charges=num(float(r["charges"]) + float(r["slippage"])),
                   net=num(r["net"]))
              for r in read_csv(os.path.join(data_dir, "trades.csv"))]
    daily = [dict(day=r["day"], strategy=r["strategy"], legs=int(r["legs"]), net=num(r["net"]))
             for r in read_csv(os.path.join(data_dir, "daily.csv"))]
    months = defaultdict(lambda: defaultdict(float))
    for r in daily:
        months[r["day"][:7]][r["strategy"]] += r["net"] or 0.0
    monthly = [dict(month=m, by_strategy={k: round(v, 2) for k, v in s.items()},
                    total=round(sum(s.values()), 2), days=len({r["day"] for r in daily if r["day"][:7] == m}))
               for m, s in sorted(months.items())]
    momentum = read_json(os.path.join(data_dir, "momentum.json"))
    if momentum is not None:
        momentum["trades"] = read_csv(os.path.join(data_dir, "momentum_trades.csv"))[-200:]
    gold = read_json(os.path.join(data_dir, "gold.json"))
    if gold is not None:
        gold["trades"] = read_csv(os.path.join(data_dir, "gold_trades.csv"))[-50:]
    intraday = read_json(os.path.join(data_dir, "kronos_intraday.json"))
    return dict(live=read_json(os.path.join(data_dir, "live.json")), trades=trades[-500:], daily=daily,
                monthly=monthly, total=round(sum(r["net"] or 0.0 for r in daily), 2), momentum=momentum, gold=gold,
                kronos=kronos(data_dir, trades), kronos_intraday=intraday)


def kronos(data_dir, trades):
    """Kronos paper futures trades, its signals, and Sensex Directional with Kronos picking the side.

    Side rule for Directional (it sells a call and a put): Kronos BUY -> keep only the
    put legs (a rising market hurts the sold call); SELL -> keep only the call legs;
    no signal -> keep both. Approximate: the overall stop of the real run is not re-simulated.
    """
    snap = read_json(os.path.join(data_dir, "kronos.json"))
    if snap is None:
        return None
    signals = read_csv(os.path.join(data_dir, "kronos_signals.csv"))
    snap["signals"] = signals[-300:]
    snap["trades"] = [dict(r, **{k: num(r[k]) for k in ("entry", "exit", "points", "gross", "charges", "net", "p_up")},
                           qty=int(float(r["qty"])))
                      for r in read_csv(os.path.join(data_dir, "kronos_trades.csv"))][-500:]
    side_by_day = {r["settled"]: r for r in signals if r["sym"] == "SENSEX" and r["settled"]}
    by_day = defaultdict(list)
    for t in trades:
        if t["strategy"] == "Sensex Directional":
            by_day[t["day"]].append(t)
    rows = []
    first = min(side_by_day) if side_by_day else None
    for day in sorted(by_day):
        if first is None or day < first:          # before Kronos started there is nothing to compare
            continue
        sig = side_by_day.get(day)
        side = sig["side"] if sig else "NONE"
        keep = {"BUY": {"PE"}, "SELL": {"CE"}}.get(side, {"CE", "PE"})
        legs = by_day[day]
        rows.append(dict(day=day, side=side if sig else "no signal", p_up=num(sig["p_up"]) if sig else None,
                         original=num(sum(t["net"] or 0 for t in legs)),
                         kronos=num(sum(t["net"] or 0 for t in legs if t["kind"] in keep))))
    snap["directional"] = rows
    return snap


def read_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def make_handler(token, data_dir):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body):
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path.split("?")[0] != "/api/dashboard":
                return self._send(404, {"error": "not found"})
            given = self.headers.get("Authorization", "").removeprefix("Bearer ").strip()
            if not hmac.compare_digest(given.encode(), token.encode()):
                return self._send(401, {"error": "unauthorized"})
            self._send(200, build(data_dir))

        def log_message(self, fmt, *args):      # no request logs (they would include IPs only, but keep quiet)
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "paper_data"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    a = ap.parse_args()
    data_dir = a.data if os.path.isabs(a.data) else os.path.join(HERE, a.data)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(load_token(), data_dir))
    print(f"dashboard feed on http://{a.host}:{a.port}/api/dashboard (data: {data_dir})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
