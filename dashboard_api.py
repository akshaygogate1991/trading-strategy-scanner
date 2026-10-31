"""Read-only JSON feed of the paper trades, for the Vercel dashboard.

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
    live = None
    p = os.path.join(data_dir, "live.json")
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as fh:
                live = json.load(fh)
        except (OSError, ValueError):
            live = None
    return dict(live=live, trades=trades[-500:], daily=daily, monthly=monthly,
                total=round(sum(r["net"] or 0.0 for r in daily), 2))


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
