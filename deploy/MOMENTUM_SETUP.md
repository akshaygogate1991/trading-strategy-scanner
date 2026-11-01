# Momentum paper portfolio on the VPS

`momentum_paper.py` keeps a paper portfolio of ₹5 lakh in 20 momentum stocks. It places no orders.

- **Schedule.** It runs every weekday at 17:45 IST.
- **First trading day of each month.** It picks the top 20 from the last month-end close and paper-buys or sells at that day's open.
- **Other days.** It updates prices only.

The results appear in the **Momentum swing** card on the Vercel dashboard.

## Setup (on the VPS, as root)

**1. Get the new code and install yfinance:**

```bash
su - trader -c "cd ~/trading-strategy && git pull && .venv/bin/pip install -r deploy/requirements-vps.txt"
```

**2. Check that it can download prices.** This prints today's 20 picks and saves nothing. It takes a few minutes:

```bash
su - trader -c "cd ~/trading-strategy && .venv/bin/python momentum_paper.py --dry-run"
```

**3. Turn on the evening timer:**

```bash
cp /home/trader/trading-strategy/deploy/momentum-paper.service /home/trader/trading-strategy/deploy/momentum-paper.timer /etc/systemd/system/ && systemctl daemon-reload && systemctl enable --now momentum-paper.timer && systemctl list-timers momentum-paper.timer --no-pager
```

**4. Restart the dashboard feed** so it also serves the momentum data:

```bash
systemctl restart dashboard-api
```

The first paper buys happen at the first weekday 17:45 run.

## Checking it

| What | Command |
|---|---|
| Last run | `journalctl -u momentum-paper -n 20 --no-pager` |
| Run now by hand | `systemctl start momentum-paper` |
| Holdings file | `cat /home/trader/trading-strategy/paper_data/momentum.json` |

If a log line says "no NSE bar for today", that day was a holiday, or Yahoo had not posted the day's prices yet. It simply tries again the next evening.
