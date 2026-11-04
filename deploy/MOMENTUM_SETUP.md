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

## Every 3 months: refresh the stock list (on the laptop)

Yahoo blocks much of a 2,400-stock download from a server. So the VPS downloads only the 700 most traded stocks, listed in `momentum_candidates.csv`.

- **Why that is enough:** in the backtest, a list up to 3 months old gave exactly the same picks as using all stocks.
- **When to refresh:** every 3 months. The dashboard shows a yellow warning once the list is over 100 days old.

On the laptop, run this in PowerShell (not in the VPS window):

```bash
cd "D:\ML project\trading strategy"; python momentum_paper.py --refresh-candidates; git commit -am "Refresh momentum stock list"; git push
```

Then, on the VPS:

```bash
su - trader -c "cd ~/trading-strategy && git pull"
```

If more than 3% of the 700 stocks fail to download on a rebalance day, the VPS does not trade that evening. It tries again at the next evening run.

## Checking it

| What | Command |
|---|---|
| Last run | `journalctl -u momentum-paper -n 20 --no-pager` |
| Run now by hand | `systemctl start momentum-paper` |
| Holdings file | `cat /home/trader/trading-strategy/paper_data/momentum.json` |

If a log line says "no NSE bar for today", that day was a holiday, or Yahoo had not posted the day's prices yet. It simply tries again the next evening.

## Live prices on the dashboard (every 15 minutes in market hours)

`momentum-live.timer` runs `momentum_paper.py --live` at 09:15, 09:30 ... 15:45 IST on weekdays. It only re-prices the 20 holdings at Yahoo's latest 5-minute price (which can lag a few minutes) and rewrites `momentum.json`. It never buys or sells, and the evening run stays the record. The evening run also retries at 19:00 and 21:00 IST in case Yahoo is late at 17:45.

Setup (on the VPS, as root):

```bash
su - trader -c "cd ~/trading-strategy && git pull" && cp /home/trader/trading-strategy/deploy/momentum-live.service /home/trader/trading-strategy/deploy/momentum-live.timer /home/trader/trading-strategy/deploy/momentum-paper.timer /etc/systemd/system/ && systemctl daemon-reload && systemctl enable --now momentum-live.timer && systemctl restart momentum-paper.timer && systemctl list-timers 'momentum*' --no-pager
```

Check it: `journalctl -u momentum-live -n 10 --no-pager`. Run it now by hand: `systemctl start momentum-live`.

## Gold (GOLDBEES) tab

`gold_paper.py` keeps a separate paper GOLDBEES holding: bought once for Rs 1,27,142 on
4 Nov 2026 and kept, with no selling, no top-ups and no timing (gold_test.py: every timing rule
did worse than holding). It has nothing to do with the momentum portfolio; it only runs in the
same two services (evening record and the 15-minute live re-pricing).
Files: `paper_data/gold_state.json`, `gold_trades.csv`, `gold.json`.

Update it on the VPS (as root, after pushing from the laptop):

```
su - trader -c "cd ~/trading-strategy && git pull" && cp /home/trader/trading-strategy/deploy/momentum-paper.service /home/trader/trading-strategy/deploy/momentum-live.service /etc/systemd/system/ && systemctl daemon-reload && systemctl restart dashboard-api && systemctl start momentum-paper && journalctl -u momentum-paper -n 3 --no-pager
```

The last line shows `GOLDBEES ..., value Rs ...`. Vercel redeploys the tab by itself after the push.
