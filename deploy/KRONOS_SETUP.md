# Kronos paper trades on the VPS

`kronos_paper.py` runs every weekday at 17:55 IST. It places no orders.

- **Settles yesterday's signals.** It paper-trades 1 lot of Nifty, Bank Nifty and Sensex futures at today's open and closes each trade at today's close.
- **Makes tomorrow's signals.** Kronos looks at the last 400 daily candles and imagines 20 possible next candles. It signals BUY if at least 60% of them close higher, SELL if at most 40% do, and otherwise no trade.

The results appear on the dashboard's **Paper Trades with Kronos** tab. That tab also shows Sensex Directional with Kronos picking which side to sell.

Kronos is the open-source model from github.com/shiyu-coder/Kronos (MIT licence). It needs PyTorch, which runs on the CPU and takes about 200 MB of disk space and about 1 GB of RAM while it runs.

## Setup (on the VPS, as root)

**1. Check that there is room.** You need at least 1.5 GB "available" memory and 3 GB free disk:

```bash
free -h && df -h /home
```

**2. Get the new code, PyTorch (CPU) and the Kronos code:**

```bash
su - trader -c "cd ~/trading-strategy && git pull && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu && .venv/bin/pip install einops safetensors tqdm huggingface_hub && git clone --depth 1 https://github.com/shiyu-coder/Kronos.git ~/Kronos"
```

**3. Test run.** This prints the signals for the next session and saves nothing. The first time, it downloads the model (about 100 MB), so allow a few minutes:

```bash
su - trader -c "cd ~/trading-strategy && KRONOS_DIR=/home/trader/Kronos .venv/bin/python kronos_paper.py --dry-run"
```

**4. Turn on the evening timer and restart the dashboard feed:**

```bash
cp /home/trader/trading-strategy/deploy/kronos-paper.service /home/trader/trading-strategy/deploy/kronos-paper.timer /etc/systemd/system/ && systemctl daemon-reload && systemctl enable --now kronos-paper.timer && systemctl restart dashboard-api && systemctl list-timers kronos-paper.timer --no-pager
```

The first signals are made at the next weekday 17:55 run. The first trades settle the evening after that.

## Checking it

| What | Command |
|---|---|
| Last run | `journalctl -u kronos-paper -n 20 --no-pager` |
| Run now by hand | `systemctl start kronos-paper` |
| Signals | `cat /home/trader/trading-strategy/paper_data/kronos_signals.csv` |
| Trades | `cat /home/trader/trading-strategy/paper_data/kronos_trades.csv` |

## Intraday Kronos paper trades (forecast in the live market, trade at once)

`kronos_intraday_paper.py` runs at 09:46, 10:11, 11:01, 12:31, 13:31, 14:31 IST (forecasts for 09:45,
10:10, 11:00, 12:30, 13:30, 14:30) and 15:46 (settles). For Nifty, Bank Nifty and Sensex it asks
Kronos for the next 4 fifteen-minute candles (10 paths, 160 candles of history) and takes a paper
trade of 1 lot at once: BUY if >= 60 % of paths end up, SELL if <= 40 %, exit after 30 and after
60 minutes (two separate paper trades). Prices from Yahoo 5-minute candles; no login, no orders.
It runs at low CPU priority so the Sensex paper trader keeps its CPU.
The results are in the Kronos tab, "Intraday Kronos", as a time x index table.
The same rule is backtested on Jul 2024 - Jul 2026 by `kronos_intraday_test.py` (laptop).

Turn it on (as root on the VPS, after pushing from the laptop):

```
su - trader -c "cd ~/trading-strategy && git pull" && cp /home/trader/trading-strategy/deploy/kronos-intraday.service /home/trader/trading-strategy/deploy/kronos-intraday.timer /etc/systemd/system/ && systemctl daemon-reload && systemctl enable --now kronos-intraday.timer && systemctl restart dashboard-api && systemctl list-timers kronos-intraday.timer --no-pager
```

Check a run: `journalctl -u kronos-intraday -n 20 --no-pager`. Try it by hand outside market hours
(saves nothing): `su - trader -c "cd ~/trading-strategy && KRONOS_DIR=/home/trader/Kronos .venv/bin/python kronos_intraday_paper.py --dry-run"`
