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
su - trader -c "cd ~/trading-strategy && git pull && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu && .venv/bin/pip install einops safetensors tqdm && git clone --depth 1 https://github.com/shiyu-coder/Kronos.git ~/Kronos"
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
