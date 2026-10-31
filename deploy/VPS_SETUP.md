# Running the paper trader on the Hetzner VPS

`paper_trader.py` **never places orders**. It reads live Sensex option prices from Angel One, runs:
- **Sensex Directional** every trading day;
- **Chota Packet Sensex** on expiry days only;

and writes paper trades to `paper_data/`. It starts by itself every weekday at 09:05 IST, so your laptop can stay off.

Commands below assume Ubuntu on the VPS. Replace `VPS_IP` with your server's IP address.

---

## 1. One-time server setup (on the VPS)

```bash
ssh root@VPS_IP
apt update && apt install -y python3-venv python3-pip git
timedatectl                        # "System clock synchronized: yes" must show
adduser --disabled-password --gecos "" trader
```

## 2. Get the code (on the VPS, as the trader user)

First push the latest code from your laptop with `git push`. Then on the VPS:

```bash
su - trader
git clone https://github.com/akshaygogate1991/trading-strategy-scanner.git ~/trading-strategy
cd ~/trading-strategy
python3 -m venv .venv
.venv/bin/pip install -r deploy/requirements-vps.txt
mkdir -p .streamlit strategies
exit
```

## 3. Copy the two private things (from your laptop, in PowerShell)

These are never in git: your Angel One login, and the seminar strategy files.

```bash
scp "D:\ML project\trading strategy\.streamlit\secrets.toml" root@VPS_IP:/home/trader/trading-strategy/.streamlit/
scp -r "D:\ML project\trading strategy\strategies" root@VPS_IP:/home/trader/trading-strategy/
```

Then lock them down on the VPS:

```bash
ssh root@VPS_IP
chown -R trader:trader /home/trader/trading-strategy
chmod 600 /home/trader/trading-strategy/.streamlit/secrets.toml
```

## 4. Test the connection (works on any day, places nothing)

```bash
su - trader -c "cd ~/trading-strategy && .venv/bin/python paper_trader.py --check"
```

You should see `Angel One login: OK`, the Sensex price, and a few option prices.

**If the login fails:** Angel One SmartAPI may ask you to register the server's IP address in your SmartAPI app settings. Brokers now require a fixed IP for API trading, and a VPS has one. The same applies to Flattrade later.

**If your Streamlit dashboard uses the same API key:** a login from the VPS might log the dashboard out, or the reverse. If that happens, create a second API key in SmartAPI just for the VPS.

## 5. Start it automatically every weekday

```bash
cp /home/trader/trading-strategy/deploy/paper-trader.service /etc/systemd/system/
cp /home/trader/trading-strategy/deploy/paper-trader.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now paper-trader.timer
systemctl list-timers paper-trader.timer      # shows the next 09:05 IST start
```

**Automatic handling:**
- **Weekends and holidays:** it exits without trading. On a holiday it reads the exchange's own timestamp to tell.
- **Crash mid-day:** it restarts in 30 seconds and restores its open positions from the minute bars it saves every minute.
- **Late start** (e.g. the VPS rebooted at 11:00): a strategy whose entry time has already passed skips that day instead of entering late.
- **Morning, if it ran yesterday:** it won't trade the same day twice.

## 6. Watching it

```bash
journalctl -u paper-trader -f                                  # live output
tail -f /home/trader/trading-strategy/paper_data/events.log    # every entry / exit
cat /home/trader/trading-strategy/paper_data/daily.csv         # one line per strategy per day
cd /home/trader/trading-strategy && .venv/bin/python summarize_trades.py paper_data/trades.csv
```

`paper_data/bars/` keeps every day's real minute bars. That's our own data, collected from now on, and any day can be replayed through the backtester:

```bash
.venv/bin/python paper_trader.py --replay-saved paper_data/bars/2026-11-03.parquet
```

## 7. Updating later

```bash
su - trader -c "cd ~/trading-strategy && git pull"
systemctl restart paper-trader.timer
```

## Security checklist

- Log in with SSH keys and turn off password login.
- Run `ufw allow OpenSSH && ufw enable` to allow SSH only.
- `secrets.toml` must be readable only by `trader` (`chmod 600`, step 3).
- Nothing in this setup can place an order. Order placement through Flattrade will be a separate, explicit step later.
