# Paper-trade dashboard on Vercel

```
 VPS                                                    Vercel
 paper_trader.py ──writes──> paper_data/live.json       dashboard/index.html  (the page)
                             trades.csv, daily.csv            │ asks /api/data with your password
 dashboard_api.py (127.0.0.1:8787, needs a token)   <──  dashboard/api/data.js (checks the password,
 Caddy: https://46-225-42-103.sslip.io (HTTPS)               adds the secret token, fetches the VPS)
```

- **Read-only.** The dashboard only reads. Nothing on it can place an order or change a file.
- **Two locks.** You type a password on the page. Vercel then calls the VPS with a separate secret token, which never reaches your browser.
- **Update speed.** The page refreshes every 15 seconds during market hours, and every 2 minutes otherwise.

---

## Part A: on the VPS (as root)

**1. Get the new code**

```bash
su - trader -c "cd ~/trading-strategy && git pull"
```

**2. Make the secret token.** Only the trader user can read it:

```bash
su - trader -c "cd ~/trading-strategy && umask 077 && openssl rand -hex 32 > .dashboard_token && cat .dashboard_token"
```

Copy the printed token: you'll paste it into Vercel in part B. Don't share it anywhere else.

**3. Start the feed, and restart it if the server reboots**

```bash
cp /home/trader/trading-strategy/deploy/dashboard-api.service /etc/systemd/system/ && systemctl daemon-reload && systemctl enable --now dashboard-api && systemctl status dashboard-api --no-pager
```

**4. Set up HTTPS with Caddy.** First check that nothing else already uses ports 80 and 443. Docker or PM2 may be running something:

```bash
ss -tlnp | grep -E ':(80|443) '
```

- **Nothing printed:** install Caddy:

  ```bash
  apt install -y caddy && cp /home/trader/trading-strategy/deploy/Caddyfile /etc/caddy/Caddyfile && systemctl restart caddy
  ```

- **Something printed** (for example nginx or a Docker container): stop here and send me that line. The feed will then go through the existing web server instead.

**5. Open the ports.** In the Hetzner Cloud console, under **Firewalls**, allow TCP 80 and 443 inbound if you use a Hetzner firewall. If `ufw status` says active, also run:

```bash
ufw allow 80,443/tcp
```

**6. Test the feed.** It should say `unauthorized`, which means the feed is up and locked:

```bash
curl -s https://46-225-42-103.sslip.io/api/dashboard
```

## Part B: on Vercel (in the browser)

1. Go to vercel.com and sign in with GitHub.
2. Click **Add New → Project** and import **trading-strategy-scanner**.
3. Set **Root Directory** to `dashboard`. Set **Framework Preset** to **Other**, and leave the build settings empty.
4. Under **Environment Variables**, add these three:

   | Name | Value |
   |---|---|
   | `DASH_PASSWORD` | a strong password you choose, for logging in to the page |
   | `VPS_API_URL` | `https://46-225-42-103.sslip.io/api/dashboard` |
   | `VPS_TOKEN` | the token from part A, step 2 |

5. Click **Deploy**.
6. Open the `…vercel.app` link and enter your password.

Every later `git push` redeploys the page automatically.

## What the page shows

- **Tiles:**
  - today's live P&L (booked trades plus open positions at the last price);
  - this month;
  - total since start;
  - the Sensex level.
- **Today's positions:** shown for each strategy. Each leg shows:
  - the leg (e.g. SELL CE 77300) and its quantity;
  - entry time and entry price;
  - **premium** (entry price × quantity);
  - the last price, or the exit time and price;
  - the **stop** (current cut-off price, including the trail);
  - status and net P&L.

  Legs still waiting to enter show their trigger price.
- **Month-wise profit/loss:** a bar chart plus a table, split by strategy.
- **Cumulative net P&L:** a line chart.
- **Trade history:** every leg trade, with charges and net. You can filter by strategy and by month.
- **Warning:** a yellow banner appears if the VPS hasn't updated for more than 4 minutes during market hours.

## If something is wrong

| What the page shows | Check on the VPS |
|---|---|
| "Wrong password" | `DASH_PASSWORD` in Vercel (redeploy after changing it) |
| "VPS feed answered 401" | `VPS_TOKEN` in Vercel must equal `.dashboard_token` |
| "VPS feed unreachable" | `systemctl status caddy dashboard-api`, firewall ports 80/443 |
| No live positions | `journalctl -u paper-trader -n 50` |

## Checking it locally (optional)

```bash
python dashboard_api.py --data paper_data_replay      # needs DASH_TOKEN or .dashboard_token
```
