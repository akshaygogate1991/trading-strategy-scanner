// Vercel serverless function: checks the dashboard password, then fetches the
// paper-trade feed from the VPS with a secret token. The token never reaches
// the browser. Environment variables (set in the Vercel project settings):
//   DASH_PASSWORD  the password you type on the dashboard
//   VPS_API_URL    https://46-225-42-103.sslip.io/api/dashboard
//   VPS_TOKEN      the same token as .dashboard_token on the VPS
const crypto = require("crypto");

const digest = (s) => crypto.createHash("sha256").update(String(s)).digest();

module.exports = async (req, res) => {
  res.setHeader("Cache-Control", "no-store");
  const { DASH_PASSWORD, VPS_API_URL, VPS_TOKEN } = process.env;
  if (!DASH_PASSWORD || !VPS_API_URL || !VPS_TOKEN) {
    return res.status(500).json({ error: "Dashboard is not configured (environment variables missing)" });
  }
  const given = req.headers["x-dash-password"] || "";
  if (!crypto.timingSafeEqual(digest(given), digest(DASH_PASSWORD))) {
    await new Promise((r) => setTimeout(r, 800)); // slows down password guessing
    return res.status(401).json({ error: "Wrong password" });
  }
  try {
    const r = await fetch(VPS_API_URL, {
      headers: { Authorization: `Bearer ${VPS_TOKEN}` },
      signal: AbortSignal.timeout(8000),
    });
    if (!r.ok) return res.status(502).json({ error: `VPS feed answered ${r.status}` });
    return res.status(200).json(await r.json());
  } catch (e) {
    return res.status(502).json({ error: "VPS feed unreachable" });
  }
};
