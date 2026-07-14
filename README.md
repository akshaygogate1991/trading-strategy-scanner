# Nifty Strategy Scanner

Streamlit scanner for a conservative Indian market swing strategy:

- 18 EMA pullback continuation
- 20-day breakout with volume confirmation
- Objective Elliott Wave swing filter
- Nifty trend filter
- ATR/swing-low stop loss
- 1.5R, 2.5R, and 3R targets
- Position sizing from capital and risk percentage

## Run

```powershell
pip install -r requirements.txt
streamlit run app.py
```

Open the Streamlit URL, then click **Scan Market**.

## Expected Scan Time

Nifty 100 plus Nifty and Bank Nifty usually takes about 1-3 minutes on public Yahoo Finance data. Streamlit caches data for 20 minutes, so repeated checks are faster.

## Important

This is a decision-support scanner, not automatic financial advice. It is designed to show only rule-matching setups and to say "No trade available" when the market does not match the strategy.
