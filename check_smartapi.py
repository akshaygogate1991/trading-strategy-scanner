"""Quick one-time check that SmartAPI login + data fetch works.

Run from the project folder:   python check_smartapi.py

It logs in with your .streamlit/secrets.toml credentials (DATA ONLY),
pulls a few recent daily candles for RELIANCE and the Nifty 50 index,
and prints them. If you see rows, the connection works.
"""
import smartapi_data as sd


def main():
    print("Connecting to Angel One SmartAPI (data only)...")
    for ticker in ("RELIANCE.NS", "^NSEI"):
        try:
            df = sd.get_history(ticker, "1d", 30)
        except Exception as exc:
            print(f"  {ticker}: ERROR -> {exc}")
            continue
        if df is None or df.empty:
            print(f"  {ticker}: no data returned")
        else:
            last = df.tail(3)
            print(f"  {ticker}: {len(df)} candles, last close = {df['Close'].iloc[-1]:.2f}")
            print(last.to_string())
    print("\nDone. If you saw candles above, SmartAPI is working.")


if __name__ == "__main__":
    main()
