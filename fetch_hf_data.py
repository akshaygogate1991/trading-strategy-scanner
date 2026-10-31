"""Download 1-minute index option data (CC-BY-NC) from Hugging Face
dataset thetrademarkk/india-index-options-1m into hf_data/.
One parquet per expiry (weekly for NIFTY/SENSEX, monthly for BANKNIFTY and stocks); already-downloaded files are skipped."""
import json, os, sys, urllib.parse, urllib.request

REPO = "https://huggingface.co/datasets/thetrademarkk/india-index-options-1m"
API = "https://huggingface.co/api/datasets/thetrademarkk/india-index-options-1m/tree/main"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_data")
INDICES = {"NIFTY", "BANKNIFTY", "SENSEX"}
START, END = "2025-06-01", "2026-07-31"   # override: python fetch_hf_data.py START END [TICKER...]


def get(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)


def main():
    global START, END
    if len(sys.argv) >= 3:
        START, END = sys.argv[1], sys.argv[2]
    for sym in sys.argv[3:] or ["NIFTY", "SENSEX"]:
        if sym in INDICES:
            get(f"{REPO}/resolve/main/index/{sym}.parquet", os.path.join(OUT, "index", f"{sym}.parquet"))
            folder = f"options/{sym}"
        else:                      # stock options: monthly, no spot file in the dataset
            folder = f"stocks_options/{urllib.parse.quote(sym)}"
        os.makedirs(os.path.join(OUT, sym), exist_ok=True)
        files = json.load(urllib.request.urlopen(f"{API}/{folder}"))
        for f in files:
            name = f["path"].split("/")[-1]
            if START <= name[:10] <= END and f["size"] > 100_000:
                get(f"{REPO}/resolve/main/{urllib.parse.quote(f['path'])}", os.path.join(OUT, sym, name))
                print(sym, name, f["size"] // 1_000_000, "MB", flush=True)


if __name__ == "__main__":
    sys.exit(main())
