# Running the Kronos option test on Google Colab (free GPU)

On the laptop CPU one 5-minute forecast takes about 9 minutes (about 3 days for the full
test). A free Colab T4 GPU is about 20x faster. No passwords or keys are needed: Kronos
downloads from Hugging Face without a login.

1. Zip these from `D:\ML project` (keep the folder names):
   - `Kronos\` (the whole folder)
   - `trading strategy\kronos_options_test.py`
   - `trading strategy\hf_data\index\SENSEX.parquet`
   - `trading strategy\logs\batch\chotasx_expiry.csv`
   - `trading strategy\export_sensexbasket_trades.csv`
   - `trading strategy\logs\oos_export_sensexbasket_trades.csv`
2. Open https://colab.research.google.com, New notebook, Runtime > Change runtime type > T4 GPU.
3. Upload the zip (folder icon on the left), then run in a cell:

```
!unzip -q kronos_test.zip
!pip -q install einops huggingface_hub safetensors
%cd "trading strategy"
!python kronos_options_test.py forecast 5m chota
```

   It prints `device: cuda` at the start. If it says `cpu`, the GPU runtime is not on.
4. When it finishes (or if Colab disconnects, re-run step 3; it continues where it stopped),
   download `logs/kronos_options_forecasts.csv` and save it on the laptop as
   `D:\ML project\trading strategy\logs\kronos_options_forecasts_colab.csv`
   (a new name, so the laptop's own 15-minute file is kept; the report reads both).
   Colab deletes its files when the session ends, so download before closing it.
5. On the laptop: `python kronos_options_test.py report`
