# Preprocessing Module

`src/preprocessing` prepares raw OHLCV rows for the training and backtest
pipelines. The public repository defaults to an offline synthetic fixture so
the workflow can run without redistributing licensed market data.

## Components

| Module | Role |
|---|---|
| `pipeline.py` | Orchestrates offline fixture processing or optional download-based collection. |
| `data_collector.py` | Downloads index constituents and Yahoo Finance OHLCV rows when real-data mode is used. |
| `data_processor.py` | Adds rolling momentum, volatility, RSI, and MACD indicators. |
| `fama_french_loader.py` | Lazily downloads and merges Fama-French 5-Factor data for real-data runs. |
| `data_splitter.py` | Saves train/test CSV files after split selection. |
| `indicators.py` | Contains deterministic rolling indicator calculations. |

## Public Smoke Path

```powershell
python main.py --mode preprocess
```

With `config/sample.yaml`, this reads
`tests/fixtures/sample_prices.csv`, computes rolling indicators, and writes:

- `data/train_data.csv`
- `data/test_data.csv`

Both output files are generated artifacts and are ignored by git.

## Real-Data Path

Set `data.raw_prices_path` to a local raw OHLCV CSV, or remove it and use the
download path. Required raw CSV columns are:

```text
Date, Symbol, Open, High, Low, Close, Volume
```

Optional columns such as `Sector`, `Industry`, `Mkt_RF`, `SMB`, `HML`, `RMW`,
`CMA`, and `RF` are preserved when present.
