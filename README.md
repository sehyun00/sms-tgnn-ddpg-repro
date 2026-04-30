# SMS TGNN-DDPG Reproducibility Package

This repository is the public, clean-room reproduction package for the
TGNN, DDPG, and Hybrid TGNN-DDPG portfolio backtesting experiments.

It contains code and a deterministic toy fixture only. Full market data,
trained checkpoints, logs, plots, and manuscript files are intentionally not
tracked.

## What Is Included

- `src/models`: TGNN, DDPG, and Hybrid TGNN-DDPG model implementations.
- `src/preprocessing`: offline fixture preprocessing and optional market-data collection.
- `src/training`: dataset, replay buffer, and training loops.
- `src/backtest`: strategy execution, metrics, and visualization helpers.
- `config/sample.yaml`: CPU smoke-test configuration.
- `tests/fixtures/sample_prices.csv`: deterministic toy OHLCV data for tests.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Reproduction Workflow

The default config uses the bundled sample fixture and writes generated files
to ignored `data/` and `results/` directories.

```powershell
python main.py --mode preprocess
python main.py --mode train --config config/sample.yaml
python main.py --mode backtest --config config/sample.yaml
python run_multiseed.py --dry-run --config config/sample.yaml
pytest tests
```

For real experiments, replace `data.raw_prices_path` with a local raw price CSV
or remove it and use the downloader path. Do not commit generated
`train_data.csv`, `test_data.csv`, `results/`, or checkpoint files.

## Data Policy

The public repository does not redistribute Yahoo Finance, Fama-French, or
other derived research datasets. The included fixture is synthetic and exists
only to verify code paths. Any reported paper result must be regenerated from
the user's own licensed/downloaded data and supported by the resulting CSV
artifacts.

## Research Integrity

- Seeds are fixed for Python, NumPy, and PyTorch in `main.py` and
  `run_multiseed.py`.
- PyTorch CUDNN deterministic mode is enabled by default.
- The preprocessing command writes train/test splits by date and does not fit
  scalers or selectors on test data.
- Backtesting transfer-learning code must not fine-tune on test data.
- README performance numbers are intentionally omitted until generated result
  CSVs are available in the local ignored `results/` directory.
