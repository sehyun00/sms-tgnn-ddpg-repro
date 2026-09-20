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

## Major-Revision Workflow

The corrected paper-scale workflow is separate from the legacy smoke path. It
uses future 1/5/21/63-day returns, history-only graphs, sequential net-return RL
rewards, frozen TGNN/DDPG branches, a validation-only state gate, TD3 and
classical baselines, post-hoc cost sweeps, and run-level provenance.

Run the complete CPU smoke audit before starting a paper-scale job:

```powershell
python scripts/run_revision_experiments.py --config config/revision_smoke.yaml --stage prepare
python scripts/run_revision_experiments.py --config config/revision_smoke.yaml --stage train --model all --seed 42 --resume
python scripts/run_revision_experiments.py --config config/revision_smoke.yaml --stage backtest --model all --seed 42 --resume
python scripts/run_revision_experiments.py --config config/revision_smoke.yaml --stage aggregate
```

The full configuration is `config/paper_revision.yaml` and requires a CUDA
runtime. It can be run on a Windows NVIDIA workstation with the local wrapper
below or through `notebooks/colab_revision.ipynb`. `SMS_PREPARED_DIR` and
`SMS_RESULTS_DIR` may point prepared data and resumable run artifacts to another
local disk or mounted Google Drive. See `docs/revision-execution.md` for the
experiment matrix and evidence contract.

For a coauthor's Windows NVIDIA workstation, run the complete experiment with
one restart-safe command:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_professor_windows.ps1
```

The wrapper creates a virtual environment, installs CUDA-enabled PyTorch,
checks the GPU and test suite, runs the complete revision matrix with resume,
and creates a handoff ZIP without raw market data. See
`docs/professor-local-run.md` for setup, output paths, and the primary-only
option.

The legacy workflow can use a local `data.raw_prices_path`; the paper workflow
uses the archived-universe downloader specified in `config/paper_revision.yaml`.
Do not commit generated datasets, `results/`, or checkpoint files.

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
- A run is reportable only when its `run.json` status is `SUCCESS` and the
  aggregate table can trace it through `ledger.csv`; model failures never fall
  back to equal weighting.
