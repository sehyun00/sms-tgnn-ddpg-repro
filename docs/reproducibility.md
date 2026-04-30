# Reproducibility Notes

The public smoke workflow is deterministic and fixture-based:

1. `scripts/create_sample_data.py` creates synthetic OHLCV data with fixed seed
   `42`.
2. `main.py --mode preprocess` computes rolling indicators and writes ignored
   `data/train_data.csv` and `data/test_data.csv`.
3. `main.py --mode train --config config/sample.yaml` trains one CPU epoch.
4. `run_multiseed.py --dry-run` verifies seed-specific result routing without
   launching expensive training.

Do not cite paper metrics from this fixture. It is a code-path verification
asset, not an empirical result.
