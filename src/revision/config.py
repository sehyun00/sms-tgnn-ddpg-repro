from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FREQUENCY_HORIZONS = {
    "daily": 1,
    "weekly": 5,
    "monthly": 21,
    "quarterly": 63,
}


def _resolve_path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


def load_revision_config(path: str | Path) -> Dict[str, Any]:
    """Load and validate one immutable revision experiment specification."""
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")

    required = ["project", "paths", "data", "folds", "model", "training", "backtest"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing configuration sections: {missing}")

    config = copy.deepcopy(config)
    config["_meta"] = {"config_path": str(config_path), "project_root": str(PROJECT_ROOT)}
    for key in ("prepared_dir", "results_dir"):
        config["paths"][key] = _resolve_path(config["paths"][key])
    if os.environ.get("SMS_PREPARED_DIR"):
        config["paths"]["prepared_dir"] = str(Path(os.environ["SMS_PREPARED_DIR"]).resolve())
    if os.environ.get("SMS_RESULTS_DIR"):
        config["paths"]["results_dir"] = str(Path(os.environ["SMS_RESULTS_DIR"]).resolve())
    raw_path = config["data"].get("raw_prices_path")
    if raw_path:
        config["data"]["raw_prices_path"] = _resolve_path(raw_path)

    frequencies = config["backtest"].get("frequencies", [])
    unknown_frequencies = sorted(set(frequencies) - set(FREQUENCY_HORIZONS))
    if unknown_frequencies:
        raise ValueError(f"Unsupported rebalancing frequencies: {unknown_frequencies}")

    target_horizons = config["data"].get("target_horizons", [])
    if sorted(target_horizons) != [1, 5, 21, 63]:
        raise ValueError("data.target_horizons must be exactly [1, 5, 21, 63]")

    exact_test = config["data"]["universe"]["exact_test_symbols"]
    if config["data"].get("provider") == "yfinance" and len(exact_test) != 10:
        raise ValueError("The paper-scale configuration must retain exactly ten test symbols")
    if len(set(exact_test)) != len(exact_test):
        raise ValueError("Test symbols contain duplicates")

    costs = config["backtest"].get("cost_bps", [])
    if any(float(cost) < 0 for cost in costs):
        raise ValueError("Transaction costs cannot be negative")
    if float(config["backtest"].get("primary_cost_bps", -1)) not in [float(x) for x in costs]:
        raise ValueError("backtest.primary_cost_bps must be present in backtest.cost_bps")

    positive_training_fields = (
        "tgnn_max_epochs",
        "tgnn_early_stopping_patience",
        "rl_max_epochs",
        "rl_early_stopping_patience_evaluations",
        "gradient_steps_per_epoch",
        "evaluation_interval",
        "checkpoint_interval",
        "batch_size",
        "buffer_size",
    )
    non_positive = [
        field for field in positive_training_fields if int(config["training"].get(field, 0)) <= 0
    ]
    if non_positive:
        raise ValueError(f"Training budget values must be positive: {non_positive}")

    for fold_name, fold in config["folds"].items():
        dates = [fold[part][side] for part in ("train", "validation", "test") for side in ("start", "end")]
        if dates != sorted(dates):
            raise ValueError(f"Fold {fold_name} dates must be chronological and non-overlapping")

    return config


def canonical_config(config: Dict[str, Any]) -> str:
    public_config = {key: value for key, value in config.items() if key != "_meta"}
    return json.dumps(public_config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
