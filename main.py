import argparse
import copy
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
PROJECT_ROOT = Path(__file__).resolve().parent


def configure_stdio() -> None:
    """Use UTF-8 console output on Windows when supported."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def _resolve_repo_path(value: Optional[str]) -> Optional[str]:
    """Resolve repo-relative paths while preserving empty values."""
    if not value:
        return value
    path = Path(value)
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


def load_config(path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load a YAML config relative to the repository root."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return normalize_paths(config)


def normalize_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize config paths so commands work from any current directory."""
    config = copy.deepcopy(config)
    paths = config.setdefault("paths", {})
    for key in ("data_dir", "results_dir"):
        paths[key] = _resolve_repo_path(paths.get(key, key.replace("_dir", "")))

    data_conf = config.setdefault("data", {})
    for key in ("raw_prices_path", "sample_prices_path"):
        if key in data_conf:
            data_conf[key] = _resolve_repo_path(data_conf[key])
    return config


def set_seed(seed: int, deterministic: bool = True, benchmark: bool = False) -> None:
    """Fix Python, NumPy, and PyTorch RNGs for reproducible experiments."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark if deterministic else True
    print(f"[seed] fixed={seed}, deterministic={deterministic}")


def configure_reproducibility(config: Dict[str, Any]) -> None:
    repro = config.get("model", {}).get("reproducibility", {})
    set_seed(
        seed=int(repro.get("seed", config.get("project", {}).get("seed", 42))),
        deterministic=bool(repro.get("deterministic", True)),
        benchmark=bool(repro.get("cudnn_benchmark", False)),
    )


def run_preprocess(config: Dict[str, Any], csv_path: Optional[str] = None) -> None:
    from src.preprocessing.pipeline import Pipeline

    data_conf = config.get("data", {})
    raw_prices = csv_path or data_conf.get("raw_prices_path") or data_conf.get(
        "sample_prices_path"
    )
    pipeline = Pipeline(
        csv_path=raw_prices,
        output_dir=config["paths"]["data_dir"],
        start_year=int(data_conf.get("start_year", 2006)),
        end_year=int(data_conf.get("end_year", 2025)),
    )
    pipeline.run(config)


def _models_for(config: Dict[str, Any]) -> list[str]:
    selected = config["project"].get("selected_model", "tgnn").lower()
    return ["tgnn", "ddpg", "hybrid"] if selected == "all" else [selected]


def run_train_mode(config: Dict[str, Any]) -> None:
    from src.pipelines.train_pipeline import run_train

    for model_name in _models_for(config):
        model_config = copy.deepcopy(config)
        model_config["project"]["selected_model"] = model_name
        run_train(model_config)


def run_backtest_mode(
    config: Dict[str, Any], model_path: Optional[str] = None
) -> None:
    from src.pipelines.backtest_pipeline import run_backtest

    for model_name in _models_for(config):
        model_config = copy.deepcopy(config)
        model_config["project"]["selected_model"] = model_name
        run_backtest(model_config, model_path=model_path)


def run_compare_mode() -> None:
    script = PROJECT_ROOT / "scripts" / "generate_comparison_chart.py"
    subprocess.run([sys.executable, str(script)], cwd=str(PROJECT_ROOT), check=True)


def main() -> None:
    configure_stdio()
    parser = argparse.ArgumentParser(description="SMS TGNN-DDPG reproducibility CLI")
    parser.add_argument(
        "--mode",
        choices=["preprocess", "train", "backtest", "full", "compare"],
        default="train",
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--csv", default=None, help="Optional raw price CSV path.")
    parser.add_argument("--model_path", default=None, help="Optional checkpoint path.")
    args = parser.parse_args()

    config = load_config(args.config)
    configure_reproducibility(config)

    if args.mode == "preprocess":
        run_preprocess(config, csv_path=args.csv)
    elif args.mode == "train":
        run_train_mode(config)
    elif args.mode == "backtest":
        run_backtest_mode(config, model_path=args.model_path)
    elif args.mode == "full":
        run_preprocess(config, csv_path=args.csv)
        run_train_mode(config)
        run_backtest_mode(config, model_path=args.model_path)
    elif args.mode == "compare":
        run_compare_mode()


if __name__ == "__main__":
    main()
