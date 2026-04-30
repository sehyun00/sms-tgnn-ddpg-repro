import argparse
import copy
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from main import load_config

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SEEDS = [0, 42, 123, 456, 789]


def configure_stdio() -> None:
    """Use UTF-8 console output on Windows when supported."""
    import sys

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Fix all RNGs used by Python, NumPy, and PyTorch."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    print(f"[seed] fixed={seed}, deterministic={deterministic}")


def redirect_results(config: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Route one seed's outputs to an isolated ignored results directory."""
    seed_config = copy.deepcopy(config)
    seed_config["paths"]["results_dir"] = str(
        PROJECT_ROOT / "results" / "multiseed" / f"seed_{seed}"
    )
    return seed_config


def dry_run(seeds: List[int], config: Dict[str, Any]) -> None:
    """Verify seed routing and config mutation without running experiments."""
    for seed in seeds:
        seed_config = redirect_results(config, seed)
        seed_config["model"]["reproducibility"]["seed"] = seed
        set_seed(seed)
        print(
            f"[dry-run] seed={seed} results_dir={seed_config['paths']['results_dir']}"
        )


def run_single_seed(seed: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run Train -> Backtest for one seed and collect metrics."""
    from src.pipelines.backtest_pipeline import run_backtest
    from src.pipelines.train_pipeline import run_train

    seed_config = redirect_results(config, seed)
    Path(seed_config["paths"]["results_dir"]).mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    models = ["tgnn", "ddpg", "hybrid"]
    for model_name in models:
        model_config = copy.deepcopy(seed_config)
        model_config["project"]["selected_model"] = model_name
        model_config["model"]["reproducibility"]["seed"] = seed
        run_train(model_config)

    collected: Dict[str, Any] = {}
    col_map = {
        "CAGR": "CAGR (%)",
        "Sharpe": "Sharpe Ratio",
        "MDD": "MDD (%)",
        "Total_Return": "Total Return (%)",
    }

    for model_name in models:
        bt_config = copy.deepcopy(seed_config)
        bt_config["project"]["selected_model"] = model_name
        bt_config["model"]["reproducibility"]["seed"] = seed
        run_backtest(bt_config, model_path=None)

        metrics_path = (
            Path(seed_config["paths"]["results_dir"])
            / model_name
            / "logs"
            / "backtest_metrics.csv"
        )
        if not metrics_path.exists():
            continue
        df = pd.read_csv(metrics_path).rename(columns=col_map)
        for _, row in df.iterrows():
            strategy = row["Strategy"]
            if strategy == "Benchmark" and strategy in collected:
                continue
            collected[strategy] = row.to_dict()
            collected[strategy]["seed"] = seed
    return collected


def aggregate_results(all_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Aggregate per-seed metrics into mean/std rows by strategy."""
    strategies = sorted({name for result in all_results for name in result.keys()})
    numeric_cols = ["CAGR (%)", "Sharpe Ratio", "MDD (%)", "Total Return (%)"]
    rows = []

    for strategy in strategies:
        row: Dict[str, Any] = {"Strategy": strategy}
        for col in numeric_cols:
            vals = [
                float(result[strategy][col])
                for result in all_results
                if strategy in result and col in result[strategy]
            ]
            if vals:
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                row[f"{col} Mean"] = round(mean_val, 4)
                row[f"{col} Std"] = round(std_val, 4)
                row[f"{col} (mean+-std)"] = f"{mean_val:.2f} +- {std_val:.2f}"
            else:
                row[f"{col} Mean"] = None
                row[f"{col} Std"] = None
                row[f"{col} (mean+-std)"] = "N/A"
        rows.append(row)
    return pd.DataFrame(rows)


def parse_seeds(raw: str) -> List[int]:
    """Parse a comma-separated seed list."""
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    configure_stdio()
    parser = argparse.ArgumentParser(description="Run multi-seed reproducibility.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    seeds = parse_seeds(args.seeds)
    if args.dry_run:
        dry_run(seeds, config)
        return

    all_results = []
    failed = []
    for seed in seeds:
        result = run_single_seed(seed, config)
        if result:
            all_results.append(result)
        else:
            failed.append(seed)

    if not all_results:
        raise RuntimeError("No seed produced metrics.")

    out_dir = PROJECT_ROOT / "results" / "multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df = aggregate_results(all_results)
    summary_df.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    meta = {
        "seeds": seeds,
        "completed_seeds": [seed for seed in seeds if seed not in failed],
        "failed_seeds": failed,
        "n_strategies": len(summary_df),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
