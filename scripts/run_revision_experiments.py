from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.revision.config import load_revision_config
from src.revision.runner import execute


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run leakage-controlled SMS major-revision experiments."
    )
    parser.add_argument("--config", default="config/paper_revision.yaml")
    parser.add_argument("--stage", required=True, choices=["prepare", "train", "backtest", "aggregate"])
    parser.add_argument("--fold", default="primary", choices=["primary", "secondary"])
    parser.add_argument("--model", default="all", help="One model, comma-separated models, or all")
    parser.add_argument("--seed", default="all", help="One seed, comma-separated seeds, or all")
    parser.add_argument("--frequency", default="all", help="One frequency, comma-separated frequencies, or all")
    parser.add_argument("--graph", default="primary", help="sector, correlation, combined, comma-separated, all, or primary")
    parser.add_argument("--universe", default="n10", choices=["n5", "n10", "n15"])
    parser.add_argument("--resume", action="store_true", help="Skip successful runs and continue interrupted checkpoints")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_revision_config(args.config)
    result = execute(
        config=config,
        stage=args.stage,
        fold=args.fold,
        model=args.model,
        seed=args.seed,
        frequency=args.frequency,
        graph=args.graph,
        universe=args.universe,
        resume=args.resume,
    )
    if isinstance(result, dict):
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    elif isinstance(result, list):
        print("\n".join(str(item) for item in result if item is not None))
    else:
        print(result)


if __name__ == "__main__":
    main()
