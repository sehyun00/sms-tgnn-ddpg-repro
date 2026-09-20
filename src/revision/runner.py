from __future__ import annotations

from typing import Any, Dict, Iterable

import torch

from .backtest import BASELINES, backtest_model
from .data import load_prepared, prepare_data
from .statistics import aggregate_results
from .training import train_model


TRAINABLE_MODELS = ["tgnn", "ddpg", "td3", "hybrid"]
BACKTEST_MODELS = [
    "tgnn",
    "ddpg",
    "td3",
    "hybrid",
    "hybrid_fixed",
    "equal_weight",
    "min_variance",
    "risk_parity",
]


def _selected(raw: str, available: Iterable[str]) -> list[str]:
    choices = list(available)
    if raw == "all":
        return choices
    selected = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(choices))
    if unknown:
        raise ValueError(f"Unknown choices {unknown}; expected one of {choices}")
    return selected


def _seeds(config: Dict[str, Any], fold: str, raw: str, universe: str = "n10") -> list[int]:
    configured = [int(seed) for seed in config["folds"][fold]["seeds"]]
    if raw == "all":
        if universe in {"n5", "n15"}:
            return [42] if 42 in configured else configured[:1]
        return configured
    requested = [int(value.strip()) for value in raw.split(",") if value.strip()]
    unknown = sorted(set(requested) - set(configured))
    if unknown:
        raise ValueError(f"Seeds are not configured for fold {fold}: {unknown}")
    return requested


def _frequencies(config: Dict[str, Any], fold: str, raw: str) -> list[str]:
    configured = list(config["backtest"]["frequencies"])
    if fold == "secondary":
        configured = [config["backtest"]["primary_frequency"]]
    return _selected(raw, configured) if raw != "all" else configured


def _graphs(config: Dict[str, Any], raw: str) -> list[str]:
    if raw == "primary":
        return [config["backtest"]["primary_graph"]]
    return _selected(raw, config["backtest"]["graph_modes"])


def _require_device(config: Dict[str, Any], stage: str) -> None:
    if stage in {"train", "backtest"} and str(config["project"].get("device", "cpu")).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "This paper-scale configuration requires CUDA. Use config/revision_smoke.yaml locally "
                "or enable a GPU runtime in Colab."
            )


def execute(
    config: Dict[str, Any],
    stage: str,
    fold: str = "primary",
    model: str = "all",
    seed: str = "all",
    frequency: str = "all",
    graph: str = "primary",
    universe: str = "n10",
    resume: bool = False,
) -> Any:
    if stage == "prepare":
        return prepare_data(config)
    if fold not in config["folds"]:
        raise ValueError(f"Unknown fold: {fold}")
    _require_device(config, stage)
    panel, factors, universes, manifest = load_prepared(config)
    data_hash = manifest["data_hash"]

    if stage == "train":
        models = _selected(model, TRAINABLE_MODELS)
        seeds = _seeds(config, fold, seed)
        frequencies = _frequencies(config, fold, frequency)
        graphs = _graphs(config, graph)
        outputs = []
        # Fixed dependency order guarantees Hybrid sees completed, frozen branches.
        for current_seed in seeds:
            for current_model in TRAINABLE_MODELS:
                if current_model not in models:
                    continue
                if current_model == "tgnn":
                    for current_graph in graphs:
                        outputs.append(
                            train_model(
                                config, panel, factors, universes, data_hash, fold, current_model,
                                current_seed, "all", current_graph, resume
                            )
                        )
                elif current_model in {"ddpg", "td3"}:
                    for current_frequency in frequencies:
                        outputs.append(
                            train_model(
                                config, panel, factors, universes, data_hash, fold, current_model,
                                current_seed, current_frequency, "none", resume
                            )
                        )
                else:
                    for current_graph in graphs:
                        outputs.append(
                            train_model(
                                config, panel, factors, universes, data_hash, fold, current_model,
                                current_seed, "all", current_graph, resume
                            )
                        )
        return outputs

    if stage == "backtest":
        models = _selected(model, BACKTEST_MODELS)
        seeds = _seeds(config, fold, seed, universe=universe)
        frequencies = _frequencies(config, fold, frequency)
        graphs = _graphs(config, graph)
        outputs = []
        for current_model in models:
            model_seeds = [0] if current_model in BASELINES else seeds
            for current_seed in model_seeds:
                for current_frequency in frequencies:
                    model_graphs = graphs if current_model in {"tgnn", "hybrid", "hybrid_fixed"} else ["none"]
                    for current_graph in model_graphs:
                        outputs.append(
                            backtest_model(
                                config, panel, factors, universes, data_hash, fold, current_model,
                                current_seed, current_frequency, current_graph, universe, resume
                            )
                        )
        return outputs

    if stage == "aggregate":
        return aggregate_results(config, factors, data_hash)
    raise ValueError(f"Unknown stage: {stage}")
