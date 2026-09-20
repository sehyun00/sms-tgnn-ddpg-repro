from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
import torch

from .dataset import PanelWindowDataset
from .ledger import completed_run, make_run_id, run_directory, tracked_run
from .models import AlphaMLP, mix_portfolios, normalize_summaries
from .portfolio import baseline_weights, compute_metrics, run_weight_backtest
from .training import (
    _actor_weights,
    checkpoint_path,
    close_frame,
    head_for_frequency,
    load_rl,
    load_tgnn,
    make_dataset,
    risk_free_series,
    tgnn_weights,
)
from .utils import atomic_write_json, set_global_seed, sha256_file


NEURAL_MODELS = {"tgnn", "ddpg", "td3", "hybrid", "hybrid_fixed"}
BASELINES = {"equal_weight", "min_variance", "risk_parity"}


def costs_for_run(
    config: Dict[str, Any], fold: str, model: str, graph: str, universe: str
) -> list[float]:
    primary_cost = float(config["backtest"]["primary_cost_bps"])
    graph_ablation = model in {"tgnn", "hybrid", "hybrid_fixed"} and (
        graph != config["backtest"]["primary_graph"]
    )
    if fold == "secondary" or universe != "n10" or graph_ablation:
        return [primary_cost]
    return [float(value) for value in config["backtest"]["cost_bps"]]


def backtest_specification(
    config: Dict[str, Any],
    fold: str,
    model: str,
    seed: int,
    frequency: str,
    graph: str,
    universe: str,
) -> Dict[str, Any]:
    if model in BASELINES | {"ddpg", "td3"}:
        graph = "none"
    return {
        "stage": "backtest",
        "fold": fold,
        "universe": universe,
        "model": model,
        "seed": int(seed),
        "frequency": frequency,
        "graph": graph,
        "cost_bps": costs_for_run(config, fold, model, graph, universe),
    }


def _load_alpha(config: Dict[str, Any], path: Path, device: torch.device) -> tuple[AlphaMLP, Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Alpha checkpoint is missing: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = AlphaMLP(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        minimum=float(checkpoint["minimum"]),
        maximum=float(checkpoint["maximum"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint


def _alpha_values(
    model: AlphaMLP,
    checkpoint: Dict[str, Any],
    dataset: PanelWindowDataset,
    dates: Sequence[pd.Timestamp],
) -> pd.Series:
    summaries = np.vstack([dataset.raw_state_summary(date) for date in dates])
    normalized = normalize_summaries(summaries, checkpoint["mean"], checkpoint["scale"])
    with torch.no_grad():
        tensor = torch.from_numpy(normalized).to(next(model.parameters()).device)
        values = model(tensor).squeeze(-1).cpu().numpy()
    if (values < float(checkpoint["minimum"]) - 1e-7).any() or (
        values > float(checkpoint["maximum"]) + 1e-7
    ).any():
        raise RuntimeError("Alpha inference escaped configured bounds")
    return pd.Series(values, index=pd.DatetimeIndex(dates), name="alpha")


def _long_weights(
    final: pd.DataFrame,
    alpha: pd.Series | None = None,
    tgnn: pd.DataFrame | None = None,
    ddpg: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date in final.index:
        for symbol in final.columns:
            row: dict[str, Any] = {
                "decision_date": date,
                "Symbol": symbol,
                "target_weight": float(final.loc[date, symbol]),
            }
            if alpha is not None:
                row["alpha"] = float(alpha.loc[date])
            if tgnn is not None:
                row["tgnn_weight"] = float(tgnn.loc[date, symbol])
            if ddpg is not None:
                row["ddpg_weight"] = float(ddpg.loc[date, symbol])
            rows.append(row)
    return pd.DataFrame(rows)


def _verify_alpha_sources(
    checkpoint: Dict[str, Any], tgnn_path: Path, ddpg_path: Path, frequency: str
) -> None:
    sources = checkpoint["source_checkpoints"]
    expected_tgnn = sources["tgnn"]
    expected_ddpg = sources["ddpg"][frequency]
    actual_tgnn = sha256_file(tgnn_path)
    actual_ddpg = sha256_file(ddpg_path)
    if expected_tgnn != actual_tgnn or expected_ddpg != actual_ddpg:
        raise RuntimeError("Hybrid base checkpoint hashes differ from the alpha calibration manifest")


def generate_target_weights(
    config: Dict[str, Any],
    panel: pd.DataFrame,
    symbols: Sequence[str],
    fold: str,
    model_name: str,
    seed: int,
    frequency: str,
    graph: str,
) -> tuple[pd.DataFrame, pd.Series | None, pd.DataFrame | None, pd.DataFrame | None, Dict[str, Any]]:
    dataset = make_dataset(config, panel, symbols, fold, "test", graph, require_targets=False)
    dates = dataset.decision_dates(frequency)
    if len(dates) < 2:
        raise RuntimeError(f"Backtest requires at least two {frequency} decisions")
    training_symbols = json.loads(
        (Path(config["paths"]["prepared_dir"]) / "universe_manifest.json").read_text(encoding="utf-8")
    )["train"]
    metadata: Dict[str, Any] = {"decisions": len(dates), "symbols": list(symbols)}

    if model_name in BASELINES:
        close = close_frame(panel, symbols)
        frame = baseline_weights(
            close,
            dates,
            model_name,
            int(config["backtest"]["covariance_lookback"]),
        )
        return frame, None, None, None, metadata

    if model_name == "tgnn":
        path = checkpoint_path(config, fold, "tgnn", seed, "all", graph)
        model = load_tgnn(config, symbols, path)
        frame = tgnn_weights(model, dataset, dates, head_for_frequency(config, frequency))
        metadata["checkpoint_sha256"] = sha256_file(path)
        return frame, None, None, None, metadata

    if model_name in {"ddpg", "td3"}:
        path = checkpoint_path(config, fold, model_name, seed, frequency, "none")
        agent = load_rl(config, symbols, path, model_name)
        frame = _actor_weights(agent, dataset, dates)
        metadata["checkpoint_sha256"] = sha256_file(path)
        return frame, None, None, None, metadata

    if model_name not in {"hybrid", "hybrid_fixed"}:
        raise ValueError(f"Unknown model: {model_name}")

    tgnn_path = checkpoint_path(config, fold, "tgnn", seed, "all", graph)
    ddpg_path = checkpoint_path(config, fold, "ddpg", seed, frequency, "none")
    tgnn_model = load_tgnn(config, symbols, tgnn_path)
    ddpg_model = load_rl(config, symbols, ddpg_path, "ddpg")
    tgnn_frame = tgnn_weights(tgnn_model, dataset, dates, head_for_frequency(config, frequency))
    ddpg_frame = _actor_weights(ddpg_model, dataset, dates)
    if model_name == "hybrid_fixed":
        alpha = pd.Series(0.5, index=pd.DatetimeIndex(dates), name="alpha")
        metadata["alpha_mode"] = "fixed-0.5"
    else:
        alpha_path = checkpoint_path(config, fold, "hybrid", seed, "all", graph)
        alpha_model, alpha_checkpoint = _load_alpha(config, alpha_path, tgnn_model.device)
        _verify_alpha_sources(alpha_checkpoint, tgnn_path, ddpg_path, frequency)
        alpha = _alpha_values(alpha_model, alpha_checkpoint, dataset, dates)
        metadata["alpha_mode"] = "learned-state-only"
        metadata["alpha_checkpoint_sha256"] = sha256_file(alpha_path)
    alpha_matrix = alpha.to_numpy()[:, None]
    final = alpha_matrix * tgnn_frame.to_numpy() + (1.0 - alpha_matrix) * ddpg_frame.to_numpy()
    final = final / final.sum(axis=1, keepdims=True)
    final_frame = pd.DataFrame(final, index=pd.DatetimeIndex(dates), columns=list(symbols))
    metadata["tgnn_checkpoint_sha256"] = sha256_file(tgnn_path)
    metadata["ddpg_checkpoint_sha256"] = sha256_file(ddpg_path)
    metadata["training_symbols"] = training_symbols
    return final_frame, alpha, tgnn_frame, ddpg_frame, metadata


def run_xai_faithfulness(
    config: Dict[str, Any],
    panel: pd.DataFrame,
    symbols: Sequence[str],
    fold: str,
    seed: int,
    graph: str,
    frequency: str,
    final_weights: pd.DataFrame,
    alpha_values: pd.Series,
    directory: Path,
) -> None:
    dataset = make_dataset(config, panel, symbols, fold, "test", graph, require_targets=False)
    dates = list(final_weights.index[:-1])
    count = min(int(config["xai"]["dates"]), len(dates))
    if count == 0:
        return
    selected_indices = np.unique(np.linspace(0, len(dates) - 1, count, dtype=int))
    selected_dates = [dates[index] for index in selected_indices]
    tgnn_path = checkpoint_path(config, fold, "tgnn", seed, "all", graph)
    ddpg_path = checkpoint_path(config, fold, "ddpg", seed, frequency, "none")
    tgnn_model = load_tgnn(config, symbols, tgnn_path)
    ddpg_model = load_rl(config, symbols, ddpg_path, "ddpg")
    rng = np.random.default_rng(int(config["xai"]["seed"]))
    mask_fraction = float(config["xai"]["mask_fraction"])
    random_repetitions = int(config["xai"]["random_masks_per_date"])
    closes = close_frame(panel, symbols)
    rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []

    for date in selected_dates:
        sample = dataset.sample_for_date(date)
        prices = sample["prices"].unsqueeze(0).to(tgnn_model.device)
        macro = sample["macro"].unsqueeze(0).to(tgnn_model.device)
        adjacency = sample["adj_matrix"].unsqueeze(0).to(tgnn_model.device)
        with torch.no_grad():
            tgnn_original, _, attention = tgnn_model.get_portfolio_weights(
                prices,
                adjacency,
                macro=macro,
                target_head=head_for_frequency(config, frequency),
                temperature=float(config["model"]["softmax_temperature"]),
                return_attn_weights=True,
            )
            ddpg_original = ddpg_model.actor(sample["features"].unsqueeze(0).to(ddpg_model.device))[0]
        importance = attention[0, :, :, -1, :].mean(dim=(0, 1)).detach().cpu().numpy()
        importance = importance / max(float(importance.sum()), 1e-12)
        for lag, value in enumerate(importance[::-1]):
            attention_rows.append({"Date": date, "lag": lag, "normalized_attention": float(value)})
        n_mask = max(1, int(np.ceil(len(importance) * mask_fraction)))
        top_positions = np.argsort(importance)[-n_mask:]
        original = torch.as_tensor(final_weights.loc[date].to_numpy(copy=True), dtype=torch.float32, device=tgnn_model.device)
        alpha = torch.tensor([[float(alpha_values.loc[date])]], dtype=torch.float32, device=tgnn_model.device)
        next_date = final_weights.index[final_weights.index.get_loc(date) + 1]
        forward_returns = torch.as_tensor(
            closes.loc[next_date].to_numpy() / closes.loc[date].to_numpy() - 1.0,
            dtype=torch.float32,
            device=tgnn_model.device,
        )

        def evaluate_mask(positions: np.ndarray) -> tuple[float, float]:
            masked = prices.clone()
            masked[:, :, positions, :] = 0.0
            with torch.no_grad():
                masked_tgnn, _ = tgnn_model.get_portfolio_weights(
                    masked,
                    adjacency,
                    macro=macro,
                    target_head=head_for_frequency(config, frequency),
                    temperature=float(config["model"]["softmax_temperature"]),
                )
                mixed = mix_portfolios(masked_tgnn, ddpg_original, alpha)[0]
            weight_change = float(0.5 * torch.abs(mixed - original).sum().item())
            return_change = float(torch.abs(torch.dot(mixed - original, forward_returns)).item())
            return weight_change, return_change

        top_weight_change, top_return_change = evaluate_mask(top_positions)
        random_values = [
            evaluate_mask(rng.choice(len(importance), size=n_mask, replace=False))
            for _ in range(random_repetitions)
        ]
        rows.append(
            {
                "Date": date,
                "top_weight_change": top_weight_change,
                "top_forward_return_change": top_return_change,
                "random_weight_change_mean": float(np.mean([value[0] for value in random_values])),
                "random_forward_return_change_mean": float(np.mean([value[1] for value in random_values])),
                "top_exceeds_random_weight": top_weight_change > np.mean([value[0] for value in random_values]),
                "top_exceeds_random_return": top_return_change > np.mean([value[1] for value in random_values]),
            }
        )
    pd.DataFrame(rows).to_csv(directory / "xai_faithfulness.csv", index=False)
    pd.DataFrame(attention_rows).to_csv(directory / "attention_lags.csv", index=False)


def backtest_model(
    config: Dict[str, Any],
    panel: pd.DataFrame,
    factors: pd.DataFrame,
    universes: Dict[str, Any],
    data_hash: str,
    fold: str,
    model: str,
    seed: int,
    frequency: str,
    graph: str,
    universe: str,
    resume: bool,
) -> Path:
    if universe not in {"n5", "n10", "n15"}:
        raise ValueError("universe must be n5, n10, or n15")
    if model not in NEURAL_MODELS | BASELINES:
        raise ValueError(f"Unknown backtest model: {model}")
    symbols = universes[universe]
    if not symbols:
        raise RuntimeError(f"Universe {universe} is empty")
    specification = backtest_specification(config, fold, model, seed, frequency, graph, universe)
    run_id = make_run_id(specification)
    required = ["metrics.csv", "weights.csv.gz"]
    for cost in specification["cost_bps"]:
        required.extend(
            [
                f"daily_cost_{float(cost):g}bps.csv.gz",
                f"decisions_cost_{float(cost):g}bps.csv",
            ]
        )
    if resume and completed_run(config, run_id, required, data_hash=data_hash):
        return run_directory(config, run_id)

    set_global_seed(seed, bool(config["project"].get("deterministic", True)))
    with tracked_run(config, data_hash, specification) as (_, directory):
        weights, alpha, tgnn_frame, ddpg_frame, metadata = generate_target_weights(
            config, panel, symbols, fold, model, seed, frequency, specification["graph"] if specification["graph"] != "none" else "combined"
        )
        long_weights = _long_weights(weights, alpha, tgnn_frame, ddpg_frame)
        long_weights.to_csv(
            directory / "weights.csv.gz", index=False, compression={"method": "gzip", "mtime": 0}
        )
        closes = close_frame(panel, symbols)
        test_end = pd.Timestamp(config["folds"][fold]["test"]["end"])
        evaluated_close = closes.loc[(closes.index >= weights.index[0]) & (closes.index <= test_end)]
        rf = risk_free_series(factors)
        metric_rows: list[dict[str, Any]] = []
        artifacts: Dict[str, Any] = {"weights": "weights.csv.gz", "daily": {}, "decisions": {}}
        if tgnn_frame is not None and ddpg_frame is not None:
            tgnn_daily = run_weight_backtest(evaluated_close, tgnn_frame, 0.0).daily["gross_return"]
            ddpg_daily = run_weight_backtest(evaluated_close, ddpg_frame, 0.0).daily["gross_return"]
            branch_daily = pd.concat(
                [tgnn_daily.rename("tgnn_gross_return"), ddpg_daily.rename("ddpg_gross_return")],
                axis=1,
                join="inner",
            )
            branch_daily.reset_index().to_csv(directory / "branch_daily_returns.csv", index=False)
            artifacts["branch_daily_returns"] = "branch_daily_returns.csv"
        for cost in specification["cost_bps"]:
            output = run_weight_backtest(
                evaluated_close,
                weights,
                float(cost),
                initial_capital=float(config["backtest"]["initial_capital"]),
            )
            daily_name = f"daily_cost_{float(cost):g}bps.csv.gz"
            output.daily.reset_index().to_csv(
                directory / daily_name, index=False, compression={"method": "gzip", "mtime": 0}
            )
            decisions_name = f"decisions_cost_{float(cost):g}bps.csv"
            output.decisions.to_csv(directory / decisions_name, index=False)
            metrics = compute_metrics(output.daily, rf, output.decisions)
            metric_rows.append({"cost_bps": float(cost), **metrics})
            artifacts["daily"][str(float(cost))] = daily_name
            artifacts["decisions"][str(float(cost))] = decisions_name
        pd.DataFrame(metric_rows).to_csv(directory / "metrics.csv", index=False)
        atomic_write_json(directory / "artifacts.json", {**artifacts, "metadata": metadata})

        should_run_xai = (
            model == "hybrid"
            and alpha is not None
            and fold == "primary"
            and universe == "n10"
            and frequency == config["backtest"]["primary_frequency"]
            and specification["graph"] == config["backtest"]["primary_graph"]
            and int(seed) == 42
        )
        if should_run_xai:
            run_xai_faithfulness(
                config,
                panel,
                symbols,
                fold,
                seed,
                specification["graph"],
                frequency,
                weights,
                alpha,
                directory,
            )
        return directory
