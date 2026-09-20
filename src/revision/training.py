from __future__ import annotations

import copy
import math
from collections import deque
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.ddpg import DDPGAgent
from src.models.td3 import TD3Agent
from src.models.tgnn import TGNN
from src.models.tgnn.loss import combined_loss

from .dataset import PanelWindowDataset
from .ledger import completed_run, make_run_id, matching_run_provenance, run_directory, tracked_run
from .models import AlphaMLP, assert_frozen, freeze_module, normalize_summaries
from .portfolio import (
    compute_metrics,
    drift_weights,
    interval_net_returns,
    portfolio_turnover,
    run_weight_backtest,
)
from .replay import IndexedPanelReplayBuffer
from .utils import atomic_write_json, set_global_seed, sha256_file


def research_model_config(config: Dict[str, Any], symbols: Sequence[str]) -> Dict[str, Any]:
    model_config = copy.deepcopy(config)
    model_config["data"]["stock_universes"] = list(symbols)
    model_config["data"]["factors"] = bool(model_config["data"].get("macro_features"))
    model_config["project"]["selected_model"] = "revision"
    return model_config


def target_columns(config: Dict[str, Any]) -> list[str]:
    return list(config["data"]["target_heads"].values())


def head_for_frequency(config: Dict[str, Any], frequency: str) -> str:
    horizon_by_frequency = {"daily": 1, "weekly": 5, "monthly": 21, "quarterly": 63}
    horizon = horizon_by_frequency[frequency]
    for head, column in config["data"]["target_heads"].items():
        if column == f"ForwardReturn{horizon}D":
            return head
    raise KeyError(f"No target head configured for {frequency}")


def make_dataset(
    config: Dict[str, Any],
    panel: pd.DataFrame,
    symbols: Sequence[str],
    fold: str,
    split: str,
    graph: str,
    require_targets: bool,
) -> PanelWindowDataset:
    bounds = config["folds"][fold][split]
    return PanelWindowDataset(
        panel=panel,
        symbols=symbols,
        feature_columns=config["data"]["features"],
        macro_columns=config["data"]["macro_features"],
        target_columns=target_columns(config),
        start=bounds["start"],
        end=bounds["end"],
        window_size=int(config["data"]["window_size"]),
        graph_lookback=int(config["data"]["graph_lookback"]),
        graph_mode=graph,
        require_targets=require_targets,
    )


def close_frame(panel: pd.DataFrame, symbols: Sequence[str]) -> pd.DataFrame:
    close = panel[panel["Symbol"].isin(symbols)].pivot(index="Date", columns="Symbol", values="Close")
    return close.reindex(columns=list(symbols)).sort_index().astype(float)


def risk_free_series(factors: pd.DataFrame) -> pd.Series:
    values = factors.set_index("Date")["RF"].sort_index().astype(float)
    values.name = "RF"
    return values


def training_specification(
    config: Dict[str, Any],
    fold: str,
    model: str,
    seed: int,
    frequency: str,
    graph: str,
) -> Dict[str, Any]:
    if model == "tgnn":
        frequency = "all"
    if model in {"ddpg", "td3"}:
        graph = "none"
    if model == "hybrid":
        frequency = "all"
    return {
        "stage": "train",
        "fold": fold,
        "universe": "train",
        "model": model,
        "seed": int(seed),
        "frequency": frequency,
        "graph": graph,
        "cost_bps": float(config["training"]["training_cost_bps"]),
    }


def checkpoint_path(
    config: Dict[str, Any], fold: str, model: str, seed: int, frequency: str, graph: str
) -> Path:
    specification = training_specification(config, fold, model, seed, frequency, graph)
    return run_directory(config, make_run_id(specification)) / "checkpoint.pt"


def _batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _tgnn_epoch(
    model: TGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    heads: Sequence[str],
    mse_weight: float,
    ranking_weight: float,
) -> float:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for raw_batch in loader:
            batch = _batch_to_device(raw_batch, model.device)
            if training:
                optimizer.zero_grad()
            loss = torch.zeros((), device=model.device)
            for target_index, head in enumerate(heads):
                predictions, _ = model(
                    batch["prices"], batch["adj_matrix"], macro=batch["macro"], target_type=head
                )
                loss = loss + combined_loss(
                    predictions,
                    batch["labels"][:, :, target_index],
                    active_mask=batch["active_mask"],
                    alpha=mse_weight,
                    beta=ranking_weight,
                )
            loss = loss / len(heads)
            if not torch.isfinite(loss):
                raise RuntimeError("TGNN loss became non-finite")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            losses.append(float(loss.item()))
    if not losses:
        raise RuntimeError("TGNN loader produced no batches")
    return float(np.mean(losses))


def train_tgnn(
    config: Dict[str, Any],
    panel: pd.DataFrame,
    symbols: Sequence[str],
    fold: str,
    seed: int,
    graph: str,
    directory: Path,
    resume: bool,
) -> Path:
    model_config = research_model_config(config, symbols)
    train_data = make_dataset(config, panel, symbols, fold, "train", graph, require_targets=True)
    validation_data = make_dataset(config, panel, symbols, fold, "validation", graph, require_targets=True)
    train_loader = DataLoader(
        train_data, batch_size=int(config["training"]["batch_size"]), shuffle=True, num_workers=0
    )
    validation_loader = DataLoader(
        validation_data, batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=0
    )
    model = TGNN(model_config)
    learning_rate = float(config["model"]["tgnn"]["learning_rate"])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    latest_path = directory / "latest.pt"
    start_epoch = 0
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    stale = 0
    if resume and latest_path.exists():
        latest = torch.load(latest_path, map_location=model.device, weights_only=False)
        model.load_state_dict(latest["model_state"])
        optimizer.load_state_dict(latest["optimizer_state"])
        start_epoch = int(latest["epoch"]) + 1
        best_loss = float(latest["best_loss"])
        best_state = latest.get("best_state")
        history = list(latest.get("history", []))
        stale = int(latest.get("stale", 0))

    settings = config["model"]["tgnn"]
    max_epochs = int(config["training"]["tgnn_max_epochs"])
    patience = int(config["training"]["tgnn_early_stopping_patience"])
    checkpoint_interval = int(config["training"]["checkpoint_interval"])
    for epoch in range(start_epoch, max_epochs):
        train_loss = _tgnn_epoch(
            model,
            train_loader,
            optimizer,
            model.heads,
            float(settings["mse_weight"]),
            float(settings["ranking_weight"]),
        )
        validation_loss = _tgnn_epoch(
            model,
            validation_loader,
            None,
            model.heads,
            float(settings["mse_weight"]),
            float(settings["ranking_weight"]),
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-10:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if (epoch + 1) % checkpoint_interval == 0 or epoch + 1 == max_epochs:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_loss": best_loss,
                    "best_state": best_state,
                    "history": history,
                    "stale": stale,
                },
                latest_path,
            )
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("TGNN training never produced a finite validation checkpoint")
    final_path = directory / "checkpoint.pt"
    torch.save(
        {
            "model": "tgnn",
            "model_state": best_state,
            "best_validation_loss": best_loss,
            "heads": model.heads,
            "symbols": list(symbols),
            "seed": seed,
            "graph": graph,
        },
        final_path,
    )
    pd.DataFrame(history).to_csv(directory / "training_history.csv", index=False)
    return final_path


def _actor_weights(
    agent: Any,
    dataset: PanelWindowDataset,
    dates: Sequence[pd.Timestamp],
    state_cache: Dict[pd.Timestamp, np.ndarray] | None = None,
) -> pd.DataFrame:
    rows: list[np.ndarray] = []
    agent.eval()
    with torch.no_grad():
        for date in dates:
            if state_cache is None:
                state = dataset.sample_for_date(date)["features"].unsqueeze(0).to(agent.device)
            else:
                state = (
                    torch.from_numpy(state_cache[pd.Timestamp(date)])
                    .unsqueeze(0)
                    .to(device=agent.device, dtype=torch.float32)
                )
            weights = agent.actor(state)[0][0].detach().cpu().numpy()
            rows.append(weights)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates), columns=dataset.symbols)


def _save_rl_latest(agent: Any, path: Path, episode: int, best_score: float, best_state: Any, history: list, stale: int) -> None:
    optimizer_states = {name: value.state_dict() for name, value in vars(agent).items() if name.endswith("_optimizer")}
    torch.save(
        {
            "episode": episode,
            "model_state": agent.state_dict(),
            "optimizer_states": optimizer_states,
            "buffer": list(agent.buffer.buffer),
            "best_score": best_score,
            "best_state": best_state,
            "history": history,
            "stale": stale,
            "update_steps": getattr(agent, "update_steps", None),
        },
        path,
    )


def _restore_rl_latest(agent: Any, path: Path) -> tuple[int, float, Any, list, int]:
    latest = torch.load(path, map_location=agent.device, weights_only=False)
    agent.load_state_dict(latest["model_state"])
    for name, state in latest.get("optimizer_states", {}).items():
        optimizer = getattr(agent, name, None)
        if optimizer is not None:
            optimizer.load_state_dict(state)
    agent.buffer.buffer = deque(latest.get("buffer", []), maxlen=agent.buffer.capacity)
    if latest.get("update_steps") is not None:
        agent.update_steps = int(latest["update_steps"])
    return (
        int(latest["episode"]) + 1,
        float(latest["best_score"]),
        latest.get("best_state"),
        list(latest.get("history", [])),
        int(latest.get("stale", 0)),
    )


def train_rl(
    config: Dict[str, Any],
    panel: pd.DataFrame,
    factors: pd.DataFrame,
    symbols: Sequence[str],
    fold: str,
    seed: int,
    frequency: str,
    model_name: str,
    directory: Path,
    resume: bool,
) -> Path:
    model_config = research_model_config(config, symbols)
    train_data = make_dataset(config, panel, symbols, fold, "train", "combined", require_targets=False)
    validation_data = make_dataset(config, panel, symbols, fold, "validation", "combined", require_targets=False)
    agent = DDPGAgent(model_config) if model_name == "ddpg" else TD3Agent(model_config)
    agent.buffer = IndexedPanelReplayBuffer(
        train_data,
        capacity=int(config["training"]["buffer_size"]),
        state_cache_size=len(train_data),
    )
    train_dates = train_data.decision_dates(frequency)
    validation_dates = validation_data.decision_dates(frequency)
    if len(train_dates) < 2 or len(validation_dates) < 2:
        raise RuntimeError(f"{frequency} training requires at least two train and validation decisions")
    closes = close_frame(panel, symbols)
    risk_free = risk_free_series(factors)
    validation_state_cache = {
        pd.Timestamp(date): validation_data.sample_for_date(date)["features"].numpy().astype(np.float16)
        for date in validation_dates
    }
    cost_bps = float(config["training"]["training_cost_bps"])
    settings = config["model"][model_name]
    max_epochs = int(config["training"]["rl_max_epochs"])
    gradient_steps = int(config["training"]["gradient_steps_per_epoch"])
    evaluation_interval = int(config["training"]["evaluation_interval"])
    checkpoint_interval = int(config["training"]["checkpoint_interval"])
    patience_evaluations = int(config["training"]["rl_early_stopping_patience_evaluations"])
    latest_path = directory / "latest.pt"
    start_episode = 0
    best_score = -math.inf
    best_state = None
    history: list[dict[str, float | int]] = []
    stale = 0
    if resume and latest_path.exists():
        start_episode, best_score, best_state, history, stale = _restore_rl_latest(agent, latest_path)

    for episode in range(start_episode, max_epochs):
        progress = episode / max(1, max_epochs - 1)
        noise = float(settings["noise_start"]) + progress * (
            float(settings["noise_end"]) - float(settings["noise_start"])
        )
        pretrade: np.ndarray | None = None
        rewards: list[float] = []
        turnovers: list[float] = []
        concentrations: list[float] = []
        update_metrics: list[Dict[str, float]] = []
        for step, (date, next_date) in enumerate(zip(train_dates[:-1], train_dates[1:])):
            state_index = train_data.index_for_date(date)
            next_index = train_data.index_for_date(next_date)
            state = agent.buffer.state(state_index)
            action = agent.select_action(state, noise_std=noise)
            turnover = portfolio_turnover(action, pretrade)
            asset_returns = closes.loc[next_date].to_numpy() / closes.loc[date].to_numpy() - 1.0
            gross = float(np.dot(action, asset_returns))
            cost = turnover * cost_bps / 10_000.0
            net = float((1.0 - cost) * (1.0 + gross) - 1.0)
            if net <= -1.0 or not np.isfinite(net):
                raise RuntimeError(f"Non-finite RL reward at {date.date()}")
            reward = float(np.log1p(net))
            done = float(step == len(train_dates) - 2)
            agent.buffer.push(state_index, action, reward, next_index, done)
            rewards.append(reward)
            turnovers.append(turnover)
            concentrations.append(float(np.square(action).sum()))
            pretrade = drift_weights(action, asset_returns)

        # Keep DDPG and TD3 optimization budgets identical across every
        # rebalancing frequency. Trajectory length changes replay contents,
        # not the number of gradient updates.
        for _ in range(gradient_steps):
            metrics = agent.update()
            if metrics is None:
                break
            update_metrics.append(metrics)

        row: dict[str, float | int] = {
            "episode": episode,
            "episode_reward": float(np.sum(rewards)),
            "mean_reward": float(np.mean(rewards)),
            "noise": noise,
            "mean_turnover": float(np.mean(turnovers)),
            "mean_concentration": float(np.mean(concentrations)),
        }
        metric_names = sorted({key for metrics in update_metrics for key in metrics})
        for name in metric_names:
            values = [metrics[name] for metrics in update_metrics if name in metrics and np.isfinite(metrics[name])]
            row[name] = float(np.mean(values)) if values else float("nan")

        should_evaluate = (episode + 1) % evaluation_interval == 0 or episode + 1 == max_epochs
        if should_evaluate:
            weights = _actor_weights(agent, validation_data, validation_dates, validation_state_cache)
            evaluated_close = closes.loc[
                (closes.index >= validation_dates[0]) & (closes.index <= config["folds"][fold]["validation"]["end"])
            ]
            output = run_weight_backtest(evaluated_close, weights, cost_bps)
            validation_metrics = compute_metrics(output.daily, risk_free, output.decisions)
            score = float(validation_metrics["sharpe"])
            row["validation_sharpe"] = score
            if np.isfinite(score) and score > best_score + 1e-10:
                best_score = score
                best_state = {key: value.detach().cpu().clone() for key, value in agent.state_dict().items()}
                stale = 0
            else:
                stale += 1
        history.append(row)

        if (episode + 1) % checkpoint_interval == 0 or episode + 1 == max_epochs:
            _save_rl_latest(agent, latest_path, episode, best_score, best_state, history, stale)
        if stale >= patience_evaluations:
            break

    if best_state is None:
        # A one-episode smoke run can still be valid when its Sharpe is undefined;
        # preserve the trained state but make that limitation explicit.
        if max_epochs == 1:
            best_state = {key: value.detach().cpu().clone() for key, value in agent.state_dict().items()}
        else:
            raise RuntimeError(f"{model_name.upper()} training never produced a finite validation checkpoint")
    final_path = directory / "checkpoint.pt"
    torch.save(
        {
            "model": model_name,
            "model_state": best_state,
            "best_validation_sharpe": best_score,
            "symbols": list(symbols),
            "seed": seed,
            "frequency": frequency,
        },
        final_path,
    )
    pd.DataFrame(history).to_csv(directory / "training_history.csv", index=False)
    return final_path


def load_tgnn(
    config: Dict[str, Any], symbols: Sequence[str], path: Path
) -> TGNN:
    if not path.exists():
        raise FileNotFoundError(f"TGNN checkpoint is missing: {path}")
    model = TGNN(research_model_config(config, symbols))
    checkpoint = torch.load(path, map_location=model.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


def load_rl(
    config: Dict[str, Any], symbols: Sequence[str], path: Path, model_name: str
) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"{model_name.upper()} checkpoint is missing: {path}")
    model_config = research_model_config(config, symbols)
    agent = DDPGAgent(model_config) if model_name == "ddpg" else TD3Agent(model_config)
    checkpoint = torch.load(path, map_location=agent.device, weights_only=False)
    agent.load_state_dict(checkpoint["model_state"], strict=True)
    agent.eval()
    return agent


def tgnn_weights(
    model: TGNN,
    dataset: PanelWindowDataset,
    dates: Sequence[pd.Timestamp],
    head: str,
) -> pd.DataFrame:
    rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for date in dates:
            sample = dataset.sample_for_date(date)
            weights, _ = model.get_portfolio_weights(
                sample["prices"].unsqueeze(0).to(model.device),
                sample["adj_matrix"].unsqueeze(0).to(model.device),
                macro=sample["macro"].unsqueeze(0).to(model.device),
                target_head=head,
                temperature=float(model.config["model"].get("softmax_temperature", 1.0)),
            )
            rows.append(weights[0].cpu().numpy())
    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates), columns=dataset.symbols)


def train_alpha(
    config: Dict[str, Any],
    panel: pd.DataFrame,
    symbols: Sequence[str],
    fold: str,
    seed: int,
    graph: str,
    directory: Path,
) -> Path:
    tgnn_path = checkpoint_path(config, fold, "tgnn", seed, "all", graph)
    tgnn = load_tgnn(config, symbols, tgnn_path)
    freeze_module(tgnn)
    summaries: list[np.ndarray] = []
    labels: list[float] = []
    sample_weights: list[float] = []
    sample_rows: list[dict[str, Any]] = []
    source_checkpoints = {"tgnn": sha256_file(tgnn_path), "ddpg": {}}
    closes = close_frame(panel, symbols)
    training_cost = float(config["training"]["training_cost_bps"])

    for frequency in config["backtest"]["frequencies"]:
        ddpg_path = checkpoint_path(config, fold, "ddpg", seed, frequency, "none")
        ddpg = load_rl(config, symbols, ddpg_path, "ddpg")
        freeze_module(ddpg)
        assert_frozen(tgnn, ddpg)
        source_checkpoints["ddpg"][frequency] = sha256_file(ddpg_path)
        dataset = make_dataset(config, panel, symbols, fold, "validation", graph, require_targets=False)
        dates = dataset.decision_dates(frequency)
        if len(dates) < 2:
            raise RuntimeError(f"Alpha calibration needs at least two {frequency} decisions")
        tgnn_frame = tgnn_weights(tgnn, dataset, dates, head_for_frequency(config, frequency))
        ddpg_frame = _actor_weights(ddpg, dataset, dates)
        tgnn_returns = interval_net_returns(closes, tgnn_frame, training_cost)
        ddpg_returns = interval_net_returns(closes, ddpg_frame, training_cost)
        for date in tgnn_returns.index.intersection(ddpg_returns.index):
            difference = float(tgnn_returns.loc[date] - ddpg_returns.loc[date])
            summaries.append(dataset.raw_state_summary(date))
            labels.append(float(difference > 0.0))
            sample_weights.append(abs(difference))
            sample_rows.append(
                {
                    "Date": date,
                    "frequency": frequency,
                    "tgnn_interval_net_return": float(tgnn_returns.loc[date]),
                    "ddpg_interval_net_return": float(ddpg_returns.loc[date]),
                    "label_tgnn_better": float(difference > 0.0),
                    "absolute_gap": abs(difference),
                }
            )

    x = np.vstack(summaries).astype(np.float32)
    y = np.asarray(labels, dtype=np.float32).reshape(-1, 1)
    raw_weights = np.asarray(sample_weights, dtype=np.float32).reshape(-1, 1)
    if len(x) < 2:
        raise RuntimeError("Alpha calibration requires at least two samples")
    dates = pd.DatetimeIndex([row["Date"] for row in sample_rows])
    frequencies = np.asarray([str(row["frequency"]) for row in sample_rows])
    order = np.lexsort((frequencies, dates.asi8))
    x, y, raw_weights = x[order], y[order], raw_weights[order]
    sample_rows = [sample_rows[index] for index in order]
    dates = dates[order]

    unique_dates = dates.unique().sort_values()
    if len(unique_dates) >= 5:
        cutoff_position = min(len(unique_dates) - 1, max(1, round(len(unique_dates) * 0.8)))
        cutoff_date = unique_dates[cutoff_position]
        train_indices = np.flatnonzero(dates < cutoff_date)
        validation_indices = np.flatnonzero(dates >= cutoff_date)
    else:
        cutoff_date = None
        train_indices = np.arange(len(x))
        validation_indices = np.arange(len(x))
    if len(train_indices) == 0 or len(validation_indices) == 0:
        raise RuntimeError("Alpha chronological calibration split is empty")

    mean = x[train_indices].mean(axis=0)
    scale = x[train_indices].std(axis=0)
    normalized = normalize_summaries(x, mean, scale)
    mean_gap = float(raw_weights[train_indices].mean())
    weights = np.clip(raw_weights / max(mean_gap, 1e-8), 0.0, 10.0)
    if not np.any(weights[train_indices] > 0.0):
        weights = np.ones_like(raw_weights)

    settings = config["model"]["alpha"]
    model = AlphaMLP(
        input_dim=x.shape[1],
        hidden_dim=int(settings["hidden_dim"]),
        minimum=float(settings["minimum"]),
        maximum=float(settings["maximum"]),
    ).to(tgnn.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(settings["learning_rate"]))
    tensors = {
        "x": torch.from_numpy(normalized).to(tgnn.device),
        "y": torch.from_numpy(y).to(tgnn.device),
        "w": torch.from_numpy(weights).to(tgnn.device),
    }
    best_loss = math.inf
    best_state = None
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(int(settings["epochs"])):
        model.train()
        optimizer.zero_grad()
        logits = model.logits(tensors["x"][train_indices])
        per_sample = F.binary_cross_entropy_with_logits(logits, tensors["y"][train_indices], reduction="none")
        train_loss = (per_sample * tensors["w"][train_indices]).mean()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_logits = model.logits(tensors["x"][validation_indices])
            validation_per_sample = F.binary_cross_entropy_with_logits(
                validation_logits, tensors["y"][validation_indices], reduction="none"
            )
            validation_loss = (
                validation_per_sample * tensors["w"][validation_indices]
            ).sum() / tensors["w"][validation_indices].sum().clamp_min(1e-8)
        value = float(validation_loss.item())
        history.append({"epoch": epoch, "train_loss": float(train_loss.item()), "validation_loss": value})
        if value < best_loss - 1e-10:
            best_loss = value
            best_state = {key: item.detach().cpu().clone() for key, item in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= int(settings["patience"]):
            break
    if best_state is None:
        raise RuntimeError("Alpha calibration failed to produce a checkpoint")
    final_path = directory / "checkpoint.pt"
    torch.save(
        {
            "model": "alpha",
            "model_state": best_state,
            "input_dim": int(x.shape[1]),
            "mean": mean,
            "scale": scale,
            "minimum": float(settings["minimum"]),
            "maximum": float(settings["maximum"]),
            "hidden_dim": int(settings["hidden_dim"]),
            "source_checkpoints": source_checkpoints,
            "calibration_holdout_start": cutoff_date.isoformat() if cutoff_date is not None else None,
            "best_validation_loss": best_loss,
            "seed": seed,
            "graph": graph,
        },
        final_path,
    )
    pd.DataFrame(sample_rows).to_csv(directory / "alpha_samples.csv", index=False)
    pd.DataFrame(history).to_csv(directory / "training_history.csv", index=False)
    return final_path


def train_model(
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
    resume: bool,
) -> Path | None:
    if model in {"equal_weight", "min_variance", "risk_parity", "hybrid_fixed"}:
        return None
    specification = training_specification(config, fold, model, seed, frequency, graph)
    run_id = make_run_id(specification)
    required = ["checkpoint.pt", "training_history.csv"]
    if model == "hybrid":
        required.append("alpha_samples.csv")
    if resume and completed_run(config, run_id, required, data_hash=data_hash):
        return run_directory(config, run_id) / "checkpoint.pt"
    resume_checkpoint = resume and matching_run_provenance(config, run_id, data_hash=data_hash)

    set_global_seed(seed, bool(config["project"].get("deterministic", True)))
    symbols = universes["train"]
    with tracked_run(config, data_hash, specification) as (_, directory):
        if model == "tgnn":
            path = train_tgnn(
                config, panel, symbols, fold, seed, specification["graph"], directory, resume_checkpoint
            )
        elif model in {"ddpg", "td3"}:
            path = train_rl(
                config,
                panel,
                factors,
                symbols,
                fold,
                seed,
                specification["frequency"],
                model,
                directory,
                resume_checkpoint,
            )
        elif model == "hybrid":
            path = train_alpha(config, panel, symbols, fold, seed, specification["graph"], directory)
        else:
            raise ValueError(f"Unknown trainable model: {model}")
        atomic_write_json(
            directory / "artifacts.json",
            {"checkpoint": str(path), "checkpoint_sha256": sha256_file(path)},
        )
        return path
