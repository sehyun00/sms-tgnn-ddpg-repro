from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch

from src.revision.config import load_revision_config
from src.revision.data import engineer_features, load_prepared, prepare_data
from src.revision.dataset import PanelWindowDataset, build_adjacency, select_period_end_dates


def test_forward_returns_are_future_not_historical():
    dates = pd.bdate_range("2020-01-01", periods=4)
    prices = pd.DataFrame(
        {
            "Date": dates,
            "Symbol": "AAA",
            "Sector": "Technology",
            "Open": [100.0, 110.0, 121.0, 133.1],
            "High": [100.0, 110.0, 121.0, 133.1],
            "Low": [100.0, 110.0, 121.0, 133.1],
            "Close": [100.0, 110.0, 121.0, 133.1],
            "Volume": 1000,
        }
    )
    engineered = engineer_features(prices, [1])
    assert np.isclose(engineered.loc[0, "ForwardReturn1D"], 0.10)
    assert engineered.loc[0, "ForwardDate1D"] == dates[1]
    assert np.isnan(engineered.loc[len(engineered) - 1, "ForwardReturn1D"])


def test_graph_modes_are_symmetric_and_history_only():
    returns = np.array(
        [
            [0.01, 0.01, -0.01],
            [0.02, 0.02, -0.02],
            [-0.01, -0.01, 0.01],
        ]
    )
    sectors = ["Tech", "Tech", "Health"]
    sector = build_adjacency(returns, sectors, "sector")
    correlation = build_adjacency(returns, sectors, "correlation")
    combined = build_adjacency(returns, sectors, "combined")
    assert sector[0, 1] == 1.0 and sector[0, 2] == 0.0
    assert correlation[0, 1] > 0.99 and correlation[0, 2] == 0.0
    assert np.allclose(combined, 0.5 * sector + 0.5 * correlation)
    assert np.allclose(combined, combined.T)
    assert np.allclose(np.diag(combined), 1.0)


def test_rebalance_schedule_enters_on_first_fold_date():
    dates = pd.bdate_range("2021-01-04", "2021-04-30")
    quarterly = select_period_end_dates(dates, "quarterly")
    assert quarterly[0] == dates[0]
    assert pd.Timestamp("2021-03-31") in quarterly
    assert quarterly[-1] == pd.Timestamp("2021-04-30")


def test_prepare_fixture_records_hashes_and_builds_windows(tmp_path):
    config = load_revision_config("config/revision_smoke.yaml")
    config = copy.deepcopy(config)
    config["paths"]["prepared_dir"] = str(tmp_path / "prepared")
    manifest = prepare_data(config)
    panel, _, universes, loaded_manifest = load_prepared(config)
    assert manifest["data_hash"] == loaded_manifest["data_hash"]
    assert universes["n10"] == ["AAA", "BBB", "CCC"]
    dataset = PanelWindowDataset(
        panel,
        symbols=universes["train"],
        feature_columns=config["data"]["features"],
        macro_columns=config["data"]["macro_features"],
        target_columns=list(config["data"]["target_heads"].values()),
        start=config["folds"]["primary"]["train"]["start"],
        end=config["folds"]["primary"]["train"]["end"],
        window_size=config["data"]["window_size"],
        graph_lookback=config["data"]["graph_lookback"],
        graph_mode="combined",
        require_targets=True,
    )
    sample = dataset[0]
    assert sample["features"].shape[0] == 3
    assert sample["labels"].shape == (3, 4)
    assert sample["adj_matrix"].shape == (3, 3)
    assert sample["date"] >= config["folds"]["primary"]["train"]["start"]
    last_position = dataset.positions[-1]
    assert dataset.target_date_array[last_position].max() <= np.datetime64(
        config["folds"]["primary"]["train"]["end"]
    )
    decision_date = pd.Timestamp(sample["date"])
    changed_panel = panel.copy()
    changed_panel.loc[changed_panel["Date"] > decision_date, "Close"] *= 10.0
    changed_dataset = PanelWindowDataset(
        changed_panel,
        symbols=universes["train"],
        feature_columns=config["data"]["features"],
        macro_columns=config["data"]["macro_features"],
        target_columns=list(config["data"]["target_heads"].values()),
        start=config["folds"]["primary"]["train"]["start"],
        end=config["folds"]["primary"]["train"]["end"],
        window_size=config["data"]["window_size"],
        graph_lookback=config["data"]["graph_lookback"],
        graph_mode="combined",
        require_targets=False,
    )
    changed_sample = changed_dataset.sample_for_date(decision_date)
    assert torch.equal(sample["features"], changed_sample["features"])
    assert torch.equal(sample["adj_matrix"], changed_sample["adj_matrix"])
