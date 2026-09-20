from __future__ import annotations

import inspect

import numpy as np
import torch

from src.models.td3 import TD3Agent
from src.models.tgnn import TGNN
from src.revision.config import load_revision_config
from src.revision.models import AlphaMLP, assert_frozen, freeze_module, mix_portfolios


def test_alpha_is_bounded_state_only_and_branches_can_be_frozen():
    model = AlphaMLP(input_dim=7, hidden_dim=16, minimum=0.1, maximum=0.9)
    values = model(torch.randn(8, 7))
    assert torch.all(values >= 0.1) and torch.all(values <= 0.9)
    assert list(inspect.signature(model.forward).parameters) == ["state_summary"]
    branch = torch.nn.Linear(3, 2)
    freeze_module(branch)
    assert_frozen(branch)
    assert not branch.training


def test_hybrid_mixing_stays_on_simplex():
    tgnn = torch.tensor([[0.8, 0.2]])
    ddpg = torch.tensor([[0.1, 0.9]])
    mixed = mix_portfolios(tgnn, ddpg, torch.tensor([0.25]))
    assert torch.allclose(mixed.sum(dim=1), torch.ones(1))
    assert np.isclose(mixed[0, 0].item(), 0.275)


def test_td3_uses_asset_agnostic_simplex_actions():
    config = load_revision_config("config/revision_smoke.yaml")
    config["data"]["stock_universes"] = ["AAA", "BBB", "CCC"]
    config["data"]["factors"] = True
    agent = TD3Agent(config)
    feature_count = len(config["data"]["features"]) + len(config["data"]["macro_features"])
    states = np.random.default_rng(42).normal(
        size=(2, 3, config["data"]["window_size"], feature_count)
    ).astype(np.float32)
    actions = np.full((2, 3), 1.0 / 3.0, dtype=np.float32)
    agent.buffer.push_batch(states, actions, np.array([0.01, 0.02]), states, np.array([0.0, 1.0]))
    metrics = agent.update()
    assert metrics is not None
    predicted = agent.predict({"features": torch.from_numpy(states)})
    assert predicted.shape == (2, 3)
    assert torch.allclose(predicted.sum(dim=1), torch.ones(2), atol=1e-5)


def test_tgnn_preserves_attention_heads_for_xai():
    config = load_revision_config("config/revision_smoke.yaml")
    config["data"]["stock_universes"] = ["AAA", "BBB", "CCC"]
    config["data"]["factors"] = True
    model = TGNN(config)
    batch, assets, window = 1, 3, config["data"]["window_size"]
    prices = torch.randn(batch, assets, window, len(config["data"]["features"]))
    macro = torch.randn(batch, assets, window, len(config["data"]["macro_features"]))
    adjacency = torch.eye(assets).unsqueeze(0)
    _, _, attention = model.get_portfolio_weights(
        prices, adjacency, macro=macro, target_head="Return1D", return_attn_weights=True
    )
    assert attention.shape == (
        batch,
        assets,
        config["model"]["tgnn"]["num_heads"],
        window,
        window,
    )
