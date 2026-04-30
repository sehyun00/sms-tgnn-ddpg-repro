import os
import sys

import torch
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.ddpg import DDPGAgent
from src.models.hybrid import HybridAgent
from src.models.tgnn import TGNN


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "../config/sample.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["project"]["device"] = "cpu"
    return config


def feature_count(config):
    macro = 5 if config["data"].get("factors", False) else 0
    return len(config["data"]["features"]) + macro


def test_tgnn_forward_pass():
    torch.manual_seed(42)
    config = load_config()
    model = TGNN(config)

    batch_size = 2
    num_stocks = len(config["data"]["stock_universes"])
    seq_len = config["data"]["window_size"]
    num_price_features = len(config["data"]["features"])

    prices = torch.randn(batch_size, num_stocks, seq_len, num_price_features)
    macro = torch.randn(batch_size, num_stocks, seq_len, 5)
    adj = torch.ones(batch_size, num_stocks, num_stocks)

    preds, embeddings = model(prices, adj, macro=macro)

    assert preds.shape == (batch_size, num_stocks)
    assert embeddings.shape == (batch_size, num_stocks, 96)


def test_hybrid_forward_pass():
    torch.manual_seed(42)
    config = load_config()
    model = HybridAgent(config)

    batch_size = 2
    num_stocks = len(config["data"]["stock_universes"])
    seq_len = config["data"]["window_size"]
    x = torch.randn(batch_size, num_stocks, seq_len, feature_count(config))
    adj = torch.ones(batch_size, num_stocks, num_stocks)

    weights = model.predict({"features": x, "adj_matrix": adj})

    assert weights.shape == (batch_size, num_stocks)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size), atol=1e-5)


def test_ddpg_forward_pass():
    torch.manual_seed(42)
    config = load_config()
    model = DDPGAgent(config)

    batch_size = 2
    num_stocks = len(config["data"]["stock_universes"])
    seq_len = config["data"]["window_size"]
    x = torch.randn(batch_size, num_stocks, seq_len, feature_count(config))

    weights = model.predict({"features": x})

    assert weights.shape == (batch_size, num_stocks)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size), atol=1e-5)
