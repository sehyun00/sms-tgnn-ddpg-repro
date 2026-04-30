from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from src.models.layers import GlobalPoolHead

GCN_OUT_DIM = 64
MACRO_DIM = 32
DDPG_EMB_DIM = 64


def init_ensemble(agent: Any, config: Dict[str, Any]) -> None:
    """Initialize HybridAgent alpha controls and ensemble network."""
    model_cfg = config["model"]
    agent.alpha_min = model_cfg.get("hybrid_alpha_min", 0.2)
    agent.alpha_max = model_cfg.get("hybrid_alpha_max", 0.8)
    agent.alpha_mode = model_cfg.get("hybrid_alpha_mode", "fixed")
    agent.horizon_dim = model_cfg.get("hybrid_horizon_dim", 8)

    tgnn_emb_dim = GCN_OUT_DIM + (MACRO_DIM if agent.tgnn.use_factors else 0)
    input_dim = tgnn_emb_dim + DDPG_EMB_DIM + 2
    if agent.alpha_mode == "dynamic":
        input_dim += agent.horizon_dim

    agent.horizon_embedding = nn.Embedding(4, agent.horizon_dim)
    agent.ensemble_net = GlobalPoolHead(
        input_dim=input_dim, hidden_dim=128, output_dim=1
    )
    params = list(agent.ensemble_net.parameters()) + list(
        agent.horizon_embedding.parameters()
    )
    agent.ensemble_optimizer = optim.Adam(params, lr=agent.lr_actor)
    print(
        "[INFO] Hybrid ensemble input_dim="
        f"{input_dim} (tgnn={tgnn_emb_dim}, ddpg={DDPG_EMB_DIM})"
    )


def build_ensemble_input(
    agent: Any,
    ddpg_emb: torch.Tensor,
    tgnn_emb: torch.Tensor,
    ddpg_weights: torch.Tensor,
    tgnn_weights: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    """Concatenate per-asset embeddings and weights for alpha prediction."""
    inputs = [
        ddpg_emb,
        tgnn_emb,
        ddpg_weights.unsqueeze(-1),
        tgnn_weights.unsqueeze(-1),
    ]
    if agent.alpha_mode == "dynamic":
        batch_size, num_assets = ddpg_weights.shape
        horizon_tensor = torch.tensor([horizon], device=agent.device)
        horizon_emb = agent.horizon_embedding(horizon_tensor)
        horizon_emb = horizon_emb.unsqueeze(0).expand(batch_size, num_assets, -1)
        inputs.append(horizon_emb)
    return torch.cat(inputs, dim=-1)


def mix_weights(
    agent: Any,
    ensemble_in: torch.Tensor,
    ddpg_weights: torch.Tensor,
    tgnn_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Predict alpha and mix TGNN/DDPG portfolio weights on the simplex."""
    raw_alpha = agent.ensemble_net(ensemble_in)
    alpha = torch.clamp(torch.sigmoid(raw_alpha), agent.alpha_min, agent.alpha_max)
    final_weights = alpha * tgnn_weights + (1.0 - alpha) * ddpg_weights
    final_weights = final_weights / (final_weights.sum(dim=-1, keepdim=True) + 1e-8)
    return final_weights, alpha
