from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class AlphaMLP(nn.Module):
    """State-only gate; no frequency or horizon identifier is accepted."""

    def __init__(self, input_dim: int, hidden_dim: int, minimum: float = 0.1, maximum: float = 0.9):
        super().__init__()
        if not 0.0 <= minimum < maximum <= 1.0:
            raise ValueError("Alpha bounds must satisfy 0 <= minimum < maximum <= 1")
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max(8, hidden_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(8, hidden_dim // 2), 1),
        )

    def logits(self, state_summary: torch.Tensor) -> torch.Tensor:
        return self.network(state_summary)

    def forward(self, state_summary: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(self.logits(state_summary))
        return self.minimum + (self.maximum - self.minimum) * probability


def normalize_summaries(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    scale = np.where(np.asarray(scale) > 1e-8, scale, 1.0)
    normalized = (np.asarray(values) - np.asarray(mean)) / scale
    if not np.isfinite(normalized).all():
        raise RuntimeError("Alpha state normalization produced non-finite values")
    return normalized.astype(np.float32)


def mix_portfolios(tgnn: torch.Tensor, ddpg: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    if alpha.ndim == 1:
        alpha = alpha.unsqueeze(-1)
    weights = alpha * tgnn + (1.0 - alpha) * ddpg
    weights = torch.clamp(weights, min=0.0)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False


def assert_frozen(*modules: nn.Module) -> None:
    trainable = [name for module in modules for name, parameter in module.named_parameters() if parameter.requires_grad]
    if trainable:
        raise RuntimeError(f"Base branches must be frozen before alpha calibration: {trainable[:5]}")
