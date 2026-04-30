import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from src.models.layers import SharedFactorEncoder, ScoreHead, PortfolioSoftmax


class Actor(nn.Module):
    """
    Asset-Agnostic DDPG Actor: State -> Action (Portfolio Weights)
    Uses a Shared Network (Scoring Function) to handle variable number of stocks.
    """

    def __init__(
        self, num_features: int, hidden_dim: int = 64, temperature: float = 1.0
    ):
        """
        Args:
            num_features: Input feature dimension per stock (T * F)
            hidden_dim: Hidden dimension for scoring network
            temperature: Softmax temperature scaling (default: 1.0)
        """
        super().__init__()
        self.num_features = num_features

        # Shared Encoder per Stock
        # Input: [Batch, N, num_features] -> Output: [Batch, N, 64]
        self.encoder = SharedFactorEncoder(num_features, 64)

        # Scoring Network (Shared Weights across Stocks)
        # Input: [Batch, N, 64] -> Output: [Batch, N, 1]
        self.score_net = ScoreHead(64, hidden_dim)

        # Softmax with Temperature
        self.softmax_layer = PortfolioSoftmax(temperature)

        # Constraints
        self.min_weight = 0.0
        self.max_weight = 1.0

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class DDPGActor(Actor):
    # Wrapper to handle input shape adaptation
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Asset-Agnostic Actor.

        Args:
            x: [Batch, N, T, F] or [Batch, N, T*F] - Input State

        Returns:
            weights: [Batch, N] - Portfolio weights summing to 1
            entropy: [Batch] - Entropy for regularization
        """
        batch_size = x.shape[0]
        # X shape handling: [Batch, N, T, F] -> [Batch, N, -1]
        if x.dim() == 4:
            x_flat = x.reshape(batch_size, x.shape[1], -1)
        else:
            x_flat = x  # [Batch, N, Feature]

        # 1. Encode Features (Shared) -> [Batch, N, 64]
        enc = self.encoder(x_flat)

        # 2. Score Each Stock (Shared) -> [Batch, N, 1]
        scores = self.score_net(enc).squeeze(-1)  # [Batch, N]

        # 3. Portfolio Construction (Softmax over N stocks)
        # This handles variable N automatically.
        weights = self.softmax_layer(scores)

        # 4. Constraints & Normalization (Simplex)
        weights = torch.clamp(weights, min=self.min_weight, max=self.max_weight)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # 5. Entropy (for Exploration Bonus)
        entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1)

        return weights, entropy
