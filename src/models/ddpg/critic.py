import torch
import torch.nn as nn
from src.models.layers import SharedFactorEncoder, GlobalPoolHead


class Critic(nn.Module):
    """
    Asset-Agnostic Critic using Deep Sets Architecture.
    Permutation Invariant: Q(State, Action) relies on aggregate portfolio state, not fixed ordering.
    """

    def __init__(self, num_features: int, action_dim: int = 1, hidden_dim: int = 64):
        """
        Args:
            num_features: Input feature dimension per stock
            action_dim: 1 (Scalar weight per stock)
            hidden_dim: Hidden dimension for Deep Sets
        """
        super().__init__()

        # 1. State Encoder (Shared)
        self.encoder = SharedFactorEncoder(num_features, 64)

        # 2. Global Pooling Head (Deep Sets: Local -> Global -> Output)
        # Input to Pooler: 64 (Encoded State) + 1 (Action) = 65
        self.pool_net = GlobalPoolHead(
            input_dim=64 + action_dim, hidden_dim=hidden_dim, output_dim=1
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [Batch, N, T, F]
            action: [Batch, N] (Portfolio Weights)
        Returns:
            Q-Value: [Batch, 1]
        """
        batch_size = state.shape[0]

        # 1. Encode State -> [Batch, N, 64]
        if state.dim() == 4:
            x_flat = state.reshape(batch_size, state.shape[1], -1)
        else:
            x_flat = state

        enc = self.encoder(x_flat)

        # 2. Concatenate Action -> [Batch, N, 65]
        if action.dim() == 2:
            action = action.unsqueeze(-1)

        # [Batch, N, 65]
        local_input = torch.cat([enc, action], dim=-1)

        # 3. Global Pooling & Prediction -> [Batch, 1]
        q_value = self.pool_net(local_input)

        return q_value
