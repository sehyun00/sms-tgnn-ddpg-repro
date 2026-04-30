import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """
    Graph Convolutional Layer.
    Z = ReLU(D^-1/2 * A * D^-1/2 * X * W)
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        D = torch.sum(adj, dim=-1)
        D_inv_sqrt = torch.pow(D + 1e-6, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0

        norm_adj = D_inv_sqrt.unsqueeze(-1) * adj * D_inv_sqrt.unsqueeze(-2)
        support = self.linear(x)
        output = torch.matmul(norm_adj, support)
        return F.relu(output)


class TemporalAttention(nn.Module):
    """
    Multi-head Self Attention for Temporal features.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor, return_attn_weights: bool = False) -> torch.Tensor:
        # x: [batch, T, N, D] where N is number of nodes
        batch, T, N, D = x.shape
        # Permute to [batch*N, T, D] for processing time series of each node independently
        x_reshaped = x.permute(0, 2, 1, 3).reshape(batch * N, T, D)

        attn_out, attn_weights = self.attention(x_reshaped, x_reshaped, x_reshaped)
        
        # Reshape attention weights: [batch*N, T, T] -> [batch, N, T, T]
        attn_weights = attn_weights.reshape(batch, N, T, T)

        # Take the last time step: [batch*N, D] -> [batch, N, D]
        out = attn_out[:, -1, :].reshape(batch, N, D)
        
        if return_attn_weights:
            return out, attn_weights
        return out


# --- Shared Components for Asset-Agnostic Models ---


class SharedFactorEncoder(nn.Module):
    """
    Shared Factor Encoder for input features.
    Processes each stock independently using the same weights.
    Input: [Batch, N, NumFeatures] -> Output: [Batch, N, HiddenDim]
    """

    def __init__(self, num_features: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ScoreHead(nn.Module):
    """
    Scoring Network for Stock Selection (Shared Weights).
    Input: [Batch, N, HiddenDim] (Encoded State)
    Output: [Batch, N, 1] (Score)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GlobalPoolHead(nn.Module):
    """
    Global Pooling Layer to aggregate portfolio state.
    Used for Critics or Ensembles to get a fixed-size representation
    regardless of the number of stocks N.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Final projection after pooling
        self.final = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, N, InputDim]
        # 1. Transform per-stock features
        h = self.encoder(x)  # [Batch, N, HiddenDim]

        # 2. Global Pooling (Mean + Max)
        # This makes it Permutation Invariant & input-size agnostic
        mean_pool = h.mean(dim=1)  # [Batch, HiddenDim]
        # max_pool = h.max(dim=1)[0] # Optional: Combine mean & max if needed

        # 3. Final Prediction (e.g., Q-Value or Alpha)
        out = self.final(mean_pool)  # [Batch, OutputDim]
        return out


class PortfolioSoftmax(nn.Module):
    """
    Softmax layer with Temperature Scaling for Portfolio Weights.
    Formula: pi_i = exp(s_i / T) / sum(exp(s_j / T))

    Args:
        temperature (float): scalar to scale logits.
                             T > 1: Smoother distribution (High Entropy)
                             T < 1: Sharper distribution (Low Entropy)
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = max(temperature, 1e-6)  # Prevent division by zero

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: [Batch, N] unnormalized logits
        Returns:
            weights: [Batch, N] summing to 1
        """
        return F.softmax(scores / self.temperature, dim=-1)
