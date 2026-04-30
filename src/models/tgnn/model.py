import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from src.models.base_model import BaseModel
from src.models.layers import GraphConvLayer, TemporalAttention


class TGNN(BaseModel):
    """
    Context-Aware TGNN (Temporal Graph Neural Network).

    Implements separate processing paths for Local (Price) and Global (Macro) features
    to prevent signal dilution (over-smoothing) in the Graph Convolution layers.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # 1. Feature Dimension Logic
        # Calculate split dimensions based on config
        self.raw_price_features = len(config["data"]["features"])

        self.use_factors = "factors" in config["data"] and config["data"]["factors"]
        if self.use_factors:
            self.macro_features = 5  # Fama-French 5-Factor
        else:
            self.macro_features = 0

        self.num_stocks = len(config["data"]["stock_universes"])
        self.tgnn_cfg = config["model"]["tgnn"]
        self.hidden_dim = self.tgnn_cfg.get("hidden_dim", 128)
        self.num_heads = self.tgnn_cfg.get("num_heads", 8)
        self.dropout_rate = self.tgnn_cfg.get("dropout", 0.3)

        # Architecture Dimensions
        # GCN Path (Local Price Info)
        gcn_hidden_dims = [128, 128, 64]

        # Macro Path (Global Market Info)
        macro_hidden_dim = 32

        # ---------------------------------------------------------
        # Path A: Local Feature Encoder (Price -> GCN)
        # ---------------------------------------------------------
        self.input_proj = nn.Linear(self.raw_price_features, gcn_hidden_dims[0])
        self.input_ln = nn.LayerNorm(gcn_hidden_dims[0])

        self.gcn_layers = nn.ModuleList(
            [
                GraphConvLayer(gcn_hidden_dims[i], gcn_hidden_dims[i + 1])
                for i in range(len(gcn_hidden_dims) - 1)
            ]
        )

        self.gcn_lns = nn.ModuleList(
            [
                nn.LayerNorm(gcn_hidden_dims[i + 1])
                for i in range(len(gcn_hidden_dims) - 1)
            ]
        )

        # Temporal Attention for Node Embeddings
        self.temporal_attn = TemporalAttention(gcn_hidden_dims[-1], self.num_heads)

        # ---------------------------------------------------------
        # Path B: Global Context Encoder (Macro -> MLP)
        # ---------------------------------------------------------
        if self.use_factors:
            self.macro_encoder = nn.Sequential(
                nn.Linear(self.macro_features, 64),
                nn.ReLU(),
                nn.Linear(64, macro_hidden_dim),
                nn.ReLU(),
            )
        else:
            self.macro_encoder = None
            macro_hidden_dim = 0

        # ---------------------------------------------------------
        # Path C: Fusion & Prediction
        # ---------------------------------------------------------
        # Input to predictor = Node Embedding + Global Context
        fusion_dim = gcn_hidden_dims[-1] + macro_hidden_dim

        self.heads = ["Momentum1M", "Momentum3M", "Momentum6M", "Momentum12M"]
        self.predictors = nn.ModuleDict(
            {head: self._make_predictor(fusion_dim) for head in self.heads}
        )

        self.to(self.device)

    def _make_predictor(self, input_dim: int) -> nn.Sequential:
        """
        Predictor Head
        """
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        macro: torch.Tensor = None,
        target_type: str = "Momentum1M",
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Context-Aware Forward Pass.

        Args:
            x: [Batch, N, T, F_price] - Local Price Features
            adj: [Batch, N, N] - Adjacency Matrix
            macro: [Batch, N, T, F_macro] - Global Macro Features (Optional)
            target_type: Prediction Head
        """
        batch, N, T, F = x.shape

        # ----------------------
        # Path A: Local (GCN)
        # ----------------------
        gcn_outputs = []
        for t in range(T):
            x_t = x[:, :, t, :]  # [B, N, F]
            h = self.input_proj(x_t)
            h = self.input_ln(h)

            # GCN + Residual
            for gcn, ln in zip(self.gcn_layers, self.gcn_lns):
                h_new = gcn(h, adj)
                h_new = ln(h_new)

                if h.shape[-1] == h_new.shape[-1]:
                    h = h + h_new
                else:
                    h = h_new

            gcn_outputs.append(h)

        # Temporal Attention -> Node Embeddings
        # Stack: [B, T, N, D]
        temporal_features = torch.stack(gcn_outputs, dim=1)
        
        # Check if we need attention weights (for XAI)
        return_attn = kwargs.get("return_attn_weights", False)
        if return_attn:
            node_embeddings, attn_weights = self.temporal_attn(temporal_features, return_attn_weights=True)
        else:
            node_embeddings = self.temporal_attn(temporal_features)
            attn_weights = None

        # ----------------------
        # Path B: Global (Macro)
        # ----------------------
        global_context_emb = None
        if self.use_factors and macro is not None:
            # Macro is [B, N, T, F_macro] -> We want to summarize T
            # Since Macro is same for all N, we can take mean over N if we want,
            # but usually it's cleaner to process per-node to keep tensor structure simple.

            # Simple aggregation over Time: Mean or Last?
            # Let's use Last Step Macro for immediate regime context
            macro_last = macro[:, :, -1, :]  # [B, N, F_macro]
            global_context_emb = self.macro_encoder(macro_last)  # [B, N, D_macro]
        elif self.use_factors and macro is None:
            # Fallback if macro expected but not provided (shouldn't happen with correct trainer)
            batch_size, num_nodes = node_embeddings.shape[:2]
            global_context_emb = torch.zeros(batch_size, num_nodes, 32).to(self.device)

        # ----------------------
        # Path C: Fusion
        # ----------------------
        if global_context_emb is not None:
            # Concatenate [Node, Macro]
            combined_embedding = torch.cat(
                [node_embeddings, global_context_emb], dim=-1
            )
        else:
            combined_embedding = node_embeddings

        if target_type in self.predictors:
            predictions = self.predictors[target_type](combined_embedding).squeeze(-1)
        else:
            predictions = self.predictors["Momentum1M"](combined_embedding).squeeze(-1)

        if return_attn:
            return predictions, combined_embedding, attn_weights
        return predictions, combined_embedding

    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            x = (
                batch["prices"].to(self.device)
                if "prices" in batch
                else batch["features"].to(self.device)
            )
            macro = batch["macro"].to(self.device) if "macro" in batch else None
            adj = batch["adj_matrix"].to(self.device)
            preds, _ = self.forward(x, adj, macro=macro, target_type="Momentum1M")
        return preds.cpu()

    def get_portfolio_weights(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        macro: torch.Tensor = None,
        target_head: str = "Momentum1M",
        temperature: float = 1.0,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        import torch.nn.functional as F

        if return_attn_weights:
            scores, embeddings, attn_weights = self.forward(
                x, adj, macro=macro, target_type=target_head, return_attn_weights=True
            )
            weights = F.softmax(scores / temperature, dim=-1)
            return weights, embeddings, attn_weights
        else:
            scores, embeddings = self.forward(
                x, adj, macro=macro, target_type=target_head
            )
            weights = F.softmax(scores / temperature, dim=-1)
            return weights, embeddings
