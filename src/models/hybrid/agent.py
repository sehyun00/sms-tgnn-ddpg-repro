from typing import Any, Dict, Tuple

import numpy as np
import torch

from src.models.base_model import BaseModel
from src.models.ddpg.agent import DDPGAgent
from src.models.tgnn.model import TGNN

from .checkpoint import load_hybrid_weights, load_tgnn_checkpoint, save_hybrid
from .mixing import build_ensemble_input, init_ensemble, mix_weights


class HybridAgent(BaseModel):
    """Composition-based TGNN + DDPG portfolio agent."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.config = config

        ddpg_cfg = config["model"].get("ddpg", {})
        training_cfg = config["training"]
        self.gamma = ddpg_cfg.get("gamma", training_cfg.get("gamma", 0.99))
        self.tau = ddpg_cfg.get("tau", training_cfg.get("tau", 0.005))
        self.lr_actor = ddpg_cfg.get("actor_lr", training_cfg.get("lr_actor", 1e-4))
        self.lr_critic = ddpg_cfg.get("critic_lr", training_cfg.get("lr_critic", 1e-3))
        self.batch_size = ddpg_cfg.get("batch_size", training_cfg.get("batch_size", 64))
        self.temperature = config["model"].get("softmax_temperature", 1.0)

        self.ddpg = DDPGAgent(config)
        self.tgnn = TGNN(config)
        load_tgnn_checkpoint(self)
        init_ensemble(self, config)
        self.to(self.device)

    @property
    def buffer(self):
        """Expose the DDPG replay buffer used by the trainer."""
        return self.ddpg.buffer

    @property
    def actor(self):
        """Expose the DDPG actor for backtest compatibility."""
        return self.ddpg.actor

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        target_head: str = "Momentum1M",
        horizon: int = 0,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ensemble portfolio weights and the TGNN mixing alpha."""
        batch_size, num_assets = x.shape[:2]
        ddpg_weights, _ = self.ddpg.actor(x)

        x_flat = x.reshape(batch_size, num_assets, -1) if x.dim() == 4 else x
        ddpg_emb = self.ddpg.actor.encoder(x_flat)

        price_feature_count = len(self.config["data"]["features"])
        prices = x[:, :, :, :price_feature_count]
        macro = x[:, :, :, price_feature_count:]
        macro = macro if macro.shape[-1] > 0 else None

        return_attn = bool(kwargs.get("return_attn_weights", False))
        if return_attn:
            tgnn_weights, tgnn_emb, attn = self.tgnn.get_portfolio_weights(
                prices.to(self.tgnn.device),
                adj.to(self.tgnn.device),
                macro=macro.to(self.tgnn.device) if macro is not None else None,
                target_head=target_head,
                temperature=self.temperature,
                return_attn_weights=True,
            )
        else:
            tgnn_weights, tgnn_emb = self.tgnn.get_portfolio_weights(
                prices.to(self.tgnn.device),
                adj.to(self.tgnn.device),
                macro=macro.to(self.tgnn.device) if macro is not None else None,
                target_head=target_head,
                temperature=self.temperature,
            )
            attn = None

        ensemble_in = build_ensemble_input(
            self, ddpg_emb, tgnn_emb, ddpg_weights, tgnn_weights, horizon
        )
        final_weights, alpha = mix_weights(
            self, ensemble_in, ddpg_weights, tgnn_weights
        )
        if return_attn:
            return final_weights, alpha, attn
        return final_weights, alpha

    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict portfolio weights for an evaluation batch."""
        self.eval()
        with torch.no_grad():
            x = batch["features"].to(self.device)
            adj = batch["adj_matrix"].to(self.device)
            weights, _ = self.forward(x, adj)
        return weights.cpu()

    def select_action(
        self, state_feat: np.ndarray, state_adj: np.ndarray, noise_std: float = 0.1
    ) -> np.ndarray:
        """Select one portfolio action, optionally with Dirichlet exploration."""
        self.eval()
        with torch.no_grad():
            feat_tensor = torch.as_tensor(state_feat, dtype=torch.float32).unsqueeze(0)
            adj_tensor = torch.as_tensor(state_adj, dtype=torch.float32).unsqueeze(0)
            weights, _ = self.forward(feat_tensor.to(self.device), adj_tensor.to(self.device))
            action = weights.cpu().numpy()[0]

        if noise_std > 0:
            concentration = np.clip(action / (noise_std + 1e-8), 0.1, 100.0)
            action = np.random.dirichlet(concentration)
        return action

    def update(self):
        """Update DDPG internals and the ensemble alpha network."""
        ddpg_result = self.ddpg.update()
        if ddpg_result is None or len(self.buffer) < self.batch_size:
            return ddpg_result

        states, _, _, _, _ = self.buffer.sample(self.batch_size)
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        batch_size, num_assets = states.shape[:2]
        adj = torch.ones(batch_size, num_assets, num_assets, device=self.device)

        self.train()
        final_weights, alpha = self.forward(states, adj)
        q_value = self.ddpg.critic(states, final_weights)
        alpha_reg = 0.1 * ((alpha - 0.5) ** 2).mean()
        ensemble_loss = -q_value.mean() + alpha_reg

        self.ensemble_optimizer.zero_grad()
        ensemble_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ensemble_net.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.horizon_embedding.parameters(), 1.0)
        self.ensemble_optimizer.step()

        return {
            "critic_loss": ddpg_result["critic_loss"],
            "actor_loss": ddpg_result["actor_loss"],
            "ensemble_loss": ensemble_loss.item(),
            "alpha_mean": alpha.mean().item(),
        }

    def save(self, path: str) -> None:
        """Save HybridAgent trainable components."""
        save_hybrid(self, path)

    def load(self, path: str, strict: bool = True) -> None:
        """Load HybridAgent trainable components."""
        load_hybrid_weights(self, path, strict=strict)

    def get_portfolio_weights(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        target_head: str = "Momentum1M",
        horizon: int = 0,
        return_attn_weights: bool = False,
    ):
        """Backtesting helper matching the TGNN interface."""
        if return_attn_weights:
            final_weights, _, attn = self.forward(
                x, adj, target_head=target_head, horizon=horizon, return_attn_weights=True
            )
            return final_weights, attn
        final_weights, _ = self.forward(x, adj, target_head=target_head, horizon=horizon)
        return final_weights
