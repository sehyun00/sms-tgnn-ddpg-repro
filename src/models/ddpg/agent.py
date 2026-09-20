import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Any, Dict
from src.models.base_model import BaseModel
from .actor import DDPGActor
from .critic import Critic
from src.training.replay_buffer import ReplayBuffer


class DDPGAgent(BaseModel):
    """
    DDPG Agent Wrapper with Full RL Logic (Actor-Critic).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.num_stocks = len(config["data"]["stock_universes"])

        # Determine effective Num Features (F * T)
        raw_features = len(config["data"]["features"])
        # Fama-French 5-Factor: Mkt_RF, SMB, HML, RMW, CMA
        FAMA_FRENCH_FACTORS = 5
        if config["data"].get("factors", False):
            raw_features += FAMA_FRENCH_FACTORS

        window_size = config["data"]["window_size"]
        self.effective_num_features = raw_features * window_size

        # Hyperparameters
        # Hyperparameters (Prioritize model.ddpg config)
        ddpg_conf = config["model"].get("ddpg", {})
        training_conf = config["training"]

        self.gamma = ddpg_conf.get("gamma", training_conf.get("gamma", 0.99))
        self.tau = ddpg_conf.get("tau", training_conf.get("tau", 0.005))
        self.lr_actor = ddpg_conf.get("actor_lr", training_conf.get("lr_actor", 1e-4))
        self.lr_critic = ddpg_conf.get(
            "critic_lr", training_conf.get("lr_critic", 1e-3)
        )
        self.batch_size = ddpg_conf.get(
            "batch_size", training_conf.get("batch_size", 64)
        )
        self.temperature = config["model"].get("softmax_temperature", 1.0)
        self.entropy_coefficient = float(ddpg_conf.get("entropy_coefficient", 0.01))

        # Networks
        # Networks
        # Asset-Agnostic: No num_stocks dependency in architecture
        self.actor = DDPGActor(
            self.effective_num_features, temperature=self.temperature
        )
        self.critic = Critic(
            self.effective_num_features, action_dim=1
        )  # action_dim per stock is 1

        # Target Networks
        self.actor_target = DDPGActor(
            self.effective_num_features, temperature=self.temperature
        )
        self.critic_target = Critic(self.effective_num_features, action_dim=1)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.to(self.device)
        self.actor_target.to(self.device)
        self.critic_target.to(self.device)

        # Optimizers (Internal management for RL loop)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        # Replay Buffer (config에서 크기 읽기)
        buffer_size = config["training"].get("buffer_size", 10000)
        self.buffer = ReplayBuffer(capacity=buffer_size)

    def forward(self, x: torch.Tensor, adj: torch.Tensor = None) -> torch.Tensor:
        # Return prediction for compatibility
        # x is [B, N, T, F]
        weights, _ = self.actor(x)
        return weights

    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.actor.eval()
        with torch.no_grad():
            x = batch["features"].to(self.device)  # [B, N, T, F]
            # adj not used in DDPG
            weights, _ = self.actor(x)
        return weights.cpu()

    def select_action(self, state: np.ndarray, noise_std: float = 0.1) -> np.ndarray:
        """
        Select action with Dirichlet noise for exploration.
        state: [N, T, F] (Single sample)
        """
        self.actor.eval()
        with torch.no_grad():
            state_tensor = (
                torch.FloatTensor(state).unsqueeze(0).to(self.device)
            )  # [1, N, T, F]
            weights, _ = self.actor(state_tensor)
            action = weights.cpu().numpy()[0]  # [N]

        if noise_std > 0:
            # Dirichlet Noise for Simplex Constraint
            # concentration = action / (noise^2) roughly
            concentration = action / (noise_std + 1e-8)
            concentration = np.clip(concentration, 0.1, 100.0)
            action = np.random.dirichlet(concentration)

        return action

    def update(self):
        """
        Perform one step of Actor-Critic update using Replay Buffer.
        """
        if len(self.buffer) < self.batch_size:
            return None

        # Sample Batch
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        states = torch.FloatTensor(states).to(self.device)  # [B, N, T, F]
        actions = torch.FloatTensor(actions).to(self.device)  # [B, N]
        rewards = torch.FloatTensor(rewards).to(self.device)  # [B, 1]
        next_states = torch.FloatTensor(next_states).to(self.device)  # [B, N, T, F]
        dones = torch.FloatTensor(dones).to(self.device)  # [B, 1]

        # ----------------------
        # Critic Update
        # ----------------------
        with torch.no_grad():
            next_actions, _ = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # ----------------------
        # Actor Update
        # ----------------------
        pred_actions, pred_entropy = self.actor(states)
        # We want to MAXIMIZE Q(s, a), so Minimize -Q
        actor_loss = -self.critic(states, pred_actions).mean()

        # Entropy Regularization (Optional, from legacy)
        entropy_bonus = self.entropy_coefficient * pred_entropy.mean()
        total_actor_loss = actor_loss - entropy_bonus

        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # ----------------------
        # Target Soft Update
        # ----------------------
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": total_actor_loss.item(),
        }

    def _soft_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
