from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.base_model import BaseModel
from src.models.ddpg.actor import DDPGActor
from src.models.ddpg.critic import Critic
from src.training.replay_buffer import ReplayBuffer


def project_simplex(actions: torch.Tensor) -> torch.Tensor:
    actions = torch.clamp(actions, min=1e-8)
    return actions / actions.sum(dim=-1, keepdim=True)


class TD3Agent(BaseModel):
    """Asset-agnostic TD3 using the same actor, reward, and action space as DDPG."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        raw_features = len(config["data"]["features"])
        raw_features += len(config["data"].get("macro_features", []))
        effective_features = raw_features * int(config["data"]["window_size"])
        settings = config["model"]["td3"]
        temperature = float(config["model"].get("softmax_temperature", 1.0))

        self.actor = DDPGActor(effective_features, temperature=temperature).to(self.device)
        self.actor_target = DDPGActor(effective_features, temperature=temperature).to(self.device)
        self.critic1 = Critic(effective_features, action_dim=1).to(self.device)
        self.critic2 = Critic(effective_features, action_dim=1).to(self.device)
        self.critic1_target = Critic(effective_features, action_dim=1).to(self.device)
        self.critic2_target = Critic(effective_features, action_dim=1).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=float(settings["actor_lr"]))
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=float(settings["critic_lr"]))
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=float(settings["critic_lr"]))
        self.gamma = float(settings["gamma"])
        self.tau = float(settings["tau"])
        self.batch_size = int(settings["batch_size"])
        self.target_noise = float(settings["target_noise"])
        self.noise_clip = float(settings["noise_clip"])
        self.policy_delay = int(settings["policy_delay"])
        self.buffer = ReplayBuffer(capacity=int(config["training"]["buffer_size"]))
        self.update_steps = 0

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        return self.actor(x)[0]

    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.actor(batch["features"].to(self.device))[0].cpu()

    def select_action(self, state: np.ndarray, noise_std: float = 0.1) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self.actor(tensor)[0][0].cpu().numpy()
        if noise_std > 0:
            concentration = np.clip(action / (noise_std + 1e-8), 0.1, 100.0)
            action = np.random.dirichlet(concentration)
        return action

    def update(self) -> Dict[str, float] | None:
        if len(self.buffer) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            target_action = self.actor_target(next_states)[0]
            noise = torch.randn_like(target_action).mul(self.target_noise).clamp(-self.noise_clip, self.noise_clip)
            target_action = project_simplex(target_action + noise)
            target_q = torch.minimum(
                self.critic1_target(next_states, target_action),
                self.critic2_target(next_states, target_action),
            )
            target_q = rewards + (1.0 - dones) * self.gamma * target_q

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        critic1_loss = nn.functional.mse_loss(current_q1, target_q)
        critic2_loss = nn.functional.mse_loss(current_q2, target_q)
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()

        self.update_steps += 1
        actor_loss_value = float("nan")
        if self.update_steps % self.policy_delay == 0:
            actor_actions = self.actor(states)[0]
            actor_loss = -self.critic1(states, actor_actions).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            actor_loss_value = float(actor_loss.item())
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)

        return {
            "critic1_loss": float(critic1_loss.item()),
            "critic2_loss": float(critic2_loss.item()),
            "actor_loss": actor_loss_value,
        }

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for target_parameter, parameter in zip(target.parameters(), source.parameters()):
            target_parameter.data.copy_(self.tau * parameter.data + (1.0 - self.tau) * target_parameter.data)
