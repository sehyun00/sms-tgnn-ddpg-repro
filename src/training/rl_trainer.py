import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional

from .replay_buffer import ReplayBuffer
from .environment import PortfolioEnvironment


class RLTrainer:
    """
    Reinforcement Learning Trainer (DDPG/Hybrid).
    Handles the interaction between Agent and Environment,
    and performs Actor-Critic updates.
    """

    def __init__(self, agent, env: PortfolioEnvironment, config: Dict[str, Any]):
        self.agent = agent
        self.env = env
        self.config = config
        self.device = agent.device

        # Hyperparameters
        self.gamma = config["training"].get("gamma", 0.99)
        self.tau = config["training"].get("tau", 0.005)  # Soft update
        self.batch_size = config["training"].get("batch_size", 64)
        self.lr_actor = config["training"].get("lr_actor", 1e-4)
        self.lr_critic = config["training"].get("lr_critic", 1e-3)
        self.max_grad_norm = config["training"].get("max_grad_norm", 1.0)

        # Buffer
        self.buffer = ReplayBuffer(capacity=10000)

        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.agent.actor.parameters(), lr=self.lr_actor
        )
        self.critic_optimizer = optim.Adam(
            self.agent.critic.parameters(), lr=self.lr_critic
        )

        # Logging
        # Organize results by model name
        import os

        model_name = config["project"].get("selected_model", "default_rl")
        self.results_dir = os.path.join(config["paths"]["results_dir"], model_name)

        from datetime import datetime
        import logging

        os.makedirs(self.results_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = self._setup_logger()

    def _setup_logger(self):
        import logging  # Re-import or ensure global
        import os

        logger = logging.getLogger(f"RLTrainer_{self.timestamp}")
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(
            os.path.join(self.results_dir, f"train_rl_{self.timestamp}.log")
        )
        ch = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

    def save_checkpoint(self, name):
        import os

        save_path = os.path.join(self.results_dir, f"{name}_{self.timestamp}.pth")
        torch.save(
            {
                "actor": self.agent.actor.state_dict(),
                "critic": self.agent.critic.state_dict(),
                "config": self.config,
            },
            save_path,
        )
        self.logger.info(f"Checkpoint saved: {save_path}")

    def train_episode(self, noise_std: float = 0.1) -> Dict[str, float]:
        state = self.env.reset()
        episode_reward = 0
        episode_steps = 0

        # Tracking losses
        critic_losses = []
        actor_losses = []

        while True:
            # 1. Select Action (with exploration noise)
            feat, adj = self._process_state(state)

            self.agent.actor.eval()
            with torch.no_grad():
                if hasattr(self.agent, "actor"):
                    # Handle return signature difference (weights, aux)
                    # Handle argument difference (Hybrid: (x, adj), DDPG: (x))
                    try:
                        if adj is not None:
                            action_out = self.agent.actor(feat, adj)
                        else:
                            action_out = self.agent.actor(feat)
                    except TypeError:
                        # Fallback for DDPG if it doesn't accept adj but adj is present
                        action_out = self.agent.actor(feat)

                    if isinstance(action_out, tuple):
                        action = action_out[0]
                    else:
                        action = action_out
                else:
                    raise ValueError("Agent must have an 'actor' module.")

            action = action.cpu().numpy()[0]  # Remove batch dim

            # Add Noise
            if noise_std > 0:
                noise = np.random.normal(0, noise_std, size=action.shape)
                action = np.clip(action + noise, 0, 1)
                action = action / (action.sum() + 1e-8)  # Re-normalize

            # 2. Step Environment
            next_state, reward, done, info = self.env.step(action)

            # 3. Store in Buffer
            self.buffer.push(state, action, reward, next_state, done)

            # 4. Update Network (if buffer has enough samples)
            if len(self.buffer) > self.batch_size:
                c_loss, a_loss = self._update_parameters()
                critic_losses.append(c_loss)
                actor_losses.append(a_loss)

            state = next_state
            episode_reward += reward
            episode_steps += 1

            if done:
                break

        return {
            "episode_reward": episode_reward,
            "avg_critic_loss": np.mean(critic_losses) if critic_losses else 0.0,
            "avg_actor_loss": np.mean(actor_losses) if actor_losses else 0.0,
            "epsilon": noise_std,
        }

    def _process_state(self, state_dict: Dict[str, Any]) -> torch.Tensor:
        """
        Convert state dict to tensor input for Actor.
        Returns: (features, adj) tuple
        """
        features = (
            torch.FloatTensor(state_dict["features"]).unsqueeze(0).to(self.device)
        )  # [1, N, T, F]

        adj = None
        if "adj_matrix" in state_dict:
            adj = (
                torch.FloatTensor(state_dict["adj_matrix"]).unsqueeze(0).to(self.device)
            )  # [1, N, N]

        return features, adj

    def _update_parameters(self):
        self.agent.actor.train()
        self.agent.critic.train()

        # Sample Batch
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        s_feat, s_adj = self._batch_stack_states(states)
        ns_feat, ns_adj = self._batch_stack_states(next_states)

        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # --- Critic Update ---
        with torch.no_grad():
            # Target Action
            try:
                if ns_adj is not None:
                    next_actions = self.agent.actor_target(ns_feat, ns_adj)
                else:
                    next_actions = self.agent.actor_target(ns_feat)
            except TypeError:
                next_actions = self.agent.actor_target(ns_feat)

            if isinstance(next_actions, tuple):
                next_actions = next_actions[0]

            flat_next_state = self._flatten_state(ns_feat, ns_adj)

            # Target Q
            target_q = self.agent.critic_target(flat_next_state, next_actions)
            target_y = rewards + (1 - dones) * self.gamma * target_q

        # Current Q
        flat_state = self._flatten_state(s_feat, s_adj)
        current_q = self.agent.critic(flat_state, actions)
        critic_loss = nn.MSELoss()(current_q, target_y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.agent.critic.parameters(), self.max_grad_norm
        )
        self.critic_optimizer.step()

        # --- Actor Update ---
        try:
            if s_adj is not None:
                pred_actions = self.agent.actor(s_feat, s_adj)
            else:
                pred_actions = self.agent.actor(s_feat)
        except TypeError:
            pred_actions = self.agent.actor(s_feat)

        if isinstance(pred_actions, tuple):
            pred_actions = pred_actions[0]

        # Q(state, actor(state))
        actor_loss = -self.agent.critic(flat_state, pred_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.agent.actor.parameters(), self.max_grad_norm
        )
        self.actor_optimizer.step()

        # --- Soft Update ---
        self._soft_update(self.agent.actor, self.agent.actor_target)
        self._soft_update(self.agent.critic, self.agent.critic_target)

        return critic_loss.item(), actor_loss.item()

    def _batch_stack_states(self, state_batch):
        """Stack list of state dicts into batches"""
        # DEBUG
        if len(state_batch) > 0:
            s0 = state_batch[0]
            # print(f"DEBUG: state_batch type: {type(state_batch)}, elem type: {type(s0)}")
            if not isinstance(s0, dict):
                # print(f"CRITICAL ERROR: State is not dict! Type: {type(s0)}")
                # Convert if it's a numpy object wrapping a dict
                if isinstance(s0, np.ndarray) and s0.shape == ():
                    state_batch = [s.item() for s in state_batch]

        features = torch.FloatTensor(np.array([s["features"] for s in state_batch])).to(
            self.device
        )

        adj = None
        if "adj_matrix" in state_batch[0]:
            adj = torch.FloatTensor(
                np.array([s["adj_matrix"] for s in state_batch])
            ).to(self.device)

        return features, adj

    def _flatten_state(self, features, adj):
        """Flatten state for Critic input"""
        batch = features.shape[0]
        flat_feat = features.reshape(batch, -1)

        agent_name = self.agent.__class__.__name__
        if "Hybrid" in agent_name:
            if adj is not None:
                flat_adj = adj.reshape(batch, -1)
                return torch.cat([flat_feat, flat_adj], dim=-1)

        return flat_feat

    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )
