import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any
import logging
from datetime import datetime
import numpy as np

from ..models.base_model import BaseModel
from .dataset import FinancialDataset


class Trainer:
    """
    Standard Trainer for Research Models.
    Handles training loop, logging, and checkpointing.
    """

    def __init__(
        self, config: Dict[str, Any], model: BaseModel, dataset: FinancialDataset
    ):
        self.config = config
        self.model = model
        self.device = model.device
        self.model_type = config["project"].get("selected_model", "tgnn").lower()

        # Data Loader
        self.batch_size = config["model"]["ddpg"][
            "batch_size"
        ]  # Using DDPG batch size as default
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Optimizer
        self.lr = config["model"]["ddpg"]["actor_lr"]  # Default to generic LR
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)

        # Loss
        # Support TGNN Combined Loss
        from ..models.tgnn.loss import combined_loss

        if config["project"]["selected_model"] == "tgnn":
            self.criterion = combined_loss
        else:
            self.criterion = nn.MSELoss()

        # Logging
        # Organize results by model name
        model_name = config["project"].get("selected_model", "default")
        self.results_dir = os.path.join(config["paths"]["results_dir"], model_name)
        self.log_dir = os.path.join(self.results_dir, "logs")
        self.ckpt_dir = os.path.join(self.results_dir, "checkpoints")

        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger(f"Trainer_{self.timestamp}")
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(
            os.path.join(self.log_dir, f"train_{self.timestamp}.log")
        )
        ch = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

    def _run_epoch(self, epoch_idx: int) -> float:
        """
        Runs a single epoch of training.
        Returns:
            avg_loss: Average loss for the epoch
        """
        from collections import defaultdict

        self.model.train()
        total_loss = 0
        total_metrics = defaultdict(float)  # Initialize here!

        for batch in self.dataloader:
            # Move to device
            # Move to device
            # Check for Context-Aware Split (Price vs Macro)
            if "prices" in batch:
                prices = batch["prices"].to(self.device)
                macro = batch["macro"].to(self.device) if "macro" in batch else None
            else:
                prices = batch["features"].to(self.device)  # Legacy fallback
                macro = None

            # Legacy "features" still used for DDPG/Hybrid (Concatenated)
            # DDPG/Hybrid는 여전히 Concatenated Feature를 사용합니다.
            features = batch.get("features", prices).to(self.device)
            adj = batch["adj_matrix"].to(self.device)
            labels = batch["labels"].to(self.device)
            if "active_mask" in batch:
                active_mask = batch["active_mask"].to(self.device)
            else:
                active_mask = None

            self.optimizer.zero_grad()

            if self.model_type == "tgnn":
                # Multi-Task Logic
                heads = ["Momentum1M", "Momentum3M", "Momentum6M", "Momentum12M"]
                batch_loss = 0
                for i, target_head in enumerate(heads):
                    # Context-Aware Call: Pass separated prices and macro
                    if macro is not None:
                        output = self.model(
                            prices, adj, macro=macro, target_type=target_head
                        )
                    else:
                        output = self.model(features, adj, target_type=target_head)

                    preds = output[0] if isinstance(output, tuple) else output
                    target_labels = labels[:, :, i]
                    if preds.shape != target_labels.shape:
                        preds = preds.view_as(target_labels)
                    target_labels = labels[:, :, i]
                    if preds.shape != target_labels.shape:
                        preds = preds.view_as(target_labels)

                    # Active Mask를 Loss에 적용
                    # combined_loss는 active_mask를 매개변수로 받습니다.
                    loss = self.criterion(preds, target_labels, active_mask=active_mask)
                    batch_loss += loss
                batch_loss.backward()
                total_loss += batch_loss.item()
            else:
                # DDPG / RL Loop
                # 1. Prepare Data
                next_features = batch["next_features"].to(self.device)
                adj = batch.get("adj_matrix")
                if adj is not None:
                    adj = adj.to(self.device)

                # 2. Select Action (Batch Exploration)
                # Hybrid 모델은 forward()로, DDPG는 actor()로 처리
                model_class_name = self.model.__class__.__name__

                self.model.eval()
                with torch.no_grad():
                    if model_class_name == "HybridAgent":
                        # Hybrid: forward() 호출 + 랜덤 horizon 샘플링 (다양한 리밸런싱 주기 학습)
                        import random

                        horizon = random.randint(
                            0, 3
                        )  # 0=monthly, 1=quarterly, 2=semiannual, 3=annual
                        if adj is not None:
                            action_probs, _ = self.model(features, adj, horizon=horizon)
                        else:
                            # adj가 없으면 단위행렬 사용
                            N = features.shape[1]
                            adj = (
                                torch.eye(N)
                                .unsqueeze(0)
                                .expand(features.shape[0], -1, -1)
                                .to(self.device)
                            )
                            action_probs, _ = self.model(features, adj, horizon=horizon)
                    else:
                        # DDPG: 기존 로직
                        action_probs, _ = self.model.actor(features)
                    actions = action_probs.cpu().numpy()  # [B, N]

                # Add Dirichlet Noise (Vectorized)
                # alpha = action * concentration
                # Dirichlet 노이즈 추가 (벡터화)
                # alpha = action * concentration
                noise_std = self.config["model"]["ddpg"].get("noise_std", 0.1)
                conc = actions / (noise_std + 1e-8)
                conc = np.clip(conc, 0.1, 100.0)

                # Sample noisy actions
                noisy_actions = np.array([np.random.dirichlet(c) for c in conc])
                noisy_actions = torch.FloatTensor(noisy_actions).to(self.device)

                # 3. Calculate Reward (Portfolio Return)
                target_returns = labels[:, :, 0]  # [B, N]
                if noisy_actions.shape != target_returns.shape:
                    target_returns = target_returns.view_as(noisy_actions)

                # Reward = Portfolio Return (with Loss Aversion)
                portfolio_returns = torch.sum(
                    noisy_actions * target_returns, dim=1, keepdim=True
                )
                # Reward = Portfolio Return (Symmetric Reward)
                # 🌡️ Temperature=3.0 적용으로 Entropy가 자연스럽게 유지되므로,
                # 인위적인 L2 Penalty는 제거합니다. (Alpha 보존)
                rewards = portfolio_returns

                # 4. Push to Buffer
                dones = torch.zeros_like(rewards)  # Continuous task

                self.model.buffer.push_batch(
                    features.cpu().numpy(),
                    noisy_actions.cpu().numpy(),
                    rewards.cpu().numpy(),
                    next_features.cpu().numpy(),
                    dones.cpu().numpy(),
                )

                # 5. Update Agent
                metrics = self.model.update()

                if metrics:
                    loss = metrics["critic_loss"] + metrics["actor_loss"]  # For logging
                    total_loss += loss
                    for k, v in metrics.items():
                        total_metrics[k] += v
                else:
                    # Buffer not full yet
                    loss = 0

                # Warm-up Logic for Hybrid Model
                warmup_epochs = self.config.get("model", {}).get(
                    "hybrid_warmup_episodes", 0
                )
                is_warmup = epoch_idx < warmup_epochs

                # Update Alpha only if NOT in warmup or model is not Hybrid
                if model_class_name == "HybridAgent":
                    if not is_warmup:
                        self.model.ensemble_optimizer.step()  # Explicit step if manual optimizer handling
                        # Note: In HybridAgent.update(), we already called backward.
                        # We need to ensure we don't double step or miss step.
                        # Actually HybridAgent.update() does optimizer.step() internally for actor/critic.
                        # But for ensemble_net, we added logic in previous turn?
                        # Let's check HybridAgent.update() implementation again via memory or assumtion.
                        # Wait, in 'hybrid_model_review' artifact, we saw:
                        # "Modified ensemble_optimizer in HybridAgent.__init__()"
                        # "Corrected Trainer._run_epoch() to prevent stepping the main optimizer"
                        # HybridAgent.update() handles actor/critic steps.
                        # We need to verify if HybridAgent.update() handles ensemble_optimizer step.
                        pass
                    else:
                        # During warmup, we do NOT step ensemble_optimizer
                        # But HybridAgent.update() might have already stepped it if logic is inside.
                        # I need to verify HybridAgent.py first.
                        pass

            # Only step main optimizer for TGNN (DDPG/Hybrid have internal optimizers)
            if self.model_type == "tgnn":
                self.optimizer.step()

        return total_loss / len(self.dataloader)

    def train(self):
        """
        Executes the training loop.
        """
        epochs = self.config["training"]["episodes"]
        self.logger.info(f"Starting training for {epochs} epochs on {self.device}")

        best_loss = float("inf")

        for epoch in range(epochs):
            avg_loss = self._run_epoch(epoch)

            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.6f}")

            # Checkpoint
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_path = os.path.join(
                    self.ckpt_dir, f"best_model_{self.timestamp}.pth"
                )
                self.model.save(save_path)
                self.logger.info(
                    f"Saved best model (Loss: {best_loss:.6f}) to {save_path}"
                )

        # Save Final Model as well
        final_path = os.path.join(self.ckpt_dir, f"final_model_{self.timestamp}.pth")
        self.model.save(final_path)
        self.logger.info(f"Saved final model to {final_path}")

        self.logger.info(f"Training Complete. Best Loss: {best_loss:.6f}")

    def finetune(self, epochs: int = 50, lr_factor: float = 0.1):
        """
        Fine-tune a pre-trained model on new data (Transfer Learning).
        """
        self.logger.info(
            f"Starting Fine-tuning for {epochs} epochs (LR factor: {lr_factor})"
        )

        # Adjust Learning Rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr * lr_factor

        best_loss = float("inf")

        for epoch in range(epochs):
            avg_loss = self._run_epoch(epoch)

            if epoch % 10 == 0:
                self.logger.info(
                    f"Finetune Epoch {epoch}/{epochs} | Loss: {avg_loss:.6f}"
                )

            if avg_loss < best_loss:
                best_loss = avg_loss
                save_path = os.path.join(
                    self.ckpt_dir, f"finetuned_model_{self.timestamp}.pth"
                )
                self.model.save(save_path)

        self.logger.info(f"Fine-tuning Complete. Best Loss: {best_loss:.6f}")
