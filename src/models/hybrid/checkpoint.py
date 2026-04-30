import glob
import os
from typing import Any

import torch


def _freeze_tgnn(agent: Any) -> None:
    agent.tgnn.eval()
    for param in agent.tgnn.parameters():
        param.requires_grad = False


def load_tgnn_checkpoint(agent: Any) -> None:
    """Load and freeze the optional pretrained TGNN encoder."""
    ckpt_path = (
        agent.config.get("model", {}).get("hybrid", {}).get("tgnn_checkpoint")
    )
    if ckpt_path in (None, "", "none", "None"):
        print("[INFO] HybridAgent initialized without a pretrained TGNN checkpoint.")
        return

    if ckpt_path == "auto":
        ckpt_dir = os.path.join(
            agent.config["paths"]["results_dir"], "tgnn", "checkpoints"
        )
        files = glob.glob(os.path.join(ckpt_dir, "best_model_*.pth"))
        if not files:
            print(f"[WARN] No TGNN checkpoint found in {ckpt_dir}.")
            return
        ckpt_path = max(files, key=os.path.getmtime)

    if not os.path.exists(ckpt_path):
        print(f"[WARN] TGNN checkpoint path does not exist: {ckpt_path}")
        return

    state_dict = torch.load(ckpt_path, map_location=agent.device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    agent.tgnn.load_state_dict(state_dict)
    _freeze_tgnn(agent)
    print(f"[OK] TGNN checkpoint loaded and frozen: {ckpt_path}")


def save_hybrid(agent: Any, path: str) -> None:
    """Save the trainable HybridAgent components."""
    torch.save(
        {
            "ddpg_actor": agent.ddpg.actor.state_dict(),
            "ddpg_critic": agent.ddpg.critic.state_dict(),
            "ensemble": agent.ensemble_net.state_dict(),
            "horizon_embedding": agent.horizon_embedding.state_dict(),
            "config": agent.config,
        },
        path,
    )


def load_hybrid_weights(agent: Any, path: str, strict: bool = True) -> None:
    """Load HybridAgent weights across explicit or prefixed checkpoint formats."""
    checkpoint = torch.load(path, map_location=agent.device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    if "ddpg_actor" in checkpoint:
        agent.ddpg.actor.load_state_dict(checkpoint["ddpg_actor"], strict=strict)
        if "ddpg_critic" in checkpoint:
            agent.ddpg.critic.load_state_dict(checkpoint["ddpg_critic"], strict=strict)
        if "ensemble" in checkpoint:
            agent.ensemble_net.load_state_dict(checkpoint["ensemble"], strict=False)
        if "horizon_embedding" in checkpoint:
            agent.horizon_embedding.load_state_dict(
                checkpoint["horizon_embedding"], strict=False
            )
        print(f"[OK] HybridAgent loaded: {path}")
        return

    actor_state = {
        k.replace("ddpg.actor.", "").replace("actor.", ""): v
        for k, v in state_dict.items()
        if k.startswith("ddpg.actor.") or k.startswith("actor.")
    }
    critic_state = {
        k.replace("ddpg.critic.", "").replace("critic.", ""): v
        for k, v in state_dict.items()
        if k.startswith("ddpg.critic.") or k.startswith("critic.")
    }
    ensemble_state = {
        k.replace("ensemble_net.", ""): v
        for k, v in state_dict.items()
        if k.startswith("ensemble_net.")
    }

    loaded = []
    if actor_state:
        agent.ddpg.actor.load_state_dict(actor_state, strict=False)
        loaded.append("actor")
    if critic_state:
        agent.ddpg.critic.load_state_dict(critic_state, strict=False)
        loaded.append("critic")
    if ensemble_state:
        agent.ensemble_net.load_state_dict(ensemble_state, strict=False)
        loaded.append("ensemble")

    if loaded:
        print(f"[OK] HybridAgent loaded parts ({', '.join(loaded)}): {path}")
    else:
        print(f"[WARN] No compatible HybridAgent weights found in {path}.")
