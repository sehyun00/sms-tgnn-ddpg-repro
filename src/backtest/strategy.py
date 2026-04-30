import torch
import numpy as np
from typing import Dict, Any, Tuple


class StrategyHandler:
    """
    Handles model inference and weight calculation strategies.
    """

    def __init__(self, n_stocks: int, device):
        self.n_stocks = n_stocks
        self.device = device

    def get_weights(
        self,
        strategy_type: str,
        model,
        window: Dict[str, Any],
        target_head: str = None,
        horizon: int = 0,
        return_attn_weights: bool = False,
    ) -> Any:
        """
        Determines portfolio weights based on strategy type.

        Args:
            horizon: 리밸런싱 주기 (0=monthly, 1=quarterly, 2=semiannual, 3=annual)
        """
        if strategy_type == "buy_and_hold":
            return np.ones(self.n_stocks) / self.n_stocks

        elif strategy_type == "model":
            return self._model_strategy(model, window, target_head, horizon, return_attn_weights)

        else:
            # Default fallback
            return np.ones(self.n_stocks) / self.n_stocks

    def _model_strategy(
        self, model, window: Dict[str, Any], target_head: str, horizon: int = 0, return_attn_weights: bool = False
    ) -> Any:
        """
        Executes model inference and applies weighting logic.

        [모델별 출력 형식 차이]
        - DDPG: Actor가 이미 Softmax 적용된 weights 반환 → 그대로 사용
        - Hybrid: forward() 호출하여 동적 alpha 적용
        - TGNN: Raw scores 반환 → _calculate_softmax_weights()로 변환 필요

        Args:
            horizon: 리밸런싱 주기 (0=monthly, 1=quarterly, 2=semiannual, 3=annual)

        Returns:
            np.ndarray: 포트폴리오 비중 (sum ≈ 1.0)
        """
        # 1. Prepare Input
        # 1. Prepare Input
        # Context-Aware inputs (Dual-Path)
        if "prices" in window and "macro" in window:
            prices = torch.FloatTensor(window["prices"]).to(self.device)  # [N, T, F_p]
            macro = torch.FloatTensor(window["macro"]).to(self.device)  # [N, T, F_m]

            # Add Batch Dim: [1, N, T, F]
            if prices.dim() == 3:
                prices = prices.unsqueeze(0)
            if macro.dim() == 3:
                macro = macro.unsqueeze(0)

            # Hybrid 모델은 Concatenated Input (Price + Macro) 필요
            x_input = torch.cat([prices, macro], dim=-1)
            macro_input = macro
        else:
            # Legacy inputs
            features = torch.FloatTensor(window["features"]).to(self.device)
            if features.dim() == 2:
                features = features.unsqueeze(0)
            elif features.dim() == 3:
                features = features.unsqueeze(0)
            x_input = features
            macro_input = None

        adj = None
        if "adj_matrix" in window:
            adj = torch.FloatTensor(window["adj_matrix"]).unsqueeze(0).to(self.device)

        # 2. Inference
        model.eval()
        with torch.no_grad():
            # Hybrid 모델 전용 처리 (동적 Alpha 지원)
            model_class_name = model.__class__.__name__
            if model_class_name == "HybridAgent":
                # Hybrid uses internal buffering or needs update to accept macro if it relies on TGNN
                # Ideally Hybrid should also accept split features, but for now let's assume it handles legacy or is updated separately.
                # If Hybrid calls TGNN internally, it might need to pass macro.
                # Let's pass what we have; Hybrid forward might need alignment.
                # For now, keeping Hybrid flow simple as user focus is TGNN verification.
                if adj is not None:
                    weights_out = model(
                        x_input, adj, target_head=target_head, horizon=horizon, return_attn_weights=return_attn_weights
                    )
                else:
                    N = x_input.shape[1]
                    adj = torch.eye(N).unsqueeze(0).to(self.device)
                    weights_out = model(
                        x_input, adj, target_head=target_head, horizon=horizon, return_attn_weights=return_attn_weights
                    )
                
                if return_attn_weights and isinstance(weights_out, tuple):
                    # weights_out = (final_weights, alpha, attn_weights)
                    return weights_out[0].cpu().numpy()[0], weights_out[2]
                elif isinstance(weights_out, tuple):
                    return weights_out[0].cpu().numpy()[0]
                return weights_out.cpu().numpy()[0]

            # DDPG 및 기타 RL 모델
            elif hasattr(model, "actor"):
                try:
                    if adj is not None:
                        output = model.actor(x_input, adj)
                    else:
                        output = model.actor(x_input)
                except TypeError:
                    output = model.actor(x_input)

                if isinstance(output, tuple):
                    raw_weights = output[0].cpu().numpy()[0]
                    return raw_weights
                else:
                    return output.cpu().numpy()[0]

            else:
                # TGNN / Supervised Models
                # Pass macro if available
                if macro_input is not None and "prices" in locals():
                    output = model(
                        prices, adj, macro=macro_input, target_type=target_head, return_attn_weights=return_attn_weights
                    )
                elif macro_input is not None:
                    output = model(
                        x_input, adj, macro=macro_input, target_type=target_head, return_attn_weights=return_attn_weights
                    )
                else:
                    output = model(x_input, adj, target_type=target_head, return_attn_weights=return_attn_weights)

                if return_attn_weights and isinstance(output, tuple):
                    # output = (preds, combined_embedding, attn_weights)
                    scores = output[0].cpu().numpy()[0]
                    return self._calculate_softmax_weights(scores), output[2]
                elif isinstance(output, tuple):
                    scores = output[0].cpu().numpy()[0]
                    return self._calculate_softmax_weights(scores)
                
                scores = output.cpu().numpy()[0]  # [N]

                return self._calculate_softmax_weights(scores)

    def _calculate_softmax_weights(self, scores: np.ndarray) -> np.ndarray:
        """
        Full Universe Softmax Strategy (Research Standard).
        Applies Softmax to ALL stock scores to determine weights.
        """
        # score-based weighting (Softmax)
        # 1. Normalize scores (Z-score) to prevent softmax saturation
        if scores.std() > 1e-6:
            z_scores = (scores - scores.mean()) / scores.std()
        else:
            z_scores = scores - scores.mean()

        # 2. Apply Softmax with temperature (optional, default 1.0)
        # Using a slight temperature > 1 can smooth out extreme bets if needed, but 1.0 is standard.
        exp_scores = np.exp(z_scores)
        weights = exp_scores / np.sum(exp_scores)

        return weights
