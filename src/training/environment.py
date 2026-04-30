import numpy as np
from typing import Dict, Any, Tuple, List


class PortfolioEnvironment:
    """
    Reinforcement Learning Environment for Portfolio Management.
    - Simulates portfolio rebalancing steps.
    - Calculates rewards based on returns, risk (Sharpe, MDD), and constraints.
    - Based on HybridPortfolioEnv references.
    """

    def __init__(self, dataset, windows=None, initial_cash: float = 1_000_000):
        """
        Args:
            dataset: Dataset object (TGNNDataset compatible)
            windows: List of window dictionaries (if None, use dataset's default)
            initial_cash: Initial portfolio value
        """
        self.dataset = dataset
        self.windows = windows if windows else dataset.windows
        self.initial_cash = initial_cash
        self.portfolio_value = initial_cash
        self.current_step = 0
        self.n_steps = len(self.windows)

        # RL Parameters
        self.gamma = 2.0  # CRRA Risk Aversion
        self.cost_bps = 0.0005  # 5bps Transaction Cost

        # State tracking
        self.n_stocks = len(dataset.symbols)
        self.prev_weights = np.zeros(self.n_stocks)
        self.return_history = []
        self.current_mdd = 0.0

    def reset(self):
        self.current_step = 0
        self.portfolio_value = self.initial_cash
        self.prev_weights = np.zeros(self.n_stocks)
        self.return_history = []
        self.current_mdd = 0.0
        return self._get_state(self.current_step)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        w = self.windows[self.current_step]
        returns = w.get("labels", np.zeros(self.n_stocks))  # Momentum1M usually
        features = w.get(
            "features", None
        )  # Full features for reward calculation if needed

        # 1. Action Normalization
        action = np.array(action, dtype=np.float64)
        action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        action = np.clip(action, 0, 1)
        action_sum = np.sum(action)
        if action_sum > 0:
            action = action / action_sum
        else:
            action = np.ones(len(action)) / len(action)

        # 2. Returns Processing
        returns = np.array(returns, dtype=np.float64)
        returns = np.nan_to_num(returns, nan=0.0)  # Delisted/Missing = 0%
        # Clip returns for safety
        returns = np.clip(returns, -0.95, 2.0)

        # Portfolio Return
        portfolio_return = float(np.dot(action, returns))
        portfolio_return = np.clip(portfolio_return, -0.8, 1.0)

        # Transaction Cost
        turnover = np.sum(np.abs(action - self.prev_weights))
        cost = turnover * self.cost_bps
        net_return = portfolio_return - cost
        net_return = np.clip(net_return, -0.8, 1.0)

        # 3. Value Update
        self.portfolio_value *= 1 + net_return
        if self.portfolio_value < 1000.0:  # Safeguard
            self.portfolio_value = 1000.0

        self.current_step += 1
        done = self.current_step >= self.n_steps
        self.return_history.append(net_return)

        # 4. MDD Tracking
        if len(self.return_history) >= 12:
            cumulative_returns = np.cumprod(1 + np.array(self.return_history))
            peak = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - peak) / (peak + 1e-8)
            self.current_mdd = abs(np.min(drawdowns))
        else:
            self.current_mdd = 0.0

        # 5. Reward Calculation
        reward = self._calculate_reward(action, returns, net_return, turnover)

        self.prev_weights = action

        # 6. Next State
        if not done:
            next_state = self._get_state(self.current_step)
        else:
            # Create zero state dict manually
            s0 = self._get_state(0)
            next_state = {k: np.zeros_like(v) for k, v in s0.items()}

        info = {
            "portfolio_value": self.portfolio_value,
            "return": net_return,
            "date": w.get("date"),
            "turnover": turnover,
            "cost": cost,
            "current_mdd": self.current_mdd,
        }

        return next_state, reward, done, info

    def _get_state(self, idx: int):
        """Helper to extract state from dataset window"""
        # Adapted from dataset.get_state logic
        # For TGNN/Hybrid: State is usually features + adj
        # For DDPG: Flattened features
        # The trainer/agent handles the shape. We return the raw Window content usually.
        # But wait, step() needs to return numpy array for ReplayBuffer.
        # Dataset.__getitem__ returns tensors.
        # Let's use dataset.__getitem__(idx) but convert to numpy.

        # However, new dataset structure returns a dict {"features": ..., "adj_matrix": ...}
        # ReplayBuffer stores tuples. The agent needs to define what 'state' is.
        # Hybrid Agent: forward(x, adj). So state = (x, adj).
        # DDPG Agent: forward(x). So state = x.

        # To make this generic, we return the DATA DICT (converted to numpy) or TUPLE?
        # ReplayBuffer expects a single object 'state'.
        # We can store the dict in the buffer!

        if idx >= len(self.windows):
            w = self.windows[0]
        else:
            w = self.windows[idx]

        # Convert torch tensors to numpy if they are tensors
        state_dict = {}
        for k, v in w.items():
            if k in ["features", "adj_matrix", "active_mask"]:
                if hasattr(v, "numpy"):
                    state_dict[k] = v.numpy()
                elif isinstance(v, (np.ndarray, list)):
                    state_dict[k] = np.array(v)
                else:
                    state_dict[k] = v

        return state_dict

    def _calculate_reward(self, action, returns, net_return, turnover):
        """
        Complex Reward Function from HybridPortfolioEnv.
        Includes: Return, Sharpe, MDD Penalty, Concentration Penalty, Turnover Penalty.
        """
        # 1. Early Stage (Return Focus)
        if len(self.return_history) < 6:
            reward = net_return * 200.0 - turnover * 2.0
            return np.clip(reward, -200, 200)

        returns_array = np.array(self.return_history[-12:])
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array) + 1e-8

        # 2. Components
        return_reward = mean_return * 200.0

        sharpe = mean_return / std_return
        sharpe = np.clip(sharpe, -5, 5)
        sharpe_reward = sharpe * 50.0

        # MDD Penalty
        mdd_penalty = 0
        if self.current_mdd > 0.40:
            mdd_penalty = 50.0 * (self.current_mdd - 0.40)
        elif self.current_mdd > 0.25:
            mdd_penalty = 10.0 * (self.current_mdd - 0.25)
        elif self.current_mdd < 0.15:
            mdd_penalty = -20.0  # Bonus

        # Concentration Penalty
        concentration = np.sum(action**2)
        concentration_penalty = 0
        if concentration > 0.25:
            concentration_penalty = 30.0 * (concentration - 0.25)

        diversity_bonus = 0
        if 0.10 <= concentration <= 0.20:
            diversity_bonus = 15.0

        turnover_penalty = turnover * 0.5

        # Final Sum
        reward = (
            return_reward
            + sharpe_reward
            + diversity_bonus
            - mdd_penalty
            - concentration_penalty
            - turnover_penalty
        )

        return float(np.clip(reward, -200, 200))
