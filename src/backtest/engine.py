import numpy as np
import pandas as pd
import os
import time
from typing import Dict, Any

from .strategy import StrategyHandler
from .visualization import Visualizer
from .metrics import compute_metrics


class Backtester:
    """
    Refactored Backtester Engine.
    Orchestrates simulation, strategy execution, and result aggregation.
    """

    def __init__(self, config: Dict[str, Any], model, dataset):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.device = getattr(model, "device", "cpu")
        self.symbols = dataset.symbols
        self.n_stocks = len(self.symbols)

        # Output Setup
        model_name = config["project"].get("selected_model", "default")
        results_dir = os.path.join(config["paths"]["results_dir"], model_name)
        self.visualizer = Visualizer(results_dir)

        # Strategy Setup
        self.strategy_handler = StrategyHandler(self.n_stocks, self.device)

        # Simulation Params
        self.initial_capital = 1_000_000

        # Prepare Data
        self.test_windows = self._get_test_windows()
        # Shift returns to t+1 to align with execution at t (Close)
        self.daily_returns_df = self._precalculate_returns().shift(-1).fillna(0.0)

        # Subset Masking Setup
        self.valid_mask = None
        if "test_valid_subset" in config["data"]:
            subset_symbols = config["data"]["test_valid_subset"]
            self.valid_mask = np.zeros(self.n_stocks)
            mapped_count = 0
            for sym in subset_symbols:
                if sym in self.symbols:
                    idx = self.symbols.index(sym)
                    self.valid_mask[idx] = 1.0
                    mapped_count += 1
            print(
                f"      [Backtester] Subset Masking Active: {mapped_count}/{self.n_stocks} stocks valid."
            )

    def _get_test_windows(self):
        test_split_date = pd.Timestamp(
            self.config["training"].get("test_split_date", "2021-01-01")
        )
        windows = [
            w
            for w in self.dataset.windows
            if pd.to_datetime(w["date"]) >= test_split_date
        ]

        if not windows:
            print(
                f"⚠️ Warning: No test windows found after {test_split_date}. Using last 20%."
            )
            split_idx = int(len(self.dataset.windows) * 0.8)
            windows = self.dataset.windows[split_idx:]

        return windows

    def _precalculate_returns(self):
        print("      [Init] Pre-calculating Daily Returns...")
        # Create empty DF with all unique dates
        dates = self.dataset.df.index.unique().sort_values()
        # 명시적으로 Timestamp로 변환 (target_date와 타입 일치 보장)
        dates = pd.to_datetime(dates)
        daily_returns = pd.DataFrame(index=dates)

        for sym in self.symbols:
            # Filter symbol data
            sym_df = self.dataset.df[self.dataset.df["Symbol"] == sym]
            if "Close" in sym_df.columns:
                # Calculate pct_change properly
                # Note: pct_change on filtered DF preserves index (Date)
                returns = sym_df["Close"].pct_change()
                # 인덱스도 Timestamp로 통일
                returns.index = pd.to_datetime(returns.index)
                daily_returns[sym] = returns
            else:
                daily_returns[sym] = 0.0

        daily_returns.fillna(0.0, inplace=True)
        return daily_returns

    def _should_rebalance(
        self, current_date: pd.Timestamp, prev_date: pd.Timestamp, interval_type: str
    ) -> bool:
        """
        날짜 기반 리밸런싱 판단.

        Args:
            current_date: 현재 거래일
            prev_date: 이전 거래일
            interval_type: "daily", "monthly", "quarterly", "semiannual", "annual"

        Returns:
            True if rebalancing should occur
        """
        if prev_date is None:
            return True  # 첫 거래일은 항상 리밸런싱

        if interval_type == "daily":
            return True
        elif interval_type == "monthly":
            return current_date.month != prev_date.month
        elif interval_type == "quarterly":
            return (current_date.month - 1) // 3 != (prev_date.month - 1) // 3
        elif interval_type == "semiannual":
            return (current_date.month - 1) // 6 != (prev_date.month - 1) // 6
        elif interval_type == "annual":
            return current_date.year != prev_date.year
        else:
            # Fallback: treat as daily
            return True

    def run_strategy(
        self,
        strategy_type: str = "model",
        target_head: str = "Momentum1M",
        rebalance_freq: str = "monthly",
    ) -> Dict[str, Any]:
        """
        Runs strategy with specified rebalancing frequency.

        Args:
            strategy_type: "buy_and_hold" or "model"
            target_head: TGNN head name (e.g., "Momentum1M")
            rebalance_freq: "daily", "monthly", "quarterly", "semiannual", "annual"
        """
        print(f"   [RUN] Running Strategy Simulation: {strategy_type.upper()} ({target_head}) | Freq: {rebalance_freq}")

        capital = self.initial_capital
        portfolio_values = [capital]
        dates = []
        trade_logs = []
        latencies = []
        attention_data = []

        # Initial Weights
        current_weights = np.ones(self.n_stocks) / self.n_stocks

        # Apply initial mask if exists (only for model strategies, not buy_and_hold)
        if strategy_type != "buy_and_hold" and self.valid_mask is not None:
            current_weights = current_weights * self.valid_mask
            current_weights /= np.sum(current_weights) + 1e-8

        prev_date = None  # 날짜 기반 리밸런싱을 위한 이전 날짜 추적

        for i, window in enumerate(self.test_windows):
            target_date = pd.Timestamp(window["date"])

            # 1. Update Weights (Rebalance) - 날짜 기반 판단
            trade_type = "Hold"
            should_rebalance = self._should_rebalance(
                target_date, prev_date, rebalance_freq
            )

            # Buy & Hold: Only rebalance at the first step
            if strategy_type == "buy_and_hold":
                should_rebalance = i == 0

            if should_rebalance:
                trade_type = "Rebalance"
                try:
                    prev_weights = current_weights.copy()

                    # 리밸런싱 주기를 horizon 정수로 변환 (동적 Alpha용)
                    horizon_map = {
                        "monthly": 0,
                        "quarterly": 1,
                        "semiannual": 2,
                        "annual": 3,
                    }
                    horizon = horizon_map.get(rebalance_freq, 0)

                    start_time = time.time()
                    
                    # Request attention weights for model strategies
                    inference_out = self.strategy_handler.get_weights(
                        strategy_type, self.model, window, target_head, horizon=horizon, return_attn_weights=True
                    )
                    
                    if isinstance(inference_out, tuple):
                        current_weights, attn_weights = inference_out
                        if attn_weights is not None:
                            attention_data.append({
                                "Date": target_date,
                                "Weights": attn_weights.cpu().numpy()
                            })
                    else:
                        current_weights = inference_out
                    
                    latency = (time.time() - start_time) * 1000  # ms
                    latencies.append(latency)

                    # Apply Subset Masking (Zero out missing stocks) - only for model strategies
                    if strategy_type != "buy_and_hold" and self.valid_mask is not None:
                        current_weights = current_weights * self.valid_mask

                    # Force Normalize to prevent Cash Drag (Model might output sum < 1.0 due to float prec)
                    current_weights /= np.sum(current_weights) + 1e-8

                    if i > 0:
                        turnover = np.sum(np.abs(current_weights - prev_weights)) / 2

                        # [Research Fix] Apply Transaction Cost
                        # 비용 = 회전율 * 자본 * 수수료율
                        cost_rate = self.config["backtest"].get("transaction_cost", 0.0)
                        cost = capital * turnover * cost_rate
                        capital -= cost

                        # if turnover >= 1e-4:
                        #     print(
                        #         f"      [Rebalance] Step {i}: Turnover {turnover:.4f} | Cost: {cost:.2f}"
                        #     )

                except Exception as e:
                    print(f"      [Error] Weight calc failed at step {i}: {e}")
                    # On error, we might fallback to equal weights or keep previous
                    # For safety/research parity, fallback to equal is mostly standard or hold
                    current_weights = np.ones(self.n_stocks) / self.n_stocks

            # 2. Log Trade
            strategy_name = strategy_type
            if strategy_type == "model":
                strategy_name = f"{strategy_type}_{target_head}_{rebalance_freq}"

            log_entry = {
                "Date": target_date,
                "Strategy": strategy_name,
                "Type": trade_type,
            }

            # Determine symbols to log
            if (
                self.valid_mask is not None
                and "test_valid_subset" in self.config["data"]
            ):
                logging_symbols = self.config["data"]["test_valid_subset"]
            else:
                logging_symbols = self.symbols

            # Add weights to log
            for sym in logging_symbols:
                if sym in self.symbols:
                    idx = self.symbols.index(sym)
                    if idx < len(current_weights):
                        log_entry[sym] = round(float(current_weights[idx]), 4)

            trade_logs.append(log_entry)

            # 3. Calculate Return
            # Get daily return vector for this date (already shifted to t+1)
            try:
                if target_date in self.daily_returns_df.index:
                    daily_returns = self.daily_returns_df.loc[
                        target_date, self.symbols
                    ].values
                else:
                    daily_returns = np.zeros(self.n_stocks)
            except Exception:
                daily_returns = np.zeros(self.n_stocks)

            # PnL Update
            port_ret = np.dot(current_weights, daily_returns)
            capital *= 1 + port_ret

            portfolio_values.append(capital)
            dates.append(target_date)

            # 4. Drift Weights (Price Impact)
            # w_i_new = w_i_old * (1 + r_i) / (1 + port_ret)
            if (1 + port_ret) != 0:
                current_weights = current_weights * (1 + daily_returns) / (1 + port_ret)

            # Buy&Hold는 드리프트 그대로 유지,  model 전략은 재정규화
            if strategy_type != "buy_and_hold":
                current_weights /= np.sum(current_weights)

            # 날짜 기반 리밸런싱을 위한 이전 날짜 업데이트
            prev_date = target_date

        # Summary Statistics
        if latencies:
            avg_latency = np.mean(latencies)
            std_latency = np.std(latencies)
            print(f"      [Performance] Avg Latency: {avg_latency:.2f}ms (+/- {std_latency:.2f}ms)")
            
            # Save latency info to results
            perf_path = os.path.join(self.visualizer.results_dir, "performance.txt")
            with open(perf_path, "w") as f:
                f.write(f"Strategy: {strategy_type} ({target_head})\n")
                f.write(f"Average Inference Latency: {avg_latency:.4f} ms\n")
                f.write(f"Latency Std Dev: {std_latency:.4f} ms\n")
                f.write(f"Total Rebalancing Events: {len(latencies)}\n")

        return {
            "dates": dates,
            "portfolio_values": portfolio_values[1:],  # align with dates
            "trade_logs": trade_logs,
            "final_capital": capital,
            "latencies": latencies,
            "attention_data": attention_data,
        }

    def compute_metrics(self, results):
        return compute_metrics(results["portfolio_values"], self.initial_capital)

    def save_and_plot(self, results):
        self.visualizer.save_logs(results, self.symbols)
        self.visualizer.plot_comparison(results, initial_capital=self.initial_capital)
        
        # [XAI] Plot Attention Map if available (take first non-empty)
        for name, res in results.items():
            if "attention_data" in res and res["attention_data"]:
                self.visualizer.plot_attention_heatmap(res["attention_data"], self.symbols)
                break
        
        # [Research Integrity] Print Defense Logic
        print("\n" + "-"*50)
        print("📌 [Research Integrity] Backtest Summary Notes")
        print("1. Universe: S&P 500 (Selected for High Liquidity & Low Slippage)")
        print("2. Bias Defense: This study focuses on Relation Learning effectiveness.")
        print("3. Limitation: Survival bias acknowledged. Future work with PiT data suggested.")
        print("-"*50)
