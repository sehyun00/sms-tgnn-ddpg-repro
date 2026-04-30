import os
import pandas as pd
from typing import Dict, Any, Optional

from src.training.dataset import FinancialDataset
from src.backtest import Backtester
from src.models.tgnn.model import TGNN
from src.models.ddpg.agent import DDPGAgent
from src.models.hybrid.agent import HybridAgent
from src.training.trainer import Trainer


def run_backtest(config: Dict[str, Any], model_path: Optional[str] = None):
    """
    Executes the Backtesting Pipeline.
    1. Load Test Data
    2. Check Universe Mismatch (Transfer Learning)
    3. Initialize Model & Load Weights (or Finetune)
    4. Run Backtester Strategy
    """
    print("\n" + "=" * 50)
    print("      Starting Backtest Pipeline")
    print("=" * 50)

    # 1. Load Data (Test)
    print("\n[1/4] Loading Test Data...")
    # Fix: Use data_dir from config instead of hardcoded "data/"
    data_dir = config["paths"]["data_dir"]
    test_data_path = os.path.join(data_dir, "test_data.csv")
    if not os.path.exists(test_data_path):
        print(
            f"❌ Error: {test_data_path} not found. Please run 'python main.py --mode preprocess' first."
        )
        return

    test_df = pd.read_csv(test_data_path)

    if "Date" in test_df.columns:
        test_df["Date"] = pd.to_datetime(test_df["Date"])
        test_df.set_index("Date", inplace=True)

    # Clean NaNs
    test_df.dropna(inplace=True)

    print(f"      Data Shape: {test_df.shape}")

    # 2. Create Dataset
    print("\n[2/4] Creating Dataset...")

    # CRITICAL FIX for DDPG & Transfer Learning:
    # 1. Capture User's Desired Test Universe (from config)
    user_test_universe = config["data"].get("stock_universes", [])
    if user_test_universe is None:
        user_test_universe = []

    # 2. Load "trained_universe.json" to know what the model expects
    model_type = config["project"].get("selected_model", "tgnn").lower()
    results_dir = os.path.join(config["paths"]["results_dir"], model_type)
    universe_path = os.path.join(results_dir, "trained_universe.json")

    trained_universe = []
    if os.path.exists(universe_path):
        import json

        with open(universe_path, "r") as f:
            trained_universe = json.load(f)
        print(f"      Loaded Training Universe Info: {len(trained_universe)} symbols.")

    # 3. Determine Universe to use for Dataset Creation
    # If User specified a universe in config, prioritize that (Test Subset)
    # If User left it empty, they might want to test on EVERYTHING in test_data.csv
    # test_data에서 실제 종목 추론 (Asset-Agnostic 모델에서 항상 사용)
    inferred = set(test_df["Symbol"].unique()) if test_df is not None else set()

    if user_test_universe and len(user_test_universe) > 0:
        print(f"      User specified Test Universe: {len(user_test_universe)} symbols.")
        # user_test_universe와 test_data의 교집합 확인
        valid_user = set(user_test_universe) & inferred
        if len(valid_user) < len(user_test_universe):
            print(
                f"      ⚠️ Only {len(valid_user)} of {len(user_test_universe)} symbols exist in test_data."
            )
            inferred = valid_user if valid_user else inferred
        else:
            inferred = set(user_test_universe)
    elif test_df is not None and not test_df.empty:
        # Detect Intersection (Valid Test vs Trained)
        trained = set(trained_universe) if trained_universe else set()

        valid_subset = sorted(list(inferred & trained))
        excluded = sorted(list(inferred - trained))

        print(f"      Inferred Test Data: {len(inferred)} symbols.")

        # If intersection exists, enable masking for those stocks
        if valid_subset:
            print(
                f"      ✅ Intersection Detected! Enabling Subset Masking for {len(valid_subset)} stocks."
            )
            if excluded:
                print(
                    f"      [EXCL] Excluded {len(excluded)} symbols not in training set: {excluded}"
                )

            config["data"]["test_valid_subset"] = valid_subset

        # Fix: Ensure stock_universes is updated with inferred universe!
        # Otherwise dataset will be empty if defaults are used.
        if not config["data"]["stock_universes"]:
            print(
                f"      [INFO] Using Inferred Universe ({len(inferred)} symbols) for Backtest."
            )
            config["data"]["stock_universes"] = sorted(list(inferred))

    if trained_universe and len(trained_universe) > 0:
        # 모든 모델을 Asset-Agnostic으로 처리 (Transfer Learning 지원)
        # TGNN도 인코더만 재사용하고 test_data 종목에 맞게 동작
        print(
            f"      [INFO] Model ({model_type}) is Asset-Agnostic. Using Test Data Universe ({len(inferred)} symbols) instead of Training Universe."
        )
        # Use Inferred Test Universe
        config["data"]["stock_universes"] = sorted(list(inferred))

    # For Dataset creation, we pass the Test DF
    # The dataset class will filter strictly if config has stock_universes
    dataset = FinancialDataset(config, test_df, mode="test")
    print(f"      Windows Created: {len(dataset)}")

    # 3. Initialize Model & Load Weights
    print(f"\n[3/4] Loading Model ({model_type.upper()})...")

    # Check for Mismatch (Transfer Learning Scenario)
    # We compare loaded 'trained_universe' (from JSON) vs 'dataset.symbols' (Test Data)
    is_transfer_learning = False

    # If we have trained info, compare.
    if trained_universe:
        train_n = len(trained_universe)
        test_n = len(dataset.symbols)

        if train_n != test_n:
            print(
                f"      [WARN] Universe Mismatch detected: Train({train_n}) vs Test({test_n})"
            )
            print(
                f"      [SYNC] Enabling Transfer Learning (Partial Load + Fine-tuning)..."
            )
            is_transfer_learning = True

            # Ensure config matches TEST universe for Model Initialization
            config["data"]["stock_universes"] = dataset.symbols

    # Initialize Model (with Test Universe size if Transfer Learning, else Train size)
    if model_type == "hybrid":
        model = HybridAgent(config)
    elif model_type == "ddpg":
        model = DDPGAgent(config)
    else:
        model = TGNN(config)

    # Determine model path if not provided
    if not model_path:
        # Try to find best model in results dir
        try:
            if os.path.exists(results_dir):
                import glob

                ckpt_dir = os.path.join(results_dir, "checkpoints")

                # 1. Search in checkpoints dir (New Structure)
                files = glob.glob(os.path.join(ckpt_dir, "best_model_*.pth"))

                # 2. Fallback to root dir (Old Structure)
                if not files:
                    files = glob.glob(os.path.join(results_dir, "best_model_*.pth"))

                if files:
                    # Find latest by creation time
                    model_path = max(files, key=os.path.getctime)
                    print(f"      Found latest model: {os.path.basename(model_path)}")
                else:
                    print(
                        f"⚠️ No best_model_*.pth found in {ckpt_dir} or {results_dir}."
                    )
        except Exception as e:
            print(f"⚠️ Error finding model: {e}")

    if model_path and os.path.exists(model_path):
        print(f"      Loading weights from: {model_path}")
        if is_transfer_learning:
            # Partial Load
            model.load(model_path, strict=False)
            print("      [OK] Partial weights loaded (Encoder transferred).")

            # [Research Rule] DO NOT Fine-tune on Test Data (Look-ahead Bias)
            print(
                "      [LOCK] Finite-tuning disabled to prevent Data Leakage (Research Integrity)."
            )
            # print("\n[3.5] Fine-tuning Model on Test Data...")
            # trainer = Trainer(config, model, dataset)
            # trainer.finetune(epochs=50, lr_factor=0.1)
            # print("      ✅ Fine-tuning completed.")
        else:
            # Strict Load
            model.load(model_path, strict=True)
    else:
        if model_path:
            print(f"⚠️ Warning: Model path {model_path} does not exist.")

    # 4. Run Backtest
    print("\n[4/4] Running Backtest Strategy...")
    backtester = Backtester(config, model, dataset)

    # Run strategies
    results = {}

    # 1. Buy & Hold (Benchmark)
    results["Benchmark"] = backtester.run_strategy(strategy_type="buy_and_hold")

    # 2. Model Strategy
    # Check if model has multiple heads (like TGNN)
    if hasattr(model, "heads"):
        # 2. TGNN Rebalancing Strategies (Horizon Matching)
        print(f"\n[Model: TGNN Rebalancing Strategies (Horizon Matching)]")
        # Map: Frequency Name -> Target Head
        horizon_map = {
            "monthly": "Momentum1M",
            "quarterly": "Momentum3M",
            "semiannual": "Momentum6M",
            "annual": "Momentum12M",
        }

        for freq_name, target_head in horizon_map.items():
            run_name = f"TGNN_{freq_name.capitalize()}"
            print(
                f"\n[Strategy: {freq_name.capitalize()} Rebalancing | Head: {target_head}]"
            )
            results[run_name] = backtester.run_strategy(
                strategy_type="model",
                target_head=target_head,
                rebalance_freq=freq_name,
            )

    else:
        # RL (DDPG) or Hybrid - Single Policy
        # Iterate over Rebalancing Frequencies
        print(f"\n[Model: {model_type.upper()} Rebalancing Strategies]")

        rebalance_freqs = ["monthly", "quarterly", "semiannual", "annual"]

        for freq_name in rebalance_freqs:
            run_name = f"{model_type.upper()}_{freq_name.capitalize()}"
            print(f"\n[Strategy: {freq_name.capitalize()} Rebalancing]")
            results[run_name] = backtester.run_strategy(
                strategy_type="model",
                target_head="Portfolio",
                rebalance_freq=freq_name,
            )

    # Visualization and Logging
    print("\n[5/5] Saving Results and Visualization...")

    # Use the visualizer instance embedded in backtester
    # 1. Save Trade Logs (Aggregated)
    backtester.visualizer.save_logs(results, dataset.symbols)

    # 2. Save Metrics CSV (for generate_comparison_chart.py)
    backtester.visualizer.save_metrics(results)

    # 3. Plot Comparison
    # Visualizer knows the results_dir from backtester init
    backtester.visualizer.plot_comparison(
        results, initial_capital=backtester.initial_capital
    )

    # 4. [XAI] Plot Attention Heatmap (TGNN 또는 Hybrid 모델)
    # hasattr(model, "heads")  → TGNN standalone
    # hasattr(model, "tgnn")   → HybridAgent (내부 TGNN 포함)
    if hasattr(model, "heads") or hasattr(model, "tgnn"):
        for name, res in results.items():
            if res.get("attention_data"):
                backtester.visualizer.plot_attention_heatmap(
                    res["attention_data"], dataset.symbols
                )
                print(f"[OK] Attention heatmap generated for: {name}")
                break

    # Print Final Summary
    print("\n📊 Final Metrics:")

    # Re-extract metrics for printing
    metrics_list = []
    for name, res in results.items():
        m = backtester.compute_metrics(res)
        m["Strategy"] = name
        metrics_list.append(m)

    for m in metrics_list:
        print(
            f"  [{m['Strategy']}] Return: {m['Total_Return']:.2f}% | CAGR: {m['CAGR']:.2f}% | MDD: {m['MDD']:.2f}% | Sharpe: {m['Sharpe']:.2f}"
        )

    print("\n🎉 Backtest Completed Successfully!")
