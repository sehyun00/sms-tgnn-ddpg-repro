import os
import pandas as pd
from typing import Dict, Any

from src.training.dataset import FinancialDataset
from src.training.trainer import Trainer
from src.models.tgnn.model import TGNN
from src.models.ddpg.agent import DDPGAgent
from src.models.hybrid.agent import HybridAgent


def run_train(config: Dict[str, Any]):
    """
    Executes the Training Pipeline.
    1. Load Data
    2. Create Dataset
    3. Initialize Model
    4. Run Training
    """
    print("\n" + "=" * 50)
    print("🚀 Starting Training Pipeline")
    print("=" * 50)

    # 1. Load Data (Train)
    print("\n[1/4] Loading Training Data...")
    data_dir = config["paths"]["data_dir"]
    train_data_path = os.path.join(data_dir, "train_data.csv")
    if not os.path.exists(train_data_path):
        print(
            f"❌ Error: {train_data_path} not found. Please run 'python main.py --mode preprocess' first."
        )
        return

    train_df = pd.read_csv(train_data_path)

    # 📝 Data Quality Fixes
    if "Date" in train_df.columns:
        train_df["Date"] = pd.to_datetime(train_df["Date"])
        train_df.set_index("Date", inplace=True)

    train_df.dropna(inplace=True)

    # 2. Create Dataset
    print("\n[2/4] Creating Dataset...")
    if "Symbol" in train_df.columns:
        data_symbols = train_df["Symbol"].unique().tolist()
        print(f"      Found {len(data_symbols)} symbols in data file.")

        target_universe = config["data"].get("stock_universes", [])

        if target_universe and len(target_universe) > 0:
            print(
                f"      Running on Config Universe: {len(target_universe)} symbols (Filtering...)"
            )
            full_df = train_df[train_df["Symbol"].isin(target_universe)]
            actual_symbols = full_df["Symbol"].unique().tolist()
            config["data"]["stock_universes"] = actual_symbols
            print(f"      Final Training Universe: {len(actual_symbols)} symbols")
        else:
            print("      Running on ALL available symbols (Config universe is empty).")
            config["data"]["stock_universes"] = data_symbols
            full_df = train_df

    dataset = FinancialDataset(config, full_df, mode="train")
    print(f"      Windows Created: {len(dataset)}")

    # 3. Initialize Model
    model_type = config["project"].get("selected_model", "tgnn").lower()
    print(f"\n[3/4] Initializing Model ({model_type.upper()})...")

    if model_type == "hybrid":
        model = HybridAgent(config)
    elif model_type == "ddpg":
        model = DDPGAgent(config)
    else:
        model = TGNN(config)

    # 4. Train
    print("\n[4/4] Starting Training Loop...")

    # Save Actual Universe used for Training to JSON
    # This is critical for DDPG to load correct input size later
    import json

    results_dir = os.path.join(config["paths"]["results_dir"], model_type)
    os.makedirs(results_dir, exist_ok=True)
    universe_path = os.path.join(results_dir, "trained_universe.json")
    with open(universe_path, "w") as f:
        json.dump(config["data"]["stock_universes"], f)
    print(f"      Saved Training Universe to: {universe_path}")

    trainer = Trainer(config, model, dataset)
    trainer.train()

    print("\n✅ Training Pipeline Completed Successfully!")
