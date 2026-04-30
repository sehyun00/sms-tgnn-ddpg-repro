import os
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.data_processor import DataProcessor
from src.preprocessing.pipeline import Pipeline


def load_config():
    config_path = Path(__file__).resolve().parents[1] / "config" / "sample.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_data_processor_adds_indicators():
    config = load_config()
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "sample_prices.csv"
    df = pd.read_csv(fixture_path)
    one_symbol = df[df["Symbol"] == "AAA"].copy()
    one_symbol["Date"] = pd.to_datetime(one_symbol["Date"])
    one_symbol = one_symbol.set_index("Date")

    processed = DataProcessor(config).add_technical_indicators(one_symbol)

    for col in ["Momentum1M", "Momentum3M", "Momentum6M", "Momentum12M", "RSI", "MACD"]:
        assert col in processed.columns
    assert processed["RSI"].dropna().between(0, 100).all()


def test_offline_pipeline_writes_date_split(tmp_path):
    config = load_config()
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "sample_prices.csv"
    config["paths"]["data_dir"] = str(tmp_path)

    Pipeline(csv_path=str(fixture_path), output_dir=str(tmp_path)).run(config)

    train = pd.read_csv(tmp_path / "train_data.csv", parse_dates=["Date"])
    test = pd.read_csv(tmp_path / "test_data.csv", parse_dates=["Date"])
    split_date = pd.Timestamp(config["training"]["test_split_date"])

    assert not train.empty
    assert not test.empty
    assert train["Date"].max() < split_date
    assert test["Date"].min() >= split_date
    assert {"Mkt_RF", "SMB", "HML", "RMW", "CMA"}.issubset(train.columns)
