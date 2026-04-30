from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "tests" / "fixtures" / "sample_prices.csv"


def build_sample_prices() -> pd.DataFrame:
    """Create deterministic synthetic OHLCV rows for smoke tests."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", "2021-06-30")
    symbols = [
        ("AAA", "Technology", 100.0, 0.0008),
        ("BBB", "Healthcare", 80.0, 0.0005),
        ("CCC", "Industrials", 60.0, 0.0003),
    ]
    rows = []

    for symbol, sector, base_price, drift in symbols:
        price = base_price
        for i, date in enumerate(dates):
            seasonal = 0.002 * np.sin(i / 18.0)
            shock = rng.normal(drift + seasonal, 0.01)
            close = max(1.0, price * (1.0 + shock))
            open_price = price * (1.0 + rng.normal(0.0, 0.002))
            high = max(open_price, close) * (1.0 + abs(rng.normal(0.002, 0.001)))
            low = min(open_price, close) * (1.0 - abs(rng.normal(0.002, 0.001)))
            volume = int(1_000_000 + 50_000 * np.sin(i / 10.0) + rng.normal(0, 5000))
            rows.append(
                {
                    "Date": date.strftime("%Y-%m-%d"),
                    "Symbol": symbol,
                    "Sector": sector,
                    "Industry": f"{sector} Sample",
                    "Open": round(open_price, 4),
                    "High": round(high, 4),
                    "Low": round(low, 4),
                    "Close": round(close, 4),
                    "Volume": max(volume, 1),
                    "Mkt_RF": round(0.0002 + 0.0001 * np.sin(i / 21.0), 6),
                    "SMB": round(0.0001 * np.cos(i / 17.0), 6),
                    "HML": round(0.0001 * np.sin(i / 13.0), 6),
                    "RMW": round(0.0001 * np.cos(i / 19.0), 6),
                    "CMA": round(0.0001 * np.sin(i / 23.0), 6),
                    "RF": 0.00001,
                }
            )
            price = close

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    build_sample_prices().to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
