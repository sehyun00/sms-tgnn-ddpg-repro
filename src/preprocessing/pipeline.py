"""
전처리 파이프라인 모듈
데이터 수집부터 저장까지 전체 흐름을 통합 실행합니다.

실행 방법:
    # config의 raw_prices_path 또는 다운로드 경로 사용
    python main.py --mode preprocess

    # CSV 파일 지정
    python main.py --mode preprocess --csv tests/fixtures/sample_prices.csv
"""

import pandas as pd
from pathlib import Path
from typing import Optional

from .data_collector import DataCollector
from .data_processor import DataProcessor
from .data_splitter import DataSplitter
from .fama_french_loader import FamaFrenchLoader


class Pipeline:
    """전처리 파이프라인 통합 클래스."""

    def __init__(
        self,
        csv_path: Optional[str] = None,
        output_dir: str = "data",
        start_year: int = 2006,
        end_year: int = 2025,
    ):
        """
        Args:
            csv_path: 종목 리스트 CSV 경로 (None이면 자동 크롤링)
            output_dir: 출력 디렉토리
            start_year: 시작 연도
            end_year: 종료 연도
        """
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.start_year = start_year
        self.end_year = end_year

        self.collector = DataCollector()
        self.ff_loader = FamaFrenchLoader()

    def run(self, config: Optional[dict] = None) -> None:
        """
        전체 파이프라인을 실행합니다.

        Args:
            config: 설정 딕셔너리 (선택적)
        """
        config = config or {}
        if self.csv_path and self._is_raw_price_csv(Path(self.csv_path)):
            self._run_offline_prices(Path(self.csv_path), config)
            return

        print("=" * 60)
        print("🚀 전처리 파이프라인 시작")
        print("=" * 60)

        # 1. 종목 로드 (CSV 또는 자동 크롤링)
        if self.csv_path and Path(self.csv_path).exists():
            print(f"\n📂 CSV에서 종목 로드: {self.csv_path}")
            self.collector.load_stocks_from_csv(self.csv_path)
        else:
            print("\n📡 S&P 500 종목 자동 크롤링 모드")
            self.collector.load_stocks_auto(target="sp500")

        # 2. 생존 종목 필터링
        survivors = self.collector.filter_survivor_stocks(
            start_year=self.start_year, end_year=self.end_year
        )

        train_candidate_stocks = survivors["train"]
        test_candidate_stocks = survivors["test"]

        if not train_candidate_stocks:
            print("❌ Train 종목이 없습니다. 종료합니다.")
            return

        # 3. Fama-French 데이터 다운로드
        print("\n📊 Fama-French 5 Factor 다운로드 중...")
        try:
            self.ff_loader.download_factors(
                start_date=f"{self.start_year}-01-01",
                end_date=f"{self.end_year}-12-31",  # Fama-French loader handles this internally/pandas-datareader usually checks range
            )
        except Exception as e:
            print(f"⚠️ Fama-French 다운로드 실패: {e}")

        # 4. Train 데이터 수집 (2006-2020)
        print(
            f"\n📊 Train 데이터 수집 중 ({len(train_candidate_stocks)}개 종목, 2006-2020)..."
        )
        train_data_list = self._collect_and_process(
            train_candidate_stocks,
            start_date="2006-01-01",
            end_date="2021-01-01",  # 2020-12-31 포함을 위해 +1일
            config=config,
        )

        # 5. Test 데이터 수집 (2021-현재)
        print(
            f"\n📊 Test 데이터 수집 중 ({len(test_candidate_stocks)}개 종목, 2021-현재)..."
        )
        test_data_list = self._collect_and_process(
            test_candidate_stocks,
            start_date="2021-01-01",
            end_date="2026-01-01",  # 2025-12-31 포함을 위해 +1일
            config=config,
        )

        if not train_data_list:
            print("❌ Train 데이터가 없습니다.")
            return

        # 6. 병합
        train_full_df = pd.concat(train_data_list, ignore_index=True)
        test_full_df = (
            pd.concat(test_data_list, ignore_index=True)
            if test_data_list
            else pd.DataFrame()
        )

        print(f"\n📊 Train 총: {len(train_full_df):,}행")
        print(f"📊 Test 총: {len(test_full_df):,}행")

        # 7. Fama-French 병합
        try:
            train_full_df = self.ff_loader.merge_factors(train_full_df)
            if not test_full_df.empty:
                test_full_df = self.ff_loader.merge_factors(test_full_df)
            print("✅ Fama-French 팩터 병합 완료")
        except Exception as e:
            print(f"⚠️ Fama-French 병합 실패: {e}")

        # 8. 섹터별 분할
        if not test_full_df.empty:
            splitter = DataSplitter(
                train_full_df,
                test_full_df,
                train_candidate_stocks,
                test_candidate_stocks,
            )

            final_train_df, final_test_df, train_symbols, test_symbols = (
                splitter.split_by_sector(train_per_sector=5, test_total=10)
            )

            # 9. 저장
            splitter.save_datasets(final_train_df, final_test_df, self.output_dir)

            print("\n" + "=" * 60)
            print("✅ 파이프라인 완료!")
            print("=" * 60)
            print(f"   Train 종목 ({len(train_symbols)}개): {train_symbols[:5]}...")
            print(f"   Test 종목 ({len(test_symbols)}개): {test_symbols}")
        else:
            # Test 데이터 없이 Train만 저장
            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            train_full_df.to_csv(output_path / "train_data.csv", index=False)
            print(f"\n✅ Train 데이터만 저장: {output_path / 'train_data.csv'}")

    def _is_raw_price_csv(self, path: Path) -> bool:
        """Return True when the CSV contains local OHLCV rows, not only tickers."""
        if not path.exists() or path.suffix.lower() != ".csv":
            return False
        try:
            columns = set(pd.read_csv(path, nrows=1).columns)
        except Exception:
            return False
        return {"Date", "Symbol", "Open", "High", "Low", "Close", "Volume"}.issubset(
            columns
        )

    def _run_offline_prices(self, raw_path: Path, config: dict) -> None:
        """Build train/test CSV files from a local raw price fixture."""
        print("=" * 60)
        print("Offline preprocessing from local price CSV")
        print("=" * 60)
        print(f"Input: {raw_path}")

        df = pd.read_csv(raw_path)
        required = {"Date", "Symbol", "Open", "High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Raw price CSV is missing columns: {sorted(missing)}")

        df["Date"] = pd.to_datetime(df["Date"])
        processor = DataProcessor(config)
        factor_cols = ["Mkt_RF", "SMB", "HML", "RMW", "CMA"]
        processed = []

        for symbol, group in df.sort_values(["Symbol", "Date"]).groupby("Symbol"):
            stock_df = group.set_index("Date").copy()
            stock_df = processor.add_technical_indicators(stock_df)
            stock_df["Date"] = stock_df.index
            stock_df["Symbol"] = symbol
            if "Sector" not in stock_df:
                stock_df["Sector"] = "Unknown"
            if "Industry" not in stock_df:
                stock_df["Industry"] = "Unknown"
            for col in factor_cols:
                if col not in stock_df:
                    stock_df[col] = 0.0
            processed.append(stock_df.reset_index(drop=True))

        full_df = pd.concat(processed, ignore_index=True)
        needed = (
            list(config.get("data", {}).get("features", []))
            + factor_cols
            + ["Momentum1M", "Momentum3M", "Momentum6M", "Momentum12M"]
        )
        full_df = full_df.dropna(subset=[c for c in needed if c in full_df.columns])

        split_date = pd.to_datetime(
            config.get("training", {}).get("test_split_date", "2021-01-01")
        )
        train_df = full_df[full_df["Date"] < split_date].copy()
        test_df = full_df[full_df["Date"] >= split_date].copy()
        if train_df.empty or test_df.empty:
            raise ValueError(
                "Offline preprocessing produced an empty train or test split. "
                "Adjust training.test_split_date or provide a longer raw CSV."
            )

        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(output_path / "train_data.csv", index=False)
        test_df.to_csv(output_path / "test_data.csv", index=False)
        print(f"Saved train rows: {len(train_df):,} -> {output_path / 'train_data.csv'}")
        print(f"Saved test rows:  {len(test_df):,} -> {output_path / 'test_data.csv'}")

    def _collect_and_process(
        self, stocks_info: list, start_date: str, end_date: str, config: Optional[dict]
    ) -> list:
        """종목 데이터 수집 및 전처리."""
        data_list = []
        processor = DataProcessor(config or {})
        total = len(stocks_info)

        for idx, stock_info in enumerate(stocks_info):
            symbol = stock_info["Symbol"]
            df = self.collector.fetch_daily_data(symbol, start_date, end_date)

            if df is None or df.empty:
                continue

            # 전처리
            df = processor.add_technical_indicators(df)

            # 메타데이터
            df["Sector"] = stock_info.get("Sector", "Unknown")
            df["Industry"] = stock_info.get("Industry", "Unknown")
            df["Date"] = df.index

            data_list.append(df)
            print(f"  ({idx + 1}/{total}) ✅ {symbol}: {len(df)}행")

        return data_list


def main():
    """CLI 진입점."""
    import argparse

    # 경로 설정
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent  # → project root

    DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data"

    parser = argparse.ArgumentParser(
        description="SMS Backtesting 전처리 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # config의 raw_prices_path 또는 다운로드 경로 사용
  python main.py --mode preprocess

  # CSV 파일 지정
  python main.py --mode preprocess --csv tests/fixtures/sample_prices.csv
        """,
    )

    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="종목 리스트 CSV 경로 (생략 시 자동 크롤링)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"출력 디렉토리 (기본값: {DEFAULT_OUTPUT_DIR})",
    )

    args = parser.parse_args()

    pipeline = Pipeline(csv_path=args.csv, output_dir=args.output)
    pipeline.run()


if __name__ == "__main__":
    main()
