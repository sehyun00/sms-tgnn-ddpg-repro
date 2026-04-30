"""
데이터 분할 모듈
Train/Test 데이터를 섹터별로 분할하고 저장합니다.
"""

import pandas as pd
import os
from typing import List, Tuple, Dict, Any


class DataSplitter:
    """Train/Test 데이터 분할 클래스."""

    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_stocks_info: List[Dict[str, Any]],
        test_stocks_info: List[Dict[str, Any]],
    ):
        """
        Args:
            train_df: Train 전체 데이터
            test_df: Test 전체 데이터
            train_stocks_info: Train 종목 정보 리스트
            test_stocks_info: Test 종목 정보 리스트
        """
        self.train_df = train_df
        self.test_df = test_df
        self.train_stocks_info = train_stocks_info
        self.test_stocks_info = test_stocks_info

    def split_by_sector(
        self, train_per_sector: int = 5, test_total: int = 10
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
        """
        섹터별로 종목을 선택하여 분할합니다.
        전략: Test Set을 먼저 선정하고, Train Set은 Test에 포함되지 않은 종목 중에서 선택합니다. (Disjoint Split)

        Args:
            train_per_sector: 섹터당 Train 종목 수
            test_total: Test 총 종목 수

        Returns:
            (train_df, test_df, train_symbols, test_symbols)
        """
        print(
            "\n✂️ 섹터별 종목 선택 중... (전략: Test Set 우선 선정 -> Train Set은 Test 제외하고 선정)"
        )

        # 1. Test 종목 우선 선택 (Generalization Test를 위해 대표 종목 선점)
        test_symbols_selected = []
        test_sectors = self.test_df["Sector"].unique()
        stocks_per_sector = max(1, test_total // len(test_sectors))

        print(f"\n[Test] 총 {test_total}개 (섹터당 약 {stocks_per_sector}개)")
        for sector in sorted(test_sectors):
            sector_symbols = self.test_df[self.test_df["Sector"] == sector][
                "Symbol"
            ].unique()

            # 데이터 품질 체크 (Row count >= 600)
            valid_symbols = []
            for sym in sector_symbols:
                count = len(self.test_df[self.test_df["Symbol"] == sym])
                if count >= 600:
                    valid_symbols.append(sym)

            # 데이터 수(Count) 기준으로 정렬하여 상위 종목 선택
            # Test 기간에 가장 데이터가 풍부하고 거래가 활발한(추정) 종목 선택
            sorted_symbols = sorted(
                valid_symbols,
                key=lambda x: len(self.test_df[self.test_df["Symbol"] == x]),
                reverse=True,
            )

            selected = sorted_symbols[:stocks_per_sector]
            test_symbols_selected.extend(selected)
            print(f"   [{sector:30s}] {len(selected):2d}개")

            if len(test_symbols_selected) >= test_total:
                break

        test_symbols_selected = test_symbols_selected[:test_total]

        # 2. Train 종목 선택 (Test Set 제외)
        train_symbols_selected = []
        train_sectors = self.train_df["Sector"].unique()

        print(f"\n[Train] 섹터당 최대 {train_per_sector}개 (Test 종목 제외)")
        for sector in sorted(train_sectors):
            sector_symbols = sorted(
                self.train_df[self.train_df["Sector"] == sector]["Symbol"].unique()
            )

            valid_symbols = []
            for sym in sector_symbols:
                # Disjoint Condition: Test Set에 이미 뽑힌 종목은 절대 Train에 넣지 않음
                if sym in test_symbols_selected:
                    continue

                count = len(self.train_df[self.train_df["Symbol"] == sym])
                if count >= 1000:
                    valid_symbols.append(sym)

            # 여기서도 데이터 많은 순 등으로 정렬 가능하지만, 기존 로직(순서대로) 유지하거나 품질순 정렬 추가 가능
            # 일단 valid_symbols 순서대로 (보통 티커송출순/알파벳순 등일 수 있음)
            selected = valid_symbols[:train_per_sector]
            train_symbols_selected.extend(selected)
            print(f"   [{sector:30s}] {len(selected):2d}개")

        # DataFrame 필터링
        final_train_df = self.train_df[
            self.train_df["Symbol"].isin(train_symbols_selected)
        ].copy()

        final_test_df = self.test_df[
            self.test_df["Symbol"].isin(test_symbols_selected)
        ].copy()

        print("\n✅ 최종 선택 (Disjoint Split):")
        print(
            f"   Train: {len(train_symbols_selected)}개 종목, {len(final_train_df):,}행"
        )
        print(
            f"   Test:  {len(test_symbols_selected)}개 종목, {len(final_test_df):,}행"
        )

        # 교집합 확인 (검증)
        intersection = set(train_symbols_selected) & set(test_symbols_selected)
        if intersection:
            print(f"⚠️ 경고: Train/Test 중복 종목 발생! {intersection}")
        else:
            print("✨ 검증 완료: Train과 Test 종목이 완벽히 분리되었습니다.")

        return (
            final_train_df,
            final_test_df,
            train_symbols_selected,
            test_symbols_selected,
        )

    def save_datasets(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str = "."
    ) -> None:
        """
        Train/Test 데이터를 CSV로 저장합니다.

        Args:
            train_df: Train 데이터
            test_df: Test 데이터
            output_dir: 출력 디렉토리
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        train_path = os.path.join(output_dir, "train_data.csv")
        test_path = os.path.join(output_dir, "test_data.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"\n💾 저장 완료: {train_path} ({len(train_df):,}행)")
        print(f"💾 저장 완료: {test_path} ({len(test_df):,}행)")
