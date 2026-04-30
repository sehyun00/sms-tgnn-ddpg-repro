"""
데이터 수집 모듈
NASDAQ 종목 자동 추출 및 Yahoo Finance를 통한 주가 데이터 수집을 담당합니다.
"""

import pandas as pd
import yfinance as yf
import requests
import os
import time
from typing import Dict, List, Any, Optional


class DataCollector:
    """NASDAQ 종목 수집 및 주가 데이터 다운로드 클래스."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 설정 딕셔너리 (선택적)
        """
        self.config = config or {}
        self.stocks: List[Dict[str, str]] = []

    def fetch_nasdaq_from_wikipedia(self, target: str = "sp500") -> pd.DataFrame:
        """
        Wikipedia에서 주요 지수 구성 종목을 크롤링합니다.

        Args:
            target: 'nasdaq100' 또는 'sp500'

        Returns:
            종목 정보 DataFrame (ticker, name, sector, industry)
        """
        print(f"\n📡 Wikipedia에서 {target.upper()} 종목 크롤링 중...")

        urls = {
            "nasdaq100": "https://en.wikipedia.org/wiki/Nasdaq-100",
            "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        }

        url = urls.get(target, urls["nasdaq100"])

        try:
            # 403 방지를 위한 User-Agent 헤더
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            response = requests.get(url, headers=headers)
            response.raise_for_status()

            tables = pd.read_html(response.text)

            # 종목 테이블 찾기
            nasdaq_df = None
            for table in tables:
                if "Ticker" in table.columns or "Symbol" in table.columns:
                    nasdaq_df = table
                    break

            if nasdaq_df is None:
                raise ValueError("종목 테이블을 찾을 수 없습니다")

            # 컬럼명 표준화
            nasdaq_df.columns = nasdaq_df.columns.str.strip()

            if "Ticker" in nasdaq_df.columns:
                ticker_col = "Ticker"
            else:
                ticker_col = "Symbol"

            # 컬럼 매핑
            column_mapping = {
                ticker_col: "ticker",
                "Company": "name",
                "GICS Sector": "sector",
                "GICS Sub-Industry": "industry",
            }

            available_cols = [
                col for col in column_mapping.keys() if col in nasdaq_df.columns
            ]
            nasdaq_df = nasdaq_df[available_cols].copy()
            nasdaq_df = nasdaq_df.rename(columns=column_mapping)

            nasdaq_df = nasdaq_df.dropna(subset=["ticker"])
            nasdaq_df["ticker"] = nasdaq_df["ticker"].str.strip()

            print(f"✅ {len(nasdaq_df)}개 종목 크롤링 완료")
            return nasdaq_df

        except Exception as e:
            print(f"❌ Wikipedia 크롤링 실패: {e}")
            return pd.DataFrame()

    def enrich_with_yfinance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        yfinance로 섹터/산업 정보를 보완합니다.

        Args:
            df: 기본 종목 정보 DataFrame

        Returns:
            보완된 종목 정보 DataFrame
        """
        print("\n🔍 yfinance로 종목 정보 보완 중...")

        enriched_data = []
        total = len(df)

        for idx, row in df.iterrows():
            ticker = row["ticker"]

            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                name = row.get("name", info.get("longName", ticker))
                sector = row.get("sector", info.get("sector", "Unknown"))
                industry = row.get("industry", info.get("industry", "Unknown"))

                # NaN 체크
                if pd.isna(sector) or sector == "":
                    sector = info.get("sector", "Unknown")
                if pd.isna(industry) or industry == "":
                    industry = info.get("industry", "Unknown")

                enriched_data.append(
                    {
                        "ticker": ticker,
                        "name": name,
                        "sector": sector,
                        "industry": industry,
                    }
                )

                print(f"  ({idx + 1}/{total}) ✅ {ticker}: {sector}")
                time.sleep(0.1)  # API 제한 방지

            except Exception:
                print(f"  ({idx + 1}/{total}) ⚠️ {ticker}: 실패")
                enriched_data.append(
                    {
                        "ticker": ticker,
                        "name": row.get("name", ticker),
                        "sector": row.get("sector", "Unknown"),
                        "industry": row.get("industry", "Unknown"),
                    }
                )

        result_df = pd.DataFrame(enriched_data)
        print(f"\n✅ {len(result_df)}개 종목 정보 보완 완료")
        return result_df

    def load_stocks_from_csv(self, file_path: str) -> List[Dict[str, str]]:
        """
        CSV 파일에서 종목 리스트를 로드합니다.

        Args:
            file_path: CSV 파일 경로 (필수 컬럼: Symbol, Sector)

        Returns:
            종목 정보 리스트
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        df = pd.read_csv(file_path)

        if "Ticker" in df.columns:
            df = df.rename(columns={"Ticker": "Symbol"})
        if "ticker" in df.columns:
            df = df.rename(columns={"ticker": "Symbol"})
        if "sector" in df.columns:
            df = df.rename(columns={"sector": "Sector"})

        required_cols = ["Symbol", "Sector"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV에 필수 컬럼이 필요합니다: {required_cols}")

        self.stocks = df.to_dict("records")
        print(f"✅ {len(self.stocks)}개 종목 로드 완료: {file_path}")
        return self.stocks

    def load_stocks_auto(self, target: str = "sp500") -> List[Dict[str, str]]:
        """
        CSV 없이 Wikipedia에서 자동으로 종목을 로드합니다.

        Args:
            target: 'nasdaq100' 또는 'sp500'

        Returns:
            종목 정보 리스트
        """
        df = self.fetch_nasdaq_from_wikipedia(target)

        if df.empty:
            raise ValueError("종목 리스트를 가져올 수 없습니다")

        # yfinance로 섹터 정보 보완
        df = self.enrich_with_yfinance(df)

        # 컬럼명 통일
        df = df.rename(
            columns={"ticker": "Symbol", "sector": "Sector", "industry": "Industry"}
        )

        self.stocks = df.to_dict("records")
        print(f"\n✅ 총 {len(self.stocks)}개 종목 자동 로드 완료")
        return self.stocks

    def filter_survivor_stocks(
        self, start_year: int = 2006, end_year: int = 2025
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Train/Test 기간별로 생존 종목을 필터링합니다.

        Args:
            start_year: 시작 연도
            end_year: 종료 연도

        Returns:
            {'train': [...], 'test': [...]} 형태의 종목 정보
        """
        print(f"\n🔍 생존 종목 필터링 중 ({start_year}-{end_year})...")

        survivors = {"train": [], "test": []}
        train_end = 2020
        test_start = 2021

        total = len(self.stocks)
        for idx, stock in enumerate(self.stocks):
            symbol = stock["Symbol"]

            try:
                hist = yf.download(
                    symbol,
                    start=f"{start_year}-01-01",
                    end=f"{end_year + 1}-01-01",
                    progress=False,
                    auto_adjust=True,
                )

                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = hist.columns.droplevel(1)

                if hist.empty:
                    continue

                # Train 기간 확인 (2006-2020)
                train_data = hist[hist.index.year <= train_end]

                # 1. 데이터 시작일 체크 (Survivorship Bias 방지)
                # 요청하신 대로 start_year(2006)부터 데이터가 존재하는 종목만 선별
                if train_data.empty or train_data.index[0].year > start_year:
                    # print(f"  ({idx + 1}/{total}) ⚠️ {symbol}: {start_year}년 데이터 부재 (Start: {train_data.index[0].year if not train_data.empty else 'N/A'})")
                    continue

                # 2. 데이터 길이 체크
                if len(train_data) >= 5 * 200:  # 최소 5년 데이터
                    survivors["train"].append(stock)

                # Test 기간 확인 (2021-2025)
                test_data = hist[hist.index.year >= test_start]
                if len(test_data) >= 3 * 200:  # 최소 3년 데이터
                    survivors["test"].append(stock)

                print(
                    f"  ({idx + 1}/{total}) ✅ {symbol}: Train={len(train_data)}, Test={len(test_data)}"
                )

            except Exception as e:
                print(f"  ({idx + 1}/{total}) ❌ {symbol}: 오류 ({e})")

        print(f"\n   ✅ Train 후보: {len(survivors['train'])}개")
        print(f"   ✅ Test 후보: {len(survivors['test'])}개")

        return survivors

    def fetch_daily_data(
        self, symbol: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        특정 종목의 일별 OHLCV 데이터를 가져옵니다.

        Args:
            symbol: 종목 심볼
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)

        Returns:
            OHLCV DataFrame 또는 None
        """
        try:
            df = yf.download(
                symbol, start=start_date, end=end_date, progress=False, auto_adjust=True
            )

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            if df.empty:
                return None

            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df["Symbol"] = symbol
            return df

        except Exception as e:
            print(f"❌ {symbol} 데이터 수집 실패: {e}")
            return None
