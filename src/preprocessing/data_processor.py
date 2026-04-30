import pandas as pd
from typing import Dict, Any
from .indicators import TechnicalIndicators
from .fama_french_loader import FamaFrenchLoader


class DataProcessor:
    """
    Processes raw stock data into 5-Factor model data.
    Acts as a Facade for Indicators and External Fama-French Data.

    [주의] Fama-French 5-Factor는 외부 데이터에서 로드합니다.
    자체 팩터 계산(레거시 FactorCalculator)은 제거되었습니다.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataProcessor with configuration.
        """
        self.config = config
        self.ff_loader = FamaFrenchLoader()

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds technical indicators using TechnicalIndicators module.
        """
        return TechnicalIndicators.add_all_indicators(df)

    def merge_fama_french_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Downloads and merges Fama-French 5 Factors.
        Fama-French 컬럼: Mkt_RF, SMB, HML, RMW, CMA, RF
        """
        # Download if not already valid
        if self.ff_loader.ff_data is None:
            self.ff_loader.download_factors()

        return self.ff_loader.merge_factors(df)
