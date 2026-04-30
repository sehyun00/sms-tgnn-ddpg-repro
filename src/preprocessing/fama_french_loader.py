import pandas as pd

from typing import Optional
import logging


class FamaFrenchLoader:
    """
    Downloads and manages Fama-French 5 Factor data.
    """

    def __init__(self):
        self.ff_data: Optional[pd.DataFrame] = None
        self.logger = logging.getLogger(__name__)

    def download_factors(
        self, start_date: str = "2006-01-01", end_date: str = "2025-12-31"
    ) -> Optional[pd.DataFrame]:
        """
        Downloads Fama-French 5 Factor data from Kenneth French Data Library.
        """
        self.logger.info("📥 Downloading Fama-French 5 Factors...")
        try:
            import pandas_datareader.data as web

            # Kenneth French Data Library
            # 'F-F_Research_Data_5_Factors_2x3_daily'
            ds = web.DataReader(
                "F-F_Research_Data_5_Factors_2x3_daily",
                "famafrench",
                start=start_date,
                end=end_date,
            )

            # Index 0 is daily data (dict return)
            self.ff_data = ds[0].copy()

            # Percent to Decimal (0.5% -> 0.005)
            self.ff_data = self.ff_data / 100.0

            # Standardize Index
            self.ff_data.index.name = "Date"
            if self.ff_data.index.tz is not None:
                self.ff_data.index = self.ff_data.index.tz_localize(None)

            # Rename Columns (Mkt-RF -> Mkt_RF)
            self.ff_data.columns = [
                col.replace("-", "_").replace(" ", "") for col in self.ff_data.columns
            ]

            self.logger.info(f"✅ Fama-French data loaded: {len(self.ff_data)} rows")
            return self.ff_data

        except Exception as e:
            self.logger.error(f"❌ Failed to download Fama-French data: {e}")
            return None

    def merge_factors(self, stock_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges stock data with Fama-French data on Date.
        """
        if self.ff_data is None:
            self.logger.warning("⚠️ Fama-French data is missing. Skipping merge.")
            return stock_df

        stock_df = stock_df.copy()

        # Date Format
        stock_df["Date"] = pd.to_datetime(stock_df["Date"])
        if stock_df["Date"].dt.tz is not None:
            stock_df["Date"] = stock_df["Date"].dt.tz_localize(None)

        # Left Join
        merged_df = pd.merge(stock_df, self.ff_data, on="Date", how="left")

        # Fill NA (ffill for holidays)
        cols_to_fill = self.ff_data.columns
        merged_df[cols_to_fill] = merged_df[cols_to_fill].fillna(method="ffill")

        return merged_df
