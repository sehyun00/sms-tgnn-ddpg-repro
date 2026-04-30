import pandas as pd
import numpy as np


class TechnicalIndicators:
    """
    Calculates technical indicators for stock data.
    """

    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Adds all technical indicators to the DataFrame."""
        df = df.copy()

        # 1. Momentum
        df = TechnicalIndicators.add_momentum(df)

        # 2. Volatility
        df = TechnicalIndicators.add_volatility(df)

        # 3. RSI
        df = TechnicalIndicators.add_rsi(df)

        # 4. MACD
        df = TechnicalIndicators.add_macd(df)

        return df

    @staticmethod
    def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
        # 1M(20), 3M(60), 6M(120), 12M(252)
        df["Momentum1M"] = df["Close"].pct_change(periods=20)
        df["Momentum3M"] = df["Close"].pct_change(periods=60)
        df["Momentum6M"] = df["Close"].pct_change(periods=120)
        df["Momentum12M"] = df["Close"].pct_change(periods=252)
        return df

    @staticmethod
    def add_volatility(df: pd.DataFrame) -> pd.DataFrame:
        # 20-day annualized volatility
        df["Volatility"] = df["Close"].pct_change().rolling(window=20).std() * np.sqrt(
            252
        )
        return df

    @staticmethod
    def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # Fill NA with neutral 50
        df["RSI"] = df["RSI"].fillna(50)
        return df

    @staticmethod
    def add_macd(
        df: pd.DataFrame, short: int = 12, long: int = 26, signal: int = 9
    ) -> pd.DataFrame:
        short_ema = df["Close"].ewm(span=short, adjust=False).mean()
        long_ema = df["Close"].ewm(span=long, adjust=False).mean()

        df["MACD"] = short_ema - long_ema
        df["Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["Signal"]
        return df
