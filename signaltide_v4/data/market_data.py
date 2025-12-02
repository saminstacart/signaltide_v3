"""
Market data provider with OHLCV, returns, volatility, and beta calculations.
"""

import logging
from typing import List, Optional, Dict, Union
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from .base import PITDataManager, DataCache
from signaltide_v4.config.settings import get_settings

logger = logging.getLogger(__name__)


class MarketDataProvider(PITDataManager):
    """
    Provider for market data (prices, returns, volatility, beta).

    Provides:
    - OHLCV data
    - Daily/monthly returns
    - Rolling volatility
    - Market beta (vs SPY)
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize market data provider."""
        super().__init__(db_path)
        self._cache = DataCache(maxsize=256)
        self._spy_returns_cache: Optional[pd.Series] = None

    def get_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        as_of_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Get OHLCV data for tickers."""
        return self.get_prices(tickers, start_date, end_date)

    def get_prices(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get price data for tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            columns: Specific columns to return (default: all OHLCV)

        Returns:
            DataFrame with MultiIndex (date, ticker) and price columns
        """
        if not tickers:
            return pd.DataFrame()

        # Create cache key
        cache_key = f"prices_{','.join(sorted(tickers))}_{start_date}_{end_date}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        # Build query
        placeholders = ','.join(['?' for _ in tickers])
        query = f"""
            SELECT date, ticker, open, high, low, close, volume, closeadj
            FROM sharadar_prices
            WHERE ticker IN ({placeholders})
            AND date BETWEEN ? AND ?
            ORDER BY date, ticker
        """

        params = tuple(tickers) + (start_date, end_date)
        df = self.execute_query(query, params)

        if len(df) == 0:
            return pd.DataFrame()

        # Convert date and set as index
        df['date'] = pd.to_datetime(df['date'], format='mixed')
        df = df.set_index(['date', 'ticker'])

        # Filter columns if specified
        if columns:
            df = df[[c for c in columns if c in df.columns]]

        self._cache.set(cache_key, df)
        return df

    def get_prices_at_date(
        self,
        tickers: List[str],
        date: str
    ) -> pd.Series:
        """Get closing prices at specific date."""
        prices = self.get_prices(tickers, date, date, columns=['close'])
        if len(prices) == 0:
            return pd.Series(dtype=float)
        return prices['close'].droplevel('date')

    def get_returns(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        period: str = 'daily'
    ) -> pd.DataFrame:
        """
        Get returns for tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date (lookback for first return)
            end_date: End date
            period: 'daily' or 'monthly'

        Returns:
            DataFrame with returns (date x ticker)
        """
        # Need extra lookback for first return
        lookback_start = (pd.Timestamp(start_date) - timedelta(days=5)).strftime('%Y-%m-%d')

        prices = self.get_prices(tickers, lookback_start, end_date, columns=['closeadj'])
        if len(prices) == 0:
            return pd.DataFrame()

        # Handle duplicate index entries before unstacking
        price_series = prices['closeadj']
        if price_series.index.duplicated().any():
            # Keep last value for each (date, ticker) combination
            price_series = price_series[~price_series.index.duplicated(keep='last')]

        # Unstack to date x ticker
        prices_wide = price_series.unstack(level='ticker')

        if period == 'monthly':
            # Resample to month-end
            prices_wide = prices_wide.resample('ME').last()

        # Calculate returns
        returns = prices_wide.pct_change()

        # Filter to requested date range
        returns = returns[returns.index >= start_date]
        returns = returns[returns.index <= end_date]

        return returns

    def get_cumulative_returns(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        skip_recent_days: int = 0
    ) -> pd.Series:
        """
        Get cumulative returns over period.

        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            skip_recent_days: Days to skip at end (for momentum skip)

        Returns:
            Series with cumulative returns per ticker
        """
        # Adjust end date if skipping recent days
        if skip_recent_days > 0:
            adj_end = (pd.Timestamp(end_date) - timedelta(days=skip_recent_days))
            end_date = adj_end.strftime('%Y-%m-%d')

        prices = self.get_prices(tickers, start_date, end_date, columns=['closeadj'])
        if len(prices) == 0:
            return pd.Series(dtype=float)

        # Handle duplicate index entries before unstacking
        price_series = prices['closeadj']
        if price_series.index.duplicated().any():
            price_series = price_series[~price_series.index.duplicated(keep='last')]

        # Pivot to date x ticker
        prices_wide = price_series.unstack(level='ticker')

        # Get first and last prices
        first_prices = prices_wide.iloc[0]
        last_prices = prices_wide.iloc[-1]

        # Calculate cumulative returns
        cum_returns = (last_prices / first_prices) - 1

        return cum_returns

    def get_volatility(
        self,
        tickers: List[str],
        as_of_date: str,
        lookback_days: int = 252
    ) -> pd.Series:
        """
        Get annualized volatility for tickers.

        Args:
            tickers: List of ticker symbols
            as_of_date: Calculate as of this date
            lookback_days: Number of trading days for calculation

        Returns:
            Series with annualized volatility per ticker
        """
        # Calculate lookback start
        start_date = (pd.Timestamp(as_of_date) - timedelta(days=int(lookback_days * 1.5)))
        start_date = start_date.strftime('%Y-%m-%d')

        returns = self.get_returns(tickers, start_date, as_of_date, period='daily')
        if len(returns) == 0:
            return pd.Series(dtype=float)

        # Take last lookback_days
        returns = returns.tail(lookback_days)

        # Annualized volatility
        volatility = returns.std() * np.sqrt(252)

        return volatility

    def get_beta(
        self,
        tickers: List[str],
        as_of_date: str,
        lookback_days: int = 252
    ) -> pd.Series:
        """
        Get market beta (vs SPY) for tickers.

        Args:
            tickers: List of ticker symbols
            as_of_date: Calculate as of this date
            lookback_days: Number of trading days

        Returns:
            Series with beta per ticker
        """
        # Calculate lookback start
        start_date = (pd.Timestamp(as_of_date) - timedelta(days=int(lookback_days * 1.5)))
        start_date = start_date.strftime('%Y-%m-%d')

        # Get stock returns
        stock_returns = self.get_returns(tickers, start_date, as_of_date, period='daily')
        if len(stock_returns) == 0:
            return pd.Series(dtype=float)

        # Get SPY returns
        spy_returns = self.get_returns(['SPY'], start_date, as_of_date, period='daily')
        if len(spy_returns) == 0:
            return pd.Series(1.0, index=tickers)  # Default beta = 1

        spy_returns = spy_returns['SPY']

        # Align dates
        common_dates = stock_returns.index.intersection(spy_returns.index)
        stock_returns = stock_returns.loc[common_dates].tail(lookback_days)
        spy_returns = spy_returns.loc[common_dates].tail(lookback_days)

        # Calculate beta for each stock
        betas = {}
        spy_var = spy_returns.var()

        if spy_var == 0:
            return pd.Series(1.0, index=tickers)

        for ticker in stock_returns.columns:
            stock_ret = stock_returns[ticker].dropna()
            common = stock_ret.index.intersection(spy_returns.index)
            if len(common) < 60:  # Minimum observations
                betas[ticker] = 1.0
                continue

            cov = stock_ret.loc[common].cov(spy_returns.loc[common])
            betas[ticker] = cov / spy_var

        return pd.Series(betas)

    def get_month_end_dates(
        self,
        start_date: str,
        end_date: str
    ) -> List[str]:
        """Get list of month-end trading dates."""
        query = """
            SELECT DISTINCT date
            FROM sharadar_prices
            WHERE ticker = 'SPY'
            AND date BETWEEN ? AND ?
            ORDER BY date
        """
        df = self.execute_query(query, (start_date, end_date))
        if len(df) == 0:
            return []

        df['date'] = pd.to_datetime(df['date'], format='mixed')
        df = df.set_index('date')

        # Get last trading day of each month
        month_ends = df.resample('ME').last()

        return [d.strftime('%Y-%m-%d') for d in month_ends.index]
