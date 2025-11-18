"""
DataManager: Central interface for all market data access.

This is the single source of truth for market data in SignalTide v3.
Enforces point-in-time data access to prevent lookahead bias.
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from functools import lru_cache
import hashlib
import config
from data.database import Database


class DataCache:
    """
    Simple in-memory cache for frequently accessed data.

    Uses LRU eviction and size limits from config.
    """

    def __init__(self, max_size_mb: int = 500):
        """
        Initialize cache.

        Args:
            max_size_mb: Maximum cache size in megabytes
        """
        self.max_size_mb = max_size_mb
        self.cache: Dict[str, pd.DataFrame] = {}
        self.access_times: Dict[str, datetime] = {}

    def _make_key(self, **kwargs) -> str:
        """Create cache key from parameters."""
        key_str = str(sorted(kwargs.items()))
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, **kwargs) -> Optional[pd.DataFrame]:
        """Get data from cache if available."""
        key = self._make_key(**kwargs)
        if key in self.cache:
            self.access_times[key] = datetime.now()
            return self.cache[key].copy()  # Return copy to prevent mutation
        return None

    def put(self, data: pd.DataFrame, **kwargs):
        """Store data in cache."""
        key = self._make_key(**kwargs)
        self.cache[key] = data.copy()
        self.access_times[key] = datetime.now()
        self._evict_if_needed()

    def _evict_if_needed(self):
        """Evict least recently used items if cache is too large."""
        # Simple size check (rough estimate)
        total_size_mb = sum(df.memory_usage(deep=True).sum() for df in self.cache.values()) / 1024 / 1024

        if total_size_mb > self.max_size_mb:
            # Evict least recently used
            sorted_keys = sorted(self.access_times.items(), key=lambda x: x[1])
            to_remove = sorted_keys[0][0]
            del self.cache[to_remove]
            del self.access_times[to_remove]

    def clear(self):
        """Clear all cached data."""
        self.cache.clear()
        self.access_times.clear()

    def size_mb(self) -> float:
        """Get current cache size in MB."""
        if not self.cache:
            return 0.0
        return sum(df.memory_usage(deep=True).sum() for df in self.cache.values()) / 1024 / 1024


class DataManager:
    """
    Central interface for all market data operations.

    Key Features:
    - Point-in-time data access (prevents lookahead bias)
    - In-memory caching for performance
    - Data quality validation
    - Support for Sharadar data format (price, fundamentals, insider)

    CRITICAL: All data retrieval methods accept 'as_of' parameter to ensure
    no future data is used in backtests or signal generation.
    """

    def __init__(self, db_path: Optional[Path] = None,
                 enable_cache: bool = True):
        """
        Initialize DataManager.

        Args:
            db_path: Path to database (default: from config)
            enable_cache: Whether to enable caching (default: True)
        """
        self.db = Database(db_path)
        self.enable_cache = enable_cache and config.ENABLE_CACHE

        if self.enable_cache:
            self.cache = DataCache(max_size_mb=config.CACHE_SIZE_MB)
        else:
            self.cache = None

    def get_price_data(self, ticker: Union[str, List[str]],
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None,
                       as_of: Optional[datetime] = None,
                       validate: bool = True) -> pd.DataFrame:
        """
        Get OHLCV price data with point-in-time constraints.

        CRITICAL: The 'as_of' parameter ensures we only get data that would
        have been available at that point in time. This prevents lookahead bias.

        Args:
            ticker: Single ticker or list of tickers
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            as_of: Point-in-time constraint (only data available as of this date)
            validate: Whether to run data quality checks

        Returns:
            DataFrame with OHLCV data, indexed by date
            Columns: open, high, low, close, volume, ticker (if multiple tickers)

        Example:
            ```python
            # Get data as it would have been known on 2023-01-01
            data = dm.get_price_data('AAPL',
                                     start_date=datetime(2022, 1, 1),
                                     end_date=datetime(2022, 12, 31),
                                     as_of=datetime(2023, 1, 1))
            ```
        """
        # Check cache first
        if self.enable_cache:
            cache_key = {
                'ticker': ticker,
                'start': start_date,
                'end': end_date,
                'as_of': as_of,
                'type': 'price'
            }
            cached = self.cache.get(**cache_key)
            if cached is not None:
                # Apply same post-processing as uncached data
                if isinstance(ticker, str) and 'ticker' in cached.columns:
                    return cached.drop(columns=['ticker'])
                return cached

        # Handle single ticker vs list
        if isinstance(ticker, str):
            tickers = [ticker]
        else:
            tickers = ticker

        # Retrieve data for each ticker
        dfs = []
        for t in tickers:
            df = self.db.get_prices(
                ticker=t,
                start_date=start_date,
                end_date=end_date,
                as_of=as_of
            )

            if len(df) > 0:
                df['ticker'] = t
                dfs.append(df)

        if not dfs:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'ticker'])

        # Combine all tickers
        result = pd.concat(dfs, axis=0)

        # Validate data quality
        if validate:
            self._validate_price_data(result, tickers)

        # Cache result
        if self.enable_cache:
            self.cache.put(result, **cache_key)

        # If single ticker, drop ticker column for convenience
        if isinstance(ticker, str):
            result = result.drop(columns=['ticker'])

        return result

    def get_fundamental_data(self, ticker: Union[str, List[str]],
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None,
                           as_of: Optional[datetime] = None,
                           dimension: str = 'ARQ',
                           metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get fundamental data with point-in-time constraints.

        CRITICAL: Uses filing_date, not report_period, for point-in-time.
        We only know fundamentals after they're filed, not when the period ends.

        Args:
            ticker: Single ticker or list of tickers
            start_date: Start date (based on filing_date)
            end_date: End date (based on filing_date)
            as_of: Point-in-time constraint (based on filing_date)
            dimension: ARQ (annual), MRQ (quarterly), ART, MRT
            metrics: Specific metrics to return (None for all)

        Returns:
            DataFrame with fundamental data, indexed by filing_date
        """
        # Check cache
        if self.enable_cache:
            cache_key = {
                'ticker': ticker,
                'start': start_date,
                'end': end_date,
                'as_of': as_of,
                'dimension': dimension,
                'type': 'fundamental'
            }
            cached = self.cache.get(**cache_key)
            if cached is not None:
                if metrics:
                    return cached[metrics + ['ticker', 'filing_date']]
                return cached

        # Handle single ticker vs list
        if isinstance(ticker, str):
            tickers = [ticker]
        else:
            tickers = ticker

        # Retrieve data
        dfs = []
        for t in tickers:
            df = self.db.get_fundamentals(
                ticker=t,
                start_date=start_date,
                end_date=end_date,
                as_of=as_of,
                dimension=dimension
            )

            if len(df) > 0:
                df['ticker'] = t
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        result = pd.concat(dfs, axis=0)

        # Set index to filing_date (critical for point-in-time)
        result = result.set_index('filing_date')

        # Filter to specific metrics if requested
        if metrics:
            available_metrics = [m for m in metrics if m in result.columns]
            result = result[available_metrics + ['ticker']]

        # Cache result
        if self.enable_cache:
            self.cache.put(result, **cache_key)

        return result

    def get_insider_trades(self, ticker: Union[str, List[str]],
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None,
                          as_of: Optional[datetime] = None,
                          transaction_type: Optional[str] = None) -> pd.DataFrame:
        """
        Get insider trading data.

        Args:
            ticker: Single ticker or list of tickers
            start_date: Start date (based on filing_date)
            end_date: End date (based on filing_date)
            as_of: Point-in-time constraint
            transaction_type: Filter by transaction type (P=Purchase, S=Sale)

        Returns:
            DataFrame with insider trades
        """
        # Handle single ticker vs list
        if isinstance(ticker, str):
            tickers = [ticker]
        else:
            tickers = ticker

        # Retrieve data
        dfs = []
        for t in tickers:
            df = self.db.get_insider_trades(
                ticker=t,
                start_date=start_date,
                end_date=end_date,
                as_of=as_of
            )

            if len(df) > 0:
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        result = pd.concat(dfs, axis=0)

        # Filter by transaction type if specified
        if transaction_type:
            result = result[result['transaction_type'] == transaction_type]

        return result

    def get_combined_data(self, ticker: str,
                         start_date: datetime,
                         end_date: datetime,
                         as_of: Optional[datetime] = None,
                         include_fundamentals: bool = True,
                         include_insider: bool = False) -> pd.DataFrame:
        """
        Get combined price + fundamental + insider data aligned by date.

        This is a convenience method for signals that need multiple data types.

        Args:
            ticker: Stock ticker
            start_date: Start date
            end_date: End date
            as_of: Point-in-time constraint
            include_fundamentals: Whether to include fundamental data
            include_insider: Whether to include insider trading features

        Returns:
            DataFrame with all data types aligned by date
        """
        # Get price data
        prices = self.get_price_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            as_of=as_of
        )

        if len(prices) == 0:
            return pd.DataFrame()

        result = prices.copy()

        # Add fundamentals if requested
        if include_fundamentals:
            fundamentals = self.get_fundamental_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                as_of=as_of
            )

            if len(fundamentals) > 0:
                # Forward-fill fundamentals to match price dates
                # (fundamentals are reported quarterly/annually)
                # Drop columns that would cause conflicts
                cols_to_drop = ['ticker', 'id', 'created_at']
                fundamentals = fundamentals.drop(columns=cols_to_drop, errors='ignore')
                result = result.join(fundamentals, how='left')
                result = result.fillna(method='ffill')

        # Add insider trading features if requested
        if include_insider:
            insider = self.get_insider_trades(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                as_of=as_of
            )

            if len(insider) > 0:
                # Aggregate insider trades by date
                insider_agg = self._aggregate_insider_trades(insider)
                result = result.join(insider_agg, how='left')
                result = result.fillna(0)  # No insider activity = 0

        return result

    def _aggregate_insider_trades(self, insider_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate insider trades into daily features.

        Creates features like:
        - Net insider buying (purchases - sales)
        - Number of insiders buying/selling
        - Total value of insider trades

        Args:
            insider_df: Raw insider trading data

        Returns:
            DataFrame with aggregated features by date
        """
        if len(insider_df) == 0:
            return pd.DataFrame()

        # Group by filing_date
        agg = insider_df.groupby('filing_date').agg({
            'shares': ['sum', 'count'],
            'price_per_share': 'mean'
        })

        # Flatten column names
        agg.columns = ['_'.join(col).strip() for col in agg.columns.values]

        # Calculate net buying
        purchases = insider_df[insider_df['transaction_type'] == 'P'].groupby('filing_date')['shares'].sum()
        sales = insider_df[insider_df['transaction_type'] == 'S'].groupby('filing_date')['shares'].sum()

        agg['insider_net_buying'] = purchases.sub(sales, fill_value=0)

        return agg

    def _validate_price_data(self, df: pd.DataFrame, tickers: List[str]):
        """
        Validate price data quality.

        Checks for:
        - Missing values
        - Price anomalies (e.g., close outside high/low)
        - Volume anomalies
        - Data gaps

        Logs issues to database for review.
        """
        if len(df) == 0:
            return

        for ticker in tickers:
            ticker_data = df[df['ticker'] == ticker] if 'ticker' in df.columns else df

            # Check for missing values
            missing = ticker_data[['open', 'high', 'low', 'close', 'volume']].isna().sum()
            if missing.any():
                self.db.log_data_quality_issue(
                    table_name='prices',
                    ticker=ticker,
                    issue_type='missing_values',
                    description=f"Missing values: {missing[missing > 0].to_dict()}",
                    severity='WARNING'
                )

            # Check for price anomalies
            anomalies = (
                (ticker_data['close'] > ticker_data['high']) |
                (ticker_data['close'] < ticker_data['low'])
            )
            if anomalies.any():
                self.db.log_data_quality_issue(
                    table_name='prices',
                    ticker=ticker,
                    issue_type='price_anomaly',
                    description=f"Close price outside high/low range on {anomalies.sum()} days",
                    severity='ERROR'
                )

            # Check for zero/negative prices
            invalid_prices = (
                (ticker_data['close'] <= 0) |
                (ticker_data['high'] <= 0) |
                (ticker_data['low'] <= 0)
            )
            if invalid_prices.any():
                self.db.log_data_quality_issue(
                    table_name='prices',
                    ticker=ticker,
                    issue_type='invalid_price',
                    description=f"Zero or negative prices on {invalid_prices.sum()} days",
                    severity='ERROR'
                )

            # Check for data gaps
            if len(ticker_data) > 1:
                date_diffs = ticker_data.index.to_series().diff()
                # Allow for weekends (3 days) but flag longer gaps
                long_gaps = date_diffs > timedelta(days=7)
                if long_gaps.any():
                    n_gaps = long_gaps.sum()
                    self.db.log_data_quality_issue(
                        table_name='prices',
                        ticker=ticker,
                        issue_type='data_gaps',
                        description=f"Found {n_gaps} gaps longer than 7 days",
                        severity='WARNING'
                    )

    def validate_no_lookahead(self, ticker: str,
                             start_date: datetime,
                             end_date: datetime,
                             test_date: datetime) -> bool:
        """
        Verify no lookahead bias for a specific date.

        Tests that data retrieved with as_of=test_date doesn't change
        when we add future data.

        Args:
            ticker: Stock ticker
            start_date: Start date
            end_date: End date
            test_date: Date to test for lookahead

        Returns:
            True if no lookahead detected, False otherwise
        """
        # Get data as of test_date
        data_as_of = self.get_price_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            as_of=test_date,
            validate=False
        )

        # Get data as of a future date (should be same or more)
        future_date = test_date + timedelta(days=30)
        data_future = self.get_price_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            as_of=future_date,
            validate=False
        )

        # Data as of test_date should be subset of future data
        # (or identical if no new data was added)
        if len(data_as_of) > len(data_future):
            return False  # Lookahead detected

        # Check that overlapping dates have same values
        common_dates = data_as_of.index.intersection(data_future.index)
        if len(common_dates) > 0:
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if not data_as_of.loc[common_dates, col].equals(
                    data_future.loc[common_dates, col]
                ):
                    return False  # Values changed - lookahead detected

        return True

    def get_available_tickers(self, data_type: str = 'prices') -> List[str]:
        """
        Get list of available tickers.

        Args:
            data_type: 'prices', 'fundamentals', or 'insider_trading'

        Returns:
            Sorted list of tickers
        """
        table_map = {
            'prices': 'prices',
            'fundamentals': 'fundamentals',
            'insider_trading': 'insider_trading'
        }

        return self.db.get_tickers(table=table_map.get(data_type, 'prices'))

    def get_date_range(self, ticker: str, data_type: str = 'prices') -> Dict[str, datetime]:
        """
        Get available date range for a ticker.

        Args:
            ticker: Stock ticker
            data_type: 'prices', 'fundamentals', or 'insider_trading'

        Returns:
            Dict with 'min_date' and 'max_date'
        """
        table_map = {
            'prices': 'prices',
            'fundamentals': 'fundamentals',
            'insider_trading': 'insider_trading'
        }

        return self.db.get_date_range(ticker, table=table_map.get(data_type, 'prices'))

    def clear_cache(self):
        """Clear all cached data."""
        if self.cache:
            self.cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.cache:
            return {'enabled': False}

        return {
            'enabled': True,
            'size_mb': self.cache.size_mb(),
            'max_size_mb': self.cache.max_size_mb,
            'n_items': len(self.cache.cache)
        }
