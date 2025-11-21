"""
DataManager: Simple read-only interface to Sharadar data.

Philosophy: Keep it simple. Just fetch data with point-in-time filtering.
"""

import sqlite3
from typing import Optional, List, Union
from pathlib import Path
import pandas as pd
import sys

# Import configuration
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MARKET_DATA_DB, DB_CACHE_SIZE, get_logger
from core.db import get_read_only_connection

logger = get_logger(__name__)


class DataManager:
    """
    Simple data access layer for Sharadar data.

    Features:
    - Read-only database access
    - Point-in-time filtering (as_of parameter)
    - Basic caching
    - Clean API

    Usage:
        dm = DataManager()
        prices = dm.get_prices(['AAPL', 'MSFT'], '2020-01-01', '2024-12-31')
        fundamentals = dm.get_fundamentals('AAPL', '2020-01-01', '2024-12-31', dimension='ARQ')
        insider = dm.get_insider_trades('AAPL', '2020-01-01', '2024-12-31')
    """

    def __init__(self, db_path: Optional[Path] = None, cache_size: Optional[int] = None):
        """
        Initialize DataManager.

        Args:
            db_path: Path to Sharadar database (default: from config)
            cache_size: Number of queries to cache (default: from config)
        """
        if db_path is None:
            # Use configuration
            db_path = MARKET_DATA_DB

        if cache_size is None:
            cache_size = DB_CACHE_SIZE

        self.db_path = Path(db_path)
        self._cache = {}  # Simple dict cache
        self._cache_order = []  # For LRU
        self._cache_size = cache_size
        self._trading_calendar_cache = None  # In-memory calendar DataFrame (lazy load)
        self._trading_days_only = None  # Pre-filtered list of trading days (lazy load)

        # Verify database exists
        if not self.db_path.exists():
            logger.error(f"Database not found: {self.db_path}")
            raise FileNotFoundError(
                f"Database not found: {self.db_path}\n"
                f"Set SIGNALTIDE_DB_PATH environment variable to correct path."
            )

        logger.info(f"DataManager initialized with database: {self.db_path}")
        logger.debug(f"Cache size: {self._cache_size}")

    def _get_connection(self) -> sqlite3.Connection:
        """
        Get read-only database connection.

        Uses core.db.get_read_only_connection() to ensure:
        - Foreign keys are enabled
        - Consistent connection configuration across codebase
        """
        return get_read_only_connection(db_path=self.db_path)

    def _check_cache(self, key: str) -> Optional[pd.DataFrame]:
        """Check cache for key."""
        if key in self._cache:
            # Move to end (most recently used)
            self._cache_order.remove(key)
            self._cache_order.append(key)
            return self._cache[key].copy()
        return None

    def _add_to_cache(self, key: str, df: pd.DataFrame):
        """Add dataframe to cache."""
        # Evict oldest if full
        if len(self._cache) >= self._cache_size:
            oldest_key = self._cache_order.pop(0)
            del self._cache[oldest_key]

        self._cache[key] = df.copy()
        self._cache_order.append(key)

    def get_prices(self,
                   symbols: Union[str, List[str]],
                   start_date: str,
                   end_date: str,
                   as_of_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get OHLCV price data.

        Args:
            symbols: Ticker or list of tickers
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            as_of_date: Point-in-time date (only data available by this date)

        Returns:
            DataFrame with columns: ticker, date, open, high, low, close, volume
            Index: date (datetime)
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        logger.debug(f"Fetching prices for {symbols} from {start_date} to {end_date} (as_of_date={as_of_date})")

        # Cache key
        cache_key = f"prices_{'-'.join(symbols)}_{start_date}_{end_date}_{as_of_date}"

        # Check cache
        cached = self._check_cache(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for {cache_key}")
            return cached

        # Build query
        placeholders = ','.join('?' * len(symbols))
        query = f"""
            SELECT ticker, date, open, high, low, close, volume, closeadj
            FROM sharadar_prices
            WHERE ticker IN ({placeholders})
              AND date >= ?
              AND date <= ?
        """
        params = list(symbols) + [start_date, end_date]

        # Add point-in-time filter
        if as_of_date:
            query += " AND lastupdated <= ?"
            params.append(as_of_date)

        query += " ORDER BY date, ticker"

        # Execute
        try:
            conn = self._get_connection()
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            logger.debug(f"Retrieved {len(df)} price rows from database")
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            raise

        # Convert date to datetime (handle various formats)
        df['date'] = pd.to_datetime(df['date'], format='mixed')
        df = df.set_index('date')

        # Cache and return
        self._add_to_cache(cache_key, df)
        logger.debug(f"Cached result under key: {cache_key}")
        return df

    def get_fundamentals(self,
                        symbol: str,
                        start_date: str,
                        end_date: str,
                        dimension: str = 'ARQ',
                        as_of_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get fundamental data (SF1 dataset).

        Args:
            symbol: Ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            dimension: ARQ (quarterly), ARY (annual), MRQ, MRY
            as_of_date: Point-in-time date (uses filing date, not quarter-end)
                        REQUIRED to prevent lookahead bias

        Returns:
            DataFrame with fundamental metrics
            Index: calendardate (datetime)
        """
        # Runtime validation: as_of_date is REQUIRED for temporal discipline
        if as_of_date is None:
            logger.warning(
                f"as_of_date not provided for get_fundamentals({symbol}). "
                f"This may introduce lookahead bias! Using end_date as fallback."
            )
            as_of_date = end_date

        # Cache key
        cache_key = f"fundamentals_{symbol}_{start_date}_{end_date}_{dimension}_{as_of_date}"

        # Check cache
        cached = self._check_cache(cache_key)
        if cached is not None:
            return cached

        # Build query
        query = """
            SELECT *
            FROM sharadar_sf1
            WHERE ticker = ?
              AND dimension = ?
              AND calendardate >= ?
              AND calendardate <= ?
        """
        params = [symbol, dimension, start_date, end_date]

        # Add point-in-time filter (datekey is filing/availability date)
        if as_of_date:
            query += " AND datekey <= ?"
            params.append(as_of_date)

        query += " ORDER BY calendardate"

        # Execute
        conn = self._get_connection()
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if len(df) == 0:
            # Return empty dataframe with expected structure
            return pd.DataFrame()

        # Convert dates
        df['calendardate'] = pd.to_datetime(df['calendardate'])
        df['reportperiod'] = pd.to_datetime(df['reportperiod'])
        df = df.set_index('calendardate')

        # Cache and return
        self._add_to_cache(cache_key, df)
        return df

    def get_insider_trades(self,
                          symbol: str,
                          start_date: str,
                          end_date: str,
                          as_of_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get insider trading data.

        Args:
            symbol: Ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            as_of_date: Point-in-time date (uses filing date)
                        REQUIRED to prevent lookahead bias

        Returns:
            DataFrame with insider trades
            Index: filingdate (datetime)
        """
        # Runtime validation: as_of_date is REQUIRED for temporal discipline
        if as_of_date is None:
            logger.warning(
                f"as_of_date not provided for get_insider_trades({symbol}). "
                f"This may introduce lookahead bias! Using end_date as fallback."
            )
            as_of_date = end_date

        # Cache key
        cache_key = f"insider_{symbol}_{start_date}_{end_date}_{as_of_date}"

        # Check cache
        cached = self._check_cache(cache_key)
        if cached is not None:
            return cached

        # Build query
        query = """
            SELECT *
            FROM sharadar_insiders
            WHERE ticker = ?
              AND filingdate >= ?
              AND filingdate <= ?
        """
        params = [symbol, start_date, end_date]

        # Add point-in-time filter (filingdate is when we learned about trade)
        if as_of_date:
            query += " AND filingdate <= ?"
            params.append(as_of_date)

        query += " ORDER BY filingdate"

        # Execute
        conn = self._get_connection()
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if len(df) == 0:
            return pd.DataFrame()

        # Convert dates
        df['filingdate'] = pd.to_datetime(df['filingdate'])
        df['transactiondate'] = pd.to_datetime(df['transactiondate'])
        df = df.set_index('filingdate')

        # Cache and return
        self._add_to_cache(cache_key, df)
        return df

    def get_tickers(self,
                   category: str = 'Domestic',
                   is_delisted: bool = False) -> List[str]:
        """
        Get list of tickers matching criteria.

        Args:
            category: Ticker category (Domestic, Canadian, ADR, etc.)
            is_delisted: Include delisted tickers

        Returns:
            List of ticker symbols
        """
        # Cache key
        cache_key = f"tickers_{category}_{is_delisted}"

        # Check cache
        cached = self._check_cache(cache_key)
        if cached is not None:
            return cached['ticker'].tolist()

        # Build query
        query = "SELECT ticker FROM sharadar_tickers WHERE category = ?"
        params = [category]

        if not is_delisted:
            query += " AND isdelisted = 'N'"

        # Execute
        conn = self._get_connection()
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        # Cache and return
        self._add_to_cache(cache_key, df)
        return df['ticker'].tolist()

    def _get_trading_calendar_df(self) -> pd.DataFrame:
        """
        Get trading calendar as DataFrame (cached in-memory).

        Loads the entire dim_trading_calendar table once and caches it
        in memory for fast repeated access. This is efficient because:
        - Calendar is static (~13k rows for 2000-2035)
        - Loaded once per DataManager instance
        - Pandas operations are much faster than repeated DB queries

        Returns:
            DataFrame with columns: calendar_date, is_trading_day
        """
        if self._trading_calendar_cache is None:
            logger.debug("Loading trading calendar into memory")
            conn = self._get_connection()

            # Load only essential columns for calendar operations
            query = """
                SELECT calendar_date, is_trading_day
                FROM dim_trading_calendar
                ORDER BY calendar_date ASC
            """

            self._trading_calendar_cache = pd.read_sql_query(query, conn)
            conn.close()

            # Pre-filter to trading days only for faster operations
            self._trading_days_only = self._trading_calendar_cache[
                self._trading_calendar_cache['is_trading_day'] == 1
            ]['calendar_date'].tolist()

            logger.debug(f"Loaded {len(self._trading_calendar_cache)} calendar days "
                        f"({len(self._trading_days_only)} trading days)")

        return self._trading_calendar_cache

    def get_last_trading_date(self, as_of_date: str) -> str:
        """
        Return the last trading_date <= as_of_date using dim_trading_calendar.

        Uses in-memory cached calendar for fast lookups.

        Args:
            as_of_date: Date in 'YYYY-MM-DD' format

        Returns:
            Last trading date as 'YYYY-MM-DD' string

        Raises:
            ValueError: If no trading date exists on or before as_of_date

        Example:
            >>> dm = DataManager()
            >>> dm.get_last_trading_date('2024-06-15')  # Returns Friday if Sat
            '2024-06-14'
        """
        # Use cached calendar for fast in-memory lookup
        self._get_trading_calendar_df()  # Ensure cache is loaded

        # Filter trading days <= as_of_date
        valid_dates = [d for d in self._trading_days_only if d <= as_of_date]

        if not valid_dates:
            raise ValueError(f"No trading date found on or before {as_of_date}")

        return valid_dates[-1]  # Last (most recent) date

    def get_next_trading_date(self, as_of_date: str) -> Optional[str]:
        """
        Return the next trading_date > as_of_date, or None if none exists.

        Uses in-memory cached calendar for fast lookups.

        Args:
            as_of_date: Date in 'YYYY-MM-DD' format

        Returns:
            Next trading date as 'YYYY-MM-DD' string, or None

        Example:
            >>> dm = DataManager()
            >>> dm.get_next_trading_date('2024-06-14')  # Returns Monday if Fri
            '2024-06-17'
        """
        # Use cached calendar for fast in-memory lookup
        self._get_trading_calendar_df()  # Ensure cache is loaded

        # Filter trading days > as_of_date
        future_dates = [d for d in self._trading_days_only if d > as_of_date]

        return future_dates[0] if future_dates else None

    def get_trading_dates_between(self, start_date: str, end_date: str) -> List[str]:
        """
        Return all trading dates between start_date and end_date inclusive.

        Uses in-memory cached calendar for fast lookups.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format

        Returns:
            List of trading dates as 'YYYY-MM-DD' strings

        Example:
            >>> dm = DataManager()
            >>> dates = dm.get_trading_dates_between('2024-01-01', '2024-01-05')
            >>> len(dates)  # Should be ~4-5 trading days
            4
        """
        # Use cached calendar for fast in-memory lookup
        self._get_trading_calendar_df()  # Ensure cache is loaded

        # Filter trading days in range [start_date, end_date]
        return [d for d in self._trading_days_only if start_date <= d <= end_date]

    def get_month_end_rebalance_dates(self, start_date: str, end_date: str) -> List[str]:
        """
        Get all month-end trading dates in a range.

        Returns the last trading day of each month within the date range.
        Useful for monthly portfolio rebalancing strategies.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format

        Returns:
            List of month-end trading dates as 'YYYY-MM-DD' strings

        Example:
            >>> dm = DataManager()
            >>> dates = dm.get_month_end_rebalance_dates('2024-01-01', '2024-03-31')
            >>> dates  # Last trading day of Jan, Feb, Mar
            ['2024-01-31', '2024-02-29', '2024-03-28']
        """
        # Use cached calendar for fast in-memory lookup
        self._get_trading_calendar_df()  # Ensure cache is loaded

        # Get full calendar DataFrame
        cal_df = self._trading_calendar_cache

        # Filter to date range and trading days only
        mask = (cal_df['calendar_date'] >= start_date) & \
               (cal_df['calendar_date'] <= end_date) & \
               (cal_df['is_trading_day'] == 1)
        trading_days = cal_df[mask].copy()

        # Extract year-month
        trading_days['year_month'] = pd.to_datetime(trading_days['calendar_date']).dt.to_period('M')

        # Get last trading day of each month
        month_ends = trading_days.groupby('year_month')['calendar_date'].max().tolist()

        return sorted(month_ends)

    def get_weekly_rebalance_dates(self, start_date: str, end_date: str,
                                   day_of_week: str = 'Friday') -> List[str]:
        """
        Get weekly rebalance dates (last trading day of each week).

        Returns the last trading day of each week within the date range.
        Useful for weekly portfolio rebalancing strategies.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            day_of_week: Target day of week (default: 'Friday').
                         Options: 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'
                         If target day is holiday, uses last prior trading day.

        Returns:
            List of weekly rebalance dates as 'YYYY-MM-DD' strings

        Example:
            >>> dm = DataManager()
            >>> dates = dm.get_weekly_rebalance_dates('2024-01-01', '2024-01-31')
            >>> len(dates)  # Should be ~4-5 Fridays in January
            4
        """
        # Map day names to pandas weekday numbers (0=Monday, 4=Friday)
        day_map = {
            'Monday': 0,
            'Tuesday': 1,
            'Wednesday': 2,
            'Thursday': 3,
            'Friday': 4
        }

        if day_of_week not in day_map:
            raise ValueError(f"Invalid day_of_week: {day_of_week}. Must be Monday-Friday.")

        target_weekday = day_map[day_of_week]

        # Use cached calendar for fast in-memory lookup
        self._get_trading_calendar_df()  # Ensure cache is loaded

        # Get full calendar DataFrame
        cal_df = self._trading_calendar_cache

        # Filter to date range and trading days only
        mask = (cal_df['calendar_date'] >= start_date) & \
               (cal_df['calendar_date'] <= end_date) & \
               (cal_df['is_trading_day'] == 1)
        trading_days = cal_df[mask].copy()

        # Add weekday and week number
        trading_days['date_obj'] = pd.to_datetime(trading_days['calendar_date'])
        trading_days['weekday'] = trading_days['date_obj'].dt.dayofweek
        trading_days['year_week'] = trading_days['date_obj'].dt.isocalendar().week.astype(str) + '_' + \
                                     trading_days['date_obj'].dt.isocalendar().year.astype(str)

        weekly_dates = []

        # For each week, get the target day if it's a trading day, else last prior trading day
        for week in trading_days['year_week'].unique():
            week_data = trading_days[trading_days['year_week'] == week].sort_values('calendar_date')

            # Try to find exact target weekday
            target_day = week_data[week_data['weekday'] == target_weekday]

            if not target_day.empty:
                # Target day exists and is a trading day
                weekly_dates.append(target_day.iloc[0]['calendar_date'])
            else:
                # Target day is holiday/weekend - use last trading day before target
                days_before_target = week_data[week_data['weekday'] < target_weekday]
                if not days_before_target.empty:
                    weekly_dates.append(days_before_target.iloc[-1]['calendar_date'])
                else:
                    # Week starts after target day - use last day of week
                    weekly_dates.append(week_data.iloc[-1]['calendar_date'])

        return sorted(weekly_dates)

    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
        self._cache_order.clear()
        # Also clear trading calendar caches
        self._trading_calendar_cache = None
        self._trading_days_only = None


# Convenience function for testing with mock data
def create_mock_data_manager():
    """
    Create DataManager that uses mock data instead of real database.

    Useful for testing without database access.
    """
    from data.mock_generator import MockDataGenerator
    from datetime import datetime, timedelta

    class MockDataManager:
        """Mock DataManager using generated data."""

        def __init__(self):
            self.generator = MockDataGenerator()
            self.end_date = datetime.now()
            self.start_date = self.end_date - timedelta(days=365 * 5)

        def get_prices(self, symbols, start_date, end_date, as_of_date=None):
            if isinstance(symbols, str):
                symbols = [symbols]

            dfs = []
            for symbol in symbols:
                df = self.generator.generate_price_data(
                    symbol,
                    pd.to_datetime(start_date),
                    pd.to_datetime(end_date)
                )
                dfs.append(df)

            result = pd.concat(dfs)

            # Apply as_of_date filter if provided
            if as_of_date:
                result = result[result.index <= pd.to_datetime(as_of_date)]

            return result

        def get_fundamentals(self, symbol, start_date, end_date, dimension='ARQ', as_of_date=None):
            df = self.generator.generate_fundamentals(
                symbol,
                pd.to_datetime(start_date),
                pd.to_datetime(end_date)
            )

            if as_of_date:
                df = df[df.index <= pd.to_datetime(as_of_date)]

            return df

        def get_insider_trades(self, symbol, start_date, end_date, as_of_date=None):
            df = self.generator.generate_insider_trades(
                symbol,
                pd.to_datetime(start_date),
                pd.to_datetime(end_date)
            )

            if len(df) == 0:
                return df

            df = df.set_index('filing_date')

            if as_of_date:
                df = df[df.index <= pd.to_datetime(as_of_date)]

            return df

        def get_tickers(self, category='Domestic', is_delisted=False):
            return self.generator.tickers

        def clear_cache(self):
            pass

    return MockDataManager()


if __name__ == '__main__':
    # Test with real database
    print("Testing DataManager with Sharadar database...")

    dm = DataManager()

    # Test prices
    print("\n1. Testing price data...")
    prices = dm.get_prices(['AAPL', 'MSFT'], '2024-01-01', '2024-01-31')
    print(f"   Got {len(prices)} price rows")
    print(prices.head())

    # Test fundamentals
    print("\n2. Testing fundamental data...")
    fundamentals = dm.get_fundamentals('AAPL', '2023-01-01', '2024-12-31', dimension='ARQ')
    print(f"   Got {len(fundamentals)} fundamental rows")
    if len(fundamentals) > 0:
        print(f"   Columns: {', '.join(fundamentals.columns[:10])}...")

    # Test insider trades
    print("\n3. Testing insider trades...")
    insider = dm.get_insider_trades('AAPL', '2024-01-01', '2024-12-31')
    print(f"   Got {len(insider)} insider trade rows")
    if len(insider) > 0:
        print(insider.head())

    # Test cache
    print("\n4. Testing cache...")
    print(f"   Cache size before: {len(dm._cache)}")
    prices2 = dm.get_prices(['AAPL', 'MSFT'], '2024-01-01', '2024-01-31')  # Should hit cache
    print(f"   Cache size after: {len(dm._cache)}")

    print("\n5. Testing mock data manager...")
    mock_dm = create_mock_data_manager()
    mock_prices = mock_dm.get_prices('TICK01', '2024-01-01', '2024-01-31')
    print(f"   Got {len(mock_prices)} mock price rows")
    print(mock_prices.head())

    print("\nDataManager tests complete!")
