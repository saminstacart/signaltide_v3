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
        """Get read-only database connection."""
        conn = sqlite3.connect(f'file:{self.db_path}?mode=ro', uri=True)
        return conn

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

        Returns:
            DataFrame with fundamental metrics
            Index: calendardate (datetime)
        """
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

        Returns:
            DataFrame with insider trades
            Index: filingdate (datetime)
        """
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

    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
        self._cache_order.clear()


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
