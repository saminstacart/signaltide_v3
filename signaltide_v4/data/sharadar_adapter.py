"""
Sharadar data adapter for SignalTide v4.

Provides a bridge between v4's data abstraction and the
existing v3 Sharadar database infrastructure.

This adapter:
- Connects to the existing SQLite Sharadar database
- Implements PIT-compliant data access
- Provides caching for performance
- Maps Sharadar schema to v4 data interfaces
"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from functools import lru_cache

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SharadarAdapter:
    """
    Adapter for Sharadar data access.

    Connects to the existing SignalTide v3 Sharadar database
    and provides data in formats expected by v4 modules.
    """

    # Table names in Sharadar database
    TABLE_PRICES = 'sharadar_prices'
    TABLE_SF1 = 'sharadar_sf1'  # Fundamentals
    TABLE_TICKERS = 'sharadar_tickers'
    TABLE_INSIDERS = 'sharadar_insiders'
    TABLE_EVENTS = 'sharadar_events'

    # Fundamental metrics mapping
    SF1_METRICS = {
        # Income Statement
        'revenue': 'REVENUE',
        'gross_profit': 'GP',
        'operating_income': 'OPINC',
        'net_income': 'NETINC',
        'ebitda': 'EBITDA',
        'eps': 'EPS',
        'eps_diluted': 'EPSDIL',

        # Balance Sheet
        'total_assets': 'ASSETS',
        'total_liabilities': 'LIABILITIES',
        'equity': 'EQUITY',
        'cash': 'CASHNEQ',
        'debt': 'DEBT',
        'working_capital': 'WORKINGCAPITAL',

        # Cash Flow
        'operating_cf': 'NCFO',
        'investing_cf': 'NCFI',
        'financing_cf': 'NCFF',
        'capex': 'CAPEX',
        'fcf': 'FCF',

        # Per Share
        'book_value_ps': 'BVPS',
        'dividends_ps': 'DPS',

        # Ratios
        'roa': 'ROA',
        'roe': 'ROE',
        'gross_margin': 'GROSSMARGIN',
        'net_margin': 'NETMARGIN',

        # Valuation
        'market_cap': 'MARKETCAP',
        'pe_ratio': 'PE',
        'pb_ratio': 'PB',
        'ps_ratio': 'PS',
        'ev_ebitda': 'EVEBITDA',
    }

    def __init__(
        self,
        db_path: Optional[str] = None,
        use_cache: bool = True,
        cache_ttl_hours: int = 24,
    ):
        """
        Initialize Sharadar adapter.

        Args:
            db_path: Path to Sharadar SQLite database
            use_cache: Whether to cache query results
            cache_ttl_hours: Cache time-to-live in hours
        """
        # Get database path from environment or config
        self.db_path = db_path or os.environ.get(
            'SIGNALTIDE_DB_PATH',
            '/Users/samuelksherman/signaltide/data/signaltide.db'
        )

        self.use_cache = use_cache
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self._cache: Dict[str, tuple] = {}  # key -> (data, timestamp)

        self._connection = None

        logger.info(f"SharadarAdapter initialized: db={self.db_path}")

    def _get_connection(self):
        """Get SQLite connection (lazy initialization)."""
        if self._connection is None:
            import sqlite3
            self._connection = sqlite3.connect(self.db_path)
            logger.debug(f"Connected to database: {self.db_path}")
        return self._connection

    def _execute_query(self, query: str, params: tuple = ()) -> pd.DataFrame:
        """Execute SQL query and return DataFrame."""
        try:
            conn = self._get_connection()
            df = pd.read_sql_query(query, conn, params=params)
            return df
        except Exception as e:
            logger.error(f"Query failed: {e}")
            logger.debug(f"Query: {query}")
            return pd.DataFrame()

    def _cache_key(self, *args) -> str:
        """Generate cache key from arguments."""
        return str(args)

    def _check_cache(self, key: str) -> Optional[Any]:
        """Check cache for valid entry."""
        if not self.use_cache or key not in self._cache:
            return None

        data, timestamp = self._cache[key]
        if datetime.now() - timestamp < self.cache_ttl:
            return data
        else:
            del self._cache[key]
            return None

    def _set_cache(self, key: str, data: Any) -> None:
        """Store data in cache."""
        if self.use_cache:
            self._cache[key] = (data, datetime.now())

    def get_prices(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        adjust: bool = True,
    ) -> pd.DataFrame:
        """
        Get price data for tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            adjust: Whether to use adjusted prices

        Returns:
            DataFrame with tickers as columns, dates as index
        """
        cache_key = self._cache_key('prices', tuple(tickers), start_date, end_date, adjust)
        cached = self._check_cache(cache_key)
        if cached is not None:
            return cached

        price_col = 'closeadj' if adjust else 'close'

        # Build query with parameterized placeholders
        placeholders = ','.join(['?' for _ in tickers])
        query = f"""
            SELECT date, ticker, {price_col} as price
            FROM {self.TABLE_PRICES}
            WHERE ticker IN ({placeholders})
            AND date >= ?
            AND date <= ?
            ORDER BY date, ticker
        """

        params = tuple(tickers) + (start_date, end_date)
        df = self._execute_query(query, params)

        if df.empty:
            logger.warning(f"No price data for {len(tickers)} tickers")
            return pd.DataFrame()

        # Pivot to have tickers as columns
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        pivot = df.pivot(index='date', columns='ticker', values='price')

        self._set_cache(cache_key, pivot)
        return pivot

    def get_returns(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Get daily returns for tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with daily returns
        """
        prices = self.get_prices(tickers, start_date, end_date)

        if prices.empty:
            return pd.DataFrame()

        returns = prices.pct_change().dropna(how='all')
        return returns

    def get_fundamentals_pit(
        self,
        tickers: List[str],
        as_of_date: str,
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get point-in-time fundamentals.

        Only returns data that would have been known as of the given date,
        based on filing dates.

        Args:
            tickers: List of ticker symbols
            as_of_date: Point-in-time date
            metrics: List of metrics to retrieve (None = all)

        Returns:
            DataFrame with latest PIT fundamentals per ticker
        """
        cache_key = self._cache_key('fundamentals_pit', tuple(tickers), as_of_date)
        cached = self._check_cache(cache_key)
        if cached is not None:
            return cached

        # Map requested metrics to SF1 column names
        if metrics:
            columns = [self.SF1_METRICS.get(m, m) for m in metrics if m in self.SF1_METRICS]
        else:
            columns = list(self.SF1_METRICS.values())

        if not columns:
            return pd.DataFrame()

        col_str = ', '.join(columns)

        # Get latest filing for each ticker as of the given date
        placeholders = ','.join(['?' for _ in tickers])
        query = f"""
            WITH ranked AS (
                SELECT
                    ticker,
                    {col_str},
                    datekey,
                    filingdate,
                    ROW_NUMBER() OVER (
                        PARTITION BY ticker
                        ORDER BY filingdate DESC, datekey DESC
                    ) as rn
                FROM {self.TABLE_SF1}
                WHERE ticker IN ({placeholders})
                AND filingdate <= ?
                AND dimension = 'MRQ'
            )
            SELECT ticker, {col_str}, datekey, filingdate
            FROM ranked
            WHERE rn = 1
        """

        params = tuple(tickers) + (as_of_date,)
        df = self._execute_query(query, params)

        if df.empty:
            logger.warning(f"No PIT fundamentals for {as_of_date}")
            return pd.DataFrame()

        df = df.set_index('ticker')

        self._set_cache(cache_key, df)
        return df

    def get_insider_transactions(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Get insider transactions.

        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with insider transactions
        """
        placeholders = ','.join(['?' for _ in tickers])
        query = f"""
            SELECT
                ticker,
                filingdate,
                transactiondate,
                ownername,
                issuerticker,
                transactioncode,
                transactionshares,
                transactionpricepershare,
                transactionvalue,
                sharesownedbeforetransaction,
                sharesownedfollowingtransaction
            FROM {self.TABLE_INSIDERS}
            WHERE ticker IN ({placeholders})
            AND filingdate >= ?
            AND filingdate <= ?
            ORDER BY filingdate DESC
        """

        params = tuple(tickers) + (start_date, end_date)
        df = self._execute_query(query, params)

        return df

    def get_market_cap(
        self,
        tickers: List[str],
        as_of_date: str,
    ) -> pd.Series:
        """
        Get market capitalizations.

        Args:
            tickers: List of ticker symbols
            as_of_date: Date for market cap

        Returns:
            Series of market caps indexed by ticker
        """
        fundamentals = self.get_fundamentals_pit(
            tickers, as_of_date, metrics=['market_cap']
        )

        if fundamentals.empty:
            return pd.Series(dtype=float)

        market_cap_col = self.SF1_METRICS['market_cap']
        return fundamentals[market_cap_col]

    def get_ticker_info(
        self,
        tickers: List[str],
    ) -> pd.DataFrame:
        """
        Get ticker metadata.

        Args:
            tickers: List of ticker symbols

        Returns:
            DataFrame with ticker info
        """
        placeholders = ','.join(['?' for _ in tickers])
        query = f"""
            SELECT
                ticker,
                name,
                exchange,
                sector,
                industry,
                siccode,
                sicdesc,
                location,
                scalemarketcap,
                scalerevenue
            FROM {self.TABLE_TICKERS}
            WHERE ticker IN ({placeholders})
        """

        params = tuple(tickers)
        df = self._execute_query(query, params)

        if not df.empty:
            df = df.set_index('ticker')

        return df

    def get_sp500_constituents(
        self,
        as_of_date: Optional[str] = None,
    ) -> List[str]:
        """
        Get S&P 500 constituents.

        Note: This is approximate - Sharadar doesn't have historical index membership.
        Uses current large caps as proxy.

        Args:
            as_of_date: Date for membership (approximate)

        Returns:
            List of ticker symbols
        """
        # Try progressively looser queries until we get results
        queries = [
            # Most restrictive
            """
            SELECT DISTINCT ticker
            FROM sharadar_tickers
            WHERE scalemarketcap IN ('5 - Large', '6 - Mega')
            AND exchange IN ('NYSE', 'NASDAQ')
            AND category = 'Domestic'
            AND isdelisted = 'N'
            ORDER BY ticker
            """,
            # Without category filter (may not exist in all DBs)
            """
            SELECT DISTINCT ticker
            FROM sharadar_tickers
            WHERE scalemarketcap IN ('5 - Large', '6 - Mega')
            AND exchange IN ('NYSE', 'NASDAQ')
            AND isdelisted = 'N'
            ORDER BY ticker
            """,
            # Even looser
            """
            SELECT DISTINCT ticker
            FROM sharadar_tickers
            WHERE exchange IN ('NYSE', 'NASDAQ', 'NYSEMKT', 'NYSEARCA')
            AND isdelisted = 'N'
            ORDER BY ticker
            LIMIT 500
            """,
            # Fallback: just get active tickers
            """
            SELECT DISTINCT ticker
            FROM sharadar_tickers
            WHERE isdelisted = 'N' OR isdelisted IS NULL
            ORDER BY ticker
            LIMIT 500
            """,
        ]

        for query in queries:
            df = self._execute_query(query)
            if not df.empty and len(df) >= 10:
                logger.debug(f"Universe query returned {len(df)} tickers")
                return df['ticker'].tolist()

        logger.warning("No tickers found in any universe query")
        return []

    def get_adv(
        self,
        tickers: List[str],
        as_of_date: str,
        lookback_days: int = 20,
    ) -> pd.Series:
        """
        Get average daily volume (in dollars).

        Args:
            tickers: List of ticker symbols
            as_of_date: End date for calculation
            lookback_days: Number of days for average

        Returns:
            Series of ADV indexed by ticker
        """
        start_date = (
            pd.Timestamp(as_of_date) - pd.Timedelta(days=lookback_days * 2)
        ).strftime('%Y-%m-%d')

        placeholders = ','.join(['?' for _ in tickers])
        query = f"""
            SELECT ticker, AVG(volume * closeadj) as adv
            FROM {self.TABLE_PRICES}
            WHERE ticker IN ({placeholders})
            AND date >= ?
            AND date <= ?
            GROUP BY ticker
        """

        params = tuple(tickers) + (start_date, as_of_date)
        df = self._execute_query(query, params)

        if df.empty:
            return pd.Series(dtype=float)

        return df.set_index('ticker')['adv']

    def get_volatility(
        self,
        tickers: List[str],
        as_of_date: str,
        lookback_days: int = 252,
    ) -> pd.Series:
        """
        Get historical volatility.

        Args:
            tickers: List of ticker symbols
            as_of_date: End date for calculation
            lookback_days: Number of days for calculation

        Returns:
            Series of annualized volatility indexed by ticker
        """
        start_date = (
            pd.Timestamp(as_of_date) - pd.Timedelta(days=lookback_days * 2)
        ).strftime('%Y-%m-%d')

        returns = self.get_returns(tickers, start_date, as_of_date)

        if returns.empty:
            return pd.Series(dtype=float)

        # Last N days
        returns = returns.tail(lookback_days)

        # Annualized volatility
        volatility = returns.std() * np.sqrt(252)

        return volatility

    def close(self):
        """Close database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            logger.debug("Database connection closed")

    def __del__(self):
        """Destructor to ensure connection is closed."""
        self.close()
