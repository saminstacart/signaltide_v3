"""
SQLite database schema and operations for SignalTide v3.

Handles storage and retrieval of:
- Price data (OHLCV)
- Fundamental data (Sharadar quality metrics)
- Insider trading data
- Metadata and data versioning
"""

from typing import Optional, List, Dict, Any
import sqlite3
from datetime import datetime
from pathlib import Path
import pandas as pd
from contextlib import contextmanager
import config


class Database:
    """
    SQLite database interface for market data.

    Supports Sharadar data format with proper temporal indexing.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database (default: from config)
        """
        self.db_path = db_path or config.MARKET_DATA_DB
        self._initialize_schema()

    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.

        Ensures proper connection handling and cleanup.

        Example:
            ```python
            with db.get_connection() as conn:
                df = pd.read_sql("SELECT * FROM prices", conn)
            ```
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _initialize_schema(self):
        """Create database tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Price data table (OHLCV)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    date DATE NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    adjusted_close REAL,
                    dividend REAL DEFAULT 0.0,
                    split_ratio REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, date)
                )
            """)

            # Indexes for price data
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_prices_ticker_date
                ON prices(ticker, date)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_prices_date
                ON prices(date)
            """)

            # Fundamental data table (Sharadar format)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fundamentals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    date DATE NOT NULL,
                    report_period DATE,
                    filing_date DATE,
                    dimension TEXT,  -- ARQ, MRQ, ART, MRT

                    -- Quality metrics
                    revenue REAL,
                    gross_profit REAL,
                    operating_income REAL,
                    net_income REAL,
                    ebitda REAL,

                    -- Balance sheet
                    total_assets REAL,
                    total_liabilities REAL,
                    stockholders_equity REAL,
                    cash_and_equivalents REAL,

                    -- Cash flow
                    operating_cash_flow REAL,
                    investing_cash_flow REAL,
                    financing_cash_flow REAL,
                    free_cash_flow REAL,

                    -- Ratios and metrics
                    shares_outstanding REAL,
                    market_cap REAL,
                    pe_ratio REAL,
                    pb_ratio REAL,
                    debt_to_equity REAL,
                    current_ratio REAL,
                    quick_ratio REAL,
                    roe REAL,
                    roa REAL,

                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, date, dimension)
                )
            """)

            # Indexes for fundamental data
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_fundamentals_ticker_date
                ON fundamentals(ticker, date)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_fundamentals_filing_date
                ON fundamentals(ticker, filing_date)
            """)

            # Insider trading table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS insider_trading (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    filing_date DATE NOT NULL,
                    trade_date DATE,
                    insider_name TEXT,
                    insider_title TEXT,
                    transaction_type TEXT,  -- P (Purchase), S (Sale), etc.
                    shares REAL,
                    price_per_share REAL,
                    shares_owned_after REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, filing_date, insider_name, transaction_type, shares)
                )
            """)

            # Indexes for insider trading
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_insider_ticker_filing
                ON insider_trading(ticker, filing_date)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_insider_filing_date
                ON insider_trading(filing_date)
            """)

            # Data quality log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_quality_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    table_name TEXT NOT NULL,
                    ticker TEXT,
                    issue_type TEXT NOT NULL,
                    description TEXT,
                    severity TEXT,  -- INFO, WARNING, ERROR
                    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()

    def store_prices(self, df: pd.DataFrame, if_exists: str = 'append') -> int:
        """
        Store price data in database.

        Args:
            df: DataFrame with columns: ticker, date, open, high, low, close, volume
            if_exists: 'append' or 'replace'

        Returns:
            Number of rows inserted
        """
        # Validate required columns
        required_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Ensure date is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])

        with self.get_connection() as conn:
            # Use replace to handle duplicates
            df.to_sql('prices', conn, if_exists=if_exists, index=False,
                     method='multi', chunksize=1000)

        return len(df)

    def store_fundamentals(self, df: pd.DataFrame, if_exists: str = 'append') -> int:
        """
        Store fundamental data in database.

        Args:
            df: DataFrame with fundamental metrics
            if_exists: 'append' or 'replace'

        Returns:
            Number of rows inserted
        """
        required_cols = ['ticker', 'date', 'filing_date']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        with self.get_connection() as conn:
            df.to_sql('fundamentals', conn, if_exists=if_exists, index=False,
                     method='multi', chunksize=1000)

        return len(df)

    def store_insider_trades(self, df: pd.DataFrame, if_exists: str = 'append') -> int:
        """
        Store insider trading data.

        Args:
            df: DataFrame with insider trades
            if_exists: 'append' or 'replace'

        Returns:
            Number of rows inserted
        """
        required_cols = ['ticker', 'filing_date', 'transaction_type']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        with self.get_connection() as conn:
            df.to_sql('insider_trading', conn, if_exists=if_exists, index=False,
                     method='multi', chunksize=1000)

        return len(df)

    def get_prices(self, ticker: Optional[str] = None,
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None,
                   as_of: Optional[datetime] = None) -> pd.DataFrame:
        """
        Retrieve price data with point-in-time constraints.

        Args:
            ticker: Stock ticker (None for all)
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            as_of: Point-in-time constraint (only data available as of this date)

        Returns:
            DataFrame with price data
        """
        query = "SELECT * FROM prices WHERE 1=1"
        params = []

        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)

        if start_date:
            query += " AND date >= ?"
            params.append(start_date.strftime('%Y-%m-%d'))

        if end_date:
            query += " AND date <= ?"
            params.append(end_date.strftime('%Y-%m-%d'))

        if as_of:
            # Only return data that existed as of this date
            query += " AND created_at <= ?"
            params.append(as_of.strftime('%Y-%m-%d %H:%M:%S'))

        query += " ORDER BY ticker, date"

        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if len(df) > 0:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

        return df

    def get_fundamentals(self, ticker: Optional[str] = None,
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None,
                        as_of: Optional[datetime] = None,
                        dimension: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve fundamental data with point-in-time constraints.

        CRITICAL: Uses filing_date for point-in-time, not report_period.
        This prevents lookahead bias - we only know fundamentals after filing.

        Args:
            ticker: Stock ticker
            start_date: Start date
            end_date: End date
            as_of: Point-in-time constraint (based on filing_date)
            dimension: ARQ, MRQ, ART, MRT (None for all)

        Returns:
            DataFrame with fundamental data
        """
        query = "SELECT * FROM fundamentals WHERE 1=1"
        params = []

        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)

        if start_date:
            query += " AND filing_date >= ?"
            params.append(start_date.strftime('%Y-%m-%d'))

        if end_date:
            query += " AND filing_date <= ?"
            params.append(end_date.strftime('%Y-%m-%d'))

        if as_of:
            # Only fundamentals filed by this date
            query += " AND filing_date <= ?"
            params.append(as_of.strftime('%Y-%m-%d'))

        if dimension:
            query += " AND dimension = ?"
            params.append(dimension)

        query += " ORDER BY ticker, filing_date"

        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if len(df) > 0:
            df['date'] = pd.to_datetime(df['date'])
            df['filing_date'] = pd.to_datetime(df['filing_date'])

        return df

    def get_insider_trades(self, ticker: Optional[str] = None,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None,
                          as_of: Optional[datetime] = None) -> pd.DataFrame:
        """
        Retrieve insider trading data.

        Args:
            ticker: Stock ticker
            start_date: Start date
            end_date: End date
            as_of: Point-in-time constraint (based on filing_date)

        Returns:
            DataFrame with insider trades
        """
        query = "SELECT * FROM insider_trading WHERE 1=1"
        params = []

        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)

        if start_date:
            query += " AND filing_date >= ?"
            params.append(start_date.strftime('%Y-%m-%d'))

        if end_date:
            query += " AND filing_date <= ?"
            params.append(end_date.strftime('%Y-%m-%d'))

        if as_of:
            query += " AND filing_date <= ?"
            params.append(as_of.strftime('%Y-%m-%d'))

        query += " ORDER BY ticker, filing_date"

        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if len(df) > 0:
            df['filing_date'] = pd.to_datetime(df['filing_date'])
            if 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date'])

        return df

    def log_data_quality_issue(self, table_name: str, issue_type: str,
                               description: str, ticker: Optional[str] = None,
                               severity: str = 'WARNING'):
        """
        Log a data quality issue.

        Args:
            table_name: Which table has the issue
            issue_type: Type of issue (e.g., 'missing_data', 'anomaly')
            description: Detailed description
            ticker: Affected ticker (if applicable)
            severity: INFO, WARNING, or ERROR
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO data_quality_log
                (table_name, ticker, issue_type, description, severity)
                VALUES (?, ?, ?, ?, ?)
            """, (table_name, ticker, issue_type, description, severity))

    def get_data_quality_issues(self, table_name: Optional[str] = None,
                                severity: Optional[str] = None,
                                limit: int = 100) -> pd.DataFrame:
        """
        Retrieve data quality issues.

        Args:
            table_name: Filter by table
            severity: Filter by severity
            limit: Maximum number of issues to return

        Returns:
            DataFrame with issues
        """
        query = "SELECT * FROM data_quality_log WHERE 1=1"
        params = []

        if table_name:
            query += " AND table_name = ?"
            params.append(table_name)

        if severity:
            query += " AND severity = ?"
            params.append(severity)

        query += " ORDER BY detected_at DESC LIMIT ?"
        params.append(limit)

        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        return df

    def get_tickers(self, table: str = 'prices') -> List[str]:
        """
        Get list of available tickers.

        Args:
            table: Which table to query ('prices', 'fundamentals', 'insider_trading')

        Returns:
            Sorted list of unique tickers
        """
        with self.get_connection() as conn:
            df = pd.read_sql_query(f"SELECT DISTINCT ticker FROM {table} ORDER BY ticker", conn)

        return df['ticker'].tolist()

    def get_date_range(self, ticker: str, table: str = 'prices') -> Dict[str, datetime]:
        """
        Get date range for a ticker.

        Args:
            ticker: Stock ticker
            table: Which table to query

        Returns:
            Dict with 'min_date' and 'max_date'
        """
        with self.get_connection() as conn:
            query = f"""
                SELECT MIN(date) as min_date, MAX(date) as max_date
                FROM {table}
                WHERE ticker = ?
            """
            df = pd.read_sql_query(query, conn, params=[ticker])

        return {
            'min_date': pd.to_datetime(df['min_date'].iloc[0]) if len(df) > 0 else None,
            'max_date': pd.to_datetime(df['max_date'].iloc[0]) if len(df) > 0 else None
        }

    def set_metadata(self, key: str, value: str):
        """Set a metadata value."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO metadata (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (key, value))

    def get_metadata(self, key: str) -> Optional[str]:
        """Get a metadata value."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM metadata WHERE key = ?", (key,))
            result = cursor.fetchone()

        return result['value'] if result else None
