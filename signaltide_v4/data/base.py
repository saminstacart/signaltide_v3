"""
Point-in-Time (PIT) compliant data manager base class.

All data access goes through this layer to ensure no lookahead bias.
Data is only available after filing_date + 1 day.
"""

import sqlite3
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from functools import lru_cache

import pandas as pd
import numpy as np

from signaltide_v4.config.settings import get_settings

logger = logging.getLogger(__name__)


class PITDataManager(ABC):
    """
    Abstract base class for Point-in-Time compliant data access.

    Key principle: Data is only available after it was publicly known.
    For SEC filings, this means filing_date + 1 trading day.
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize with database connection."""
        settings = get_settings()
        self.db_path = db_path or settings.db_path
        self.filing_lag_days = settings.filing_lag_days
        self._conn: Optional[sqlite3.Connection] = None
        logger.info(f"Initialized PITDataManager with db: {self.db_path}")

    @property
    def conn(self) -> sqlite3.Connection:
        """Lazy database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def get_pit_date(self, as_of_date: Union[str, datetime, pd.Timestamp]) -> pd.Timestamp:
        """
        Get the Point-in-Time cutoff date.

        Data filed on day T is available on day T+1.
        We use filing_date to determine when data was publicly known.
        """
        if isinstance(as_of_date, str):
            as_of_date = pd.Timestamp(as_of_date)
        elif isinstance(as_of_date, datetime):
            as_of_date = pd.Timestamp(as_of_date)
        return as_of_date

    def validate_pit_compliance(
        self,
        data: pd.DataFrame,
        as_of_date: Union[str, datetime, pd.Timestamp],
        date_column: str = 'filing_date'
    ) -> pd.DataFrame:
        """
        Filter data to ensure Point-in-Time compliance.

        Only returns data that was known as of as_of_date.
        """
        if isinstance(as_of_date, str):
            as_of_date = pd.Timestamp(as_of_date)

        if date_column not in data.columns and date_column != data.index.name:
            logger.warning(f"Date column '{date_column}' not found, returning unfiltered")
            return data

        # Filter to data known before as_of_date
        if date_column == data.index.name:
            mask = data.index <= as_of_date
        else:
            mask = pd.to_datetime(data[date_column]) <= as_of_date

        filtered = data[mask].copy()
        logger.debug(f"PIT filter: {len(data)} -> {len(filtered)} rows (as_of={as_of_date})")
        return filtered

    @abstractmethod
    def get_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        as_of_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get data for tickers with Point-in-Time compliance.

        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            as_of_date: Point-in-time cutoff (defaults to end_date)

        Returns:
            DataFrame with PIT-compliant data
        """
        pass

    def execute_query(self, query: str, params: tuple = ()) -> pd.DataFrame:
        """Execute SQL query and return DataFrame."""
        try:
            df = pd.read_sql_query(query, self.conn, params=params)
            return df
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise

    def get_available_tickers(self, as_of_date: str) -> List[str]:
        """Get list of tickers available as of given date."""
        query = """
            SELECT DISTINCT ticker
            FROM sharadar_prices
            WHERE date <= ?
            AND date >= date(?, '-30 days')
            ORDER BY ticker
        """
        df = self.execute_query(query, (as_of_date, as_of_date))
        return df['ticker'].tolist()


class DataCache:
    """Simple LRU cache wrapper for data queries."""

    def __init__(self, maxsize: int = 128):
        """Initialize cache."""
        self._cache: Dict[str, Any] = {}
        self._maxsize = maxsize
        self._access_order: List[str] = []

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Set item in cache."""
        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self._maxsize:
            # Remove least recently used
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        self._cache[key] = value
        self._access_order.append(key)

    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()
        self._access_order.clear()

    def __len__(self) -> int:
        """Return cache size."""
        return len(self._cache)
