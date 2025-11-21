#!/usr/bin/env python3
"""
Universe Manager - Point-in-Time Stock Universe Construction

Manages stock universe selection with strict point-in-time correctness:
- Market cap based filtering (top N, ranges)
- Sector/industry filtering
- S&P 500 proxy (top 500 by market cap)
- Manual ticker lists
- Automatic IPO/delisting boundary enforcement

References:
- Hou, Xue & Zhang (2015) "Digesting Anomalies: An Investment Approach"
- McLean & Pontiff (2016) "Does Academic Research Destroy Stock Return Predictability?"

Academic Note:
Point-in-time universe construction is CRITICAL to avoid selection bias.
Using future information to select stocks is a form of lookahead bias that
can artificially inflate backtest performance.

INVARIANT:
All point-in-time universe membership in SignalTide must go through UniverseManager.
Other modules should never query dim_universe_membership directly.
Exception: scripts/refresh_universe_membership.py populates the table.
"""

from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime, timedelta
import warnings
import pandas as pd
from functools import lru_cache

from data.data_manager import DataManager
from config import get_logger

logger = get_logger(__name__)


class UniverseManager:
    """
    Manages stock universe construction with point-in-time correctness.

    Supports multiple universe types:
    - sp500_actual: Actual S&P 500 membership from dim_universe_membership (recommended)
    - manual: Explicit ticker list
    - top_N: Top N stocks by market cap
    - market_cap_range: Stocks within a market cap range (e.g., mid-cap only)
    - sector: Filter by GICS sector
    - sp500_proxy: Top 500 stocks by market cap (DEPRECATED - use sp500_actual)
    - sp1000_proxy: Top 1000 stocks (proxy for Russell 1000)
    - nasdaq_proxy: Technology + Communication Services sectors (rough NASDAQ proxy)
    """

    def __init__(self, data_manager: Optional[DataManager] = None):
        """
        Initialize UniverseManager.

        Args:
            data_manager: Optional DataManager instance (creates one if None)
        """
        self.dm = data_manager or DataManager()
        logger.info("UniverseManager initialized")

        # Cache for market cap data (cleared when parameters change)
        self._market_cap_cache: Dict[str, pd.DataFrame] = {}

    def get_universe_tickers(
        self,
        universe_name: str,
        as_of_date: str
    ) -> List[str]:
        """
        Get tickers in universe as of a point in time using dim_universe_membership.

        This method queries the dim_universe_membership table which tracks
        actual index membership with start/end dates (Slowly Changing Dimension).

        Args:
            universe_name: Universe identifier (e.g., 'sp500_actual', 'nasdaq_actual')
            as_of_date: Point-in-time date (YYYY-MM-DD)

        Returns:
            Sorted list of tickers in the universe at that date

        Example:
            >>> um = UniverseManager()
            >>> tickers = um.get_universe_tickers('sp500_actual', '2023-01-01')
            >>> len(tickers)  # Should be ~503
            503
        """
        logger.info(f"Querying {universe_name} universe as of {as_of_date} from dim_universe_membership")

        # POINT-IN-TIME SEMANTICS INVARIANT:
        # - membership_start_date is INCLUSIVE (first date ticker is in universe)
        # - membership_end_date is EXCLUSIVE (first date ticker is NOT in universe)
        #
        # Example: DFS with start=2015-03-31, end=2025-05-19
        #   - as_of=2015-03-31: INCLUDED (start_date <= as_of and end_date > as_of)
        #   - as_of=2025-05-18: INCLUDED (last day in universe)
        #   - as_of=2025-05-19: EXCLUDED (end_date is NOT > as_of)
        #
        # This matches typical [start, end) interval semantics used in Python slicing,
        # pandas date ranges, and most temporal database systems.
        query = """
            SELECT ticker
            FROM dim_universe_membership
            WHERE universe_name = ?
              AND membership_start_date <= ?
              AND (membership_end_date IS NULL OR membership_end_date > ?)
            ORDER BY ticker;
        """

        conn = self.dm._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, (universe_name, as_of_date, as_of_date))
            results = cursor.fetchall()
            tickers = [row[0] for row in results]

            logger.info(
                f"Retrieved {len(tickers)} tickers from {universe_name} "
                f"universe as of {as_of_date}"
            )

            return tickers
        finally:
            conn.close()

    def get_universe(
        self,
        universe_type: str,
        as_of_date: str,
        manual_tickers: Optional[List[str]] = None,
        top_n: Optional[int] = None,
        min_market_cap: Optional[float] = None,
        max_market_cap: Optional[float] = None,
        sectors: Optional[List[str]] = None,
        min_price: float = 5.0,
        max_price: Optional[float] = None,
    ) -> List[str]:
        """
        Get universe of stocks as of a specific date.

        Args:
            universe_type: Type of universe:
                - 'sp500_actual': Actual S&P 500 membership (recommended)
                - 'manual': Explicit ticker list
                - 'top_N': Top N stocks by market cap
                - 'market_cap_range': Stocks within a market cap range
                - 'sector': Filter by GICS sector
                - 'sp500_proxy': Top 500 by market cap (DEPRECATED - use sp500_actual)
                - 'sp1000_proxy': Top 1000 by market cap
                - 'nasdaq_proxy': Technology + Communication Services sectors
            as_of_date: Date for point-in-time universe (YYYY-MM-DD)
            manual_tickers: List of tickers (for 'manual' type)
            top_n: Number of top stocks by market cap (for 'top_N' type)
            min_market_cap: Minimum market cap in USD (for 'market_cap_range')
            max_market_cap: Maximum market cap in USD (for 'market_cap_range')
            sectors: List of GICS sectors to include (for 'sector' type)
            min_price: Minimum stock price (default $5 to avoid penny stocks)
            max_price: Maximum stock price (optional)

        Returns:
            List of ticker symbols

        Raises:
            ValueError: If universe_type is invalid or required parameters missing

        Note:
            For S&P 500 backtests, use 'sp500_actual' instead of 'sp500_proxy'.
            The 'sp500_actual' type uses dim_universe_membership which tracks
            actual index changes, while 'sp500_proxy' is a market-cap approximation.
        """
        logger.info(f"Getting {universe_type} universe as of {as_of_date}")

        # NEW: sp500_actual uses dimensional table (recommended)
        if universe_type == 'sp500_actual':
            return self.get_universe_tickers('sp500_actual', as_of_date)

        elif universe_type == 'manual':
            if not manual_tickers:
                raise ValueError("manual_tickers required for universe_type='manual'")
            return self._get_manual_universe(manual_tickers, as_of_date)

        elif universe_type == 'top_N':
            if not top_n:
                raise ValueError("top_n required for universe_type='top_N'")
            return self._get_top_n_universe(
                as_of_date, top_n, sectors, min_price, max_price
            )

        elif universe_type == 'market_cap_range':
            if min_market_cap is None and max_market_cap is None:
                raise ValueError(
                    "min_market_cap or max_market_cap required for "
                    "universe_type='market_cap_range'"
                )
            return self._get_market_cap_range_universe(
                as_of_date, min_market_cap, max_market_cap, sectors,
                min_price, max_price
            )

        elif universe_type == 'sector':
            if not sectors:
                raise ValueError("sectors required for universe_type='sector'")
            return self._get_sector_universe(
                as_of_date, sectors, min_market_cap, min_price, max_price
            )

        elif universe_type == 'sp500_proxy':
            # DEPRECATED: Use sp500_actual for true S&P 500 membership
            warnings.warn(
                "sp500_proxy is deprecated and uses a market-cap approximation. "
                "Use universe_type='sp500_actual' for true S&P 500 membership tracking. "
                "See docs/DATA_ARCHITECTURE.md for details.",
                DeprecationWarning,
                stacklevel=2
            )
            logger.info("Using top 500 by market cap as S&P 500 proxy (DEPRECATED)")
            return self._get_top_n_universe(
                as_of_date, top_n=500, sectors=None,
                min_price=min_price, max_price=max_price
            )

        elif universe_type == 'sp1000_proxy':
            logger.info("Using top 1000 by market cap as Russell 1000 proxy")
            return self._get_top_n_universe(
                as_of_date, top_n=1000, sectors=None,
                min_price=min_price, max_price=max_price
            )

        elif universe_type == 'nasdaq_proxy':
            logger.info("Using Technology + Communication Services as NASDAQ proxy")
            return self._get_sector_universe(
                as_of_date,
                sectors=['Technology', 'Communication Services'],
                min_market_cap=min_market_cap,
                min_price=min_price,
                max_price=max_price
            )

        else:
            raise ValueError(
                f"Invalid universe_type: {universe_type}. "
                f"Must be one of: sp500_actual, manual, top_N, market_cap_range, sector, "
                f"sp500_proxy (deprecated), sp1000_proxy, nasdaq_proxy"
            )

    def _get_manual_universe(
        self, tickers: List[str], as_of_date: str
    ) -> List[str]:
        """
        Get manually specified universe, filtered for validity at as_of_date.

        Args:
            tickers: List of ticker symbols
            as_of_date: Date to check validity

        Returns:
            List of valid tickers (filters out those not yet IPO'd or already delisted)
        """
        logger.info(f"Manual universe: {len(tickers)} tickers specified")

        # Get ticker metadata to check IPO/delisting dates
        # Use MIN/MAX to handle duplicate rows with same ticker
        placeholders = ','.join(['?'] * len(tickers))
        query = f"""
            SELECT
                ticker,
                MIN(firstpricedate) as firstpricedate,
                MAX(lastpricedate) as lastpricedate,
                MAX(isdelisted) as isdelisted
            FROM sharadar_tickers
            WHERE ticker IN ({placeholders})
              AND category IN ('Domestic Common Stock', 'Domestic Common Stock Primary Class')
            GROUP BY ticker
        """

        conn = self.dm._get_connection()
        ticker_info = pd.read_sql_query(query, conn, params=tickers)
        conn.close()

        # Filter for tickers valid at as_of_date
        as_of = pd.to_datetime(as_of_date)
        valid_tickers = []

        for _, row in ticker_info.iterrows():
            first_date = pd.to_datetime(row['firstpricedate'], format='mixed') if row['firstpricedate'] else None
            last_date = pd.to_datetime(row['lastpricedate'], format='mixed') if row['lastpricedate'] else None

            # Check if ticker was trading at as_of_date
            if first_date and first_date > as_of:
                logger.debug(f"Excluding {row['ticker']}: not yet IPO'd at {as_of_date}")
                continue

            if last_date and last_date < as_of:
                logger.debug(f"Excluding {row['ticker']}: delisted before {as_of_date}")
                continue

            valid_tickers.append(row['ticker'])

        logger.info(
            f"Manual universe: {len(valid_tickers)}/{len(tickers)} tickers valid "
            f"at {as_of_date}"
        )
        return valid_tickers

    def _get_market_cap_data(
        self, as_of_date: str, sectors: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get market cap data as of a specific date (point-in-time).

        Uses Most Recent Quarterly (MRQ) dimension and respects datekey <= as_of_date.

        Args:
            as_of_date: Date for point-in-time data
            sectors: Optional list of sectors to filter

        Returns:
            DataFrame with columns: ticker, marketcap, sector, industry
        """
        cache_key = f"{as_of_date}_{sectors}"
        if cache_key in self._market_cap_cache:
            logger.debug(f"Using cached market cap data for {as_of_date}")
            return self._market_cap_cache[cache_key]

        logger.info(f"Querying market cap data as of {as_of_date}")

        # Query for most recent fundamentals as of as_of_date
        # We use a subquery to get the most recent datekey for each ticker
        sector_filter = ""
        params = [as_of_date]

        if sectors:
            sector_placeholders = ','.join(['?'] * len(sectors))
            sector_filter = f"AND t.sector IN ({sector_placeholders})"
            params.extend(sectors)

        query = f"""
            WITH latest_fundamentals AS (
                SELECT
                    ticker,
                    MAX(datekey) as latest_datekey
                FROM sharadar_sf1
                WHERE dimension = 'MRQ'
                  AND datekey <= ?
                  AND marketcap IS NOT NULL
                  AND marketcap > 0
                GROUP BY ticker
            )
            SELECT
                f.ticker,
                f.marketcap,
                t.sector,
                t.industry,
                t.category,
                t.isdelisted,
                t.firstpricedate,
                t.lastpricedate
            FROM sharadar_sf1 f
            JOIN latest_fundamentals lf
                ON f.ticker = lf.ticker
                AND f.datekey = lf.latest_datekey
            JOIN sharadar_tickers t
                ON f.ticker = t.ticker
            WHERE f.dimension = 'MRQ'
              AND t.category IN ('Domestic Common Stock', 'Domestic Common Stock Primary Class')
              {sector_filter}
            ORDER BY f.marketcap DESC
        """

        conn = self.dm._get_connection()
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        # Filter for stocks that were trading at as_of_date
        as_of = pd.to_datetime(as_of_date)
        df['firstpricedate'] = pd.to_datetime(df['firstpricedate'], format='mixed', errors='coerce')
        df['lastpricedate'] = pd.to_datetime(df['lastpricedate'], format='mixed', errors='coerce')

        # Remove stocks not yet IPO'd
        df = df[df['firstpricedate'] <= as_of]

        # Remove stocks already delisted
        df = df[(df['lastpricedate'].isna()) | (df['lastpricedate'] >= as_of)]

        logger.info(
            f"Retrieved {len(df)} stocks with market cap data as of {as_of_date}"
        )

        # Cache result
        self._market_cap_cache[cache_key] = df

        return df

    def _get_top_n_universe(
        self,
        as_of_date: str,
        top_n: int,
        sectors: Optional[List[str]] = None,
        min_price: float = 5.0,
        max_price: Optional[float] = None,
    ) -> List[str]:
        """
        Get top N stocks by market cap as of a specific date.

        Args:
            as_of_date: Date for point-in-time universe
            top_n: Number of top stocks to return
            sectors: Optional sector filter
            min_price: Minimum stock price
            max_price: Maximum stock price

        Returns:
            List of top N ticker symbols
        """
        logger.info(f"Getting top {top_n} stocks by market cap as of {as_of_date}")

        # Get market cap data
        df = self._get_market_cap_data(as_of_date, sectors)

        # Apply price filters if needed
        if min_price or max_price:
            df = self._apply_price_filters(df, as_of_date, min_price, max_price)

        # Sort by market cap and take top N
        df = df.nlargest(top_n, 'marketcap')

        tickers = df['ticker'].tolist()

        logger.info(
            f"Selected {len(tickers)} stocks (top {top_n} by market cap). "
            f"Market cap range: ${df['marketcap'].min()/1e9:.1f}B - "
            f"${df['marketcap'].max()/1e9:.1f}B"
        )

        return tickers

    def _get_market_cap_range_universe(
        self,
        as_of_date: str,
        min_market_cap: Optional[float],
        max_market_cap: Optional[float],
        sectors: Optional[List[str]] = None,
        min_price: float = 5.0,
        max_price: Optional[float] = None,
    ) -> List[str]:
        """
        Get stocks within a market cap range.

        Useful for targeting specific market cap segments:
        - Large cap: >$10B
        - Mid cap: $2B - $10B
        - Small cap: <$2B

        Args:
            as_of_date: Date for point-in-time universe
            min_market_cap: Minimum market cap in USD (None for no minimum)
            max_market_cap: Maximum market cap in USD (None for no maximum)
            sectors: Optional sector filter
            min_price: Minimum stock price
            max_price: Maximum stock price

        Returns:
            List of ticker symbols in the market cap range
        """
        logger.info(
            f"Getting stocks with market cap range "
            f"${min_market_cap/1e9 if min_market_cap else 0:.1f}B - "
            f"${max_market_cap/1e9 if max_market_cap else 'inf'}B "
            f"as of {as_of_date}"
        )

        # Get market cap data
        df = self._get_market_cap_data(as_of_date, sectors)

        # Apply market cap filters
        if min_market_cap is not None:
            df = df[df['marketcap'] >= min_market_cap]

        if max_market_cap is not None:
            df = df[df['marketcap'] <= max_market_cap]

        # Apply price filters if needed
        if min_price or max_price:
            df = self._apply_price_filters(df, as_of_date, min_price, max_price)

        tickers = df['ticker'].tolist()

        if len(df) > 0:
            logger.info(
                f"Selected {len(tickers)} stocks in market cap range. "
                f"Actual range: ${df['marketcap'].min()/1e9:.1f}B - "
                f"${df['marketcap'].max()/1e9:.1f}B"
            )
        else:
            logger.warning(f"No stocks found in specified market cap range")

        return tickers

    def _get_sector_universe(
        self,
        as_of_date: str,
        sectors: List[str],
        min_market_cap: Optional[float] = None,
        min_price: float = 5.0,
        max_price: Optional[float] = None,
    ) -> List[str]:
        """
        Get all stocks in specified sectors.

        GICS Sectors:
        - Basic Materials
        - Communication Services
        - Consumer Cyclical
        - Consumer Defensive
        - Energy
        - Financial Services
        - Healthcare
        - Industrials
        - Real Estate
        - Technology
        - Utilities

        Args:
            as_of_date: Date for point-in-time universe
            sectors: List of GICS sectors
            min_market_cap: Optional minimum market cap filter
            min_price: Minimum stock price
            max_price: Maximum stock price

        Returns:
            List of ticker symbols in the specified sectors
        """
        logger.info(f"Getting stocks in sectors {sectors} as of {as_of_date}")

        # Get market cap data filtered by sector
        df = self._get_market_cap_data(as_of_date, sectors)

        # Apply market cap filter if specified
        if min_market_cap is not None:
            df = df[df['marketcap'] >= min_market_cap]

        # Apply price filters if needed
        if min_price or max_price:
            df = self._apply_price_filters(df, as_of_date, min_price, max_price)

        tickers = df['ticker'].tolist()

        logger.info(
            f"Selected {len(tickers)} stocks in sectors {sectors}. "
            f"Market cap range: ${df['marketcap'].min()/1e9:.1f}B - "
            f"${df['marketcap'].max()/1e9:.1f}B"
        )

        return tickers

    def _apply_price_filters(
        self,
        df: pd.DataFrame,
        as_of_date: str,
        min_price: Optional[float],
        max_price: Optional[float],
    ) -> pd.DataFrame:
        """
        Apply price filters to universe.

        Args:
            df: DataFrame with ticker column
            as_of_date: Date to check prices
            min_price: Minimum price filter
            max_price: Maximum price filter

        Returns:
            Filtered DataFrame
        """
        if min_price is None and max_price is None:
            return df

        logger.info(f"Applying price filters: min=${min_price}, max=${max_price}")

        tickers = df['ticker'].tolist()

        # Get prices around as_of_date (use a 5-day window to handle non-trading days)
        as_of = pd.to_datetime(as_of_date)
        start_date = (as_of - timedelta(days=5)).strftime('%Y-%m-%d')
        end_date = as_of.strftime('%Y-%m-%d')

        # Query prices
        placeholders = ','.join(['?'] * len(tickers))
        query = f"""
            SELECT ticker, date, close
            FROM sharadar_prices
            WHERE ticker IN ({placeholders})
              AND date >= ?
              AND date <= ?
            ORDER BY date DESC
        """

        conn = self.dm._get_connection()
        prices = pd.read_sql_query(
            query, conn, params=tickers + [start_date, end_date]
        )
        conn.close()

        # Get most recent price for each ticker
        prices['date'] = pd.to_datetime(prices['date'], format='mixed')
        latest_prices = prices.sort_values('date').groupby('ticker').last()

        # Apply filters
        valid_tickers = set()
        for ticker in tickers:
            if ticker not in latest_prices.index:
                logger.debug(f"No price data for {ticker} near {as_of_date}")
                continue

            price = latest_prices.loc[ticker, 'close']

            if min_price is not None and price < min_price:
                logger.debug(f"Excluding {ticker}: price ${price:.2f} < ${min_price}")
                continue

            if max_price is not None and price > max_price:
                logger.debug(f"Excluding {ticker}: price ${price:.2f} > ${max_price}")
                continue

            valid_tickers.add(ticker)

        # Filter DataFrame
        df_filtered = df[df['ticker'].isin(valid_tickers)]

        logger.info(
            f"Price filter: {len(df_filtered)}/{len(df)} stocks passed "
            f"(min=${min_price}, max=${max_price})"
        )

        return df_filtered

    def get_universe_info(
        self, tickers: List[str], as_of_date: str
    ) -> pd.DataFrame:
        """
        Get metadata for universe stocks (sector, industry, market cap).

        Args:
            tickers: List of ticker symbols
            as_of_date: Date for point-in-time data

        Returns:
            DataFrame with ticker, sector, industry, marketcap
        """
        logger.info(f"Getting universe info for {len(tickers)} tickers")

        placeholders = ','.join(['?'] * len(tickers))

        query = f"""
            WITH latest_fundamentals AS (
                SELECT
                    ticker,
                    MAX(datekey) as latest_datekey
                FROM sharadar_sf1
                WHERE dimension = 'MRQ'
                  AND datekey <= ?
                  AND ticker IN ({placeholders})
                GROUP BY ticker
            )
            SELECT
                t.ticker,
                t.name,
                t.sector,
                t.industry,
                t.category,
                f.marketcap,
                f.revenue,
                f.assets
            FROM sharadar_tickers t
            LEFT JOIN latest_fundamentals lf ON t.ticker = lf.ticker
            LEFT JOIN sharadar_sf1 f
                ON t.ticker = f.ticker
                AND f.datekey = lf.latest_datekey
                AND f.dimension = 'MRQ'
            WHERE t.ticker IN ({placeholders})
            ORDER BY f.marketcap DESC NULLS LAST
        """

        params = [as_of_date] + tickers + tickers

        conn = self.dm._get_connection()
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        logger.info(f"Retrieved info for {len(df)} tickers")

        return df

    def clear_cache(self):
        """Clear market cap cache."""
        self._market_cap_cache.clear()
        logger.info("Cleared market cap cache")
