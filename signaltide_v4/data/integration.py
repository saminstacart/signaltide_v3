"""
Unified data integration layer for SignalTide v4.

Provides a single interface to access all data sources:
- Market data (prices, returns, volume)
- Fundamental data (financials, ratios)
- Factor data (Fama-French factors)
- Alternative data (insider transactions, events)

All data access is point-in-time (PIT) compliant.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import numpy as np

from signaltide_v4.data.sharadar_adapter import SharadarAdapter
from signaltide_v4.data.factor_data import FactorDataProvider
from signaltide_v4.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class UniverseSnapshot:
    """Point-in-time snapshot of universe data."""

    as_of_date: str
    tickers: List[str]
    prices: pd.Series  # Latest prices
    returns_1m: pd.Series  # 1-month returns
    returns_12m: pd.Series  # 12-month returns
    market_caps: pd.Series  # Market capitalizations
    volatility: pd.Series  # Trailing volatility
    volumes: pd.Series  # Average daily volume
    sector: pd.Series  # Sector classification

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        return pd.DataFrame({
            'price': self.prices,
            'return_1m': self.returns_1m,
            'return_12m': self.returns_12m,
            'market_cap': self.market_caps,
            'volatility': self.volatility,
            'volume': self.volumes,
            'sector': self.sector,
        })


@dataclass
class FundamentalSnapshot:
    """Point-in-time snapshot of fundamental data."""

    as_of_date: str
    tickers: List[str]

    # Profitability
    roa: pd.Series
    roe: pd.Series
    gross_margin: pd.Series
    net_margin: pd.Series
    operating_cf: pd.Series

    # Quality
    debt_equity: pd.Series
    current_ratio: pd.Series
    asset_turnover: pd.Series

    # Growth
    revenue_growth: pd.Series
    earnings_growth: pd.Series

    # Valuation
    pe_ratio: pd.Series
    pb_ratio: pd.Series
    ev_ebitda: pd.Series

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        return pd.DataFrame({
            'roa': self.roa,
            'roe': self.roe,
            'gross_margin': self.gross_margin,
            'net_margin': self.net_margin,
            'operating_cf': self.operating_cf,
            'debt_equity': self.debt_equity,
            'current_ratio': self.current_ratio,
            'asset_turnover': self.asset_turnover,
            'revenue_growth': self.revenue_growth,
            'earnings_growth': self.earnings_growth,
            'pe_ratio': self.pe_ratio,
            'pb_ratio': self.pb_ratio,
            'ev_ebitda': self.ev_ebitda,
        })


class DataIntegration:
    """
    Unified data access layer.

    Combines multiple data sources into a single interface
    with consistent PIT compliance.
    """

    def __init__(
        self,
        sharadar_adapter: Optional[SharadarAdapter] = None,
        factor_provider: Optional[FactorDataProvider] = None,
    ):
        """
        Initialize data integration layer.

        Args:
            sharadar_adapter: Sharadar data adapter
            factor_provider: Factor data provider
        """
        self.sharadar = sharadar_adapter or SharadarAdapter()
        self.factors = factor_provider or FactorDataProvider()

        self.settings = get_settings()

        logger.info("DataIntegration layer initialized")

    def get_universe(
        self,
        universe_type: str = 'default',
        as_of_date: Optional[str] = None,
        min_market_cap: Optional[float] = None,
        min_adv: Optional[float] = None,
    ) -> List[str]:
        """
        Get tradeable universe.

        Args:
            universe_type: Type of universe ('default', 'sp500', 'sp100', 'custom')
            as_of_date: Date for PIT membership
            min_market_cap: Minimum market cap filter
            min_adv: Minimum ADV filter

        Returns:
            List of eligible tickers
        """
        as_of_date = as_of_date or datetime.now().strftime('%Y-%m-%d')

        if universe_type == 'sp500':
            tickers = self.sharadar.get_sp500_constituents(as_of_date)
        elif universe_type == 'sp100':
            # Top 100 by market cap
            all_tickers = self.sharadar.get_sp500_constituents(as_of_date)
            market_caps = self.sharadar.get_market_cap(all_tickers, as_of_date)
            tickers = market_caps.nlargest(100).index.tolist()
        else:
            # Default: liquid large/mega caps
            tickers = self.sharadar.get_sp500_constituents(as_of_date)

        # Apply filters
        if min_market_cap or min_adv:
            tickers = self._apply_liquidity_filters(
                tickers, as_of_date, min_market_cap, min_adv
            )

        logger.info(f"Universe '{universe_type}': {len(tickers)} tickers")
        return tickers

    def _apply_liquidity_filters(
        self,
        tickers: List[str],
        as_of_date: str,
        min_market_cap: Optional[float],
        min_adv: Optional[float],
    ) -> List[str]:
        """Apply market cap and ADV filters."""
        filtered = set(tickers)

        if min_market_cap:
            market_caps = self.sharadar.get_market_cap(tickers, as_of_date)
            valid = market_caps[market_caps >= min_market_cap].index
            filtered &= set(valid)

        if min_adv:
            advs = self.sharadar.get_adv(tickers, as_of_date)
            valid = advs[advs >= min_adv].index
            filtered &= set(valid)

        return list(filtered)

    def get_universe_snapshot(
        self,
        tickers: List[str],
        as_of_date: str,
    ) -> UniverseSnapshot:
        """
        Get comprehensive snapshot of universe data.

        Args:
            tickers: List of ticker symbols
            as_of_date: Point-in-time date

        Returns:
            UniverseSnapshot with all metrics
        """
        logger.debug(f"Getting universe snapshot for {len(tickers)} tickers as of {as_of_date}")

        # Price data
        end_date = as_of_date
        start_date_1y = (
            pd.Timestamp(as_of_date) - pd.Timedelta(days=400)
        ).strftime('%Y-%m-%d')

        prices_df = self.sharadar.get_prices(tickers, start_date_1y, end_date)

        if prices_df.empty:
            return self._empty_universe_snapshot(as_of_date, tickers)

        # Latest prices
        latest_prices = prices_df.iloc[-1]

        # Returns
        returns_1m = self._calculate_returns(prices_df, 21)
        returns_12m = self._calculate_returns(prices_df, 252)

        # Volatility
        returns = prices_df.pct_change()
        volatility = returns.tail(252).std() * np.sqrt(252)

        # Market cap and ADV
        market_caps = self.sharadar.get_market_cap(tickers, as_of_date)
        volumes = self.sharadar.get_adv(tickers, as_of_date)

        # Sector info
        ticker_info = self.sharadar.get_ticker_info(tickers)
        sectors = ticker_info['sector'] if 'sector' in ticker_info.columns else pd.Series()

        return UniverseSnapshot(
            as_of_date=as_of_date,
            tickers=tickers,
            prices=latest_prices,
            returns_1m=returns_1m,
            returns_12m=returns_12m,
            market_caps=market_caps,
            volatility=volatility,
            volumes=volumes,
            sector=sectors,
        )

    def get_fundamental_snapshot(
        self,
        tickers: List[str],
        as_of_date: str,
    ) -> FundamentalSnapshot:
        """
        Get comprehensive fundamental data snapshot.

        Args:
            tickers: List of ticker symbols
            as_of_date: Point-in-time date

        Returns:
            FundamentalSnapshot with all metrics
        """
        logger.debug(f"Getting fundamental snapshot for {len(tickers)} tickers")

        # Get raw fundamentals
        metrics = [
            'roa', 'roe', 'gross_margin', 'net_margin', 'operating_cf',
            'total_assets', 'total_liabilities', 'equity', 'debt',
            'revenue', 'net_income', 'pe_ratio', 'pb_ratio',
        ]

        fundamentals = self.sharadar.get_fundamentals_pit(tickers, as_of_date, metrics)

        if fundamentals.empty:
            return self._empty_fundamental_snapshot(as_of_date, tickers)

        # Calculate derived metrics
        return FundamentalSnapshot(
            as_of_date=as_of_date,
            tickers=tickers,
            roa=self._safe_get(fundamentals, 'ROA'),
            roe=self._safe_get(fundamentals, 'ROE'),
            gross_margin=self._safe_get(fundamentals, 'GROSSMARGIN'),
            net_margin=self._safe_get(fundamentals, 'NETMARGIN'),
            operating_cf=self._safe_get(fundamentals, 'NCFO'),
            debt_equity=self._calculate_debt_equity(fundamentals),
            current_ratio=pd.Series(index=tickers, dtype=float),  # Would need more data
            asset_turnover=self._calculate_asset_turnover(fundamentals),
            revenue_growth=pd.Series(index=tickers, dtype=float),  # Needs historical
            earnings_growth=pd.Series(index=tickers, dtype=float),  # Needs historical
            pe_ratio=self._safe_get(fundamentals, 'PE'),
            pb_ratio=self._safe_get(fundamentals, 'PB'),
            ev_ebitda=self._safe_get(fundamentals, 'EVEBITDA'),
        )

    def get_insider_data(
        self,
        tickers: List[str],
        as_of_date: str,
        lookback_days: int = 90,
    ) -> pd.DataFrame:
        """
        Get insider transaction data.

        Args:
            tickers: List of ticker symbols
            as_of_date: Point-in-time date
            lookback_days: Days of history to retrieve

        Returns:
            DataFrame with insider transactions
        """
        start_date = (
            pd.Timestamp(as_of_date) - pd.Timedelta(days=lookback_days)
        ).strftime('%Y-%m-%d')

        transactions = self.sharadar.get_insider_transactions(
            tickers, start_date, as_of_date
        )

        return transactions

    def get_factor_data(
        self,
        start_date: str,
        end_date: str,
        factors: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get Fama-French factor data.

        Args:
            start_date: Start date
            end_date: End date
            factors: List of factors (default: all FF5)

        Returns:
            DataFrame with factor returns
        """
        return self.factors.get_factors(start_date, end_date)

    def get_market_data_for_backtest(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> Dict[str, pd.DataFrame]:
        """
        Get all market data needed for backtesting.

        Args:
            tickers: List of ticker symbols
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            Dict with 'prices', 'returns', 'volume' DataFrames
        """
        logger.info(f"Loading backtest data: {len(tickers)} tickers, {start_date} to {end_date}")

        # Extend start date for warmup
        warmup_start = (
            pd.Timestamp(start_date) - pd.Timedelta(days=365)
        ).strftime('%Y-%m-%d')

        prices = self.sharadar.get_prices(tickers, warmup_start, end_date)
        returns = prices.pct_change()

        return {
            'prices': prices,
            'returns': returns,
        }

    def _calculate_returns(
        self,
        prices: pd.DataFrame,
        days: int,
    ) -> pd.Series:
        """Calculate returns over N days."""
        if len(prices) < days + 1:
            return pd.Series(index=prices.columns, dtype=float)

        current = prices.iloc[-1]
        previous = prices.iloc[-(days + 1)]

        returns = (current / previous) - 1
        return returns

    def _safe_get(
        self,
        df: pd.DataFrame,
        column: str,
    ) -> pd.Series:
        """Safely get column from DataFrame."""
        if column in df.columns:
            return df[column]
        return pd.Series(index=df.index, dtype=float)

    def _calculate_debt_equity(self, df: pd.DataFrame) -> pd.Series:
        """Calculate debt-to-equity ratio."""
        debt = self._safe_get(df, 'DEBT')
        equity = self._safe_get(df, 'EQUITY')

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = debt / equity
            ratio = ratio.replace([np.inf, -np.inf], np.nan)

        return ratio

    def _calculate_asset_turnover(self, df: pd.DataFrame) -> pd.Series:
        """Calculate asset turnover ratio."""
        revenue = self._safe_get(df, 'REVENUE')
        assets = self._safe_get(df, 'ASSETS')

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = revenue / assets
            ratio = ratio.replace([np.inf, -np.inf], np.nan)

        return ratio

    def _empty_universe_snapshot(
        self,
        as_of_date: str,
        tickers: List[str],
    ) -> UniverseSnapshot:
        """Return empty snapshot."""
        empty = pd.Series(index=tickers, dtype=float)
        return UniverseSnapshot(
            as_of_date=as_of_date,
            tickers=tickers,
            prices=empty,
            returns_1m=empty,
            returns_12m=empty,
            market_caps=empty,
            volatility=empty,
            volumes=empty,
            sector=pd.Series(index=tickers, dtype=str),
        )

    def _empty_fundamental_snapshot(
        self,
        as_of_date: str,
        tickers: List[str],
    ) -> FundamentalSnapshot:
        """Return empty fundamental snapshot."""
        empty = pd.Series(index=tickers, dtype=float)
        return FundamentalSnapshot(
            as_of_date=as_of_date,
            tickers=tickers,
            roa=empty,
            roe=empty,
            gross_margin=empty,
            net_margin=empty,
            operating_cf=empty,
            debt_equity=empty,
            current_ratio=empty,
            asset_turnover=empty,
            revenue_growth=empty,
            earnings_growth=empty,
            pe_ratio=empty,
            pb_ratio=empty,
            ev_ebitda=empty,
        )

    def close(self):
        """Close connections."""
        self.sharadar.close()
        logger.debug("DataIntegration closed")
