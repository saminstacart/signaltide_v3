"""
Fundamental data provider with Point-in-Time compliance.

Provides:
- Cash-based operating profitability (CbOP) per Ball et al. (2016)
- Asset growth screen
- Buyback yield
- Standard fundamental metrics
"""

import logging
from typing import List, Optional, Dict, Union
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from .base import PITDataManager, DataCache
from signaltide_v4.config.settings import get_settings

logger = logging.getLogger(__name__)


class FundamentalDataProvider(PITDataManager):
    """
    Provider for fundamental data with strict PIT compliance.

    Key metrics:
    - Cash-based Operating Profitability (CbOP)
    - Asset Growth
    - Buyback Yield
    - Standard ratios (ROE, ROA, etc.)
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize fundamental data provider."""
        super().__init__(db_path)
        self._cache = DataCache(maxsize=256)
        settings = get_settings()
        self.filing_lag_days = settings.filing_lag_days

    def get_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        as_of_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Get fundamental data for tickers."""
        return self.get_fundamentals(tickers, start_date, end_date, as_of_date)

    def get_fundamentals(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        as_of_date: Optional[str] = None,
        dimension: str = 'ARQ'
    ) -> pd.DataFrame:
        """
        Get fundamental data with PIT compliance.

        Args:
            tickers: List of ticker symbols
            start_date: Start date for fundamental data
            end_date: End date
            as_of_date: Point-in-time cutoff (when we could know this data)
            dimension: 'ARQ' (quarterly) or 'ARY' (annual)

        Returns:
            DataFrame with fundamentals, only including data known as of as_of_date
        """
        if not tickers:
            return pd.DataFrame()

        as_of_date = as_of_date or end_date

        # Cache key
        cache_key = f"fund_{','.join(sorted(tickers[:10]))}_{start_date}_{end_date}_{as_of_date}_{dimension}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        placeholders = ','.join(['?' for _ in tickers])
        query = f"""
            SELECT ticker, calendardate, datekey, reportperiod,
                   revenue, netinc, assets, liabilities, equity,
                   ncfo, receivables, inventory, payables, depamor,
                   shareswa, sharesbas
            FROM sharadar_sf1
            WHERE ticker IN ({placeholders})
            AND dimension = ?
            AND datekey BETWEEN ? AND ?
            AND datekey <= ?
            ORDER BY ticker, datekey
        """

        # PIT: only data filed before as_of_date is available
        pit_cutoff = (pd.Timestamp(as_of_date) - timedelta(days=self.filing_lag_days))
        pit_cutoff = pit_cutoff.strftime('%Y-%m-%d')

        params = tuple(tickers) + (dimension, start_date, end_date, pit_cutoff)
        df = self.execute_query(query, params)

        if len(df) > 0:
            df['calendardate'] = pd.to_datetime(df['calendardate'])
            df['datekey'] = pd.to_datetime(df['datekey'])

        self._cache.set(cache_key, df)
        return df

    def get_cbop(
        self,
        tickers: List[str],
        as_of_date: str,
        lookback_quarters: int = 4
    ) -> pd.Series:
        """
        Calculate Cash-Based Operating Profitability (CbOP).

        Formula from Ball et al. (2016):
        CbOP = (OCF - ΔAR - ΔInventory + ΔAP) / Total Assets

        Returns:
            Series with CbOP per ticker
        """
        # Get fundamentals for lookback period
        start_date = (pd.Timestamp(as_of_date) - timedelta(days=lookback_quarters * 100))
        start_date = start_date.strftime('%Y-%m-%d')

        fundamentals = self.get_fundamentals(
            tickers, start_date, as_of_date, as_of_date=as_of_date
        )

        if len(fundamentals) == 0:
            return pd.Series(dtype=float)

        cbop_values = {}

        for ticker in tickers:
            ticker_data = fundamentals[fundamentals['ticker'] == ticker].copy()
            if len(ticker_data) < 2:
                continue

            # Sort by date and get most recent quarters
            ticker_data = ticker_data.sort_values('datekey').tail(lookback_quarters)

            if len(ticker_data) < 2:
                continue

            # Get current and prior period values
            current = ticker_data.iloc[-1]
            prior = ticker_data.iloc[-2]

            # Calculate TTM values if we have multiple quarters
            ttm_ocf = ticker_data['ncfo'].sum()

            # Calculate changes
            delta_ar = current['receivables'] - prior['receivables']
            delta_inv = current['inventory'] - prior['inventory']
            delta_ap = current['payables'] - prior['payables']

            # CbOP formula
            assets = current['assets']
            if pd.isna(assets) or assets <= 0:
                continue

            # Handle NaN values
            delta_ar = 0 if pd.isna(delta_ar) else delta_ar
            delta_inv = 0 if pd.isna(delta_inv) else delta_inv
            delta_ap = 0 if pd.isna(delta_ap) else delta_ap
            ttm_ocf = 0 if pd.isna(ttm_ocf) else ttm_ocf

            cbop = (ttm_ocf - delta_ar - delta_inv + delta_ap) / assets
            cbop_values[ticker] = cbop

        return pd.Series(cbop_values)

    def get_asset_growth(
        self,
        tickers: List[str],
        as_of_date: str
    ) -> pd.Series:
        """
        Calculate asset growth rate.

        From Cooper, Gulen & Schill (2008): negative predictor of returns.

        Returns:
            Series with asset growth per ticker
        """
        # Need 2 years of data
        start_date = (pd.Timestamp(as_of_date) - timedelta(days=500))
        start_date = start_date.strftime('%Y-%m-%d')

        fundamentals = self.get_fundamentals(
            tickers, start_date, as_of_date, as_of_date=as_of_date
        )

        if len(fundamentals) == 0:
            return pd.Series(dtype=float)

        growth_values = {}

        for ticker in tickers:
            ticker_data = fundamentals[fundamentals['ticker'] == ticker].copy()
            if len(ticker_data) < 5:  # Need ~1 year of quarterly data
                continue

            ticker_data = ticker_data.sort_values('datekey')

            # Get assets from ~1 year ago and now
            current_assets = ticker_data['assets'].iloc[-1]

            # Find data from ~1 year ago (4 quarters back)
            if len(ticker_data) >= 5:
                prior_assets = ticker_data['assets'].iloc[-5]
            else:
                prior_assets = ticker_data['assets'].iloc[0]

            if pd.isna(prior_assets) or prior_assets <= 0:
                continue
            if pd.isna(current_assets):
                continue

            growth = (current_assets / prior_assets) - 1
            growth_values[ticker] = growth

        return pd.Series(growth_values)

    def get_buyback_yield(
        self,
        tickers: List[str],
        as_of_date: str
    ) -> pd.Series:
        """
        Calculate buyback yield (net share repurchases / market cap).

        From Grullon & Michaely (2004): positive predictor.

        Returns:
            Series with buyback yield per ticker
        """
        # Get fundamentals for last year
        start_date = (pd.Timestamp(as_of_date) - timedelta(days=400))
        start_date = start_date.strftime('%Y-%m-%d')

        fundamentals = self.get_fundamentals(
            tickers, start_date, as_of_date, as_of_date=as_of_date
        )

        if len(fundamentals) == 0:
            return pd.Series(dtype=float)

        buyback_values = {}

        for ticker in tickers:
            ticker_data = fundamentals[fundamentals['ticker'] == ticker].copy()
            if len(ticker_data) < 5:
                continue

            ticker_data = ticker_data.sort_values('datekey')

            # Calculate change in shares outstanding
            current_shares = ticker_data['shareswa'].iloc[-1]

            if len(ticker_data) >= 5:
                prior_shares = ticker_data['shareswa'].iloc[-5]
            else:
                prior_shares = ticker_data['shareswa'].iloc[0]

            if pd.isna(prior_shares) or prior_shares <= 0:
                continue
            if pd.isna(current_shares) or current_shares <= 0:
                continue

            # Negative change = buyback (shares reduced)
            share_change = (prior_shares - current_shares) / prior_shares
            buyback_values[ticker] = share_change

        return pd.Series(buyback_values)

    def get_quality_metrics(
        self,
        tickers: List[str],
        as_of_date: str
    ) -> pd.DataFrame:
        """
        Get comprehensive quality metrics.

        Returns DataFrame with:
        - cbop: Cash-based operating profitability
        - asset_growth: YoY asset growth
        - buyback_yield: Net share repurchases
        - roe: Return on equity
        - roa: Return on assets
        - debt_to_equity: Leverage ratio
        """
        metrics = pd.DataFrame(index=tickers)

        # Get all metrics
        metrics['cbop'] = self.get_cbop(tickers, as_of_date)
        metrics['asset_growth'] = self.get_asset_growth(tickers, as_of_date)
        metrics['buyback_yield'] = self.get_buyback_yield(tickers, as_of_date)

        # Get basic ratios
        start_date = (pd.Timestamp(as_of_date) - timedelta(days=100))
        start_date = start_date.strftime('%Y-%m-%d')

        fundamentals = self.get_fundamentals(
            tickers, start_date, as_of_date, as_of_date=as_of_date
        )

        roe_values = {}
        roa_values = {}
        leverage_values = {}

        for ticker in tickers:
            ticker_data = fundamentals[fundamentals['ticker'] == ticker]
            if len(ticker_data) == 0:
                continue

            latest = ticker_data.sort_values('datekey').iloc[-1]

            # ROE
            if latest['equity'] > 0:
                roe_values[ticker] = latest['netinc'] / latest['equity']

            # ROA
            if latest['assets'] > 0:
                roa_values[ticker] = latest['netinc'] / latest['assets']

            # Debt to Equity
            if latest['equity'] > 0:
                leverage_values[ticker] = latest['liabilities'] / latest['equity']

        metrics['roe'] = pd.Series(roe_values)
        metrics['roa'] = pd.Series(roa_values)
        metrics['debt_to_equity'] = pd.Series(leverage_values)

        return metrics

    def get_accrual_anomaly(
        self,
        tickers: List[str],
        as_of_date: str
    ) -> pd.Series:
        """
        Calculate accrual anomaly per Sloan (1996).

        Formula: (Net Income - Operating Cash Flow) / Total Assets
        Direction: SHORT high accruals (high accruals = low quality)

        CRITICAL: Uses MRT (Trailing Twelve Months) for netinc/ncfo to avoid
        seasonality bias. ARQ would cause calendar-driven ranking biases
        (e.g., buying retailers in Q4, selling in Q1).

        References:
            Sloan, R. (1996). "Do Stock Prices Fully Reflect Information
            in Accruals and Cash Flows About Future Earnings?"
            The Accounting Review, 71(3), 289-315.

        Returns:
            Series with INVERTED accrual values (high = good quality)
        """
        if not tickers:
            return pd.Series(dtype=float)

        start_date = (pd.Timestamp(as_of_date) - timedelta(days=200)).strftime('%Y-%m-%d')

        placeholders = ','.join(['?' for _ in tickers])
        pit_cutoff = (pd.Timestamp(as_of_date) - timedelta(days=self.filing_lag_days)).strftime('%Y-%m-%d')

        # CRITICAL: dimension='MRT' for TTM flow items (netinc, ncfo)
        query = f"""
            SELECT ticker, datekey, netinc, ncfo, assets
            FROM sharadar_sf1
            WHERE ticker IN ({placeholders})
            AND dimension = 'MRT'
            AND datekey BETWEEN ? AND ?
            AND datekey <= ?
            ORDER BY ticker, datekey DESC
        """
        params = tuple(tickers) + (start_date, as_of_date, pit_cutoff)
        df = self.execute_query(query, params)

        if len(df) == 0:
            logger.warning(f"No MRT data found for accrual anomaly calculation")
            return pd.Series(dtype=float)

        accrual_values = {}
        for ticker in tickers:
            ticker_data = df[df['ticker'] == ticker]
            if len(ticker_data) == 0:
                continue

            latest = ticker_data.iloc[0]  # Already sorted DESC

            assets = latest['assets']
            if pd.isna(assets) or assets <= 0:
                continue

            net_income = latest['netinc'] if not pd.isna(latest['netinc']) else 0
            ncfo = latest['ncfo'] if not pd.isna(latest['ncfo']) else 0

            accrual = (net_income - ncfo) / assets
            # Invert: low accruals = high quality = positive signal
            accrual_values[ticker] = -accrual

        logger.debug(f"Accrual anomaly: {len(accrual_values)}/{len(tickers)} tickers scored")
        return pd.Series(accrual_values)

    def get_gross_profitability(
        self,
        tickers: List[str],
        as_of_date: str
    ) -> pd.Series:
        """
        Calculate gross profitability per Novy-Marx (2013).

        Formula: Gross Profit / Total Assets
        Direction: LONG high values (high GP = pricing power)

        CRITICAL: Uses MRT (Trailing Twelve Months) for gp to avoid seasonality.
        Retailers show massive GP in Q4 (Holiday) vs Q1 - ARQ would create
        calendar-driven ranking biases.

        References:
            Novy-Marx, R. (2013). "The Other Side of Value: The Gross
            Profitability Premium." Journal of Financial Economics, 108(1), 1-28.

        Returns:
            Series with gross profitability values (high = good quality)
        """
        if not tickers:
            return pd.Series(dtype=float)

        start_date = (pd.Timestamp(as_of_date) - timedelta(days=200)).strftime('%Y-%m-%d')

        placeholders = ','.join(['?' for _ in tickers])
        pit_cutoff = (pd.Timestamp(as_of_date) - timedelta(days=self.filing_lag_days)).strftime('%Y-%m-%d')

        # CRITICAL: dimension='MRT' for TTM flow items (gp)
        query = f"""
            SELECT ticker, datekey, gp, assets
            FROM sharadar_sf1
            WHERE ticker IN ({placeholders})
            AND dimension = 'MRT'
            AND datekey BETWEEN ? AND ?
            AND datekey <= ?
            ORDER BY ticker, datekey DESC
        """
        params = tuple(tickers) + (start_date, as_of_date, pit_cutoff)
        df = self.execute_query(query, params)

        if len(df) == 0:
            logger.warning(f"No MRT data found for gross profitability calculation")
            return pd.Series(dtype=float)

        gp_values = {}
        for ticker in tickers:
            ticker_data = df[df['ticker'] == ticker]
            if len(ticker_data) == 0:
                continue

            latest = ticker_data.iloc[0]  # Already sorted DESC
            gp = latest['gp']
            assets = latest['assets']

            if pd.isna(gp) or pd.isna(assets) or assets <= 0:
                continue

            gp_values[ticker] = gp / assets

        logger.debug(f"Gross profitability: {len(gp_values)}/{len(tickers)} tickers scored")
        return pd.Series(gp_values)

    def get_share_reduction(
        self,
        tickers: List[str],
        as_of_date: str
    ) -> pd.Series:
        """
        Calculate share reduction (buyback proxy) using sharesbas.

        Formula: (sharesbas_t - sharesbas_{t-4Q}) / sharesbas_{t-4Q}
        Direction: LONG negative values (share reduction = buybacks)

        Note: equpo/equiss unavailable in Sharadar; use share count proxy.
        ARQ is correct here - sharesbas is a stock variable (snapshot), not a flow.
        The time-diffing logic handles the YoY comparison correctly.

        Returns:
            Series with INVERTED share change (positive = buybacks)
        """
        if not tickers:
            return pd.Series(dtype=float)

        # Need 5 quarters of data (current + 4 prior)
        start_date = (pd.Timestamp(as_of_date) - timedelta(days=500)).strftime('%Y-%m-%d')

        placeholders = ','.join(['?' for _ in tickers])
        pit_cutoff = (pd.Timestamp(as_of_date) - timedelta(days=self.filing_lag_days)).strftime('%Y-%m-%d')

        # ARQ is correct for stock variables (balance sheet snapshots)
        query = f"""
            SELECT ticker, datekey, sharesbas
            FROM sharadar_sf1
            WHERE ticker IN ({placeholders})
            AND dimension = 'ARQ'
            AND datekey BETWEEN ? AND ?
            AND datekey <= ?
            ORDER BY ticker, datekey
        """
        params = tuple(tickers) + (start_date, as_of_date, pit_cutoff)
        df = self.execute_query(query, params)

        if len(df) == 0:
            logger.warning(f"No ARQ data found for share reduction calculation")
            return pd.Series(dtype=float)

        reduction_values = {}
        for ticker in tickers:
            ticker_data = df[df['ticker'] == ticker].sort_values('datekey')
            if len(ticker_data) < 5:  # Need current + 4 quarters
                continue

            current_shares = ticker_data['sharesbas'].iloc[-1]
            prior_shares = ticker_data['sharesbas'].iloc[-5]  # 4 quarters ago

            if pd.isna(prior_shares) or prior_shares <= 0:
                continue
            if pd.isna(current_shares):
                continue

            # Negative change = buyback (good)
            pct_change = (current_shares - prior_shares) / prior_shares
            # Invert so negative change (buybacks) becomes positive signal
            reduction_values[ticker] = -pct_change

        logger.debug(f"Share reduction: {len(reduction_values)}/{len(tickers)} tickers scored")
        return pd.Series(reduction_values)

    def get_intangible_yield(
        self,
        tickers: List[str],
        as_of_date: str
    ) -> pd.Series:
        """
        Calculate intangible value yield per Eisfeldt & Papanikolaou (2013).

        Formula: (R&D + 0.3 × SG&A) / Market Cap

        Capitalizes R&D and portion of SG&A (advertising, training) as intangible
        investment. High values indicate innovation-intensive firms with potential
        mispricing under GAAP accounting.

        CRITICAL: Uses MRT (Trailing Twelve Months) for rnd/sgna (flow items)
        and ARQ marketcap (snapshot). However, since MRT already includes
        marketcap in Sharadar, we use MRT for the full query.

        References:
            Eisfeldt, A. & Papanikolaou, D. (2013). "Organization Capital and
            the Cross-Section of Expected Returns." Journal of Finance, 68(4), 1365-1406.

            Peters, R. & Taylor, L. (2017). "Intangible Capital and the Investment-q
            Relation." Journal of Financial Economics, 123(2), 251-272.

        Returns:
            Series with intangible yield values (high = innovation-intensive)
        """
        if not tickers:
            return pd.Series(dtype=float)

        start_date = (pd.Timestamp(as_of_date) - timedelta(days=200)).strftime('%Y-%m-%d')

        placeholders = ','.join(['?' for _ in tickers])
        pit_cutoff = (pd.Timestamp(as_of_date) - timedelta(days=self.filing_lag_days)).strftime('%Y-%m-%d')

        # MRT for R&D and SG&A (TTM flows); marketcap included in MRT dimension
        query = f"""
            SELECT ticker, datekey, rnd, sgna, marketcap
            FROM sharadar_sf1
            WHERE ticker IN ({placeholders})
            AND dimension = 'MRT'
            AND datekey BETWEEN ? AND ?
            AND datekey <= ?
            ORDER BY ticker, datekey DESC
        """
        params = tuple(tickers) + (start_date, as_of_date, pit_cutoff)
        df = self.execute_query(query, params)

        if len(df) == 0:
            logger.warning(f"No MRT data found for intangible yield calculation")
            return pd.Series(dtype=float)

        intangible_values = {}
        for ticker in tickers:
            ticker_data = df[df['ticker'] == ticker]
            if len(ticker_data) == 0:
                continue

            latest = ticker_data.iloc[0]  # Already sorted DESC

            # R&D: NaN means no reported R&D (typical for non-tech firms) → default 0
            rnd = latest['rnd'] if pd.notna(latest['rnd']) else 0
            # SG&A: selling, general & administrative expense
            sgna = latest['sgna'] if pd.notna(latest['sgna']) else 0
            mktcap = latest['marketcap']

            if pd.isna(mktcap) or mktcap <= 0:
                continue

            # Intangible investment = R&D + 30% of SG&A (advertising, training proxy)
            intangible_investment = rnd + (0.3 * sgna)
            intangible_values[ticker] = intangible_investment / mktcap

        logger.debug(f"Intangible yield: {len(intangible_values)}/{len(tickers)} tickers scored")
        return pd.Series(intangible_values)
