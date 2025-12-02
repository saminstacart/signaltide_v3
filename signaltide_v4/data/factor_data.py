"""
Fama-French factor data provider.

Provides FF3 and FF5 factors for:
- Residual momentum calculation
- Factor attribution analysis
"""

import logging
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

from .base import PITDataManager, DataCache
from signaltide_v4.config.settings import get_settings

logger = logging.getLogger(__name__)


# Fama-French factor data (cached from Kenneth French's website)
# In production, this would be fetched from the database or API
FF_FACTORS = ['MKT-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']


class FactorDataProvider(PITDataManager):
    """
    Provider for Fama-French factor data.

    Provides:
    - FF3: MKT-RF, SMB, HML
    - FF5: MKT-RF, SMB, HML, RMW, CMA
    - Risk-free rate (RF)
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize factor data provider."""
        super().__init__(db_path)
        self._cache = DataCache(maxsize=64)
        self._factors_df: Optional[pd.DataFrame] = None

    def get_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        as_of_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Get factor data (tickers parameter ignored for factors)."""
        return self.get_ff5_factors(start_date, end_date)

    def get_ff3_factors(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Get Fama-French 3 factors.

        Returns:
            DataFrame with columns: MKT-RF, SMB, HML, RF
            Index: date
        """
        factors = self.get_ff5_factors(start_date, end_date)
        return factors[['MKT-RF', 'SMB', 'HML', 'RF']]

    def get_ff5_factors(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Get Fama-French 5 factors.

        Returns:
            DataFrame with columns: MKT-RF, SMB, HML, RMW, CMA, RF
            Index: date

        Note: In production, this data comes from the database.
        For now, we synthesize from market data as a proxy.
        """
        cache_key = f"ff5_{start_date}_{end_date}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        # Load from database - FAIL if not available (no synthetic fallback)
        factors = self._load_ff_from_db(start_date, end_date)
        if factors is None or len(factors) == 0:
            raise ValueError(
                "FF factors not found in database. "
                "Run: python signaltide_v4/data/ff_data_loader.py to populate."
            )

        # Validate factors are real (not synthetic) - SMB should have real variance
        smb_abs_mean = factors['SMB'].abs().mean()
        if smb_abs_mean < 0.001:
            raise ValueError(
                f"FF factors appear synthetic (SMB abs mean={smb_abs_mean:.6f}). "
                "Run: python signaltide_v4/data/ff_data_loader.py to get real data."
            )

        self._cache.set(cache_key, factors)
        return factors

    def _load_ff_from_db(
        self,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """Try to load FF factors from database."""
        # Check if FF factor table exists
        check_query = """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='ff_factors'
        """
        result = self.execute_query(check_query)

        if len(result) == 0:
            return None

        query = """
            SELECT date, mkt_rf, smb, hml, rmw, cma, rf
            FROM ff_factors
            WHERE date BETWEEN ? AND ?
            ORDER BY date
        """
        df = self.execute_query(query, (start_date, end_date))

        if len(df) == 0:
            return None

        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df.columns = ['MKT-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']

        return df

    def _synthesize_factors(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Synthesize FF-like factors from market data.

        This is a simplified proxy. In production, use actual FF data.
        """
        # Get SPY as market proxy
        query = """
            SELECT date, closeadj
            FROM sharadar_prices
            WHERE ticker = 'SPY'
            AND date BETWEEN ? AND ?
            ORDER BY date
        """
        spy = self.execute_query(query, (start_date, end_date))

        if len(spy) == 0:
            # Return empty DataFrame with correct structure
            return pd.DataFrame(columns=['MKT-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF'])

        spy['date'] = pd.to_datetime(spy['date'])
        spy = spy.set_index('date')

        # Calculate daily market returns
        mkt_rf = spy['closeadj'].pct_change()

        # Create DataFrame with synthetic factors
        # These are approximations - real analysis should use actual FF data
        np.random.seed(42)  # For reproducibility
        n = len(mkt_rf)

        factors = pd.DataFrame({
            'MKT-RF': mkt_rf.values,
            # SMB: historically ~2% annual, ~0.008% daily
            'SMB': np.random.normal(0.00008, 0.004, n),
            # HML: historically ~4% annual, ~0.016% daily
            'HML': np.random.normal(0.00016, 0.005, n),
            # RMW: historically ~3% annual
            'RMW': np.random.normal(0.00012, 0.003, n),
            # CMA: historically ~2% annual
            'CMA': np.random.normal(0.00008, 0.003, n),
            # RF: ~2% annual, ~0.008% daily
            'RF': np.full(n, 0.00008),
        }, index=mkt_rf.index)

        return factors.dropna()

    def get_risk_free_rate(
        self,
        start_date: str,
        end_date: str
    ) -> pd.Series:
        """Get risk-free rate series."""
        factors = self.get_ff5_factors(start_date, end_date)
        return factors['RF']

    def regress_on_ff3(
        self,
        returns: pd.Series,
        start_date: str,
        end_date: str
    ) -> Dict[str, float]:
        """
        Regress returns on FF3 factors.

        Returns:
            Dict with alpha, MKT-RF beta, SMB beta, HML beta, R-squared
        """
        from scipy import stats

        factors = self.get_ff3_factors(start_date, end_date)

        # Align dates
        common = returns.index.intersection(factors.index)
        if len(common) < 60:
            logger.warning(f"Insufficient data for FF3 regression: {len(common)} obs")
            return {'alpha': 0.0, 'MKT-RF': 1.0, 'SMB': 0.0, 'HML': 0.0, 'r_squared': 0.0}

        y = returns.loc[common] - factors.loc[common, 'RF']
        X = factors.loc[common, ['MKT-RF', 'SMB', 'HML']]

        # Add constant for alpha
        X_with_const = np.column_stack([np.ones(len(X)), X.values])

        # OLS regression
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(X_with_const, y.values, rcond=None)
            y_pred = X_with_const @ coeffs
            ss_res = np.sum((y.values - y_pred) ** 2)
            ss_tot = np.sum((y.values - y.mean()) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            return {
                'alpha': float(coeffs[0]) * 252,  # Annualized
                'MKT-RF': float(coeffs[1]),
                'SMB': float(coeffs[2]),
                'HML': float(coeffs[3]),
                'r_squared': float(r_squared),
            }
        except Exception as e:
            logger.error(f"FF3 regression failed: {e}")
            return {'alpha': 0.0, 'MKT-RF': 1.0, 'SMB': 0.0, 'HML': 0.0, 'r_squared': 0.0}

    def get_residual_returns(
        self,
        returns: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Get FF3-adjusted residual returns for momentum calculation.

        This implements the residual momentum approach from Blitz (2011).
        """
        factors = self.get_ff3_factors(start_date, end_date)

        # Align dates
        common = returns.index.intersection(factors.index)
        if len(common) < 60:
            logger.warning("Insufficient data for residual returns, returning raw returns")
            return returns

        returns_aligned = returns.loc[common]
        factors_aligned = factors.loc[common]

        residuals = pd.DataFrame(index=common, columns=returns.columns)

        for ticker in returns.columns:
            y = returns_aligned[ticker].dropna()
            if len(y) < 60:
                residuals[ticker] = returns_aligned[ticker]
                continue

            y_common = y.index.intersection(factors_aligned.index)
            y = y.loc[y_common] - factors_aligned.loc[y_common, 'RF']
            X = factors_aligned.loc[y_common, ['MKT-RF', 'SMB', 'HML']]

            # OLS regression
            try:
                X_with_const = np.column_stack([np.ones(len(X)), X.values])
                coeffs, _, _, _ = np.linalg.lstsq(X_with_const, y.values, rcond=None)
                y_pred = X_with_const @ coeffs
                residuals.loc[y_common, ticker] = y.values - y_pred
            except Exception:
                residuals[ticker] = returns_aligned[ticker]

        return residuals.astype(float)
