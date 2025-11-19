"""
Quality Signal - Fundamental quality metrics from Sharadar data.

Methodology:
- Profitability metrics (ROE, ROA, gross margin, operating margin)
- Earnings quality (accruals ratio, cash flow quality)
- Balance sheet strength (debt/equity, current ratio, quick ratio)
- Growth metrics (revenue growth, earnings growth)
- Combines multiple quality factors into composite score

Economic Rationale:
High-quality companies tend to outperform due to:
1. Sustainable competitive advantages
2. Better management quality
3. Lower risk of financial distress
4. Ability to compound earnings
5. Market under-reaction to quality

This signal uses Sharadar fundamental data with proper point-in-time
access via DataManager to prevent lookahead bias.

CRITICAL: Uses filing_date not report_period to ensure we only use
information that was actually available at the time.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from core.base_signal import BaseSignal
from data.data_manager import DataManager


class QualitySignal(BaseSignal):
    """
    Fundamental quality signal using Sharadar data.

    Combines multiple quality metrics:
    - Profitability: ROE, ROA, margins
    - Earnings quality: Accruals, cash flow
    - Balance sheet: Leverage, liquidity
    - Growth: Revenue, earnings trends

    Requires DataManager to access fundamental data with point-in-time constraints.
    """

    def __init__(self, params: Dict[str, Any],
                 data_manager: Optional[DataManager] = None,
                 name: str = 'QualitySignal'):
        """
        Initialize quality signal.

        Args:
            params: Signal parameters (see get_parameter_space)
            data_manager: DataManager instance for fundamental data access
            name: Signal name
        """
        super().__init__(params, name)
        self.data_manager = data_manager or DataManager()

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate quality signals from fundamental data.

        Args:
            data: DataFrame with price data and index as dates
                  Must have 'ticker' in columns or be single-ticker data

        Returns:
            Series with signals in [-1, 1], same index as data
        """
        # Get ticker from data
        if 'ticker' in data.columns:
            ticker = data['ticker'].iloc[0]
        else:
            # Assume single ticker - ticker must be passed somehow
            # For now, return zeros if ticker not available
            return pd.Series(0, index=data.index)

        # Get fundamental data aligned with price dates
        # Use filing_date for point-in-time accuracy
        fundamentals = self._get_fundamentals_for_dates(
            ticker=ticker,
            dates=data.index
        )

        if fundamentals is None or len(fundamentals) == 0:
            # No fundamental data available
            return pd.Series(0, index=data.index)

        # Calculate quality scores
        quality_scores = self._calculate_quality_scores(fundamentals)

        # Align quality scores with price data dates
        # Forward-fill fundamental data (quarterly updates)
        quality_series = pd.Series(quality_scores, index=fundamentals.index)
        quality_aligned = quality_series.reindex(data.index, method='ffill')

        # Normalize to [-1, 1] range using rolling rank
        rank_window = self.params.get('rank_window', 252)
        signals = quality_aligned.rolling(window=rank_window, min_periods=20).apply(
            lambda x: 2.0 * (pd.Series(x).rank().iloc[-1] / len(x)) - 1.0,
            raw=False
        )

        # Apply threshold
        threshold = self.params.get('signal_threshold', 0.0)
        if threshold > 0:
            signals = signals.where(signals.abs() > threshold, 0)

        # Fill NaN with 0
        signals = signals.fillna(0)

        # Clip to [-1, 1]
        signals = signals.clip(-1, 1)

        return signals

    def _get_fundamentals_for_dates(self, ticker: str,
                                    dates: pd.DatetimeIndex) -> Optional[pd.DataFrame]:
        """
        Get fundamental data for ticker covering the date range.

        Uses point-in-time retrieval based on filing_date.

        Args:
            ticker: Stock ticker
            dates: Dates to cover

        Returns:
            DataFrame with fundamental data or None
        """
        if len(dates) == 0:
            return None

        start_date = dates.min()
        end_date = dates.max()

        # Get fundamental data
        # as_of parameter ensures point-in-time data
        fundamentals = self.data_manager.get_fundamental_data(
            ticker=ticker,
            start_date=start_date - pd.Timedelta(days=365),  # Get 1 year prior for metrics
            end_date=end_date,
            as_of=end_date,  # Only data filed by end date
            dimension='ARQ'  # Annual/Quarterly reports
        )

        return fundamentals if len(fundamentals) > 0 else None

    def _calculate_quality_scores(self, fundamentals: pd.DataFrame) -> pd.Series:
        """
        Calculate composite quality score from fundamental metrics.

        Args:
            fundamentals: DataFrame with Sharadar fundamental data

        Returns:
            Series of quality scores (higher = better quality)
        """
        scores = pd.Series(0.0, index=fundamentals.index)

        # Profitability metrics
        if self.params.get('use_roe', True):
            roe_score = self._score_roe(fundamentals)
            scores += self.params.get('roe_weight', 0.2) * roe_score

        if self.params.get('use_roa', True):
            roa_score = self._score_roa(fundamentals)
            scores += self.params.get('roa_weight', 0.15) * roa_score

        if self.params.get('use_margins', True):
            margin_score = self._score_margins(fundamentals)
            scores += self.params.get('margin_weight', 0.2) * margin_score

        # Earnings quality (accruals)
        if self.params.get('use_accruals', True):
            accrual_score = self._score_accruals(fundamentals)
            scores += self.params.get('accrual_weight', 0.15) * accrual_score

        # Balance sheet strength
        if self.params.get('use_leverage', True):
            leverage_score = self._score_leverage(fundamentals)
            scores += self.params.get('leverage_weight', 0.15) * leverage_score

        if self.params.get('use_liquidity', True):
            liquidity_score = self._score_liquidity(fundamentals)
            scores += self.params.get('liquidity_weight', 0.15) * liquidity_score

        return scores

    def _score_roe(self, df: pd.DataFrame) -> pd.Series:
        """Score based on Return on Equity."""
        if 'roe' not in df.columns:
            return pd.Series(0, index=df.index)

        roe = df['roe'].fillna(0)
        # Higher ROE is better, but cap at reasonable levels
        roe = roe.clip(-0.5, 1.0)
        # Normalize to [0, 1]
        return (roe + 0.5) / 1.5

    def _score_roa(self, df: pd.DataFrame) -> pd.Series:
        """Score based on Return on Assets."""
        if 'roa' not in df.columns:
            return pd.Series(0, index=df.index)

        roa = df['roa'].fillna(0)
        roa = roa.clip(-0.3, 0.5)
        return (roa + 0.3) / 0.8

    def _score_margins(self, df: pd.DataFrame) -> pd.Series:
        """Score based on profit margins."""
        scores = pd.Series(0, index=df.index)

        # Gross margin
        if 'revenue' in df.columns and 'gross_profit' in df.columns:
            revenue = df['revenue'].replace(0, np.nan)
            gross_margin = (df['gross_profit'] / revenue).fillna(0)
            gross_margin = gross_margin.clip(0, 1)
            scores += 0.5 * gross_margin

        # Operating margin
        if 'revenue' in df.columns and 'operating_income' in df.columns:
            revenue = df['revenue'].replace(0, np.nan)
            operating_margin = (df['operating_income'] / revenue).fillna(0)
            operating_margin = operating_margin.clip(-0.5, 0.5)
            scores += 0.5 * (operating_margin + 0.5)

        return scores

    def _score_accruals(self, df: pd.DataFrame) -> pd.Series:
        """
        Score based on accruals (earnings quality).

        Lower accruals = higher quality (more cash-based earnings).
        """
        scores = pd.Series(0.5, index=df.index)  # Neutral default

        if ('net_income' in df.columns and
            'operating_cash_flow' in df.columns and
            'total_assets' in df.columns):

            net_income = df['net_income'].fillna(0)
            ocf = df['operating_cash_flow'].fillna(0)
            assets = df['total_assets'].replace(0, np.nan)

            # Accruals = (Net Income - Operating Cash Flow) / Total Assets
            accruals = ((net_income - ocf) / assets).fillna(0)

            # Lower accruals is better (more cash-based earnings)
            # Normalize: accruals typically in range [-0.2, 0.2]
            accruals = accruals.clip(-0.3, 0.3)
            # Invert so lower accruals = higher score
            scores = 1.0 - ((accruals + 0.3) / 0.6)

        return scores

    def _score_leverage(self, df: pd.DataFrame) -> pd.Series:
        """
        Score based on financial leverage.

        Lower debt/equity is generally better (less risky).
        """
        scores = pd.Series(0.5, index=df.index)

        if 'debt_to_equity' in df.columns:
            de_ratio = df['debt_to_equity'].fillna(1.0)
            # Lower is better, clip at reasonable range
            de_ratio = de_ratio.clip(0, 3.0)
            # Normalize: 0 debt = 1.0 score, 3.0 debt = 0.0 score
            scores = 1.0 - (de_ratio / 3.0)

        return scores

    def _score_liquidity(self, df: pd.DataFrame) -> pd.Series:
        """Score based on liquidity ratios."""
        scores = pd.Series(0.5, index=df.index)

        # Current ratio
        if 'current_ratio' in df.columns:
            current_ratio = df['current_ratio'].fillna(1.0)
            # Optimal range around 1.5-2.0
            current_ratio = current_ratio.clip(0, 4.0)
            # Score peaks at 2.0
            cr_score = 1.0 - abs(current_ratio - 2.0) / 2.0
            scores += 0.5 * cr_score

        # Quick ratio
        if 'quick_ratio' in df.columns:
            quick_ratio = df['quick_ratio'].fillna(0.8)
            quick_ratio = quick_ratio.clip(0, 3.0)
            # Score peaks at 1.0
            qr_score = 1.0 - abs(quick_ratio - 1.0) / 2.0
            scores += 0.5 * qr_score

        return scores / 1.5  # Normalize since we may have added 1.0

    def get_parameter_space(self) -> Dict[str, tuple]:
        """
        Define Optuna search space for quality signal.

        Returns:
            Dict of parameter specifications
        """
        return {
            # Which quality factors to use
            'use_roe': ('categorical', [True, False]),
            'use_roa': ('categorical', [True, False]),
            'use_margins': ('categorical', [True, False]),
            'use_accruals': ('categorical', [True, False]),
            'use_leverage': ('categorical', [True, False]),
            'use_liquidity': ('categorical', [True, False]),

            # Factor weights
            'roe_weight': ('float', 0.0, 0.5),
            'roa_weight': ('float', 0.0, 0.5),
            'margin_weight': ('float', 0.0, 0.5),
            'accrual_weight': ('float', 0.0, 0.5),
            'leverage_weight': ('float', 0.0, 0.5),
            'liquidity_weight': ('float', 0.0, 0.5),

            # Signal processing
            'signal_threshold': ('float', 0.0, 0.3),
            'rank_window': ('int', 60, 252),
        }

    def __repr__(self) -> str:
        """String representation."""
        active_factors = [k.replace('use_', '') for k, v in self.params.items()
                         if k.startswith('use_') and v]
        return f"QualitySignal(factors={','.join(active_factors)})"
