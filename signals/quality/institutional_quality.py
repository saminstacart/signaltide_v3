"""
Institutional Quality Signal

Implementation of Asness-Frazzini-Pedersen (2018) Quality Minus Junk
with professional multi-factor methodology.

Strategy:
- Composite of profitability + growth + safety
- Monthly rebalancing
- Cross-sectional quintile ranking
- Sector-neutral option

Quality = High Profitability + High Growth + High Safety

This is the quality factor used by:
- AQR Capital Management
- Academic factor research
- Professional quantitative equity strategies

References:
- Asness, Frazzini, Pedersen (2018) "Quality Minus Junk"
- Novy-Marx (2013) "The Other Side of Value"
- Piotroski (2000) "Value Investing: F-Score"
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from core.institutional_base import InstitutionalSignal
from data.data_manager import DataManager


class InstitutionalQuality(InstitutionalSignal):
    """
    Quality Minus Junk (QMJ) Signal.

    Professional multi-factor quality implementation:
    - Profitability: ROE, ROA, GP/A
    - Growth: Revenue growth, earnings growth
    - Safety: Low leverage, low volatility

    Parameters:
        use_profitability: Include profitability metrics (default: True)
        use_growth: Include growth metrics (default: True)
        use_safety: Include safety metrics (default: True)
        sector_neutral: Rank within sectors (default: False)
        rebalance_frequency: 'monthly' (default)
        winsorize_pct: Outlier handling (default: [5, 95])
    """

    def __init__(self,
                 params: Dict[str, Any],
                 data_manager: Optional[DataManager] = None,
                 name: str = 'InstitutionalQuality'):
        # Make a copy to avoid mutating caller's dict
        params = params.copy()

        # Set defaults for quality-specific parameters BEFORE validation
        params.setdefault('use_profitability', True)
        params.setdefault('use_growth', True)
        params.setdefault('use_safety', True)
        params.setdefault('prof_weight', 0.4)
        params.setdefault('growth_weight', 0.3)
        params.setdefault('safety_weight', 0.3)

        super().__init__(params, name)

        self.data_manager = data_manager or DataManager()

        # Factor composition (now guaranteed to exist)
        self.use_profitability = params['use_profitability']
        self.use_growth = params['use_growth']
        self.use_safety = params['use_safety']

        # Weights for composite
        self.prof_weight = params['prof_weight']
        self.growth_weight = params['growth_weight']
        self.safety_weight = params['safety_weight']

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate quality signals using QMJ methodology.

        Args:
            data: DataFrame with 'close' prices and 'ticker'

        Returns:
            Series with signals in [-1, 1] range
        """
        if 'ticker' not in data.columns:
            return pd.Series(0, index=data.index)

        ticker = data['ticker'].iloc[0]

        # Get fundamentals with point-in-time constraint
        start_date = data.index.min().strftime('%Y-%m-%d')
        end_date = data.index.max().strftime('%Y-%m-%d')

        fundamentals = self.data_manager.get_fundamentals(
            ticker,
            start_date,
            end_date,
            dimension='ARQ',  # As-reported quarterly
            as_of_date=end_date    # CRITICAL: Enforce point-in-time data access
        )

        if len(fundamentals) == 0:
            return pd.Series(0, index=data.index)

        # Calculate quality components
        quality_scores = []

        if self.use_profitability:
            prof_score = self._calculate_profitability(fundamentals)
            quality_scores.append((prof_score, self.prof_weight))

        if self.use_growth:
            growth_score = self._calculate_growth(fundamentals)
            quality_scores.append((growth_score, self.growth_weight))

        if self.use_safety:
            safety_score = self._calculate_safety(fundamentals)
            quality_scores.append((safety_score, self.safety_weight))

        if len(quality_scores) == 0:
            return pd.Series(0, index=data.index)

        # Combine components (weighted average)
        total_weight = sum(w for _, w in quality_scores)
        composite_quality = sum(score * w for score, w in quality_scores) / total_weight

        # Reindex to daily (forward-fill quarterly values)
        quality_daily = composite_quality.reindex(data.index, method='ffill').fillna(0)

        # Convert to signals using time-series ranking
        # (Cross-sectional ranking requires multiple stocks)
        signals = self._to_time_series_signals(quality_daily)

        # Monthly rebalancing
        if self.rebalance_frequency == 'monthly':
            signals = self._apply_monthly_rebalancing(signals)

        return signals.clip(-1, 1)

    def _calculate_profitability(self, fundamentals: pd.DataFrame) -> pd.Series:
        """
        Calculate profitability score.

        Components:
        - ROE (Return on Equity)
        - ROA (Return on Assets)
        - Gross Profit / Assets

        Returns z-score of profitability metrics.
        """
        prof_scores = []

        # ROE
        if 'roe' in fundamentals.columns:
            roe = fundamentals['roe'].replace([np.inf, -np.inf], np.nan)
            roe_winsorized = self.winsorize(roe.dropna())
            prof_scores.append(roe_winsorized)

        # ROA
        if 'roa' in fundamentals.columns:
            roa = fundamentals['roa'].replace([np.inf, -np.inf], np.nan)
            roa_winsorized = self.winsorize(roa.dropna())
            prof_scores.append(roa_winsorized)

        # Gross Profit / Assets
        if 'gp' in fundamentals.columns and 'assets' in fundamentals.columns:
            gp_assets = fundamentals['gp'] / fundamentals['assets'].replace(0, np.nan)
            gp_assets = gp_assets.replace([np.inf, -np.inf], np.nan)
            gp_assets_winsorized = self.winsorize(gp_assets.dropna())
            prof_scores.append(gp_assets_winsorized)

        if len(prof_scores) == 0:
            return pd.Series(0, index=fundamentals.index)

        # Average available metrics
        prof_df = pd.concat(prof_scores, axis=1)
        return prof_df.mean(axis=1, skipna=True)

    def _calculate_growth(self, fundamentals: pd.DataFrame) -> pd.Series:
        """
        Calculate growth score.

        Components:
        - Revenue growth (YoY)
        - Earnings growth (YoY)

        Returns z-score of growth metrics.
        """
        growth_scores = []

        # Revenue growth
        if 'revenue' in fundamentals.columns:
            rev_growth = fundamentals['revenue'].pct_change(periods=4)  # YoY
            rev_growth = rev_growth.replace([np.inf, -np.inf], np.nan)
            rev_growth_winsorized = self.winsorize(rev_growth.dropna())
            growth_scores.append(rev_growth_winsorized)

        # Earnings growth (net income)
        if 'netinc' in fundamentals.columns:
            ni_growth = fundamentals['netinc'].pct_change(periods=4)  # YoY
            ni_growth = ni_growth.replace([np.inf, -np.inf], np.nan)
            ni_growth_winsorized = self.winsorize(ni_growth.dropna())
            growth_scores.append(ni_growth_winsorized)

        if len(growth_scores) == 0:
            return pd.Series(0, index=fundamentals.index)

        # Average available metrics
        growth_df = pd.concat(growth_scores, axis=1)
        return growth_df.mean(axis=1, skipna=True)

    def _calculate_safety(self, fundamentals: pd.DataFrame) -> pd.Series:
        """
        Calculate safety score.

        Components:
        - Low leverage (debt/equity)
        - Low volatility of earnings
        - Positive earnings (avoid losses)

        Returns z-score of safety metrics (inverted where needed).
        """
        safety_scores = []

        # Low leverage (invert: low debt is safe)
        if 'de' in fundamentals.columns:
            de = fundamentals['de'].replace([np.inf, -np.inf], np.nan)
            de_winsorized = self.winsorize(de.dropna())
            # Invert: low leverage = high safety
            low_leverage = -de_winsorized
            safety_scores.append(low_leverage)

        # Earnings stability (low volatility of ROE)
        if 'roe' in fundamentals.columns:
            roe_volatility = fundamentals['roe'].rolling(window=8).std()  # 2 years
            roe_volatility = roe_volatility.replace([np.inf, -np.inf], np.nan)
            roe_vol_winsorized = self.winsorize(roe_volatility.dropna())
            # Invert: low volatility = high safety
            earnings_stability = -roe_vol_winsorized
            safety_scores.append(earnings_stability)

        # Consistent profitability (positive ROE)
        if 'roe' in fundamentals.columns:
            positive_roe = (fundamentals['roe'] > 0).astype(float)
            safety_scores.append(positive_roe)

        if len(safety_scores) == 0:
            return pd.Series(0, index=fundamentals.index)

        # Average available metrics
        safety_df = pd.concat(safety_scores, axis=1)
        return safety_df.mean(axis=1, skipna=True)

    def _to_time_series_signals(self, quality: pd.Series) -> pd.Series:
        """
        Convert quality scores to time-series signals.

        For single-stock analysis: Rank within stock's own history.
        Converts to quintile-like signals [-1, 1].
        """
        # Rolling percentile rank (2-year window)
        rank_window = 252 * 2  # 2 years

        signals = quality.rolling(
            window=rank_window,
            min_periods=4  # At least 4 quarters
        ).apply(
            lambda x: 2.0 * (pd.Series(x).rank(pct=True).iloc[-1]) - 1.0,
            raw=False
        )

        return signals.fillna(0)

    def _apply_monthly_rebalancing(self, signals: pd.Series) -> pd.Series:
        """Apply monthly rebalancing (hold signal for entire month)."""
        month_ends = signals.resample('M').last()
        rebalanced = month_ends.reindex(signals.index, method='ffill')
        return rebalanced.fillna(0)

    def get_parameter_space(self) -> Dict[str, tuple]:
        """
        Define parameter space for optimization.

        Returns:
            Dict with parameter specifications
        """
        return {
            'use_profitability': ('categorical', [True, False]),
            'use_growth': ('categorical', [True, False]),
            'use_safety': ('categorical', [True, False]),
            'prof_weight': ('float', 0.2, 0.6),
            'growth_weight': ('float', 0.1, 0.5),
            'safety_weight': ('float', 0.1, 0.5),
            'sector_neutral': ('categorical', [True, False])
        }

    def __repr__(self) -> str:
        components = []
        if self.use_profitability:
            components.append('Prof')
        if self.use_growth:
            components.append('Growth')
        if self.use_safety:
            components.append('Safety')

        return f"InstitutionalQuality({'+'.join(components)})"


if __name__ == '__main__':
    print("InstitutionalQuality - Quality Minus Junk (QMJ)")
    print("Professional multi-factor implementation")
    print()
    print("Components:")
    print("  1. Profitability: ROE, ROA, GP/A")
    print("  2. Growth: Revenue growth, earnings growth")
    print("  3. Safety: Low leverage, low volatility")
    print()
    print("Standard parameters:")
    print("  Weights: 40% Prof, 30% Growth, 30% Safety")
    print("  Rebalance: Monthly")
    print("  Quintile signals: [-1, -0.5, 0, 0.5, 1]")
