"""
Regime-Aware Allocators for Momentum + Quality Ensembles

Provides two allocator strategies:
1. OracleRegimeAllocatorMQ: Hindsight-based optimal weights per regime (research-only)
2. RuleBasedRegimeAllocatorMQ: PIT-safe rule-based weight allocation

Both allocators operate on monthly return series and output time-varying weights.

References:
    - Phase 3 M3.5 Spec: docs/ENSEMBLES_M3.5_REGIME_ALLOC_SPEC.md
    - Regime diagnostic: results/ensemble_baselines/momentum_quality_v1_regime_diagnostic.md
"""

from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass


# ============================================================================
# REGIME DEFINITIONS (Fixed boundaries from diagnostic)
# ============================================================================

REGIME_BOUNDARIES = {
    'pre_covid_expansion': ('2015-04-01', '2019-12-31'),
    'covid_crash':         ('2020-02-01', '2020-04-30'),
    'covid_recovery':      ('2020-05-01', '2021-12-31'),
    'bear_2022':           ('2022-01-01', '2022-12-31'),
    'recent':              ('2023-01-01', '2024-12-31'),
}


def get_regime_label(date: pd.Timestamp) -> str:
    """
    Map rebalance date to regime label using fixed boundaries.

    Args:
        date: Rebalance date (pd.Timestamp)

    Returns:
        Regime label string
    """
    if date < pd.Timestamp('2020-02-01'):
        return 'pre_covid_expansion'
    elif date < pd.Timestamp('2020-05-01'):
        return 'covid_crash'
    elif date < pd.Timestamp('2022-01-01'):
        return 'covid_recovery'
    elif date < pd.Timestamp('2023-01-01'):
        return 'bear_2022'
    else:
        return 'recent'


def assign_regime_labels(dates: pd.DatetimeIndex) -> pd.Series:
    """
    Assign regime labels to a series of dates.

    Args:
        dates: DatetimeIndex of rebalance dates

    Returns:
        pd.Series indexed by dates with regime label values
    """
    return pd.Series([get_regime_label(date) for date in dates], index=dates)


# ============================================================================
# ORACLE REGIME ALLOCATOR (Hindsight-Based, Research-Only)
# ============================================================================

@dataclass
class RegimeWeights:
    """Container for per-regime optimal weights."""
    momentum: float
    quality: float
    sharpe: float
    regime_name: str


class OracleRegimeAllocatorMQ:
    """
    Oracle regime allocator for Momentum + Quality v1.

    **RESEARCH-ONLY** - Uses ex-post regime labels (NOT PIT-safe).

    For each regime, performs grid search over (w_m, w_q) ∈ [0.1, 0.9] × [0.1, 0.9]
    to find weights that maximize in-regime Sharpe ratio.

    Provides true performance ceiling for regime-aware allocation.

    Args:
        momentum_returns: Monthly return series for momentum-only portfolio
        quality_returns: Monthly return series for quality-only portfolio (reconstructed)
        grid_step: Grid search step size (default: 0.1)

    Attributes:
        optimal_weights: Dict mapping regime_name -> RegimeWeights
        weight_series: Time series of (w_m, w_q) per rebalance date
        ensemble_returns: Reconstructed ensemble return series using optimal weights
    """

    def __init__(
        self,
        momentum_returns: pd.Series,
        quality_returns: pd.Series,
        grid_step: float = 0.1,
    ):
        """Initialize oracle allocator and compute optimal weights."""
        self.momentum_returns = momentum_returns
        self.quality_returns = quality_returns
        self.grid_step = grid_step

        # Validate inputs
        if not momentum_returns.index.equals(quality_returns.index):
            raise ValueError("momentum_returns and quality_returns must have matching indices")

        # Assign regime labels to dates
        self.regime_labels = assign_regime_labels(momentum_returns.index)

        # Compute optimal weights per regime
        self.optimal_weights = self._compute_optimal_weights()

        # Construct weight time series
        self.weight_series = self._build_weight_series()

        # Construct ensemble returns
        self.ensemble_returns = self._build_ensemble_returns()

    def _compute_optimal_weights(self) -> Dict[str, RegimeWeights]:
        """
        Compute optimal (w_m, w_q) per regime via grid search.

        Returns:
            Dict mapping regime_name -> RegimeWeights
        """
        optimal_weights = {}
        grid = np.arange(0.1, 1.0, self.grid_step)

        for regime_name in REGIME_BOUNDARIES.keys():
            # Get dates in this regime
            regime_mask = self.regime_labels == regime_name
            regime_dates = self.regime_labels[regime_mask].index

            if len(regime_dates) < 2:
                # Not enough data for Sharpe computation
                optimal_weights[regime_name] = RegimeWeights(
                    momentum=0.25,
                    quality=0.75,
                    sharpe=0.0,
                    regime_name=regime_name,
                )
                continue

            # Grid search
            best_sharpe = -np.inf
            best_w_m = 0.25
            best_w_q = 0.75

            for w_m in grid:
                w_q = 1.0 - w_m

                # Construct ensemble returns for this regime
                ensemble_rets = (
                    w_m * self.momentum_returns[regime_dates] +
                    w_q * self.quality_returns[regime_dates]
                )

                # Compute annualized Sharpe (monthly returns, 12 periods/year)
                mean_ret = ensemble_rets.mean()
                std_ret = ensemble_rets.std()

                if std_ret > 0:
                    sharpe = (mean_ret * 12) / (std_ret * np.sqrt(12))
                else:
                    sharpe = 0.0

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_w_m = w_m
                    best_w_q = w_q

            optimal_weights[regime_name] = RegimeWeights(
                momentum=best_w_m,
                quality=best_w_q,
                sharpe=best_sharpe,
                regime_name=regime_name,
            )

        return optimal_weights

    def _build_weight_series(self) -> pd.DataFrame:
        """
        Build time series of weights per rebalance date.

        Returns:
            DataFrame with columns ['momentum', 'quality', 'regime']
        """
        weights_list = []

        for date, regime_label in self.regime_labels.items():
            weights = self.optimal_weights[regime_label]
            weights_list.append({
                'date': date,
                'momentum': weights.momentum,
                'quality': weights.quality,
                'regime': regime_label,
            })

        return pd.DataFrame(weights_list).set_index('date')

    def _build_ensemble_returns(self) -> pd.Series:
        """
        Build ensemble return series using optimal weights.

        Returns:
            pd.Series of ensemble returns indexed by date
        """
        ensemble_rets = (
            self.weight_series['momentum'] * self.momentum_returns +
            self.weight_series['quality'] * self.quality_returns
        )
        return ensemble_rets

    def get_summary(self) -> pd.DataFrame:
        """
        Get summary of optimal weights per regime.

        Returns:
            DataFrame with regime, w_m, w_q, Sharpe
        """
        summary_list = []
        for regime_name, weights in self.optimal_weights.items():
            summary_list.append({
                'regime': regime_name,
                'w_momentum': weights.momentum,
                'w_quality': weights.quality,
                'sharpe': weights.sharpe,
            })
        return pd.DataFrame(summary_list)


# ============================================================================
# RULE-BASED REGIME ALLOCATOR (PIT-Safe, Practical)
# ============================================================================

class RuleBasedRegimeAllocatorMQ:
    """
    PIT-safe rule-based allocator for Momentum + Quality v1.

    Uses observable indicators at rebalance time to classify regime:
    - Realized 6M volatility (annualized)
    - Current drawdown from peak

    Maps regime -> weights:
    - CALM (low vol, small DD):   w_m=0.35, w_q=0.65
    - STRESS (high vol or deep DD): w_m=0.15, w_q=0.85
    - CHOPPY (otherwise):           w_m=0.25, w_q=0.75 (baseline)

    Args:
        equity_curve: Monthly equity curve for S&P 500 or portfolio (for indicators)
        vol_threshold_low: Volatility threshold for CALM (default: 0.15)
        vol_threshold_high: Volatility threshold for STRESS (default: 0.25)
        dd_threshold_calm: Drawdown threshold for CALM (default: -0.10)
        dd_threshold_stress: Drawdown threshold for STRESS (default: -0.15)

    Attributes:
        regime_series: Time series of regime classifications
        weight_series: Time series of (w_m, w_q) per rebalance date
    """

    # Regime weight presets
    REGIME_WEIGHTS = {
        'CALM':   {'momentum': 0.35, 'quality': 0.65},
        'STRESS': {'momentum': 0.15, 'quality': 0.85},
        'CHOPPY': {'momentum': 0.25, 'quality': 0.75},
    }

    def __init__(
        self,
        equity_curve: pd.Series,
        vol_threshold_low: float = 0.15,
        vol_threshold_high: float = 0.25,
        dd_threshold_calm: float = -0.10,
        dd_threshold_stress: float = -0.15,
    ):
        """Initialize rule-based allocator and classify regimes."""
        self.equity_curve = equity_curve
        self.vol_threshold_low = vol_threshold_low
        self.vol_threshold_high = vol_threshold_high
        self.dd_threshold_calm = dd_threshold_calm
        self.dd_threshold_stress = dd_threshold_stress

        # Compute PIT indicators
        self.indicators = self._compute_indicators()

        # Classify regimes
        self.regime_series = self._classify_regimes()

        # Build weight series
        self.weight_series = self._build_weight_series()

    def _compute_indicators(self) -> pd.DataFrame:
        """
        Compute PIT-safe regime indicators at each date.

        Returns:
            DataFrame with columns ['real_vol_6m', 'current_dd']
        """
        indicators_list = []

        # Compute returns
        returns = self.equity_curve.pct_change().dropna()

        for date in self.equity_curve.index:
            # Skip first year (need lookback data)
            if date < self.equity_curve.index[0] + pd.Timedelta(days=252):
                continue

            # 1. Realized 6M vol (annualized)
            # Use trailing 126 trading days ≈ 6 months
            date_idx = self.equity_curve.index.get_loc(date)
            lookback_start_idx = max(0, date_idx - 126)
            trailing_returns = returns.iloc[lookback_start_idx:date_idx + 1]

            if len(trailing_returns) > 1:
                real_vol_6m = trailing_returns.std() * np.sqrt(12)  # Annualize monthly vol
            else:
                real_vol_6m = 0.0

            # 2. Current drawdown from peak
            trailing_equity = self.equity_curve.iloc[:date_idx + 1]
            all_time_high = trailing_equity.max()
            current_price = trailing_equity.iloc[-1]
            current_dd = (current_price / all_time_high) - 1.0

            indicators_list.append({
                'date': date,
                'real_vol_6m': real_vol_6m,
                'current_dd': current_dd,
            })

        return pd.DataFrame(indicators_list).set_index('date')

    def _classify_regimes(self) -> pd.Series:
        """
        Classify regime at each date based on indicators.

        Returns:
            pd.Series indexed by date with regime labels
        """
        regime_list = []

        for date, row in self.indicators.iterrows():
            real_vol = row['real_vol_6m']
            current_dd = row['current_dd']

            # STRESS: High vol OR severe drawdown
            if real_vol > self.vol_threshold_high or current_dd < self.dd_threshold_stress:
                regime = 'STRESS'

            # CALM: Low vol AND small drawdown
            elif real_vol < self.vol_threshold_low and current_dd > self.dd_threshold_calm:
                regime = 'CALM'

            # CHOPPY: Everything else
            else:
                regime = 'CHOPPY'

            regime_list.append({'date': date, 'regime': regime})

        return pd.DataFrame(regime_list).set_index('date')['regime']

    def _build_weight_series(self) -> pd.DataFrame:
        """
        Build weight series from regime classifications.

        Returns:
            DataFrame with columns ['momentum', 'quality', 'regime']
        """
        weights_list = []

        for date, regime in self.regime_series.items():
            weights = self.REGIME_WEIGHTS[regime]
            weights_list.append({
                'date': date,
                'momentum': weights['momentum'],
                'quality': weights['quality'],
                'regime': regime,
            })

        return pd.DataFrame(weights_list).set_index('date')

    def apply_weights(
        self,
        momentum_returns: pd.Series,
        quality_returns: pd.Series,
    ) -> pd.Series:
        """
        Apply rule-based weights to construct ensemble returns.

        Args:
            momentum_returns: Monthly returns for momentum-only
            quality_returns: Monthly returns for quality-only

        Returns:
            pd.Series of ensemble returns
        """
        # Align indices (weight series may start later due to lookback)
        common_dates = self.weight_series.index.intersection(momentum_returns.index)

        ensemble_rets = (
            self.weight_series.loc[common_dates, 'momentum'] * momentum_returns.loc[common_dates] +
            self.weight_series.loc[common_dates, 'quality'] * quality_returns.loc[common_dates]
        )

        return ensemble_rets

    def get_summary(self) -> pd.DataFrame:
        """
        Get summary of regime distribution and average weights.

        Returns:
            DataFrame with regime counts and average weights
        """
        regime_counts = self.regime_series.value_counts()

        summary_list = []
        for regime in ['CALM', 'CHOPPY', 'STRESS']:
            count = regime_counts.get(regime, 0)
            weights = self.REGIME_WEIGHTS[regime]
            summary_list.append({
                'regime': regime,
                'count': count,
                'pct': count / len(self.regime_series) if len(self.regime_series) > 0 else 0.0,
                'w_momentum': weights['momentum'],
                'w_quality': weights['quality'],
            })

        return pd.DataFrame(summary_list)
