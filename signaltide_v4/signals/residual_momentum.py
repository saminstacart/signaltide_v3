"""
Residual Momentum Signal based on Blitz (2011) with volatility scaling.

References:
    Blitz, D., Huij, J., & Martens, M. (2011).
    "Residual Momentum". Journal of Empirical Finance, 18(3), 506-521.

    Barroso, P., & Santa-Clara, P. (2015).
    "Momentum Has Its Moments". Journal of Financial Economics, 116(1), 111-120.

    Da, Z., Gurun, U. G., & Warachka, M. (2014).
    "Frog in the Pan: Continuous Information and Momentum".
    Review of Financial Studies, 27(7), 2171-2218.

Key insights:
    - Momentum computed on FF3-adjusted returns is more stable (Blitz)
    - Volatility scaling reduces crash risk via inverse-vol weighting (Barroso)
    - Momentum consistency: smooth trends more persistent than jumps (Da et al.)
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import timedelta

import pandas as pd
import numpy as np

from .base import BaseSignal
from signaltide_v4.data.market_data import MarketDataProvider
from signaltide_v4.data.factor_data import FactorDataProvider
from signaltide_v4.config.settings import get_settings

logger = logging.getLogger(__name__)


class ResidualMomentumSignal(BaseSignal):
    """
    Residual Momentum signal using FF3-adjusted returns.

    Process:
    1. Get 12-month returns (skipping most recent month)
    2. Regress on FF3 factors to get residual returns
    3. Cumulate residual returns for momentum score

    This approach reduces crash risk compared to raw momentum.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        market_data: Optional[MarketDataProvider] = None,
        factor_data: Optional[FactorDataProvider] = None,
    ):
        """
        Initialize residual momentum signal.

        Args:
            params: Optional parameter overrides
            market_data: MarketDataProvider instance
            factor_data: FactorDataProvider instance

        Volatility scaling (Barroso & Santa-Clara 2015):
            When vol_scale=True, momentum scores are scaled by inverse
            realized volatility to reduce crash risk during high-vol periods.
        """
        super().__init__(name='residual_momentum', params=params)

        # Ensure params is a dict for .get() calls
        params = params or {}

        settings = get_settings()
        self.lookback_days = params.get('lookback_days', settings.momentum_lookback_days)
        self.skip_days = params.get('skip_days', settings.momentum_skip_days)

        # Volatility scaling parameters (Barroso & Santa-Clara 2015)
        self.vol_scale = params.get('vol_scale', True)  # Enable by default
        self.vol_lookback_days = params.get('vol_lookback_days', 126)  # 6 months
        self.target_vol = params.get('target_vol', 0.12)  # 12% annualized target

        # Momentum consistency parameters (Da, Gurun & Warachka 2014)
        # "Frog-in-the-Pan": smooth trends more persistent than jumps
        self.consistency_weight = params.get('consistency_weight', 0.30)  # 30% weight on consistency
        self.consistency_lookback_days = params.get('consistency_lookback_days', 252)  # 1 year of daily data

        self.market_data = market_data or MarketDataProvider()
        self.factor_data = factor_data or FactorDataProvider()

    def compute_raw_scores(
        self,
        tickers: List[str],
        as_of_date: str
    ) -> pd.Series:
        """
        Compute residual momentum scores.

        Steps:
        1. Get daily returns for lookback period
        2. Regress each stock on FF3 to get residuals
        3. Cumulate residuals (skipping recent period)
        """
        if not tickers:
            return pd.Series(dtype=float)

        # Calculate date range
        as_of = pd.Timestamp(as_of_date)
        skip_start = as_of - timedelta(days=self.skip_days)
        lookback_start = as_of - timedelta(days=self.lookback_days + 30)

        # Get daily returns
        returns = self.market_data.get_returns(
            tickers,
            lookback_start.strftime('%Y-%m-%d'),
            as_of_date,
            period='daily'
        )

        if len(returns) == 0:
            logger.warning(f"No returns data for residual momentum")
            return pd.Series(np.nan, index=tickers)

        # Get residual returns (FF3-adjusted)
        residual_returns = self.factor_data.get_residual_returns(
            returns,
            lookback_start.strftime('%Y-%m-%d'),
            as_of_date
        )

        # Skip most recent period (momentum reversal)
        if skip_start.strftime('%Y-%m-%d') in residual_returns.index:
            residual_returns = residual_returns[
                residual_returns.index < skip_start.strftime('%Y-%m-%d')
            ]

        # Cumulate residual returns
        if len(residual_returns) < 60:  # Minimum ~3 months
            logger.warning(f"Insufficient data for residual momentum: {len(residual_returns)} days")
            return pd.Series(np.nan, index=tickers)

        # Calculate cumulative residual return
        cum_residual = (1 + residual_returns).prod() - 1

        # Apply volatility scaling (Barroso & Santa-Clara 2015)
        if self.vol_scale:
            cum_residual = self._apply_vol_scaling(cum_residual, returns)

        # Apply consistency weighting (Da, Gurun & Warachka 2014)
        # "Frog-in-the-Pan": smooth trends more persistent than jumps
        if self.consistency_weight > 0:
            consistency = self._compute_consistency_score(
                returns, lookback_days=self.consistency_lookback_days
            )

            if len(consistency) > 0:
                # Normalize consistency cross-sectionally (rank-based)
                consistency_normalized = self.normalize_cross_sectional(
                    consistency, method='rank'
                )

                # Blend: (1-w) * raw_momentum + w * (raw_momentum * consistency)
                # High consistency amplifies momentum signal; low consistency dampens it
                final_score = (
                    (1 - self.consistency_weight) * cum_residual +
                    self.consistency_weight * (cum_residual * consistency_normalized)
                )
                cum_residual = final_score

        # Ensure all tickers are in result
        result = pd.Series(np.nan, index=tickers)
        for ticker in cum_residual.index:
            if ticker in result.index:
                result[ticker] = cum_residual[ticker]

        logger.debug(f"Residual momentum: {len(result.dropna())}/{len(tickers)} tickers scored")

        return result

    def _apply_vol_scaling(
        self,
        momentum_scores: pd.Series,
        returns: pd.DataFrame
    ) -> pd.Series:
        """
        Apply volatility scaling per Barroso & Santa-Clara (2015).

        Scales momentum by target_vol / realized_vol to reduce crash risk
        during high-volatility periods.

        Args:
            momentum_scores: Raw momentum scores
            returns: Daily returns DataFrame

        Returns:
            Volatility-scaled momentum scores
        """
        # Use trailing N days for realized volatility
        vol_window = min(self.vol_lookback_days, len(returns))
        if vol_window < 21:  # Minimum 1 month
            logger.warning(f"Insufficient data for vol scaling: {vol_window} days")
            return momentum_scores

        recent_returns = returns.tail(vol_window)

        # Compute realized volatility (annualized)
        realized_vol = recent_returns.std() * np.sqrt(252)

        # Compute scaling factor: target_vol / realized_vol
        # Cap at 2x to prevent extreme leverage
        vol_scale_factor = (self.target_vol / realized_vol).clip(upper=2.0)

        # Apply scaling
        scaled_scores = momentum_scores * vol_scale_factor

        # Log volatility stats
        avg_vol = realized_vol.mean()
        avg_scale = vol_scale_factor.mean()
        logger.debug(
            f"Vol scaling: avg realized vol={avg_vol:.2%}, "
            f"avg scale factor={avg_scale:.2f}"
        )

        return scaled_scores

    def _compute_consistency_score(
        self,
        returns: pd.DataFrame,
        lookback_days: int = 252
    ) -> pd.Series:
        """
        Calculate momentum consistency per Da, Gurun & Warachka (2014).

        "Frog-in-the-Pan": Smooth, continuous price increases are more
        persistent than large discrete jumps. Uses DAILY granularity
        for statistical robustness (N=252 vs N=12 for monthly).

        Args:
            returns: Daily returns DataFrame
            lookback_days: Number of trading days (default 252 = 1 year)

        Returns:
            Series of consistency scores (0-1, higher = smoother trend)
        """
        # Use last N trading days (DO NOT resample to monthly!)
        recent_returns = returns.tail(lookback_days)

        if len(recent_returns) < 60:  # Minimum ~3 months of daily data
            logger.warning(f"Insufficient data for consistency: {len(recent_returns)} days")
            return pd.Series(dtype=float)

        # Count positive DAYS / total DAYS (DAILY granularity, N=252)
        positive_days = (recent_returns > 0).sum()
        total_days = recent_returns.count()

        # Avoid division by zero
        consistency = positive_days / total_days.replace(0, np.nan)

        logger.debug(
            f"Consistency: mean={consistency.mean():.3f}, "
            f"std={consistency.std():.3f}, N={len(consistency.dropna())}"
        )

        return consistency

    def get_diagnostics(
        self,
        raw_scores: pd.Series,
        normalized_scores: pd.Series
    ) -> Dict[str, Any]:
        """Add residual momentum specific diagnostics."""
        base_diag = super().get_diagnostics(raw_scores, normalized_scores)

        valid = raw_scores.dropna()
        if len(valid) == 0:
            return base_diag

        # Add momentum-specific stats
        base_diag.update({
            'pct_positive_momentum': float((valid > 0).mean()),
            'pct_strong_momentum': float((valid > 0.20).mean()),  # >20% return
            'pct_weak_momentum': float((valid < -0.10).mean()),  # <-10% return
            'vol_scale_enabled': self.vol_scale,
            'consistency_weight': self.consistency_weight,
            'consistency_lookback_days': self.consistency_lookback_days,
        })

        return base_diag
