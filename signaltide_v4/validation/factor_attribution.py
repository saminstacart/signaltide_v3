"""
Fama-French 5-Factor Attribution Analysis.

Reference:
    Fama, E. F., & French, K. R. (2015).
    "A Five-Factor Asset Pricing Model".
    Journal of Financial Economics, 116(1), 1-22.

Key insight: Most "alpha" disappears when properly accounting for
standard factors (Market, Size, Value, Profitability, Investment).
Only genuine alpha survives FF5 regression.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np
from scipy import stats

from signaltide_v4.data.factor_data import FactorDataProvider
from signaltide_v4.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    """Container for factor attribution results."""

    # Alpha
    alpha: float  # Annualized alpha (intercept)
    alpha_t_stat: float
    alpha_p_value: float
    alpha_significant: bool

    # Factor loadings
    mkt_rf_beta: float
    smb_beta: float  # Size
    hml_beta: float  # Value
    rmw_beta: float  # Profitability
    cma_beta: float  # Investment

    # T-stats for factor loadings
    mkt_rf_t: float
    smb_t: float
    hml_t: float
    rmw_t: float
    cma_t: float

    # Model fit
    r_squared: float
    adj_r_squared: float
    n_observations: int

    # Diagnostics
    diagnostics: Dict[str, Any] = None


class FactorAttributor:
    """
    Performs Fama-French 5-factor attribution.

    Determines:
    1. Is there genuine alpha after factor adjustment?
    2. What factors explain the strategy's returns?
    3. Is the strategy just tilting on known factors?
    """

    def __init__(
        self,
        factor_data: Optional[FactorDataProvider] = None,
        significance_level: float = 0.05,
    ):
        """
        Initialize factor attributor.

        Args:
            factor_data: FactorDataProvider for FF5 data
            significance_level: Alpha threshold for significance testing
        """
        self.factor_data = factor_data or FactorDataProvider()
        self.significance_level = significance_level

        logger.info(f"FactorAttributor: significance={significance_level}")

    def attribute(
        self,
        returns: pd.Series,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> AttributionResult:
        """
        Perform FF5 factor attribution.

        Args:
            returns: Strategy daily returns (excess of risk-free)
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            AttributionResult with factor loadings and alpha
        """
        if len(returns) < 60:
            logger.warning(f"Insufficient data for attribution: {len(returns)}")
            return self._empty_result(len(returns))

        # Determine date range
        start_date = start_date or returns.index.min().strftime('%Y-%m-%d')
        end_date = end_date or returns.index.max().strftime('%Y-%m-%d')

        # Get FF5 factors
        factors = self.factor_data.get_ff5_factors(start_date, end_date)

        if len(factors) == 0:
            logger.warning("No factor data available")
            return self._empty_result(len(returns))

        # Align dates
        common_dates = returns.index.intersection(factors.index)
        if len(common_dates) < 60:
            logger.warning(f"Insufficient overlap: {len(common_dates)} dates")
            return self._empty_result(len(common_dates))

        y = returns.loc[common_dates]
        X = factors.loc[common_dates, ['MKT-RF', 'SMB', 'HML', 'RMW', 'CMA']]
        rf = factors.loc[common_dates, 'RF']

        # Convert to excess returns
        y_excess = y - rf

        # Run OLS regression
        result = self._ols_regression(y_excess, X)

        return result

    def _ols_regression(
        self,
        y: pd.Series,
        X: pd.DataFrame,
    ) -> AttributionResult:
        """Run OLS regression with proper statistics."""
        n = len(y)
        k = X.shape[1]  # Number of factors

        # Add constant for alpha
        X_with_const = np.column_stack([np.ones(n), X.values])

        try:
            # OLS: β = (X'X)^(-1) X'y
            XtX = X_with_const.T @ X_with_const
            XtX_inv = np.linalg.inv(XtX)
            Xty = X_with_const.T @ y.values
            betas = XtX_inv @ Xty

            # Residuals and statistics
            y_pred = X_with_const @ betas
            residuals = y.values - y_pred

            # R-squared
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y.values - y.mean()) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            # Adjusted R-squared
            adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)

            # Standard errors
            mse = ss_res / (n - k - 1)
            se = np.sqrt(np.diag(XtX_inv) * mse)

            # T-statistics
            t_stats = betas / se

            # P-values (two-tailed)
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n-k-1))

            # Extract results
            alpha = betas[0] * 252  # Annualize
            alpha_t = t_stats[0]
            alpha_p = p_values[0]

            result = AttributionResult(
                alpha=float(alpha),
                alpha_t_stat=float(alpha_t),
                alpha_p_value=float(alpha_p),
                alpha_significant=alpha_p < self.significance_level,

                mkt_rf_beta=float(betas[1]),
                smb_beta=float(betas[2]),
                hml_beta=float(betas[3]),
                rmw_beta=float(betas[4]),
                cma_beta=float(betas[5]),

                mkt_rf_t=float(t_stats[1]),
                smb_t=float(t_stats[2]),
                hml_t=float(t_stats[3]),
                rmw_t=float(t_stats[4]),
                cma_t=float(t_stats[5]),

                r_squared=float(r_squared),
                adj_r_squared=float(adj_r_squared),
                n_observations=n,

                diagnostics={
                    'factor_p_values': {
                        'MKT-RF': float(p_values[1]),
                        'SMB': float(p_values[2]),
                        'HML': float(p_values[3]),
                        'RMW': float(p_values[4]),
                        'CMA': float(p_values[5]),
                    },
                    'residual_std': float(np.sqrt(mse) * np.sqrt(252)),
                    'daily_alpha': float(betas[0]),
                },
            )

            logger.info(
                f"FF5 Attribution: alpha={alpha:.2%} (t={alpha_t:.2f}, p={alpha_p:.3f}), "
                f"R²={r_squared:.3f}"
            )

            return result

        except Exception as e:
            logger.error(f"OLS regression failed: {e}")
            return self._empty_result(n)

    def _empty_result(self, n: int) -> AttributionResult:
        """Return empty result for insufficient data."""
        return AttributionResult(
            alpha=0.0,
            alpha_t_stat=0.0,
            alpha_p_value=1.0,
            alpha_significant=False,

            mkt_rf_beta=1.0,
            smb_beta=0.0,
            hml_beta=0.0,
            rmw_beta=0.0,
            cma_beta=0.0,

            mkt_rf_t=0.0,
            smb_t=0.0,
            hml_t=0.0,
            rmw_t=0.0,
            cma_t=0.0,

            r_squared=0.0,
            adj_r_squared=0.0,
            n_observations=n,

            diagnostics={'error': 'insufficient_data'},
        )

    def interpret(self, result: AttributionResult) -> Dict[str, str]:
        """
        Provide human-readable interpretation of attribution results.
        """
        interpretations = {}

        # Alpha interpretation
        if result.alpha_significant:
            if result.alpha > 0:
                interpretations['alpha'] = (
                    f"POSITIVE ALPHA: {result.alpha:.1%} annualized (p={result.alpha_p_value:.3f}). "
                    "Strategy generates genuine excess returns after FF5 adjustment."
                )
            else:
                interpretations['alpha'] = (
                    f"NEGATIVE ALPHA: {result.alpha:.1%} annualized (p={result.alpha_p_value:.3f}). "
                    "Strategy underperforms after FF5 adjustment."
                )
        else:
            interpretations['alpha'] = (
                f"NO SIGNIFICANT ALPHA: {result.alpha:.1%} (p={result.alpha_p_value:.3f}). "
                "Returns are explained by common factors."
            )

        # Factor tilt interpretation
        tilts = []
        if abs(result.smb_beta) > 0.2:
            tilt = "small-cap" if result.smb_beta > 0 else "large-cap"
            tilts.append(f"{tilt} (SMB={result.smb_beta:.2f})")

        if abs(result.hml_beta) > 0.2:
            tilt = "value" if result.hml_beta > 0 else "growth"
            tilts.append(f"{tilt} (HML={result.hml_beta:.2f})")

        if abs(result.rmw_beta) > 0.2:
            tilt = "profitable" if result.rmw_beta > 0 else "unprofitable"
            tilts.append(f"{tilt} (RMW={result.rmw_beta:.2f})")

        if abs(result.cma_beta) > 0.2:
            tilt = "conservative" if result.cma_beta > 0 else "aggressive"
            tilts.append(f"{tilt} (CMA={result.cma_beta:.2f})")

        if tilts:
            interpretations['tilts'] = f"Factor tilts: {', '.join(tilts)}"
        else:
            interpretations['tilts'] = "No significant factor tilts detected."

        # Model fit
        if result.r_squared > 0.8:
            interpretations['fit'] = (
                f"High factor exposure (R²={result.r_squared:.1%}). "
                "Most returns explained by common factors."
            )
        elif result.r_squared > 0.5:
            interpretations['fit'] = (
                f"Moderate factor exposure (R²={result.r_squared:.1%}). "
                "Mix of factor and idiosyncratic returns."
            )
        else:
            interpretations['fit'] = (
                f"Low factor exposure (R²={result.r_squared:.1%}). "
                "Returns mostly idiosyncratic."
            )

        return interpretations
