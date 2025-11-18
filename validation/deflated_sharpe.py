"""
Deflated Sharpe Ratio calculation.

Based on Bailey & López de Prado (2014).
Accounts for multiple testing and selection bias.
"""

import numpy as np
from scipy import stats
from typing import Dict


class DeflatedSharpe:
    """
    Calculate Deflated Sharpe Ratio to account for multiple testing.

    When testing multiple strategies, the best Sharpe ratio is inflated
    due to selection bias. Deflated Sharpe corrects for this.
    """

    @staticmethod
    def calculate(observed_sharpe: float, n_trials: int, n_observations: int,
                  skew: float = 0.0, kurt: float = 3.0) -> Dict:
        """
        Calculate Deflated Sharpe Ratio.

        Args:
            observed_sharpe: Observed Sharpe ratio from best strategy
            n_trials: Number of strategies tested
            n_observations: Number of observations in backtest
            skew: Skewness of returns
            kurt: Excess kurtosis of returns (0 for normal)

        Returns:
            Dict with deflated Sharpe and p-value
        """
        # Expected maximum Sharpe from N random trials
        expected_max_sharpe = DeflatedSharpe._expected_max_sharpe(n_trials)

        # Standard deviation of maximum Sharpe
        std_max_sharpe = DeflatedSharpe._std_max_sharpe(n_trials)

        # Variance of Sharpe ratio estimator
        var_sharpe = DeflatedSharpe._variance_sharpe(n_observations, skew, kurt, observed_sharpe)
        std_sharpe = np.sqrt(var_sharpe)

        # Deflated Sharpe Ratio
        dsr = (observed_sharpe - expected_max_sharpe) / np.sqrt(std_max_sharpe**2 + std_sharpe**2)

        # p-value: probability of observing this or better by chance
        p_value = 1 - stats.norm.cdf(dsr)

        return {
            'observed_sharpe': observed_sharpe,
            'expected_max_sharpe': expected_max_sharpe,
            'deflated_sharpe': dsr,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'n_trials': n_trials,
            'n_observations': n_observations
        }

    @staticmethod
    def _expected_max_sharpe(n_trials: int) -> float:
        """
        Expected maximum Sharpe ratio from N independent trials.

        Based on extreme value theory.
        """
        # Euler-Mascheroni constant
        gamma = 0.5772156649

        # Expected maximum of N standard normal variables
        z_max = (1 - gamma) * stats.norm.ppf(1 - 1.0/n_trials) + \
                gamma * stats.norm.ppf(1 - 1.0/(n_trials * np.e))

        return z_max

    @staticmethod
    def _std_max_sharpe(n_trials: int) -> float:
        """
        Standard deviation of maximum Sharpe from N trials.
        """
        # Approximation
        return 1.0 / stats.norm.ppf(1 - 1.0/n_trials)

    @staticmethod
    def _variance_sharpe(n_obs: int, skew: float, kurt: float, sharpe: float) -> float:
        """
        Variance of Sharpe ratio estimator.

        Accounts for non-normality via skewness and kurtosis.
        """
        # Adjustment for non-normality
        var = (1 + sharpe**2 / 2 - skew * sharpe + (kurt - 1) / 4 * sharpe**2) / n_obs

        return max(var, 1e-10)  # Prevent negative variance
