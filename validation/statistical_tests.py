"""
Statistical significance tests for trading strategies.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple


class StatisticalTests:
    """
    Collection of statistical tests for strategy validation.
    """

    @staticmethod
    def sharpe_confidence_interval(returns: pd.Series, confidence: float = 0.95) -> Dict:
        """
        Calculate Sharpe ratio with confidence interval.

        Args:
            returns: Strategy returns
            confidence: Confidence level (default 0.95 for 95% CI)

        Returns:
            Dict with Sharpe ratio and confidence interval
        """
        if len(returns) < 2:
            return {'sharpe': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0, 'significant': False}

        # Calculate Sharpe ratio
        mean_return = returns.mean()
        std_return = returns.std()

        if std_return == 0:
            return {'sharpe': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0, 'significant': False}

        sharpe = mean_return / std_return * np.sqrt(252)  # Annualized

        # Standard error of Sharpe ratio
        n = len(returns)
        se = np.sqrt((1 + sharpe**2 / 2) / n)

        # Confidence interval
        z = stats.norm.ppf((1 + confidence) / 2)
        ci_lower = sharpe - z * se
        ci_upper = sharpe + z * se

        # Significant if lower bound > 0
        significant = ci_lower > 0

        return {
            'sharpe': sharpe,
            'se': se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence': confidence,
            'significant': significant,
            'n_observations': n
        }

    @staticmethod
    def t_test(returns: pd.Series, benchmark: float = 0.0) -> Dict:
        """
        T-test for whether returns are significantly different from benchmark.

        Args:
            returns: Strategy returns
            benchmark: Benchmark return (default 0)

        Returns:
            Dict with t-statistic, p-value, and significance
        """
        if len(returns) < 2:
            return {'t_stat': 0.0, 'p_value': 1.0, 'significant': False}

        t_stat, p_value = stats.ttest_1samp(returns, benchmark)

        return {
            't_stat': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'mean': returns.mean(),
            'std': returns.std(),
            'n': len(returns)
        }

    @staticmethod
    def autocorrelation_test(returns: pd.Series, max_lag: int = 20) -> Dict:
        """
        Test for autocorrelation in returns (should be minimal for valid strategy).

        Args:
            returns: Strategy returns
            max_lag: Maximum lag to test

        Returns:
            Dict with autocorrelation results
        """
        if len(returns) < max_lag + 2:
            return {'autocorr': [], 'significant_lags': [], 'max_autocorr': 0.0}

        autocorr = [returns.autocorr(lag=i) for i in range(1, max_lag + 1)]

        # Significance threshold (approximate)
        threshold = 2 / np.sqrt(len(returns))

        significant_lags = [i+1 for i, ac in enumerate(autocorr) if abs(ac) > threshold]

        return {
            'autocorr': autocorr,
            'significant_lags': significant_lags,
            'max_autocorr': max(autocorr, key=abs) if autocorr else 0.0,
            'threshold': threshold,
            'has_significant_autocorr': len(significant_lags) > 0
        }

    @staticmethod
    def normality_test(returns: pd.Series) -> Dict:
        """
        Test whether returns are normally distributed.

        Args:
            returns: Strategy returns

        Returns:
            Dict with normality test results
        """
        if len(returns) < 3:
            return {'is_normal': False, 'p_value': 1.0}

        # Shapiro-Wilk test
        stat, p_value = stats.shapiro(returns)

        # Skewness and kurtosis
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)

        return {
            'test': 'Shapiro-Wilk',
            'statistic': stat,
            'p_value': p_value,
            'is_normal': p_value > 0.05,
            'skewness': skew,
            'kurtosis': kurt
        }
