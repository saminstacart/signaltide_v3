"""
Statistical significance tests for trading strategies.

Provides statistical validation tools including:
- Sharpe ratio confidence intervals
- Probabilistic Sharpe Ratio (PSR)
- Bootstrap Confidence Intervals
- Minimum Track Record Length (MTR)
- Multiple Testing Corrections

This module extends with additional tests required for the multi-level
optimization pipeline.

Based on:
- Bailey & López de Prado (2012) "The Sharpe Ratio Efficient Frontier"
- Bailey & López de Prado (2014) "The Deflated Sharpe Ratio"
- Ledoit & Wolf (2008) "Robust Performance Hypothesis Testing with the Sharpe Ratio"
- Harvey et al. (2016) "...and the Cross-Section of Expected Returns"
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, List, Optional, Any

from config import get_logger

logger = get_logger(__name__)


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


# =============================================================================
# STANDALONE FUNCTIONS FOR OPTIMIZATION PIPELINE
# =============================================================================

def probabilistic_sharpe_ratio(
    returns: np.ndarray,
    benchmark_sr: float = 0.0,
    annualization: int = 12
) -> float:
    """
    Calculate Probabilistic Sharpe Ratio.

    PSR is the probability that the true (population) Sharpe ratio is
    greater than a benchmark, given the observed sample statistics.

    PSR = Phi((SR - SR*) / sigma(SR))

    where:
        - SR: Observed Sharpe ratio
        - SR*: Benchmark Sharpe ratio
        - sigma(SR): Standard error of Sharpe ratio estimator
        - Phi: Standard normal CDF

    Args:
        returns: Array of returns (typically monthly)
        benchmark_sr: Benchmark Sharpe ratio (default 0)
        annualization: Periods per year (12 for monthly)

    Returns:
        Probability that true SR > benchmark_sr (0 to 1)

    Note:
        PSR > 0.95 suggests the strategy is unlikely due to luck alone.
    """
    if len(returns) < 3:
        logger.warning("PSR requires at least 3 returns, returning 0.5")
        return 0.5

    returns = np.asarray(returns, dtype=float)
    n = len(returns)

    # Calculate sample statistics
    mean_r = np.mean(returns)
    std_r = np.std(returns, ddof=1)

    if std_r < 1e-10:
        logger.warning("Near-zero volatility, PSR undefined")
        return 0.5

    # Observed Sharpe (annualized)
    sr = (mean_r / std_r) * np.sqrt(annualization)

    # Higher moments for SR standard error
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns)  # Excess kurtosis (normal = 0)

    # Standard error of Sharpe ratio (Lo, 2002; Bailey & Lopez de Prado, 2012)
    var_sr = (1 + 0.5 * sr**2 - skew * sr + (kurt / 4) * sr**2) / n
    se_sr = np.sqrt(max(var_sr, 1e-10))

    # PSR: probability that true SR > benchmark
    z = (sr - benchmark_sr) / se_sr
    psr = stats.norm.cdf(z)

    logger.debug(
        f"PSR: SR={sr:.4f}, benchmark={benchmark_sr:.2f}, "
        f"SE={se_sr:.4f}, PSR={psr:.4f}"
    )

    return psr


def bootstrap_confidence_interval(
    returns: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    annualization: int = 12,
    seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for Sharpe ratio.

    Uses percentile bootstrap method for robustness to non-normality.

    Args:
        returns: Array of returns (typically monthly)
        n_bootstrap: Number of bootstrap samples (default 10000)
        ci: Confidence level (default 0.95 for 95% CI)
        annualization: Periods per year (12 for monthly)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (lower_bound, upper_bound) for Sharpe ratio

    Note:
        If lower_bound > 0, the strategy has statistically significant
        positive risk-adjusted returns at the given confidence level.
    """
    if len(returns) < 10:
        logger.warning("Bootstrap requires at least 10 returns")
        return (-np.inf, np.inf)

    returns = np.asarray(returns, dtype=float)
    n = len(returns)

    rng = np.random.RandomState(seed)

    # Bootstrap Sharpe ratios
    bootstrap_sharpes = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # Resample with replacement
        sample = rng.choice(returns, size=n, replace=True)

        mean_s = np.mean(sample)
        std_s = np.std(sample, ddof=1)

        if std_s > 1e-10:
            bootstrap_sharpes[i] = (mean_s / std_s) * np.sqrt(annualization)
        else:
            bootstrap_sharpes[i] = 0

    # Percentile confidence interval
    alpha = 1 - ci
    lower = np.percentile(bootstrap_sharpes, alpha / 2 * 100)
    upper = np.percentile(bootstrap_sharpes, (1 - alpha / 2) * 100)

    logger.debug(f"Bootstrap {ci:.0%} CI: [{lower:.4f}, {upper:.4f}]")

    return lower, upper


def minimum_track_record_length(
    sharpe: float,
    skew: float = 0.0,
    kurt: float = 3.0,
    target_prob: float = 0.95,
    annualization: int = 12
) -> int:
    """
    Calculate Minimum Track Record Length (MTR).

    MTR is the number of periods needed to statistically confirm
    that a strategy's Sharpe ratio is real (not due to chance).

    Based on inverting the PSR formula:
    MTR = n such that PSR(n) >= target_prob

    Args:
        sharpe: Annualized Sharpe ratio
        skew: Return skewness (default 0 for normal)
        kurt: Return kurtosis (default 3 for normal, pass raw not excess)
        target_prob: Target probability level (default 0.95)
        annualization: Periods per year (12 for monthly)

    Returns:
        Minimum number of periods (months) needed

    Note:
        A high Sharpe with short history may have high MTR, indicating
        the result could be spurious.
    """
    if sharpe <= 0:
        return int(1e9)  # Effectively infinite

    # Convert to per-period Sharpe
    sr_period = sharpe / np.sqrt(annualization)

    # Use excess kurtosis internally
    excess_kurt = kurt - 3

    # Target z-score
    z_target = stats.norm.ppf(target_prob)

    # Solve for n:
    # Variance factor
    var_factor = 1 + 0.5 * sr_period**2 - skew * sr_period + (excess_kurt / 4) * sr_period**2

    # MTR formula
    mtr = z_target**2 * var_factor / sr_period**2

    # Round up to integer
    mtr_months = int(np.ceil(max(mtr, 1)))

    logger.debug(
        f"MTR: SR={sharpe:.2f}, skew={skew:.2f}, kurt={kurt:.2f}, "
        f"prob={target_prob:.2f} -> {mtr_months} months"
    )

    return mtr_months


def sharpe_ratio_test(
    returns1: np.ndarray,
    returns2: np.ndarray,
    annualization: int = 12
) -> Dict[str, float]:
    """
    Test if two strategies have significantly different Sharpe ratios.

    Uses approximate test based on return differences.

    Args:
        returns1: Returns from strategy 1
        returns2: Returns from strategy 2
        annualization: Periods per year

    Returns:
        Dict with test_statistic, p_value, sr1, sr2
    """
    r1 = np.asarray(returns1)
    r2 = np.asarray(returns2)

    if len(r1) != len(r2):
        raise ValueError("Return series must have same length")

    n = len(r1)
    if n < 10:
        logger.warning("Sharpe test requires at least 10 observations")
        return {'test_statistic': 0, 'p_value': 1.0, 'sr1': 0, 'sr2': 0}

    # Calculate Sharpe ratios
    mu1, mu2 = np.mean(r1), np.mean(r2)
    sig1, sig2 = np.std(r1, ddof=1), np.std(r2, ddof=1)

    if sig1 < 1e-10 or sig2 < 1e-10:
        return {'test_statistic': 0, 'p_value': 1.0, 'sr1': 0, 'sr2': 0}

    sr1 = (mu1 / sig1) * np.sqrt(annualization)
    sr2 = (mu2 / sig2) * np.sqrt(annualization)

    # Difference
    d = r1 - r2
    mu_d = np.mean(d)
    sig_d = np.std(d, ddof=1)

    # Simple t-test on difference (approximate)
    se_d = sig_d / np.sqrt(n)
    t_stat = mu_d / se_d if se_d > 1e-10 else 0

    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))

    return {
        'test_statistic': t_stat,
        'p_value': p_value,
        'sr1': sr1,
        'sr2': sr2,
        'sr_difference': sr1 - sr2
    }


def multiple_testing_correction(
    p_values: List[float],
    method: str = 'fdr_bh',
    alpha: float = 0.05
) -> Tuple[np.ndarray, List[bool]]:
    """
    Apply multiple testing correction to p-values.

    Args:
        p_values: List of p-values from multiple tests
        method: Correction method
            - 'bonferroni': Family-wise error rate control
            - 'fdr_bh': Benjamini-Hochberg FDR control
            - 'fdr_by': Benjamini-Yekutieli FDR (more conservative)
        alpha: Significance level

    Returns:
        Tuple of (corrected_p_values, significant_flags)

    Note:
        'fdr_bh' is recommended for optimization with many trials.
        It controls the expected proportion of false discoveries.
    """
    p_values = np.asarray(p_values)
    n = len(p_values)

    if n == 0:
        return np.array([]), []

    if method == 'bonferroni':
        corrected = np.minimum(p_values * n, 1.0)
        significant = list(corrected < alpha)

    elif method == 'fdr_bh':
        # Benjamini-Hochberg procedure
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]

        # BH critical values
        thresholds = alpha * np.arange(1, n + 1) / n

        # Find largest k where p_(k) <= k*alpha/n
        below = sorted_p <= thresholds
        if not np.any(below):
            corrected = np.ones(n)
            significant = [False] * n
        else:
            # Corrected p-values
            corrected_sorted = np.minimum.accumulate(
                (sorted_p * n / np.arange(1, n + 1))[::-1]
            )[::-1]
            corrected_sorted = np.minimum(corrected_sorted, 1.0)

            # Restore original order
            corrected = np.zeros(n)
            corrected[sorted_idx] = corrected_sorted

            significant = list(corrected < alpha)

    elif method == 'fdr_by':
        # Benjamini-Yekutieli (more conservative)
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]

        # c(n) = sum(1/i for i in 1..n)
        c_n = np.sum(1.0 / np.arange(1, n + 1))
        thresholds = alpha * np.arange(1, n + 1) / (n * c_n)

        below = sorted_p <= thresholds
        if not np.any(below):
            corrected = np.ones(n)
            significant = [False] * n
        else:
            corrected_sorted = np.minimum.accumulate(
                (sorted_p * n * c_n / np.arange(1, n + 1))[::-1]
            )[::-1]
            corrected_sorted = np.minimum(corrected_sorted, 1.0)

            corrected = np.zeros(n)
            corrected[sorted_idx] = corrected_sorted

            significant = list(corrected < alpha)

    else:
        raise ValueError(f"Unknown method: {method}")

    return corrected, significant


def comprehensive_validation(
    returns: np.ndarray,
    n_trials: int,
    benchmark_sr: float = 0.0,
    n_bootstrap: int = 10000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run comprehensive statistical validation suite.

    Combines DSR, PSR, bootstrap CI, and MTR for complete validation.

    Args:
        returns: Strategy returns (monthly)
        n_trials: Number of optimization trials (for DSR)
        benchmark_sr: Benchmark Sharpe (default 0)
        n_bootstrap: Bootstrap samples
        seed: Random seed

    Returns:
        Dict with all validation metrics
    """
    from validation.deflated_sharpe import DeflatedSharpe

    returns = np.asarray(returns)
    n_obs = len(returns)

    # Basic statistics
    mean_r = np.mean(returns)
    std_r = np.std(returns, ddof=1)
    sharpe = (mean_r / std_r) * np.sqrt(12) if std_r > 1e-10 else 0

    skew = stats.skew(returns) if n_obs > 2 else 0.0
    kurt = stats.kurtosis(returns) + 3 if n_obs > 3 else 3.0  # Raw kurtosis

    # Deflated Sharpe Ratio
    dsr_result = DeflatedSharpe.calculate(
        observed_sharpe=sharpe,
        n_trials=n_trials,
        n_observations=n_obs,
        skew=skew,
        kurt=kurt - 3  # DeflatedSharpe expects excess kurtosis
    )

    # Probabilistic Sharpe Ratio
    psr = probabilistic_sharpe_ratio(returns, benchmark_sr)

    # Bootstrap CI
    ci_lower, ci_upper = bootstrap_confidence_interval(
        returns, n_bootstrap=n_bootstrap, seed=seed
    )

    # Minimum Track Record Length
    mtr = minimum_track_record_length(sharpe, skew, kurt) if sharpe > 0 else int(1e9)

    return {
        # Basic stats
        'sharpe': sharpe,
        'n_observations': n_obs,
        'skewness': skew,
        'kurtosis': kurt,

        # DSR
        'deflated_sharpe': dsr_result['deflated_sharpe'],
        'dsr_p_value': dsr_result['p_value'],
        'dsr_significant': dsr_result['is_significant'],
        'expected_max_sharpe': dsr_result['expected_max_sharpe'],

        # PSR
        'psr': psr,
        'psr_significant': psr > 0.95,

        # Bootstrap
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_significant': ci_lower > 0,

        # MTR
        'mtr_months': mtr,
        'mtr_years': mtr / 12,
        'sufficient_history': n_obs >= mtr,

        # Overall verdict
        'statistically_valid': (
            dsr_result['is_significant'] and
            ci_lower > 0 and
            n_obs >= mtr
        )
    }
