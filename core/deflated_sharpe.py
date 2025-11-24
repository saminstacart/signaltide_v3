"""
Deflated Sharpe Ratio implementation.

Corrects for multiple testing when selecting strategies from many trials.
Without this correction, the best Sharpe from N trials is biased upward
simply due to chance - the more trials, the more likely to find a
spuriously high Sharpe.

Reference: López de Prado, M. (2018). The Deflated Sharpe Ratio: Correcting
for Selection Bias, Backtest Overfitting, and Non-Normality.

Key insight: If you test N strategies and pick the best, you're not measuring
strategy skill - you're measuring max(noise_1, noise_2, ..., noise_N).
DSR corrects for this by asking: "Is this Sharpe significant given how many
strategies I tested?"
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional, List, Dict
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def compute_sharpe_std_error(
    sharpe: float,
    skewness: float,
    kurtosis: float,
    T: int
) -> float:
    """
    Compute the standard error of the Sharpe ratio.

    Uses the Lo (2002) formula which accounts for non-normality:
    SE(SR) = sqrt((1 + 0.5*SR^2 - skew*SR + (kurt/4)*SR^2) / T)

    Args:
        sharpe: Observed Sharpe ratio
        skewness: Skewness of returns (0 for normal)
        kurtosis: Excess kurtosis of returns (0 for normal)
        T: Number of return observations

    Returns:
        Standard error of Sharpe ratio

    Reference: Lo, A. (2002). The Statistics of Sharpe Ratios.
    """
    variance = (
        1 + 0.5 * sharpe**2
        - skewness * sharpe
        + (kurtosis / 4) * sharpe**2
    ) / T

    return np.sqrt(max(variance, 1e-10))  # Ensure non-negative


def expected_max_sharpe(
    num_trials: int,
    sharpe_std: float
) -> float:
    """
    Compute expected maximum Sharpe ratio under the null hypothesis.

    Uses the Euler-Mascheroni approximation for E[max(Z_1, ..., Z_N)]
    where Z_i ~ N(0, 1).

    Args:
        num_trials: Number of strategy variations tested (N)
        sharpe_std: Standard error of Sharpe ratio

    Returns:
        Expected maximum Sharpe under null (no skill)
    """
    if num_trials <= 1:
        return 0.0

    euler_mascheroni = 0.5772156649015329

    # Quantile for 1 - 1/N
    q1 = stats.norm.ppf(1 - 1/num_trials)
    # Quantile for 1 - 1/(N*e)
    q2 = stats.norm.ppf(1 - 1/(num_trials * np.e))

    expected_max = sharpe_std * (
        (1 - euler_mascheroni) * q1 + euler_mascheroni * q2
    )

    return expected_max


def compute_deflated_sharpe(
    observed_sharpe: float,
    num_trials: int,
    returns_skewness: float,
    returns_kurtosis: float,
    T: int,
    sharpe_std: Optional[float] = None
) -> Tuple[float, float]:
    """
    Compute the Deflated Sharpe Ratio (DSR).

    The DSR answers: "Is this Sharpe ratio statistically significant
    given how many strategies I tested?"

    A high observed Sharpe (e.g., 0.8) might not be significant if you
    tested 1000 parameter combinations - you'd expect to find at least
    one with high Sharpe just by chance.

    Args:
        observed_sharpe: The observed (backtest) Sharpe ratio
        num_trials: Number of strategy variations tested (N)
        returns_skewness: Skewness of the return series
        returns_kurtosis: Excess kurtosis of the return series
        T: Number of return observations
        sharpe_std: Standard error of Sharpe (computed if not provided)

    Returns:
        Tuple of (deflated_sharpe, p_value)
        - deflated_sharpe: The z-score of observed vs expected max
        - p_value: Probability of seeing this DSR under null hypothesis

    Example:
        >>> dsr, pval = compute_deflated_sharpe(
        ...     observed_sharpe=0.628,
        ...     num_trials=50,
        ...     returns_skewness=-0.2,
        ...     returns_kurtosis=1.5,
        ...     T=116
        ... )
        >>> print(f"DSR: {dsr:.3f}, p-value: {pval:.4f}")
    """
    if num_trials < 1:
        raise ValueError("num_trials must be >= 1")
    if T < 2:
        raise ValueError("T (number of observations) must be >= 2")

    # Compute Sharpe standard error if not provided
    if sharpe_std is None:
        sharpe_std = compute_sharpe_std_error(
            observed_sharpe, returns_skewness, returns_kurtosis, T
        )

    # Expected maximum Sharpe under null (multiple testing adjustment)
    exp_max = expected_max_sharpe(num_trials, sharpe_std)

    # Deflated Sharpe Ratio = (observed - expected_max) / std_error
    # This is a z-score: how many standard errors above the expected max?
    if sharpe_std > 0:
        deflated_sharpe = (observed_sharpe - exp_max) / sharpe_std
    else:
        deflated_sharpe = 0.0

    # P-value: probability of observing this DSR or higher under null
    p_value = 1 - stats.norm.cdf(deflated_sharpe)

    logger.debug(
        f"DSR calculation: observed={observed_sharpe:.4f}, "
        f"exp_max={exp_max:.4f}, std={sharpe_std:.4f}, "
        f"dsr={deflated_sharpe:.4f}, p={p_value:.4f}"
    )

    return deflated_sharpe, p_value


def apply_dsr_to_trials(
    trial_sharpes: List[float],
    trial_returns: List[np.ndarray],
    T: int
) -> Dict:
    """
    Apply DSR correction to a set of optimization trials.

    Use this after running Optuna or grid search to assess whether
    your best result is statistically significant or just noise.

    Args:
        trial_sharpes: List of Sharpe ratios from each trial
        trial_returns: List of return arrays from each trial
        T: Number of return observations per trial

    Returns:
        Dict with:
        - best_trial_idx: Index of best trial
        - observed_sharpe: Best Sharpe ratio
        - deflated_sharpe: DSR-corrected Sharpe
        - p_value: Statistical significance
        - is_significant: True if p < 0.05
        - num_trials: Number of trials tested

    Example:
        >>> results = apply_dsr_to_trials(
        ...     trial_sharpes=[0.3, 0.5, 0.628, 0.4, 0.35],
        ...     trial_returns=[...],  # actual return arrays
        ...     T=116
        ... )
        >>> if results['is_significant']:
        ...     print("Strategy has genuine alpha!")
    """
    if len(trial_sharpes) == 0:
        raise ValueError("trial_sharpes cannot be empty")
    if len(trial_sharpes) != len(trial_returns):
        raise ValueError("trial_sharpes and trial_returns must have same length")

    num_trials = len(trial_sharpes)
    best_idx = int(np.argmax(trial_sharpes))
    best_sharpe = trial_sharpes[best_idx]
    best_returns = np.array(trial_returns[best_idx])

    # Compute moments of best trial's returns
    skewness = float(stats.skew(best_returns))
    kurtosis = float(stats.kurtosis(best_returns))  # excess kurtosis

    dsr, p_value = compute_deflated_sharpe(
        observed_sharpe=best_sharpe,
        num_trials=num_trials,
        returns_skewness=skewness,
        returns_kurtosis=kurtosis,
        T=T
    )

    return {
        'best_trial_idx': best_idx,
        'observed_sharpe': best_sharpe,
        'deflated_sharpe': dsr,
        'p_value': p_value,
        'num_trials': num_trials,
        'returns_skewness': skewness,
        'returns_kurtosis': kurtosis,
        'is_significant': p_value < 0.05,
        'significance_level': (
            '***' if p_value < 0.01 else
            '**' if p_value < 0.05 else
            '*' if p_value < 0.10 else
            ''
        )
    }


def compute_minimum_track_record_length(
    observed_sharpe: float,
    target_sharpe: float = 0.0,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    confidence: float = 0.95
) -> int:
    """
    Compute minimum track record length (MinTRL) to reject null hypothesis.

    How long do you need to observe a strategy to be confident it has
    genuine alpha vs luck?

    Args:
        observed_sharpe: Current observed Sharpe ratio
        target_sharpe: Sharpe under null hypothesis (usually 0)
        skewness: Return skewness (0 for normal)
        kurtosis: Return kurtosis (3 for normal, excess kurtosis = kurtosis - 3)
        confidence: Desired confidence level (default 95%)

    Returns:
        Minimum number of observations needed

    Reference: Bailey & López de Prado (2014). The Sharpe Ratio Efficient Frontier.
    """
    if observed_sharpe <= target_sharpe:
        return float('inf')  # Can never reject null

    z = stats.norm.ppf(confidence)
    excess_kurtosis = kurtosis - 3

    # MinTRL formula
    numerator = (
        1 + 0.5 * observed_sharpe**2
        - skewness * observed_sharpe
        + (excess_kurtosis / 4) * observed_sharpe**2
    )
    denominator = (observed_sharpe - target_sharpe)**2

    min_trl = (z**2 * numerator) / denominator

    return int(np.ceil(min_trl))


def dsr_summary_report(
    observed_sharpe: float,
    num_trials: int,
    returns: np.ndarray,
    strategy_name: str = "Strategy"
) -> str:
    """
    Generate a human-readable DSR summary report.

    Args:
        observed_sharpe: Observed Sharpe ratio
        num_trials: Number of trials tested
        returns: Array of strategy returns
        strategy_name: Name for the report

    Returns:
        Formatted string report
    """
    T = len(returns)
    skewness = float(stats.skew(returns))
    kurtosis = float(stats.kurtosis(returns))

    dsr, p_value = compute_deflated_sharpe(
        observed_sharpe=observed_sharpe,
        num_trials=num_trials,
        returns_skewness=skewness,
        returns_kurtosis=kurtosis,
        T=T
    )

    sharpe_std = compute_sharpe_std_error(observed_sharpe, skewness, kurtosis, T)
    exp_max = expected_max_sharpe(num_trials, sharpe_std)
    min_trl = compute_minimum_track_record_length(observed_sharpe, 0, skewness, kurtosis + 3)

    significance = (
        "HIGHLY SIGNIFICANT (p < 0.01)" if p_value < 0.01 else
        "SIGNIFICANT (p < 0.05)" if p_value < 0.05 else
        "MARGINALLY SIGNIFICANT (p < 0.10)" if p_value < 0.10 else
        "NOT SIGNIFICANT"
    )

    report = f"""
================================================================================
DEFLATED SHARPE RATIO REPORT: {strategy_name}
================================================================================

INPUTS:
  Observed Sharpe:     {observed_sharpe:.4f}
  Number of Trials:    {num_trials}
  Return Observations: {T}
  Return Skewness:     {skewness:.4f}
  Return Kurtosis:     {kurtosis:.4f} (excess)

CALCULATIONS:
  Sharpe Std Error:    {sharpe_std:.4f}
  Expected Max Sharpe: {exp_max:.4f} (under null, given {num_trials} trials)

RESULTS:
  Deflated Sharpe:     {dsr:.4f}
  P-value:             {p_value:.4f}
  Significance:        {significance}

INTERPRETATION:
  The expected maximum Sharpe from {num_trials} random trials is {exp_max:.4f}.
  Your observed Sharpe of {observed_sharpe:.4f} is {observed_sharpe - exp_max:.4f} above this.
  {'This IS statistically significant - likely genuine alpha.' if p_value < 0.05 else
   'This is NOT statistically significant - could be noise from testing many parameters.'}

MINIMUM TRACK RECORD:
  To confirm this Sharpe at 95% confidence: {min_trl} observations needed
  Current observations: {T}
  {'Track record IS sufficient.' if T >= min_trl else f'Need {min_trl - T} more observations.'}

================================================================================
"""
    return report


if __name__ == '__main__':
    # Self-test with synthetic data
    print("Testing Deflated Sharpe Ratio implementation...")

    # Test 1: Basic calculation
    dsr, pval = compute_deflated_sharpe(
        observed_sharpe=0.628,
        num_trials=50,
        returns_skewness=-0.2,
        returns_kurtosis=1.5,
        T=116
    )
    print(f"\nTest 1 - Basic DSR:")
    print(f"  Observed Sharpe: 0.628")
    print(f"  Num Trials: 50")
    print(f"  Deflated Sharpe: {dsr:.4f}")
    print(f"  P-value: {pval:.4f}")
    print(f"  Significant at 5%: {pval < 0.05}")

    # Test 2: More trials should increase required Sharpe
    dsr2, pval2 = compute_deflated_sharpe(
        observed_sharpe=0.628,
        num_trials=200,  # More trials
        returns_skewness=-0.2,
        returns_kurtosis=1.5,
        T=116
    )
    print(f"\nTest 2 - Same Sharpe, more trials:")
    print(f"  Deflated Sharpe: {dsr2:.4f} (vs {dsr:.4f} with fewer trials)")
    print(f"  P-value: {pval2:.4f} (vs {pval:.4f})")
    assert dsr2 < dsr, "More trials should deflate more"

    # Test 3: Longer track record should increase significance
    dsr3, pval3 = compute_deflated_sharpe(
        observed_sharpe=0.628,
        num_trials=50,
        returns_skewness=-0.2,
        returns_kurtosis=1.5,
        T=500  # Longer track record
    )
    print(f"\nTest 3 - Same Sharpe, longer track record:")
    print(f"  Deflated Sharpe: {dsr3:.4f} (vs {dsr:.4f} with shorter record)")
    print(f"  P-value: {pval3:.4f} (vs {pval:.4f})")

    # Test 4: apply_dsr_to_trials
    np.random.seed(42)
    trial_sharpes = [0.3, 0.5, 0.628, 0.4, 0.35]
    trial_returns = [np.random.normal(0.01 * s, 0.05, 116) for s in trial_sharpes]

    results = apply_dsr_to_trials(trial_sharpes, trial_returns, T=116)
    print(f"\nTest 4 - apply_dsr_to_trials:")
    print(f"  Best trial index: {results['best_trial_idx']}")
    print(f"  Observed Sharpe: {results['observed_sharpe']:.4f}")
    print(f"  Deflated Sharpe: {results['deflated_sharpe']:.4f}")
    print(f"  P-value: {results['p_value']:.4f}")
    print(f"  Significant: {results['is_significant']}")

    # Test 5: Summary report
    returns = np.random.normal(0.008, 0.04, 116)
    report = dsr_summary_report(0.628, 50, returns, "M+Q v1")
    print(report)

    print("\nAll tests passed!")
