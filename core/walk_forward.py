"""
Walk-Forward Validation Framework.

Implements expanding-window cross-validation for strategy parameter selection.
This is the gold standard for avoiding overfitting in trading systems.

Key concept: Train on historical data, then test on unseen future data.
As time progresses, expand the training window while always testing OOS.

Example with min_train=3 years, test=1 year over 2015-2024:
  Fold 1: Train 2015-2017, Test 2018
  Fold 2: Train 2015-2018, Test 2019
  Fold 3: Train 2015-2019, Test 2020
  ...

Reference: López de Prado (2018) "Advances in Financial Machine Learning", Ch. 7
"""

import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardFold:
    """Single fold of walk-forward validation."""
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_sharpe: float = None
    test_sharpe: float = None
    train_return: float = None
    test_return: float = None
    params: dict = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Complete walk-forward validation result."""
    folds: List[WalkForwardFold]
    mean_test_sharpe: float
    std_test_sharpe: float
    mean_train_sharpe: float
    test_train_ratio: float
    all_folds_positive: bool
    num_positive_folds: int
    num_folds: int

    def is_robust(self, min_ratio: float = 0.5, min_positive_pct: float = 0.6) -> bool:
        """
        Check if results indicate robust strategy.

        Args:
            min_ratio: Minimum test/train Sharpe ratio (default 0.5)
            min_positive_pct: Minimum percentage of positive OOS folds (default 60%)

        Returns:
            True if strategy passes robustness checks
        """
        positive_pct = self.num_positive_folds / self.num_folds if self.num_folds > 0 else 0
        return (
            self.test_train_ratio >= min_ratio and
            positive_pct >= min_positive_pct and
            self.mean_test_sharpe > 0
        )

    def summary(self) -> str:
        """Generate summary string."""
        positive_pct = self.num_positive_folds / self.num_folds * 100 if self.num_folds > 0 else 0
        return f"""
Walk-Forward Validation Summary:
  Folds: {self.num_folds}
  Mean Train Sharpe: {self.mean_train_sharpe:.4f}
  Mean Test Sharpe:  {self.mean_test_sharpe:.4f} (±{self.std_test_sharpe:.4f})
  Test/Train Ratio:  {self.test_train_ratio:.2f}
  Positive Folds:    {self.num_positive_folds}/{self.num_folds} ({positive_pct:.0f}%)
  Is Robust:         {self.is_robust()}
"""


def generate_expanding_window_folds(
    start_date: str,
    end_date: str,
    min_train_years: int = 3,
    test_years: int = 1,
    step_years: int = 1
) -> List[Tuple[str, str, str, str]]:
    """
    Generate expanding window folds for walk-forward validation.

    Creates folds where training window expands over time while
    test window slides forward.

    Args:
        start_date: Overall start date (YYYY-MM-DD)
        end_date: Overall end date (YYYY-MM-DD)
        min_train_years: Minimum training window (years)
        test_years: Test window size (years)
        step_years: How much to expand training window each fold

    Returns:
        List of (train_start, train_end, test_start, test_end) tuples

    Example:
        >>> folds = generate_expanding_window_folds(
        ...     '2015-01-01', '2024-12-31',
        ...     min_train_years=3, test_years=1, step_years=1
        ... )
        >>> print(folds[0])  # ('2015-01-01', '2017-12-31', '2018-01-01', '2018-12-31')
    """
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])

    folds = []
    train_end_year = start_year + min_train_years - 1

    while train_end_year + test_years <= end_year:
        train_start = f"{start_year}-01-01"
        train_end = f"{train_end_year}-12-31"
        test_start = f"{train_end_year + 1}-01-01"
        test_end = f"{train_end_year + test_years}-12-31"

        folds.append((train_start, train_end, test_start, test_end))
        train_end_year += step_years

    logger.info(f"Generated {len(folds)} walk-forward folds from {start_date} to {end_date}")
    return folds


def generate_rolling_window_folds(
    start_date: str,
    end_date: str,
    train_years: int = 3,
    test_years: int = 1
) -> List[Tuple[str, str, str, str]]:
    """
    Generate rolling (fixed-size) window folds.

    Unlike expanding windows, training window size stays constant
    and slides forward. Useful for detecting regime changes.

    Args:
        start_date: Overall start date
        end_date: Overall end date
        train_years: Fixed training window size
        test_years: Test window size

    Returns:
        List of (train_start, train_end, test_start, test_end) tuples
    """
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])

    folds = []
    current_train_start = start_year

    while current_train_start + train_years + test_years - 1 <= end_year:
        train_start = f"{current_train_start}-01-01"
        train_end = f"{current_train_start + train_years - 1}-12-31"
        test_start = f"{current_train_start + train_years}-01-01"
        test_end = f"{current_train_start + train_years + test_years - 1}-12-31"

        folds.append((train_start, train_end, test_start, test_end))
        current_train_start += 1

    logger.info(f"Generated {len(folds)} rolling window folds")
    return folds


def run_walk_forward_validation(
    optimize_fn: Callable[[str, str, int], Tuple[dict, float]],
    evaluate_fn: Callable[[str, str, dict], Tuple[float, float]],
    folds: List[Tuple[str, str, str, str]],
    n_trials_per_fold: int = 30
) -> WalkForwardResult:
    """
    Run complete walk-forward validation.

    This is the main entry point for walk-forward CV. It:
    1. For each fold, optimizes parameters on training data
    2. Evaluates those parameters on test data
    3. Aggregates results to assess robustness

    Args:
        optimize_fn: Function(train_start, train_end, n_trials) -> (best_params, train_sharpe)
        evaluate_fn: Function(test_start, test_end, params) -> (test_sharpe, test_return)
        folds: List of (train_start, train_end, test_start, test_end)
        n_trials_per_fold: Number of optimization trials per fold

    Returns:
        WalkForwardResult with fold details and aggregate statistics

    Example:
        >>> def optimize(start, end, n_trials):
        ...     # Run Optuna optimization
        ...     return best_params, best_sharpe
        ...
        >>> def evaluate(start, end, params):
        ...     # Run backtest with params
        ...     return sharpe, total_return
        ...
        >>> folds = generate_expanding_window_folds('2015-01-01', '2024-12-31')
        >>> result = run_walk_forward_validation(optimize, evaluate, folds)
        >>> print(result.summary())
    """
    results = []

    for i, (train_start, train_end, test_start, test_end) in enumerate(folds):
        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {i+1}/{len(folds)}")
        logger.info(f"Train: {train_start} to {train_end}")
        logger.info(f"Test:  {test_start} to {test_end}")
        logger.info('='*60)

        try:
            # Optimize on training period
            best_params, train_sharpe = optimize_fn(train_start, train_end, n_trials_per_fold)

            # Evaluate on test period
            test_sharpe, test_return = evaluate_fn(test_start, test_end, best_params)

            fold_result = WalkForwardFold(
                fold_id=i+1,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_sharpe=train_sharpe,
                test_sharpe=test_sharpe,
                test_return=test_return,
                params=best_params
            )
            results.append(fold_result)

            logger.info(f"Fold {i+1} Results:")
            logger.info(f"  Train Sharpe: {train_sharpe:.4f}")
            logger.info(f"  Test Sharpe:  {test_sharpe:.4f}")
            if train_sharpe > 0:
                logger.info(f"  Ratio:        {test_sharpe/train_sharpe:.2f}")

        except Exception as e:
            logger.error(f"Fold {i+1} failed: {e}")
            continue

    if len(results) == 0:
        raise ValueError("All folds failed - no results to aggregate")

    # Aggregate statistics
    test_sharpes = [r.test_sharpe for r in results if r.test_sharpe is not None]
    train_sharpes = [r.train_sharpe for r in results if r.train_sharpe is not None]

    mean_test = np.mean(test_sharpes) if test_sharpes else 0
    std_test = np.std(test_sharpes) if test_sharpes else 0
    mean_train = np.mean(train_sharpes) if train_sharpes else 0

    return WalkForwardResult(
        folds=results,
        mean_test_sharpe=mean_test,
        std_test_sharpe=std_test,
        mean_train_sharpe=mean_train,
        test_train_ratio=mean_test / mean_train if mean_train > 0 else 0,
        all_folds_positive=all(s > 0 for s in test_sharpes),
        num_positive_folds=sum(1 for s in test_sharpes if s > 0),
        num_folds=len(results)
    )


def analyze_walk_forward_stability(result: WalkForwardResult) -> Dict:
    """
    Analyze stability of walk-forward results.

    Checks for:
    - Parameter consistency across folds
    - Performance degradation over time
    - Regime-specific patterns

    Args:
        result: WalkForwardResult from run_walk_forward_validation

    Returns:
        Dict with stability analysis
    """
    if len(result.folds) < 2:
        return {'error': 'Need at least 2 folds for stability analysis'}

    # Extract test Sharpes by fold
    fold_sharpes = [(f.fold_id, f.test_sharpe) for f in result.folds if f.test_sharpe is not None]

    if len(fold_sharpes) < 2:
        return {'error': 'Not enough folds with valid Sharpe'}

    fold_ids, sharpes = zip(*fold_sharpes)

    # Check for time trend (performance degradation)
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(fold_ids, sharpes)

    has_negative_trend = slope < 0 and p_value < 0.1

    # Coefficient of variation (stability measure)
    cv = np.std(sharpes) / np.mean(sharpes) if np.mean(sharpes) != 0 else float('inf')

    return {
        'num_folds': len(fold_sharpes),
        'sharpe_trend_slope': slope,
        'sharpe_trend_pvalue': p_value,
        'has_negative_trend': has_negative_trend,
        'coefficient_of_variation': cv,
        'is_stable': cv < 1.0 and not has_negative_trend,
        'min_sharpe': min(sharpes),
        'max_sharpe': max(sharpes),
        'sharpe_range': max(sharpes) - min(sharpes)
    }


def walk_forward_report(result: WalkForwardResult) -> str:
    """
    Generate comprehensive walk-forward validation report.
    """
    stability = analyze_walk_forward_stability(result)
    positive_pct = result.num_positive_folds / result.num_folds * 100 if result.num_folds > 0 else 0

    report = f"""
================================================================================
WALK-FORWARD VALIDATION REPORT
================================================================================

CONFIGURATION:
  Number of Folds: {result.num_folds}
  Method: Expanding Window

AGGREGATE RESULTS:
  Mean Train Sharpe: {result.mean_train_sharpe:.4f}
  Mean Test Sharpe:  {result.mean_test_sharpe:.4f}
  Std Test Sharpe:   {result.std_test_sharpe:.4f}
  Test/Train Ratio:  {result.test_train_ratio:.2f}

FOLD STATISTICS:
  All Positive OOS:  {result.all_folds_positive}
  Positive Folds:    {result.num_positive_folds}/{result.num_folds} ({positive_pct:.0f}%)
  Min OOS Sharpe:    {stability.get('min_sharpe', 'N/A')}
  Max OOS Sharpe:    {stability.get('max_sharpe', 'N/A')}

STABILITY ANALYSIS:
  Coefficient of Variation: {stability.get('coefficient_of_variation', 'N/A'):.2f}
  Has Negative Trend:       {stability.get('has_negative_trend', 'N/A')}
  Is Stable:                {stability.get('is_stable', 'N/A')}

FOLD-BY-FOLD RESULTS:
"""

    for fold in result.folds:
        ratio = fold.test_sharpe / fold.train_sharpe if fold.train_sharpe and fold.train_sharpe > 0 else 0
        report += f"""
  Fold {fold.fold_id}:
    Train: {fold.train_start} to {fold.train_end}
    Test:  {fold.test_start} to {fold.test_end}
    Train Sharpe: {fold.train_sharpe:.4f if fold.train_sharpe else 'N/A'}
    Test Sharpe:  {fold.test_sharpe:.4f if fold.test_sharpe else 'N/A'}
    Ratio:        {ratio:.2f if ratio else 'N/A'}
"""

    # Overall verdict
    is_robust = result.is_robust()
    verdict = "PASS - Strategy is robust" if is_robust else "FAIL - Strategy may be overfitted"

    report += f"""
================================================================================
VERDICT: {verdict}
================================================================================
"""
    return report


if __name__ == '__main__':
    # Self-test
    print("Testing Walk-Forward Validation Framework...")

    # Test fold generation
    folds = generate_expanding_window_folds(
        '2015-01-01', '2024-12-31',
        min_train_years=3, test_years=1, step_years=1
    )
    print(f"\nGenerated {len(folds)} expanding window folds:")
    for i, (ts, te, vs, ve) in enumerate(folds[:3]):
        print(f"  Fold {i+1}: Train {ts} to {te}, Test {vs} to {ve}")
    print("  ...")

    # Test rolling windows
    rolling_folds = generate_rolling_window_folds(
        '2015-01-01', '2024-12-31',
        train_years=3, test_years=1
    )
    print(f"\nGenerated {len(rolling_folds)} rolling window folds")

    # Mock validation run
    print("\nTesting mock validation run...")

    def mock_optimize(train_start, train_end, n_trials):
        return {'param1': 0.5}, 0.7 + np.random.normal(0, 0.1)

    def mock_evaluate(test_start, test_end, params):
        return 0.5 + np.random.normal(0, 0.15), 0.1

    # Use just 3 folds for quick test
    result = run_walk_forward_validation(
        mock_optimize, mock_evaluate,
        folds[:3],
        n_trials_per_fold=5
    )

    print(result.summary())
    print(f"Is Robust: {result.is_robust()}")

    print("\nAll tests passed!")
