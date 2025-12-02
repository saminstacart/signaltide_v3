"""
Walk-Forward Validation for time series backtesting.

Reference:
    Pardo, R. (2008). "The Evaluation and Optimization of Trading Strategies".
    Wiley Trading.

Key insight: For time series, never use k-fold CV. Instead use
rolling out-of-sample (OOS) windows that respect temporal ordering.
"""

import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from signaltide_v4.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Result for a single walk-forward fold."""

    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_sharpe: float
    test_sharpe: float
    test_return: float
    is_positive: bool  # Test Sharpe > 0
    diagnostics: Dict[str, Any] = None


@dataclass
class WalkForwardResult:
    """Container for walk-forward validation results."""

    folds: List[FoldResult]
    n_folds: int
    n_positive_folds: int
    pct_positive: float
    mean_test_sharpe: float
    std_test_sharpe: float
    mean_train_sharpe: float
    train_test_correlation: float
    is_valid: bool  # Passes minimum criteria?
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'n_folds': self.n_folds,
            'pct_positive': self.pct_positive,
            'mean_test_sharpe': self.mean_test_sharpe,
            'std_test_sharpe': self.std_test_sharpe,
            'mean_train_sharpe': self.mean_train_sharpe,
            'train_test_correlation': self.train_test_correlation,
            'is_valid': self.is_valid,
        }


class WalkForwardValidator:
    """
    Implements rolling walk-forward validation.

    Process:
    1. Train on window [t-W, t]
    2. Test on [t, t+H]
    3. Roll forward by H months
    4. Repeat until end of data
    """

    def __init__(
        self,
        train_months: Optional[int] = None,
        test_months: Optional[int] = None,
        min_folds: Optional[int] = None,
        min_positive_pct: float = 0.50,
    ):
        """
        Initialize walk-forward validator.

        Args:
            train_months: Training window size in months
            test_months: Test window size in months
            min_folds: Minimum number of OOS periods required
            min_positive_pct: Minimum percentage of positive OOS folds
        """
        settings = get_settings()

        self.train_months = train_months or settings.walk_forward_train_months
        self.test_months = test_months or settings.walk_forward_test_months
        self.min_folds = min_folds or settings.walk_forward_min_folds
        self.min_positive_pct = min_positive_pct

        logger.info(
            f"WalkForwardValidator: train={self.train_months}mo, "
            f"test={self.test_months}mo, min_folds={self.min_folds}"
        )

    def validate(
        self,
        strategy_fn: Callable[[str, str], Tuple[pd.Series, Dict]],
        start_date: str,
        end_date: str,
    ) -> WalkForwardResult:
        """
        Run walk-forward validation.

        Args:
            strategy_fn: Function(train_start, train_end) -> (returns, metrics)
            start_date: Overall start date
            end_date: Overall end date

        Returns:
            WalkForwardResult with all fold results
        """
        folds = self._generate_folds(start_date, end_date)

        if len(folds) < self.min_folds:
            logger.warning(
                f"Insufficient folds: {len(folds)} < {self.min_folds} required"
            )
            return self._empty_result(len(folds))

        fold_results = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(folds):
            try:
                # Run strategy on train period
                train_returns, train_metrics = strategy_fn(train_start, train_end)
                train_sharpe = self._calculate_sharpe(train_returns)

                # Run strategy on test period (with train parameters)
                test_returns, test_metrics = strategy_fn(test_start, test_end)
                test_sharpe = self._calculate_sharpe(test_returns)
                test_return = float((1 + test_returns).prod() - 1) if len(test_returns) > 0 else 0.0

                fold_result = FoldResult(
                    fold_id=i,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    train_sharpe=train_sharpe,
                    test_sharpe=test_sharpe,
                    test_return=test_return,
                    is_positive=test_sharpe > 0,
                    diagnostics={
                        'train_n': len(train_returns),
                        'test_n': len(test_returns),
                    },
                )

                fold_results.append(fold_result)
                logger.debug(
                    f"Fold {i}: train_sr={train_sharpe:.3f}, test_sr={test_sharpe:.3f}"
                )

            except Exception as e:
                logger.error(f"Fold {i} failed: {e}")
                continue

        return self._compile_results(fold_results)

    def validate_returns(
        self,
        returns: pd.Series,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> WalkForwardResult:
        """
        Simplified validation using pre-computed returns.

        Args:
            returns: Daily returns series with datetime index
            start_date: Optional start date override
            end_date: Optional end date override

        Returns:
            WalkForwardResult
        """
        if len(returns) == 0:
            return self._empty_result(0)

        start_date = start_date or returns.index.min().strftime('%Y-%m-%d')
        end_date = end_date or returns.index.max().strftime('%Y-%m-%d')

        folds = self._generate_folds(start_date, end_date)

        if len(folds) < self.min_folds:
            logger.warning(f"Insufficient folds: {len(folds)}")
            return self._empty_result(len(folds))

        fold_results = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(folds):
            # Get returns for each period
            train_mask = (returns.index >= train_start) & (returns.index <= train_end)
            test_mask = (returns.index >= test_start) & (returns.index <= test_end)

            train_rets = returns[train_mask]
            test_rets = returns[test_mask]

            if len(train_rets) < 20 or len(test_rets) < 5:
                continue

            train_sharpe = self._calculate_sharpe(train_rets)
            test_sharpe = self._calculate_sharpe(test_rets)
            test_return = float((1 + test_rets).prod() - 1)

            fold_results.append(FoldResult(
                fold_id=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_sharpe=train_sharpe,
                test_sharpe=test_sharpe,
                test_return=test_return,
                is_positive=test_sharpe > 0,
            ))

        return self._compile_results(fold_results)

    def _generate_folds(
        self,
        start_date: str,
        end_date: str,
    ) -> List[Tuple[str, str, str, str]]:
        """Generate walk-forward fold date ranges."""
        folds = []

        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        train_delta = pd.DateOffset(months=self.train_months)
        test_delta = pd.DateOffset(months=self.test_months)

        current_train_start = start

        while True:
            train_end = current_train_start + train_delta
            test_start = train_end
            test_end = test_start + test_delta

            if test_end > end:
                break

            folds.append((
                current_train_start.strftime('%Y-%m-%d'),
                train_end.strftime('%Y-%m-%d'),
                test_start.strftime('%Y-%m-%d'),
                test_end.strftime('%Y-%m-%d'),
            ))

            # Roll forward by test period
            current_train_start = current_train_start + test_delta

        logger.info(f"Generated {len(folds)} walk-forward folds")
        return folds

    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 5:
            return 0.0

        mean_ret = returns.mean()
        std_ret = returns.std()

        if std_ret == 0:
            return 0.0

        return float((mean_ret / std_ret) * np.sqrt(252))

    def _compile_results(
        self,
        fold_results: List[FoldResult],
    ) -> WalkForwardResult:
        """Compile fold results into summary."""
        n_folds = len(fold_results)

        if n_folds == 0:
            return self._empty_result(0)

        n_positive = sum(1 for f in fold_results if f.is_positive)
        pct_positive = n_positive / n_folds

        train_sharpes = [f.train_sharpe for f in fold_results]
        test_sharpes = [f.test_sharpe for f in fold_results]

        # Calculate correlation between train and test Sharpes
        if len(train_sharpes) > 2:
            correlation = float(np.corrcoef(train_sharpes, test_sharpes)[0, 1])
        else:
            correlation = 0.0

        is_valid = (
            pct_positive >= self.min_positive_pct and
            n_folds >= self.min_folds
        )

        return WalkForwardResult(
            folds=fold_results,
            n_folds=n_folds,
            n_positive_folds=n_positive,
            pct_positive=pct_positive,
            mean_test_sharpe=float(np.mean(test_sharpes)),
            std_test_sharpe=float(np.std(test_sharpes)),
            mean_train_sharpe=float(np.mean(train_sharpes)),
            train_test_correlation=correlation,
            is_valid=is_valid,
            diagnostics={
                'min_folds_required': self.min_folds,
                'min_positive_pct_required': self.min_positive_pct,
            },
        )

    def _empty_result(self, n_folds: int) -> WalkForwardResult:
        """Return empty result for insufficient data."""
        return WalkForwardResult(
            folds=[],
            n_folds=n_folds,
            n_positive_folds=0,
            pct_positive=0.0,
            mean_test_sharpe=0.0,
            std_test_sharpe=0.0,
            mean_train_sharpe=0.0,
            train_test_correlation=0.0,
            is_valid=False,
            diagnostics={'error': 'insufficient_folds'},
        )
