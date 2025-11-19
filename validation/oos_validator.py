"""
Out-of-Sample (OOS) Validation Framework

Implements rigorous walk-forward testing to prevent data snooping and overfitting.

Key principles:
1. Training period: Optimize parameters
2. Validation period: Select best parameters
3. Test period: TRUE out-of-sample performance (NEVER used in optimization)

References:
- Bailey & López de Prado (2014) "The Deflated Sharpe Ratio"
- Harvey & Liu (2015) "Backtests and Multiple Testing"
"""

from typing import Dict, List, Optional, Tuple, Callable
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_logger

logger = get_logger(__name__)


class OOSValidator:
    """
    Out-of-sample validation for trading strategies.

    Workflow:
    1. Split data into train/validation/test
    2. Optimize on train set
    3. Select parameters on validation set
    4. Final evaluation on test set (reported performance)
    """

    def __init__(self,
                 train_pct: float = 0.6,
                 validation_pct: float = 0.2,
                 test_pct: float = 0.2):
        """
        Initialize OOS validator.

        Args:
            train_pct: Percentage for training (default: 60%)
            validation_pct: Percentage for validation (default: 20%)
            test_pct: Percentage for out-of-sample testing (default: 20%)
        """
        if not np.isclose(train_pct + validation_pct + test_pct, 1.0):
            raise ValueError("Percentages must sum to 1.0")

        if test_pct < 0.1:
            logger.warning(
                f"Test set is only {test_pct*100:.0f}% of data. "
                "Recommend at least 10% for reliable OOS testing."
            )

        self.train_pct = train_pct
        self.validation_pct = validation_pct
        self.test_pct = test_pct

        logger.info(
            f"OOSValidator initialized: "
            f"train={train_pct*100:.0f}%, "
            f"val={validation_pct*100:.0f}%, "
            f"test={test_pct*100:.0f}%"
        )

    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets.

        Args:
            data: Time series data with DatetimeIndex

        Returns:
            (train_data, validation_data, test_data)
        """
        n = len(data)
        train_end = int(n * self.train_pct)
        val_end = int(n * (self.train_pct + self.validation_pct))

        train = data.iloc[:train_end]
        validation = data.iloc[train_end:val_end]
        test = data.iloc[val_end:]

        logger.info(
            f"Data split: "
            f"train={len(train)} ({train.index[0].date()} to {train.index[-1].date()}), "
            f"val={len(validation)} ({validation.index[0].date()} to {validation.index[-1].date()}), "
            f"test={len(test)} ({test.index[0].date()} to {test.index[-1].date()})"
        )

        return train, validation, test

    def validate_strategy(self,
                         strategy_func: Callable,
                         data: pd.DataFrame,
                         param_grid: Dict[str, List],
                         metric: str = 'sharpe_ratio') -> Dict:
        """
        Validate strategy with proper OOS methodology.

        Args:
            strategy_func: Function that takes (data, params) and returns metrics dict
            data: Full dataset
            param_grid: Dict of parameter names to lists of values
            metric: Metric to optimize ('sharpe_ratio', 'total_return', etc.)

        Returns:
            Dict with:
                - best_params: Parameters selected on validation set
                - train_performance: Performance on training set
                - validation_performance: Performance on validation set
                - oos_performance: OUT-OF-SAMPLE performance on test set
                - degradation: Performance degradation from validation to OOS
        """
        # Split data
        train, validation, test = self.split_data(data)

        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        logger.info(f"Testing {len(param_combinations)} parameter combinations")

        # Test all combinations on training set
        train_results = []
        for params in param_combinations:
            try:
                result = strategy_func(train, params)
                result['params'] = params
                train_results.append(result)
            except Exception as e:
                logger.warning(f"Strategy failed with params {params}: {e}")
                continue

        if not train_results:
            raise ValueError("All parameter combinations failed on training set")

        # Sort by metric
        train_results = sorted(
            train_results,
            key=lambda x: x.get(metric, -np.inf),
            reverse=True
        )

        logger.info(
            f"Top 5 train results ({metric}): "
            f"{[r.get(metric, 0) for r in train_results[:5]]}"
        )

        # Test top N parameters on validation set
        top_n = min(10, len(train_results))
        validation_results = []

        for result in train_results[:top_n]:
            params = result['params']
            try:
                val_result = strategy_func(validation, params)
                val_result['params'] = params
                val_result['train_metric'] = result.get(metric, 0)
                validation_results.append(val_result)
            except Exception as e:
                logger.warning(f"Strategy failed on validation with params {params}: {e}")
                continue

        if not validation_results:
            raise ValueError("All top parameters failed on validation set")

        # Select best parameters based on validation performance
        validation_results = sorted(
            validation_results,
            key=lambda x: x.get(metric, -np.inf),
            reverse=True
        )

        best_on_validation = validation_results[0]
        best_params = best_on_validation['params']

        logger.info(f"Best params on validation: {best_params}")
        logger.info(f"Validation {metric}: {best_on_validation.get(metric, 0):.4f}")

        # Final OOS test (NEVER seen before)
        try:
            oos_result = strategy_func(test, best_params)
        except Exception as e:
            logger.error(f"Strategy failed on OOS test set: {e}")
            raise

        # Calculate degradation
        val_metric = best_on_validation.get(metric, 0)
        oos_metric = oos_result.get(metric, 0)
        degradation = (val_metric - oos_metric) / abs(val_metric) if val_metric != 0 else 0

        logger.info(f"OOS {metric}: {oos_metric:.4f}")
        logger.info(f"Degradation from validation to OOS: {degradation*100:.1f}%")

        # Check for overfitting
        if degradation > 0.3:
            logger.warning(
                f"⚠️ HIGH DEGRADATION: {degradation*100:.1f}% drop from validation to OOS. "
                "Strategy may be overfit!"
            )

        return {
            'best_params': best_params,
            'train_performance': train_results[0],  # Best on train
            'validation_performance': best_on_validation,
            'oos_performance': oos_result,
            'degradation': degradation,
            'degradation_pct': degradation * 100,
            'metric_name': metric,
            'n_params_tested': len(param_combinations),
            'is_overfit': degradation > 0.3
        }

    def walk_forward_validation(self,
                               strategy_func: Callable,
                               data: pd.DataFrame,
                               param_grid: Dict[str, List],
                               window_size: int = 252,  # 1 year
                               step_size: int = 21,     # 1 month
                               metric: str = 'sharpe_ratio') -> List[Dict]:
        """
        Walk-forward validation (rolling window).

        Args:
            strategy_func: Strategy function
            data: Full dataset
            param_grid: Parameter grid
            window_size: Training window size in days
            step_size: Step size for rolling window
            metric: Metric to optimize

        Returns:
            List of results for each window
        """
        results = []
        n = len(data)

        logger.info(
            f"Starting walk-forward validation: "
            f"window={window_size}days, step={step_size}days"
        )

        for start_idx in range(0, n - window_size, step_size):
            end_idx = start_idx + window_size
            if end_idx >= n:
                break

            # Training window
            train_data = data.iloc[start_idx:end_idx]

            # Test window (next period)
            test_end = min(end_idx + step_size, n)
            test_data = data.iloc[end_idx:test_end]

            if len(test_data) < step_size // 2:
                # Not enough test data
                break

            logger.debug(
                f"Window {len(results)+1}: "
                f"train={train_data.index[0].date()} to {train_data.index[-1].date()}, "
                f"test={test_data.index[0].date()} to {test_data.index[-1].date()}"
            )

            # Optimize on training window
            param_combinations = self._generate_param_combinations(param_grid)
            train_results = []

            for params in param_combinations:
                try:
                    result = strategy_func(train_data, params)
                    result['params'] = params
                    train_results.append(result)
                except Exception:
                    continue

            if not train_results:
                logger.warning(f"No valid results for window {len(results)+1}")
                continue

            # Best params on training
            best_train = max(train_results, key=lambda x: x.get(metric, -np.inf))
            best_params = best_train['params']

            # Test on next period
            try:
                test_result = strategy_func(test_data, best_params)
            except Exception:
                logger.warning(f"Test failed for window {len(results)+1}")
                continue

            results.append({
                'window': len(results) + 1,
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'best_params': best_params,
                'train_metric': best_train.get(metric, 0),
                'test_metric': test_result.get(metric, 0),
                'degradation': (best_train.get(metric, 0) - test_result.get(metric, 0)) /
                              abs(best_train.get(metric, 0)) if best_train.get(metric, 0) != 0 else 0
            })

        logger.info(
            f"Walk-forward validation complete: {len(results)} windows tested"
        )

        # Summary statistics
        if results:
            avg_degradation = np.mean([r['degradation'] for r in results])
            avg_test_metric = np.mean([r['test_metric'] for r in results])

            logger.info(
                f"Average test {metric}: {avg_test_metric:.4f}, "
                f"Average degradation: {avg_degradation*100:.1f}%"
            )

        return results

    def _generate_param_combinations(self, param_grid: Dict[str, List]) -> List[Dict]:
        """Generate all combinations of parameters from grid."""
        import itertools

        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]

        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    def monte_carlo_permutation_test(self,
                                    strategy_func: Callable,
                                    data: pd.DataFrame,
                                    params: Dict,
                                    n_permutations: int = 1000,
                                    metric: str = 'sharpe_ratio') -> Dict:
        """
        Monte Carlo permutation test for statistical significance.

        Randomly permutes signals to test if performance is due to luck.

        Args:
            strategy_func: Strategy function
            data: Dataset
            params: Strategy parameters
            n_permutations: Number of random permutations
            metric: Metric to test

        Returns:
            Dict with:
                - actual_metric: Actual strategy performance
                - permutation_metrics: Distribution of random performance
                - p_value: Probability of achieving result by chance
                - percentile: Percentile of actual result in distribution
        """
        logger.info(f"Running Monte Carlo permutation test with {n_permutations} trials")

        # Actual performance
        actual_result = strategy_func(data, params)
        actual_metric = actual_result.get(metric, 0)

        # Permutation tests
        permutation_metrics = []

        for i in range(n_permutations):
            # Randomly shuffle signals (breaks signal-return relationship)
            shuffled_data = data.copy()
            if 'signal' in shuffled_data.columns:
                shuffled_data['signal'] = np.random.permutation(shuffled_data['signal'].values)

            try:
                perm_result = strategy_func(shuffled_data, params)
                perm_metric = perm_result.get(metric, 0)
                permutation_metrics.append(perm_metric)
            except Exception:
                continue

            if (i + 1) % 100 == 0:
                logger.debug(f"Completed {i+1}/{n_permutations} permutations")

        permutation_metrics = np.array(permutation_metrics)

        # Calculate p-value
        if metric in ['sharpe_ratio', 'total_return', 'profit_factor']:
            # Higher is better
            p_value = (permutation_metrics >= actual_metric).sum() / len(permutation_metrics)
            percentile = (permutation_metrics < actual_metric).sum() / len(permutation_metrics) * 100
        else:
            # Lower is better (e.g., max_drawdown)
            p_value = (permutation_metrics <= actual_metric).sum() / len(permutation_metrics)
            percentile = (permutation_metrics > actual_metric).sum() / len(permutation_metrics) * 100

        logger.info(
            f"Monte Carlo results: "
            f"actual={actual_metric:.4f}, "
            f"p-value={p_value:.4f}, "
            f"percentile={percentile:.1f}%"
        )

        is_significant = p_value < 0.05

        if not is_significant:
            logger.warning(
                f"⚠️ NOT STATISTICALLY SIGNIFICANT: p-value={p_value:.4f} > 0.05. "
                "Performance may be due to luck!"
            )

        return {
            'actual_metric': actual_metric,
            'permutation_metrics': permutation_metrics,
            'p_value': p_value,
            'percentile': percentile,
            'is_significant': is_significant,
            'mean_permutation': permutation_metrics.mean(),
            'std_permutation': permutation_metrics.std()
        }


if __name__ == '__main__':
    # Example usage
    print("=" * 80)
    print("Out-of-Sample Validation Framework")
    print("=" * 80)

    # Create sample data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)

    data = pd.DataFrame({
        'returns': np.random.randn(len(dates)) * 0.01,
        'signal': np.random.randn(len(dates))
    }, index=dates)

    # Simple strategy function for testing
    def simple_strategy(data_subset, params):
        """Simple test strategy."""
        threshold = params['threshold']
        positions = (data_subset['signal'] > threshold).astype(float)
        strategy_returns = positions.shift(1) * data_subset['returns']
        strategy_returns = strategy_returns.dropna()

        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0

        return {
            'sharpe_ratio': sharpe,
            'total_return': (1 + strategy_returns).prod() - 1,
            'n_trades': (positions.diff() != 0).sum()
        }

    # Initialize validator
    validator = OOSValidator()

    # Parameter grid
    param_grid = {
        'threshold': [-0.5, 0, 0.5, 1.0, 1.5]
    }

    # Run OOS validation
    print("\nRunning OOS validation...")
    oos_results = validator.validate_strategy(
        strategy_func=simple_strategy,
        data=data,
        param_grid=param_grid,
        metric='sharpe_ratio'
    )

    print(f"\nBest params: {oos_results['best_params']}")
    print(f"Validation Sharpe: {oos_results['validation_performance']['sharpe_ratio']:.4f}")
    print(f"OOS Sharpe: {oos_results['oos_performance']['sharpe_ratio']:.4f}")
    print(f"Degradation: {oos_results['degradation_pct']:.1f}%")
    print(f"Overfit: {'YES ⚠️' if oos_results['is_overfit'] else 'NO ✅'}")

    print("\n✅ OOS validation framework working correctly")
