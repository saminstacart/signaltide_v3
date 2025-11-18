"""
Monte Carlo permutation testing for signal validation.

Tests whether signal performance is due to skill vs luck.
"""

from typing import Dict, Callable
import numpy as np
import pandas as pd
from tqdm import tqdm


class MonteCarloValidator:
    """
    Monte Carlo permutation testing to verify signal skill.

    Randomly permutes signals and compares performance to actual.
    If actual performance is not significantly better than permuted,
    signal likely has no real edge.
    """

    def __init__(self, n_trials: int = 1000, random_state: int = 42):
        """
        Initialize Monte Carlo validator.

        Args:
            n_trials: Number of permutation trials
            random_state: Random seed for reproducibility
        """
        self.n_trials = n_trials
        self.random_state = random_state
        np.random.seed(random_state)

    def validate(self, signal_series: pd.Series, returns: pd.Series,
                 metric_fn: Callable = None) -> Dict:
        """
        Run Monte Carlo validation.

        Args:
            signal_series: Trading signals (values in [-1, 1])
            returns: Market returns (same index as signals)
            metric_fn: Function to calculate performance metric
                       Default: Sharpe ratio

        Returns:
            Dict with:
                - actual_metric: Actual performance
                - permuted_metrics: List of permuted performances
                - p_value: Statistical significance
                - percentile: Where actual ranks vs permuted
                - is_significant: Boolean (p < 0.05)
        """
        # Default metric: Sharpe ratio
        if metric_fn is None:
            metric_fn = self._sharpe_ratio

        # Calculate actual performance
        actual_metric = metric_fn(signal_series, returns)

        # Run permutation trials
        permuted_metrics = []
        for _ in tqdm(range(self.n_trials), desc="Monte Carlo trials"):
            # Permute signals (breaks timing relationship)
            permuted_signals = signal_series.sample(frac=1.0).values

            # Reconstruct series with original index
            permuted_series = pd.Series(permuted_signals, index=signal_series.index)

            # Calculate performance
            permuted_metric = metric_fn(permuted_series, returns)
            permuted_metrics.append(permuted_metric)

        # Calculate p-value
        # Proportion of permuted >= actual
        n_better = sum(1 for m in permuted_metrics if m >= actual_metric)
        p_value = n_better / self.n_trials

        # Calculate percentile
        percentile = sum(1 for m in permuted_metrics if m < actual_metric) / self.n_trials

        return {
            'actual_metric': actual_metric,
            'permuted_metrics': permuted_metrics,
            'permuted_mean': np.mean(permuted_metrics),
            'permuted_std': np.std(permuted_metrics),
            'p_value': p_value,
            'percentile': percentile,
            'is_significant': p_value < 0.05,
            'n_trials': self.n_trials
        }

    def _sharpe_ratio(self, signals: pd.Series, returns: pd.Series) -> float:
        """
        Calculate Sharpe ratio of strategy.

        Args:
            signals: Trading signals
            returns: Market returns

        Returns:
            Annualized Sharpe ratio
        """
        # Strategy returns = signal * market returns
        strategy_returns = signals.shift(1) * returns  # Shift to avoid lookahead

        # Remove NaN
        strategy_returns = strategy_returns.dropna()

        if len(strategy_returns) == 0 or strategy_returns.std() == 0:
            return 0.0

        # Annualized Sharpe (assuming daily data)
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)

        return sharpe

    def __repr__(self) -> str:
        return f"MonteCarloValidator(n_trials={self.n_trials})"
