"""
Tests for validation framework.
"""

import pytest
import pandas as pd
import numpy as np
from validation.purged_kfold import PurgedKFold, calculate_purge_embargo_sizes
from validation.monte_carlo import MonteCarloValidator
from validation.statistical_tests import StatisticalTests
from validation.deflated_sharpe import DeflatedSharpe


class TestPurgedKFold:
    """Test Purged K-Fold cross-validation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        data = pd.DataFrame({'value': np.random.randn(1000)}, index=dates)
        return data

    def test_purged_kfold_splits(self, sample_data):
        """Test that splits are generated correctly."""
        pkf = PurgedKFold(n_splits=5, purge_pct=0.05, embargo_pct=0.01)

        splits = list(pkf.split(sample_data))

        # Check number of splits
        assert len(splits) == 5

        # Check each split
        for train_idx, test_idx in splits:
            # No overlap between train and test
            assert len(set(train_idx) & set(test_idx)) == 0

            # Test indices are contiguous
            assert test_idx[-1] - test_idx[0] == len(test_idx) - 1

    def test_size_calculation(self):
        """Test purge and embargo size calculation."""
        info = calculate_purge_embargo_sizes(
            n_samples=1000,
            n_splits=5,
            purge_pct=0.05,
            embargo_pct=0.01
        )

        assert info['n_samples'] == 1000
        assert info['n_splits'] == 5
        assert info['purge_size'] == int(200 * 0.05)
        assert info['embargo_size'] == int(200 * 0.01)


class TestMonteCarloValidator:
    """Test Monte Carlo permutation testing."""

    @pytest.fixture
    def sample_signals_returns(self):
        """Create sample signals and returns."""
        np.random.seed(42)
        n = 252

        # Create signals with some predictive power
        returns = pd.Series(np.random.randn(n) * 0.01)
        signals = pd.Series(np.sign(returns.shift(-1)))  # Perfect foresight (for testing)

        # Align indices
        signals.index = returns.index

        return signals, returns

    def test_monte_carlo_validation(self, sample_signals_returns):
        """Test Monte Carlo validation."""
        signals, returns = sample_signals_returns

        validator = MonteCarloValidator(n_trials=100, random_state=42)

        result = validator.validate(signals, returns)

        # Check result structure
        assert 'actual_metric' in result
        assert 'permuted_metrics' in result
        assert 'p_value' in result
        assert 'is_significant' in result

        # Check permuted metrics
        assert len(result['permuted_metrics']) == 100


class TestStatisticalTests:
    """Test statistical significance tests."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns."""
        np.random.seed(42)
        return pd.Series(np.random.randn(252) * 0.01 + 0.0005)  # Slight positive bias

    def test_sharpe_confidence_interval(self, sample_returns):
        """Test Sharpe ratio with confidence interval."""
        result = StatisticalTests.sharpe_confidence_interval(sample_returns)

        assert 'sharpe' in result
        assert 'ci_lower' in result
        assert 'ci_upper' in result
        assert result['ci_lower'] < result['sharpe'] < result['ci_upper']

    def test_t_test(self, sample_returns):
        """Test t-test."""
        result = StatisticalTests.t_test(sample_returns)

        assert 't_stat' in result
        assert 'p_value' in result
        assert 'significant' in result


class TestDeflatedSharpe:
    """Test Deflated Sharpe Ratio calculation."""

    def test_deflated_sharpe(self):
        """Test DSR calculation."""
        result = DeflatedSharpe.calculate(
            observed_sharpe=2.0,
            n_trials=100,
            n_observations=252
        )

        assert 'deflated_sharpe' in result
        assert 'p_value' in result
        assert 'is_significant' in result

        # DSR should be less than observed Sharpe (haircut applied)
        assert result['deflated_sharpe'] < 2.0
