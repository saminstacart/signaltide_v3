"""
Tests for quintile mode behavior in InstitutionalSignal.

Validates that:
1. 'hard_20pct' mode selects exactly 20% per quintile
2. 'adaptive' mode allows bin merging when values cluster
3. Both modes produce valid signal distributions
"""

import pytest
import pandas as pd
import numpy as np

from core.institutional_base import InstitutionalSignal


class ConcreteInstitutionalSignal(InstitutionalSignal):
    """Concrete implementation for testing purposes."""

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Dummy implementation - not used in these tests."""
        return pd.Series(0, index=data.index)

    def get_parameter_space(self):
        """Dummy implementation - not used in these tests."""
        return {}


class TestQuintileModes:
    """Test quintile_mode parameter behavior."""

    @pytest.fixture
    def signal_hard(self):
        """Create signal with hard_20pct mode."""
        params = {
            'quintile_mode': 'hard_20pct',
            'winsorize_pct': [5, 95],
            'quintiles': True
        }
        return ConcreteInstitutionalSignal(params, name='TestSignal')

    @pytest.fixture
    def signal_adaptive(self):
        """Create signal with adaptive mode."""
        params = {
            'quintile_mode': 'adaptive',
            'winsorize_pct': [5, 95],
            'quintiles': True
        }
        return ConcreteInstitutionalSignal(params, name='TestSignal')

    def test_hard_20pct_exact_distribution(self, signal_hard):
        """Test that hard_20pct mode selects exactly 20% per quintile."""
        # Create 100 values for clean 20% splits
        values = pd.Series(np.random.randn(100))

        quintiles = signal_hard.to_quintiles(values)

        # Count distribution
        counts = quintiles.value_counts().sort_index()

        # Each quintile should have exactly 20 values (20%)
        assert len(counts) == 5, f"Expected 5 quintiles, got {len(counts)}"
        for label, count in counts.items():
            assert count == 20, f"Quintile {label} has {count} values, expected 20"

    def test_hard_20pct_with_clustered_values(self, signal_hard):
        """Test hard_20pct with clustered values (many duplicates)."""
        # Create clustered distribution: 50 values at 0, 50 values at 1
        values = pd.Series([0] * 50 + [1] * 50)

        quintiles = signal_hard.to_quintiles(values)

        # Should still have exactly 20% per bin (20 values each)
        counts = quintiles.value_counts()

        assert len(counts) == 5, f"Expected 5 quintiles even with clustering, got {len(counts)}"
        for label, count in counts.items():
            assert count == 20, f"Quintile {label} has {count} values, expected 20"

    def test_adaptive_allows_bin_merging(self, signal_adaptive):
        """Test that adaptive mode allows bins to merge with clustered values."""
        # Create clustered distribution
        values = pd.Series([0] * 50 + [1] * 50)

        quintiles = signal_adaptive.to_quintiles(values)

        # With adaptive mode, bins may merge, so we may get fewer than 5 bins
        counts = quintiles.value_counts()

        # Should have fewer than 5 bins due to merging
        assert len(counts) < 5, f"Expected <5 bins with adaptive mode, got {len(counts)}"

    def test_adaptive_with_spread_distribution(self, signal_adaptive):
        """Test adaptive mode with well-spread distribution."""
        # Create well-spread distribution (no clustering)
        values = pd.Series(np.arange(100))

        quintiles = signal_adaptive.to_quintiles(values)

        counts = quintiles.value_counts().sort_index()

        # Should have 5 quintiles with spread distribution
        assert len(counts) == 5, f"Expected 5 quintiles with spread values, got {len(counts)}"

        # Each should have ~20 values (exactly 20 for this case)
        for label, count in counts.items():
            assert count == 20, f"Quintile {label} has {count} values"

    def test_both_modes_produce_valid_labels(self, signal_hard, signal_adaptive):
        """Test that both modes produce standard quintile labels."""
        values = pd.Series(np.random.randn(100))
        expected_labels = {-1.0, -0.5, 0.0, 0.5, 1.0}

        # Hard mode
        quintiles_hard = signal_hard.to_quintiles(values)
        unique_hard = set(quintiles_hard.unique())
        assert unique_hard == expected_labels, f"Hard mode labels: {unique_hard}"

        # Adaptive mode
        quintiles_adaptive = signal_adaptive.to_quintiles(values)
        unique_adaptive = set(quintiles_adaptive.unique())
        assert unique_adaptive.issubset(expected_labels), \
            f"Adaptive mode labels {unique_adaptive} not subset of {expected_labels}"

    def test_hard_20pct_with_500_stocks(self, signal_hard):
        """Test hard_20pct with realistic S&P 500 size."""
        # S&P 500 typically has ~500 stocks
        values = pd.Series(np.random.randn(505))

        quintiles = signal_hard.to_quintiles(values)

        counts = quintiles.value_counts().sort_index()

        # 505 / 5 = 101 per quintile
        assert len(counts) == 5
        for label, count in counts.items():
            assert count == 101, f"Quintile {label} has {count} values, expected 101"

    def test_quintile_mode_parameter_inheritance(self):
        """Test that quintile_mode parameter is properly inherited."""
        # Default should be 'adaptive'
        params_default = {'quintiles': True}
        signal_default = ConcreteInstitutionalSignal(params_default, name='TestSignal')
        assert signal_default.quintile_mode == 'adaptive'

        # Explicit hard_20pct
        params_hard = {'quintiles': True, 'quintile_mode': 'hard_20pct'}
        signal_hard = ConcreteInstitutionalSignal(params_hard, name='TestSignal')
        assert signal_hard.quintile_mode == 'hard_20pct'

    def test_invalid_quintile_mode_raises_error(self):
        """Test that invalid quintile_mode raises ValueError."""
        params = {'quintile_mode': 'invalid_mode', 'quintiles': True}
        signal = ConcreteInstitutionalSignal(params, name='TestSignal')

        values = pd.Series(np.random.randn(100))

        with pytest.raises(ValueError, match="Unknown quintile_mode"):
            signal.to_quintiles(values)

    def test_edge_case_small_universe(self, signal_hard, signal_adaptive):
        """Test both modes with small universe (< 5 stocks)."""
        values = pd.Series([1.0, 2.0, 3.0])  # Only 3 values

        # With only 3 values, hard_20pct can still create bins (just not 5 equal bins)
        quintiles_hard = signal_hard.to_quintiles(values)
        assert len(quintiles_hard) == 3, "Should have 3 values"
        # With 3 values, rank-based qcut will create 3 bins
        assert len(quintiles_hard.unique()) == 3, "Should have 3 unique bins with 3 values"

        # Adaptive mode with 3 values will also create bins
        quintiles_adaptive = signal_adaptive.to_quintiles(values)
        assert len(quintiles_adaptive) == 3, "Should have 3 values"

    def test_mode_override_parameter(self, signal_adaptive):
        """Test that mode parameter overrides instance quintile_mode."""
        # Signal is adaptive by default
        assert signal_adaptive.quintile_mode == 'adaptive'

        values = pd.Series(np.random.randn(100))

        # Override to hard_20pct via parameter
        quintiles = signal_adaptive.to_quintiles(values, mode='hard_20pct')

        counts = quintiles.value_counts()
        assert len(counts) == 5
        for label, count in counts.items():
            assert count == 20, f"Override to hard_20pct should give exactly 20 per bin"


class TestQuintileModesIntegration:
    """Integration tests with full InstitutionalMomentum signal."""

    def test_momentum_with_hard_20pct(self):
        """Test InstitutionalMomentum with hard_20pct mode."""
        from signals.momentum.institutional_momentum import InstitutionalMomentum

        params = {
            'formation_period': 252,
            'skip_period': 21,
            'winsorize_pct': [5, 95],
            'quintiles': True,
            'quintile_mode': 'hard_20pct'
        }

        signal = InstitutionalMomentum(params)

        # Create synthetic price data for 100 stocks
        dates = pd.date_range('2020-01-01', '2022-01-01', freq='D')
        prices = pd.DataFrame({
            'close': np.random.randn(len(dates)).cumsum() + 100
        }, index=dates)

        signals = signal.generate_signals(prices)

        # Verify signal was generated
        assert len(signals) > 0
        assert signal.quintile_mode == 'hard_20pct'

    def test_momentum_with_adaptive(self):
        """Test InstitutionalMomentum with adaptive mode (default)."""
        from signals.momentum.institutional_momentum import InstitutionalMomentum

        params = {
            'formation_period': 252,
            'skip_period': 21,
            'winsorize_pct': [5, 95],
            'quintiles': True,
            'quintile_mode': 'adaptive'
        }

        signal = InstitutionalMomentum(params)

        # Create synthetic price data
        dates = pd.date_range('2020-01-01', '2022-01-01', freq='D')
        prices = pd.DataFrame({
            'close': np.random.randn(len(dates)).cumsum() + 100
        }, index=dates)

        signals = signal.generate_signals(prices)

        # Verify signal was generated
        assert len(signals) > 0
        assert signal.quintile_mode == 'adaptive'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
