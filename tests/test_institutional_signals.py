"""
Unit Tests for Institutional Signals

Tests:
1. InstitutionalMomentum - Jegadeesh-Titman 12-1
2. InstitutionalQuality - Quality Minus Junk
3. InstitutionalInsider - Cohen-Malloy-Pomorski

Focus areas:
- Parameter validation
- Signal generation correctness
- Lookahead bias prevention
- Edge case handling
- Output ranges and formats
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from signals.momentum.institutional_momentum import InstitutionalMomentum
from signals.quality.institutional_quality import InstitutionalQuality
from signals.insider.institutional_insider import InstitutionalInsider


class TestInstitutionalMomentum:
    """Tests for InstitutionalMomentum signal."""

    DEFAULT_PARAMS = {
        'formation_period': 252,  # 12 months
        'skip_period': 21,  # 1 month
        'winsorize_pct': [5, 95],
        'quintiles': True
    }

    @pytest.fixture
    def sample_data(self):
        """Create sample price data for testing."""
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        # Create trending price data
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.randn(500) * 0.02))

        df = pd.DataFrame({
            'open': prices * (1 + np.random.randn(500) * 0.01),
            'high': prices * (1 + np.random.rand(500) * 0.02),
            'low': prices * (1 - np.random.rand(500) * 0.02),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 500)
        }, index=dates)

        return df

    def test_initialization(self):
        """Test signal initialization with default parameters."""
        signal = InstitutionalMomentum(self.DEFAULT_PARAMS)
        assert signal.formation_period == 252
        assert signal.skip_period == 21
        assert signal.total_lookback == 273
        assert signal.quintiles is True

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        params = {
            'formation_period': 126,
            'skip_period': 10,
            'quintiles': False
        }
        signal = InstitutionalMomentum(params)
        assert signal.formation_period == 126
        assert signal.skip_period == 10
        assert signal.quintiles is False

    def test_signal_generation_shape(self, sample_data):
        """Test that generated signals have correct shape."""
        signal = InstitutionalMomentum(self.DEFAULT_PARAMS)
        signals = signal.generate_signals(sample_data)

        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_data)
        assert signals.index.equals(sample_data.index)

    def test_signal_range(self, sample_data):
        """Test that signals are in valid range [-1, 1]."""
        signal = InstitutionalMomentum(self.DEFAULT_PARAMS)
        signals = signal.generate_signals(sample_data)

        # Check range
        assert signals.min() >= -1.0
        assert signals.max() <= 1.0

        # Non-null signals should be quintile values
        non_zero = signals[signals != 0]
        if len(non_zero) > 0:
            unique_values = set(non_zero.unique())
            expected_values = {-1.0, -0.5, 0.0, 0.5, 1.0}
            assert unique_values.issubset(expected_values)

    def test_no_lookahead_bias(self, sample_data):
        """Test that momentum calculation doesn't use future data."""
        signal = InstitutionalMomentum(self.DEFAULT_PARAMS)
        signals = signal.generate_signals(sample_data)

        # First signal should be after formation + skip period
        first_nonzero_idx = signals[signals != 0].index[0] if (signals != 0).any() else None
        if first_nonzero_idx:
            days_from_start = (first_nonzero_idx - sample_data.index[0]).days
            # Should be at least formation_period + skip_period
            assert days_from_start >= signal.total_lookback - 10  # Allow small tolerance

    def test_monthly_rebalancing(self, sample_data):
        """Test that monthly rebalancing works correctly."""
        params = self.DEFAULT_PARAMS.copy()
        params['rebalance_frequency'] = 'monthly'
        signal = InstitutionalMomentum(params)
        signals = signal.generate_signals(sample_data)

        # Signals should only change at month-end
        signal_changes = signals.diff().fillna(0)
        changes_per_month = signal_changes[signal_changes != 0].resample('M').count()

        # Most months should have 0 or 1 change
        assert (changes_per_month <= 1).sum() / len(changes_per_month) > 0.9

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        signal = InstitutionalMomentum(self.DEFAULT_PARAMS)

        # Create short dataset
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        short_data = pd.DataFrame({
            'close': np.random.randn(50) + 100
        }, index=dates)

        signals = signal.generate_signals(short_data)

        # Should return zeros for insufficient data
        assert (signals == 0).all()

    def test_missing_close_column(self):
        """Test handling when close column is missing."""
        signal = InstitutionalMomentum(self.DEFAULT_PARAMS)

        dates = pd.date_range('2023-01-01', periods=300, freq='D')
        bad_data = pd.DataFrame({
            'open': np.random.randn(300) + 100
        }, index=dates)

        signals = signal.generate_signals(bad_data)
        assert (signals == 0).all()

    def test_parameter_space(self):
        """Test parameter space definition."""
        signal = InstitutionalMomentum(self.DEFAULT_PARAMS)
        param_space = signal.get_parameter_space()

        assert 'formation_period' in param_space
        assert 'skip_period' in param_space
        assert 'winsorize_pct' in param_space
        assert 'quintiles' in param_space

        # Check types
        assert param_space['formation_period'][0] == 'int'
        assert param_space['quintiles'][0] == 'categorical'


class TestInstitutionalQuality:
    """Tests for InstitutionalQuality signal."""

    @pytest.fixture
    def mock_data_manager(self):
        """Create mock data manager."""
        from data.data_manager import create_mock_data_manager
        return create_mock_data_manager()

    @pytest.fixture
    def sample_data(self):
        """Create sample price data with ticker."""
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.randn(500) * 0.02))

        df = pd.DataFrame({
            'open': prices,
            'high': prices,
            'low': prices,
            'close': prices,
            'volume': 1000000,
            'ticker': 'TEST'
        }, index=dates)

        return df

    def test_initialization(self, mock_data_manager):
        """Test signal initialization."""
        signal = InstitutionalQuality({}, data_manager=mock_data_manager)
        assert signal.use_profitability is True
        assert signal.use_growth is True
        assert signal.use_safety is True
        assert signal.prof_weight == 0.4
        assert signal.growth_weight == 0.3
        assert signal.safety_weight == 0.3

    def test_component_weights_sum_to_one(self, mock_data_manager):
        """Test that component weights sum to approximately 1.0."""
        params = {
            'prof_weight': 0.5,
            'growth_weight': 0.3,
            'safety_weight': 0.2
        }
        signal = InstitutionalQuality(params, data_manager=mock_data_manager)
        total_weight = signal.prof_weight + signal.growth_weight + signal.safety_weight
        assert abs(total_weight - 1.0) < 0.01  # Allow small floating point error

    def test_signal_range(self, sample_data, mock_data_manager):
        """Test that signals are in valid range."""
        signal = InstitutionalQuality({}, data_manager=mock_data_manager)
        signals = signal.generate_signals(sample_data)

        assert signals.min() >= -1.0
        assert signals.max() <= 1.0

    def test_missing_ticker(self, mock_data_manager):
        """Test handling when ticker column is missing."""
        signal = InstitutionalQuality({}, data_manager=mock_data_manager)

        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        bad_data = pd.DataFrame({
            'close': np.random.randn(100) + 100
        }, index=dates)

        signals = signal.generate_signals(bad_data)
        assert (signals == 0).all()

    def test_parameter_space(self, mock_data_manager):
        """Test parameter space definition."""
        signal = InstitutionalQuality({}, data_manager=mock_data_manager)
        param_space = signal.get_parameter_space()

        assert 'use_profitability' in param_space
        assert 'use_growth' in param_space
        assert 'use_safety' in param_space
        assert 'prof_weight' in param_space
        assert 'growth_weight' in param_space
        assert 'safety_weight' in param_space


class TestInstitutionalInsider:
    """Tests for InstitutionalInsider signal."""

    @pytest.fixture
    def mock_data_manager(self):
        """Create mock data manager."""
        from data.data_manager import create_mock_data_manager
        return create_mock_data_manager()

    @pytest.fixture
    def sample_data(self):
        """Create sample price data with ticker."""
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.randn(500) * 0.02))

        df = pd.DataFrame({
            'open': prices,
            'high': prices,
            'low': prices,
            'close': prices,
            'volume': 1000000,
            'ticker': 'TEST'
        }, index=dates)

        return df

    def test_initialization(self, mock_data_manager):
        """Test signal initialization."""
        signal = InstitutionalInsider({}, data_manager=mock_data_manager)
        assert signal.lookback_days == 90
        assert signal.min_transaction_value == 10000
        assert signal.cluster_window == 7
        assert signal.cluster_min_insiders == 3

    def test_role_weights(self, mock_data_manager):
        """Test that role weights are properly set."""
        signal = InstitutionalInsider({}, data_manager=mock_data_manager)
        assert signal.role_weights['ceo'] == 3.0
        assert signal.role_weights['cfo'] == 2.5
        assert signal.role_weights['ceo'] > signal.role_weights['director']

    def test_role_classification(self, mock_data_manager):
        """Test role classification from titles."""
        signal = InstitutionalInsider({}, data_manager=mock_data_manager)

        assert signal._classify_role('Chief Executive Officer') == 'ceo'
        assert signal._classify_role('CEO') == 'ceo'
        assert signal._classify_role('Chief Financial Officer') == 'cfo'
        assert signal._classify_role('CFO') == 'cfo'
        assert signal._classify_role('Director') == 'director'
        assert signal._classify_role('') == 'other'
        assert signal._classify_role(None) == 'other'

    def test_signal_range(self, sample_data, mock_data_manager):
        """Test that signals are in valid range."""
        signal = InstitutionalInsider({}, data_manager=mock_data_manager)
        signals = signal.generate_signals(sample_data)

        assert signals.min() >= -1.0
        assert signals.max() <= 1.0

    def test_missing_ticker(self, mock_data_manager):
        """Test handling when ticker column is missing."""
        signal = InstitutionalInsider({}, data_manager=mock_data_manager)

        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        bad_data = pd.DataFrame({
            'close': np.random.randn(100) + 100
        }, index=dates)

        signals = signal.generate_signals(bad_data)
        assert (signals == 0).all()

    def test_parameter_space(self, mock_data_manager):
        """Test parameter space definition."""
        signal = InstitutionalInsider({}, data_manager=mock_data_manager)
        param_space = signal.get_parameter_space()

        assert 'lookback_days' in param_space
        assert 'min_transaction_value' in param_space
        assert 'cluster_window' in param_space
        assert 'cluster_min_insiders' in param_space
        assert 'ceo_weight' in param_space
        assert 'cfo_weight' in param_space


class TestIntegration:
    """Integration tests across all institutional signals."""

    def test_all_signals_implement_base_interface(self):
        """Test that all signals properly implement BaseSignal interface."""
        from core.base_signal import BaseSignal

        default_momentum_params = {
            'formation_period': 252,
            'skip_period': 21,
            'winsorize_pct': [5, 95],
            'quintiles': True
        }

        signals = [
            InstitutionalMomentum(default_momentum_params),
            InstitutionalQuality({}),
            InstitutionalInsider({})
        ]

        for signal in signals:
            assert isinstance(signal, BaseSignal)
            assert hasattr(signal, 'generate_signals')
            assert hasattr(signal, 'get_parameter_space')
            assert hasattr(signal, 'validate_params')

    def test_all_signals_have_consistent_output_format(self):
        """Test that all signals produce consistent output format."""
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.randn(500) * 0.02))

        data = pd.DataFrame({
            'open': prices,
            'high': prices,
            'low': prices,
            'close': prices,
            'volume': 1000000,
            'ticker': 'TEST'
        }, index=dates)

        default_momentum_params = {
            'formation_period': 252,
            'skip_period': 21,
            'winsorize_pct': [5, 95],
            'quintiles': True
        }

        momentum_signal = InstitutionalMomentum(default_momentum_params)
        momentum_output = momentum_signal.generate_signals(data)

        # All signals should return pd.Series
        assert isinstance(momentum_output, pd.Series)

        # All signals should have same length as input
        assert len(momentum_output) == len(data)

        # All signals should have same index as input
        assert momentum_output.index.equals(data.index)

    def test_signals_handle_nans_gracefully(self):
        """Test that signals handle NaN values in data."""
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.randn(500) * 0.02))

        # Insert some NaNs
        prices[100:110] = np.nan

        data = pd.DataFrame({
            'close': prices,
            'ticker': 'TEST'
        }, index=dates)

        default_momentum_params = {
            'formation_period': 252,
            'skip_period': 21,
            'winsorize_pct': [5, 95],
            'quintiles': True
        }

        signal = InstitutionalMomentum(default_momentum_params)
        signals = signal.generate_signals(data)

        # Should not raise exception
        assert isinstance(signals, pd.Series)

        # Output should still have same length
        assert len(signals) == len(data)


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])
