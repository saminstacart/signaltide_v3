"""
Tests for BaseSignal and signal validation.
"""

import pytest
import pandas as pd
import numpy as np
from core.base_signal import ExampleMomentumSignal


class TestExampleMomentumSignal:
    """Test ExampleMomentumSignal implementation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        return data

    def test_signal_initialization(self):
        """Test signal can be initialized with parameters."""
        params = {'lookback': 20, 'threshold': 0.1}
        signal = ExampleMomentumSignal(params)
        assert signal.params == params
        assert signal.name == 'ExampleMomentumSignal'

    def test_generate_signals(self, sample_data):
        """Test signal generation."""
        params = {'lookback': 20, 'threshold': 0.1}
        signal = ExampleMomentumSignal(params)

        signals = signal.generate_signals(sample_data)

        # Check output is Series with same index
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_data)
        assert signals.index.equals(sample_data.index)

        # Check values are in [-1, 1]
        assert signals.min() >= -1.0
        assert signals.max() <= 1.0

    def test_no_lookahead_bias(self, sample_data):
        """Test that signal doesn't use future data."""
        params = {'lookback': 20, 'threshold': 0.1}
        signal = ExampleMomentumSignal(params)

        signals = signal.generate_signals(sample_data)

        # Validate no lookahead
        assert signal.validate_no_lookahead(sample_data, signals)

    def test_parameter_space(self):
        """Test parameter space definition."""
        params = {'lookback': 20, 'threshold': 0.1}
        signal = ExampleMomentumSignal(params)

        param_space = signal.get_parameter_space()

        assert 'lookback' in param_space
        assert 'threshold' in param_space
        assert param_space['lookback'][0] == 'int'
        assert param_space['threshold'][0] == 'float'

    def test_invalid_params_raise_error(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            params = {'lookback': -1, 'threshold': 0.1}
            signal = ExampleMomentumSignal(params)
