"""
Tests for MomentumSignal.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from signals.momentum.momentum_signal import MomentumSignal


class TestMomentumSignal:
    """Test MomentumSignal implementation."""

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=500, freq='D')

        # Create trending price data
        trend = np.linspace(100, 150, 500)
        noise = np.random.randn(500) * 2
        close = trend + noise

        data = pd.DataFrame({
            'open': close + np.random.randn(500) * 0.5,
            'high': close + np.abs(np.random.randn(500)),
            'low': close - np.abs(np.random.randn(500)),
            'close': close,
            'volume': np.random.randint(1000000, 5000000, 500)
        }, index=dates)

        return data

    @pytest.fixture
    def default_params(self):
        """Default parameters for momentum signal."""
        return {
            'short_lookback': 10,
            'medium_lookback': 30,
            'long_lookback': 90,
            'short_weight': 0.5,
            'medium_weight': 0.3,
            'long_weight': 0.2,
            'volume_confirmation': True,
            'volume_lookback': 20,
            'volume_boost': 0.2,
            'volatility_adjust': True,
            'volatility_window': 20,
            'use_log_returns': False,
            'signal_threshold': 0.0,
            'rank_window': 252
        }

    def test_signal_initialization(self, default_params):
        """Test signal can be initialized."""
        signal = MomentumSignal(default_params)
        assert signal.name == 'MomentumSignal'
        assert signal.params == default_params

    def test_generate_signals_shape(self, sample_price_data, default_params):
        """Test generated signals have correct shape."""
        signal = MomentumSignal(default_params)
        signals = signal.generate_signals(sample_price_data)

        assert len(signals) == len(sample_price_data)
        assert signals.index.equals(sample_price_data.index)

    def test_signals_in_range(self, sample_price_data, default_params):
        """Test signals are in [-1, 1] range."""
        signal = MomentumSignal(default_params)
        signals = signal.generate_signals(sample_price_data)

        assert signals.min() >= -1.0
        assert signals.max() <= 1.0

    def test_no_future_data_used(self, sample_price_data, default_params):
        """Test that signals don't use future data."""
        signal = MomentumSignal(default_params)

        # Generate signals on full dataset
        full_signals = signal.generate_signals(sample_price_data)

        # Generate signals on partial dataset (up to midpoint)
        midpoint = len(sample_price_data) // 2
        partial_data = sample_price_data.iloc[:midpoint]
        partial_signals = signal.generate_signals(partial_data)

        # Signals for overlapping dates should be similar
        # (may differ slightly due to rolling calculations)
        overlap_dates = partial_signals.index
        correlation = partial_signals.corr(full_signals.loc[overlap_dates])

        # Correlation should be very high (>0.95)
        assert correlation > 0.95

    def test_uptrend_gives_positive_signal(self):
        """Test that uptrending prices give positive signals."""
        # Create strong uptrend
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        prices = pd.DataFrame({
            'open': np.linspace(100, 200, 200),
            'high': np.linspace(101, 201, 200),
            'low': np.linspace(99, 199, 200),
            'close': np.linspace(100, 200, 200),
            'volume': [1000000] * 200
        }, index=dates)

        params = {
            'short_lookback': 10,
            'medium_lookback': 30,
            'long_lookback': 90,
            'short_weight': 0.5,
            'medium_weight': 0.3,
            'long_weight': 0.2,
            'volume_confirmation': False,
            'volatility_adjust': False,
            'use_log_returns': False,
            'signal_threshold': 0.0,
            'rank_window': 100
        }

        signal = MomentumSignal(params)
        signals = signal.generate_signals(prices)

        # Later signals should be predominantly positive
        late_signals = signals.iloc[-50:]
        assert late_signals.mean() > 0

    def test_parameter_space_defined(self, default_params):
        """Test parameter space is properly defined."""
        signal = MomentumSignal(default_params)
        param_space = signal.get_parameter_space()

        assert 'short_lookback' in param_space
        assert 'medium_lookback' in param_space
        assert 'long_lookback' in param_space
        assert 'volume_confirmation' in param_space

        # Check types
        assert param_space['short_lookback'][0] == 'int'
        assert param_space['volume_confirmation'][0] == 'categorical'

    def test_invalid_lookback_order_raises_error(self):
        """Test that invalid lookback order raises error."""
        bad_params = {
            'short_lookback': 100,  # Should be smallest
            'medium_lookback': 50,
            'long_lookback': 200,
            'short_weight': 0.5,
            'medium_weight': 0.3,
            'long_weight': 0.2,
            'volume_confirmation': False,
            'volatility_adjust': False,
            'use_log_returns': False,
            'signal_threshold': 0.0,
            'rank_window': 252
        }

        with pytest.raises(ValueError, match="ascending order"):
            MomentumSignal(bad_params)

    def test_volume_confirmation_effect(self, sample_price_data):
        """Test that volume confirmation affects signals."""
        base_params = {
            'short_lookback': 10,
            'medium_lookback': 30,
            'long_lookback': 90,
            'short_weight': 0.5,
            'medium_weight': 0.3,
            'long_weight': 0.2,
            'volume_confirmation': False,
            'volatility_adjust': False,
            'use_log_returns': False,
            'signal_threshold': 0.0,
            'rank_window': 252
        }

        # Without volume confirmation
        signal_no_vol = MomentumSignal(base_params)
        signals_no_vol = signal_no_vol.generate_signals(sample_price_data)

        # With volume confirmation
        vol_params = base_params.copy()
        vol_params['volume_confirmation'] = True
        vol_params['volume_lookback'] = 20
        vol_params['volume_boost'] = 0.5

        signal_with_vol = MomentumSignal(vol_params)
        signals_with_vol = signal_with_vol.generate_signals(sample_price_data)

        # Signals should be different
        assert not signals_no_vol.equals(signals_with_vol)

    def test_repr_string(self, default_params):
        """Test string representation."""
        signal = MomentumSignal(default_params)
        repr_str = repr(signal)

        assert 'MomentumSignal' in repr_str
        assert 'short=10' in repr_str
        assert 'medium=30' in repr_str
        assert 'long=90' in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
