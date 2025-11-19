"""
Tests for Simple Signals

Uses mock data to test signals without requiring database access.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data.mock_generator import create_test_data, MockDataGenerator
from signals.momentum.simple_momentum import SimpleMomentum
from signals.quality.simple_quality import SimpleQuality
from signals.insider.simple_insider import SimpleInsider
from data.data_manager import create_mock_data_manager


class TestSimpleMomentum:
    """Test SimpleMomentum signal."""

    @pytest.fixture
    def sample_data(self):
        """Create sample price data."""
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
            'volume': np.random.randint(1000000, 5000000, 500),
            'ticker': 'TEST'
        }, index=dates)

        return data

    def test_signal_initialization(self):
        """Test signal can be initialized."""
        params = {'lookback': 90, 'rank_window': 252}
        signal = SimpleMomentum(params)
        assert signal.name == 'SimpleMomentum'
        assert signal.params == params

    def test_generate_signals_shape(self, sample_data):
        """Test generated signals have correct shape."""
        params = {'lookback': 90, 'rank_window': 252}
        signal = SimpleMomentum(params)
        signals = signal.generate_signals(sample_data)

        assert len(signals) == len(sample_data)
        assert signals.index.equals(sample_data.index)

    def test_signals_in_range(self, sample_data):
        """Test signals are in [-1, 1] range."""
        params = {'lookback': 90, 'rank_window': 252}
        signal = SimpleMomentum(params)
        signals = signal.generate_signals(sample_data)

        assert signals.min() >= -1.0
        assert signals.max() <= 1.0

    def test_uptrend_gives_signal(self):
        """Test that strong uptrend generates consistent signals."""
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        prices = pd.DataFrame({
            'close': np.linspace(100, 200, 300),  # Perfect uptrend
            'ticker': 'TEST'
        }, index=dates)

        params = {'lookback': 60, 'rank_window': 200}
        signal = SimpleMomentum(params)
        signals = signal.generate_signals(prices)

        # Signals should be non-zero and consistent (all same sign)
        late_signals = signals.iloc[-50:].dropna()
        assert len(late_signals) > 0
        assert abs(late_signals.mean()) > 0.5  # Strong consistent signal

    def test_parameter_space_defined(self):
        """Test parameter space is properly defined."""
        params = {'lookback': 90, 'rank_window': 252}
        signal = SimpleMomentum(params)
        param_space = signal.get_parameter_space()

        assert 'lookback' in param_space
        assert 'rank_window' in param_space
        assert param_space['lookback'][0] == 'int'

    def test_repr(self):
        """Test string representation."""
        params = {'lookback': 90, 'rank_window': 252}
        signal = SimpleMomentum(params)
        repr_str = repr(signal)
        assert 'SimpleMomentum' in repr_str
        assert '90' in repr_str


class TestSimpleQuality:
    """Test SimpleQuality signal."""

    @pytest.fixture
    def sample_data_with_fundamentals(self):
        """Create sample data with fundamentals."""
        generator = MockDataGenerator()

        start = datetime(2020, 1, 1)
        end = datetime(2024, 12, 31)

        prices = generator.generate_price_data('TEST01', start, end)
        fundamentals = generator.generate_fundamentals('TEST01', start, end)

        return prices, fundamentals

    def test_signal_initialization(self):
        """Test signal can be initialized."""
        params = {'rank_window': 252 * 2}
        signal = SimpleQuality(params, data_manager=create_mock_data_manager())
        assert signal.name == 'SimpleQuality'

    def test_generate_signals_with_mock_data(self, sample_data_with_fundamentals):
        """Test signal generation with mock data."""
        prices, _ = sample_data_with_fundamentals

        params = {'rank_window': 252 * 2}
        mock_dm = create_mock_data_manager()
        signal = SimpleQuality(params, data_manager=mock_dm)

        signals = signal.generate_signals(prices)

        assert len(signals) == len(prices)
        assert signals.min() >= -1.0
        assert signals.max() <= 1.0

    def test_no_ticker_returns_zeros(self):
        """Test that data without ticker returns zero signals."""
        params = {'rank_window': 252 * 2}
        signal = SimpleQuality(params, data_manager=create_mock_data_manager())

        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100
        }, index=dates)

        signals = signal.generate_signals(data)
        assert (signals == 0).all()

    def test_repr(self):
        """Test string representation."""
        params = {'rank_window': 252 * 2}
        signal = SimpleQuality(params, data_manager=create_mock_data_manager())
        repr_str = repr(signal)
        assert 'SimpleQuality' in repr_str
        assert 'ROE' in repr_str


class TestSimpleInsider:
    """Test SimpleInsider signal."""

    def test_signal_initialization(self):
        """Test signal can be initialized."""
        params = {'lookback_days': 90, 'rank_window': 252}
        signal = SimpleInsider(params, data_manager=create_mock_data_manager())
        assert signal.name == 'SimpleInsider'

    def test_generate_signals_with_mock_data(self):
        """Test signal generation with mock data."""
        generator = MockDataGenerator()

        start = datetime(2020, 1, 1)
        end = datetime(2024, 12, 31)

        prices = generator.generate_price_data('TEST01', start, end)

        params = {'lookback_days': 90, 'rank_window': 252}
        mock_dm = create_mock_data_manager()
        signal = SimpleInsider(params, data_manager=mock_dm)

        signals = signal.generate_signals(prices)

        assert len(signals) == len(prices)
        assert signals.min() >= -1.0
        assert signals.max() <= 1.0

    def test_no_ticker_returns_zeros(self):
        """Test that data without ticker returns zero signals."""
        params = {'lookback_days': 90, 'rank_window': 252}
        signal = SimpleInsider(params, data_manager=create_mock_data_manager())

        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100
        }, index=dates)

        signals = signal.generate_signals(data)
        assert (signals == 0).all()

    def test_parameter_space_defined(self):
        """Test parameter space is properly defined."""
        params = {'lookback_days': 90, 'rank_window': 252}
        signal = SimpleInsider(params, data_manager=create_mock_data_manager())
        param_space = signal.get_parameter_space()

        assert 'lookback_days' in param_space
        assert 'rank_window' in param_space

    def test_repr(self):
        """Test string representation."""
        params = {'lookback_days': 90, 'rank_window': 252}
        signal = SimpleInsider(params, data_manager=create_mock_data_manager())
        repr_str = repr(signal)
        assert 'SimpleInsider' in repr_str
        assert '90' in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
