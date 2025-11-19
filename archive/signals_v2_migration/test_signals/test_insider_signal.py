"""
Tests for InsiderSignal.

Note: These tests use mock data. Full integration tests would require
actual Sharadar insider trading data in the database.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock

from signals.insider.insider_signal import InsiderSignal


class TestInsiderSignal:
    """Test InsiderSignal implementation."""

    @pytest.fixture
    def default_params(self):
        """Default parameters for insider signal."""
        return {
            'lookback_days': 180,
            'aggregation_window': 30,
            'use_recency_weight': True,
            'recency_halflife': 30,
            'use_cluster_bonus': True,
            'cluster_multiplier': 1.5,
            'signal_threshold': 0.0,
            'rank_window': 252
        }

    @pytest.fixture
    def mock_data_manager(self):
        """Create mock DataManager for testing."""
        mock_dm = Mock()

        # Create mock insider trading data
        dates = pd.date_range('2020-01-01', periods=10, freq='W')
        mock_insider_trades = pd.DataFrame({
            'filing_date': dates,
            'trade_date': dates - pd.Timedelta(days=2),
            'insider_name': [f'Insider_{i}' for i in range(10)],
            'insider_title': ['CEO', 'CFO', 'Director', 'VP', 'Officer'] * 2,
            'transaction_type': ['P', 'P', 'P', 'S', 'P', 'P', 'S', 'P', 'P', 'S'],
            'shares': [10000, 5000, 2000, 8000, 3000, 6000, 4000, 7000, 1000, 5000],
            'price_per_share': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'shares_owned_after': [50000] * 10
        })

        mock_dm.get_insider_trades.return_value = mock_insider_trades

        return mock_dm

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data."""
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        data = pd.DataFrame({
            'open': 100 + np.random.randn(252).cumsum(),
            'high': 102 + np.random.randn(252).cumsum(),
            'low': 98 + np.random.randn(252).cumsum(),
            'close': 100 + np.random.randn(252).cumsum(),
            'volume': np.random.randint(1000000, 5000000, 252),
            'ticker': 'AAPL'
        }, index=dates)
        return data

    def test_signal_initialization(self, default_params):
        """Test signal can be initialized."""
        signal = InsiderSignal(default_params)
        assert signal.name == 'InsiderSignal'
        assert hasattr(signal, 'title_weights')

    def test_parameter_space_defined(self, default_params):
        """Test parameter space is properly defined."""
        signal = InsiderSignal(default_params)
        param_space = signal.get_parameter_space()

        assert 'lookback_days' in param_space
        assert 'aggregation_window' in param_space
        assert 'use_recency_weight' in param_space
        assert 'use_cluster_bonus' in param_space

    def test_generate_signals_with_mock_data(self, sample_price_data,
                                            default_params, mock_data_manager):
        """Test signal generation with mocked insider data."""
        signal = InsiderSignal(default_params, data_manager=mock_data_manager)
        signals = signal.generate_signals(sample_price_data)

        # Should return signals with same index
        assert len(signals) == len(sample_price_data)
        assert signals.index.equals(sample_price_data.index)

        # Signals should be in range
        assert signals.min() >= -1.0
        assert signals.max() <= 1.0

    def test_trade_scoring(self, default_params):
        """Test individual trade scoring."""
        signal = InsiderSignal(default_params)

        # Test purchase by CEO
        purchase = pd.Series({
            'transaction_type': 'P',
            'insider_title': 'CEO',
            'shares': 10000,
            'price_per_share': 100
        })

        score = signal._score_trade(purchase)
        assert score > 0  # Purchase is positive

        # Test sale by VP
        sale = pd.Series({
            'transaction_type': 'S',
            'insider_title': 'VP',
            'shares': 5000,
            'price_per_share': 100
        })

        score = signal._score_trade(sale)
        assert score < 0  # Sale is negative

    def test_title_weighting(self, default_params):
        """Test that different titles have different weights."""
        signal = InsiderSignal(default_params)

        ceo_weight = signal._get_title_weight('CEO')
        director_weight = signal._get_title_weight('Director')
        vp_weight = signal._get_title_weight('VP')

        # CEO should have highest weight
        assert ceo_weight > director_weight
        assert director_weight >= vp_weight

    def test_no_ticker_returns_zeros(self, default_params):
        """Test that data without ticker returns zero signals."""
        signal = InsiderSignal(default_params)

        # Data without ticker column
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100
        }, index=dates)

        signals = signal.generate_signals(data)

        assert (signals == 0).all()

    def test_aggregation_window_validation(self):
        """Test that aggregation window validation works."""
        bad_params = {
            'lookback_days': 90,
            'aggregation_window': 180,  # Bigger than lookback!
            'use_recency_weight': True,
            'recency_halflife': 30,
            'use_cluster_bonus': True,
            'cluster_multiplier': 1.5,
            'signal_threshold': 0.0,
            'rank_window': 252
        }

        with pytest.raises(ValueError, match="aggregation_window"):
            InsiderSignal(bad_params)

    def test_repr_string(self, default_params):
        """Test string representation."""
        signal = InsiderSignal(default_params)
        repr_str = repr(signal)

        assert 'InsiderSignal' in repr_str
        assert 'lookback=' in repr_str
        assert 'window=' in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
