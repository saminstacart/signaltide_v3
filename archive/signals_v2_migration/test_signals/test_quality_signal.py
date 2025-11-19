"""
Tests for QualitySignal.

Note: These tests use mock data. Full integration tests would require
actual Sharadar fundamental data in the database.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, MagicMock

from signals.quality.quality_signal import QualitySignal


class TestQualitySignal:
    """Test QualitySignal implementation."""

    @pytest.fixture
    def default_params(self):
        """Default parameters for quality signal."""
        return {
            'use_roe': True,
            'use_roa': True,
            'use_margins': True,
            'use_accruals': True,
            'use_leverage': True,
            'use_liquidity': True,
            'roe_weight': 0.2,
            'roa_weight': 0.15,
            'margin_weight': 0.2,
            'accrual_weight': 0.15,
            'leverage_weight': 0.15,
            'liquidity_weight': 0.15,
            'signal_threshold': 0.0,
            'rank_window': 252
        }

    @pytest.fixture
    def mock_data_manager(self):
        """Create mock DataManager for testing."""
        mock_dm = Mock()

        # Create mock fundamental data
        dates = pd.date_range('2020-01-01', periods=4, freq='Q')
        mock_fundamentals = pd.DataFrame({
            'roe': [0.15, 0.18, 0.20, 0.22],
            'roa': [0.08, 0.09, 0.10, 0.11],
            'revenue': [1e9, 1.1e9, 1.2e9, 1.3e9],
            'gross_profit': [0.4e9, 0.45e9, 0.50e9, 0.55e9],
            'operating_income': [0.2e9, 0.22e9, 0.24e9, 0.26e9],
            'net_income': [0.15e9, 0.17e9, 0.19e9, 0.21e9],
            'operating_cash_flow': [0.18e9, 0.20e9, 0.22e9, 0.24e9],
            'total_assets': [2e9, 2.1e9, 2.2e9, 2.3e9],
            'debt_to_equity': [0.5, 0.48, 0.46, 0.44],
            'current_ratio': [1.8, 1.85, 1.9, 1.95],
            'quick_ratio': [1.2, 1.25, 1.3, 1.35]
        }, index=dates)

        mock_dm.get_fundamental_data.return_value = mock_fundamentals

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
        signal = QualitySignal(default_params)
        assert signal.name == 'QualitySignal'

    def test_parameter_space_defined(self, default_params):
        """Test parameter space is properly defined."""
        signal = QualitySignal(default_params)
        param_space = signal.get_parameter_space()

        assert 'use_roe' in param_space
        assert 'use_roa' in param_space
        assert 'roe_weight' in param_space

    def test_generate_signals_with_mock_data(self, sample_price_data,
                                            default_params, mock_data_manager):
        """Test signal generation with mocked fundamental data."""
        signal = QualitySignal(default_params, data_manager=mock_data_manager)
        signals = signal.generate_signals(sample_price_data)

        # Should return signals with same index
        assert len(signals) == len(sample_price_data)
        assert signals.index.equals(sample_price_data.index)

        # Signals should be in range
        assert signals.min() >= -1.0
        assert signals.max() <= 1.0

    def test_no_ticker_returns_zeros(self, default_params):
        """Test that data without ticker returns zero signals."""
        signal = QualitySignal(default_params)

        # Data without ticker column
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100
        }, index=dates)

        signals = signal.generate_signals(data)

        assert (signals == 0).all()

    def test_quality_scoring_methods(self, default_params, mock_data_manager):
        """Test individual quality scoring methods."""
        signal = QualitySignal(default_params, data_manager=mock_data_manager)

        # Create test fundamental data
        fundamentals = pd.DataFrame({
            'roe': [0.15],
            'roa': [0.08],
            'revenue': [1e9],
            'gross_profit': [0.4e9],
            'debt_to_equity': [0.5],
            'current_ratio': [1.8]
        })

        # Test ROE scoring
        roe_score = signal._score_roe(fundamentals)
        assert 0 <= roe_score.iloc[0] <= 1

        # Test ROA scoring
        roa_score = signal._score_roa(fundamentals)
        assert 0 <= roa_score.iloc[0] <= 1

        # Test leverage scoring
        leverage_score = signal._score_leverage(fundamentals)
        assert 0 <= leverage_score.iloc[0] <= 1

    def test_repr_string(self, default_params):
        """Test string representation."""
        signal = QualitySignal(default_params)
        repr_str = repr(signal)

        assert 'QualitySignal' in repr_str
        assert 'factors=' in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
