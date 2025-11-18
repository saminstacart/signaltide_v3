"""
Pytest configuration and shared fixtures.
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')

    close = 100 + np.random.randn(500).cumsum()
    high = close + np.abs(np.random.randn(500))
    low = close - np.abs(np.random.randn(500))
    open_price = close + np.random.randn(500) * 0.5

    data = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(1000, 10000, 500)
    }, index=dates)

    return data


@pytest.fixture
def sample_returns():
    """Create sample returns for testing."""
    np.random.seed(42)
    return pd.Series(np.random.randn(252) * 0.01)
