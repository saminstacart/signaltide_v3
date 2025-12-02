"""
Unit tests for SignalTide v4 signals.

Tests:
- Signal generation
- Cross-sectional normalization
- PIT compliance
"""

import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np

# Import signal classes (will fail until we have real data providers)
try:
    from signaltide_v4.signals.residual_momentum import ResidualMomentumSignal
    from signaltide_v4.signals.quality import QualitySignal
    from signaltide_v4.signals.insider import OpportunisticInsiderSignal
    from signaltide_v4.signals.tone_change import ToneChangeSignal
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


class TestSignalBase(unittest.TestCase):
    """Test base signal functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        self.as_of_date = '2023-12-31'

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Imports not available")
    def test_residual_momentum_init(self):
        """Test ResidualMomentumSignal initialization."""
        with patch('signaltide_v4.signals.residual_momentum.MarketDataProvider'):
            with patch('signaltide_v4.signals.residual_momentum.FactorDataProvider'):
                signal = ResidualMomentumSignal()
                self.assertEqual(signal.signal_name, 'residual_momentum')

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Imports not available")
    def test_quality_signal_init(self):
        """Test QualitySignal initialization."""
        with patch('signaltide_v4.signals.quality.MarketDataProvider'):
            with patch('signaltide_v4.signals.quality.FundamentalDataProvider'):
                signal = QualitySignal()
                self.assertEqual(signal.signal_name, 'quality')

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Imports not available")
    def test_insider_signal_init(self):
        """Test OpportunisticInsiderSignal initialization."""
        with patch('signaltide_v4.signals.insider.MarketDataProvider'):
            signal = OpportunisticInsiderSignal()
            self.assertEqual(signal.signal_name, 'opportunistic_insider')

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Imports not available")
    def test_tone_signal_init(self):
        """Test ToneChangeSignal initialization."""
        signal = ToneChangeSignal()
        self.assertEqual(signal.signal_name, 'tone_change')


class TestCrossSectionalNormalization(unittest.TestCase):
    """Test cross-sectional normalization."""

    def test_quintile_normalization(self):
        """Test quintile-based normalization."""
        # Create sample scores
        scores = pd.Series({
            'A': 10.0, 'B': 20.0, 'C': 30.0, 'D': 40.0, 'E': 50.0,
            'F': 60.0, 'G': 70.0, 'H': 80.0, 'I': 90.0, 'J': 100.0,
        })

        # Compute quintiles
        quintiles = pd.qcut(scores, q=5, labels=False, duplicates='drop')
        normalized = (quintiles - 2) / 2  # Maps 0-4 to -1 to 1

        # Verify range
        self.assertGreaterEqual(normalized.min(), -1.0)
        self.assertLessEqual(normalized.max(), 1.0)

    def test_zscore_normalization(self):
        """Test z-score normalization."""
        scores = pd.Series({
            'A': 10.0, 'B': 20.0, 'C': 30.0, 'D': 40.0, 'E': 50.0,
        })

        mean = scores.mean()
        std = scores.std()
        zscore = (scores - mean) / std

        # Z-scores should have mean ~0 and std ~1
        self.assertAlmostEqual(zscore.mean(), 0.0, places=5)
        self.assertAlmostEqual(zscore.std(), 1.0, places=5)


class TestPITCompliance(unittest.TestCase):
    """Test point-in-time compliance."""

    def test_date_filtering(self):
        """Test that data is correctly filtered by date."""
        # Sample data with filing dates
        data = pd.DataFrame({
            'ticker': ['AAPL', 'AAPL', 'MSFT', 'MSFT'],
            'filing_date': ['2023-01-15', '2023-04-15', '2023-02-01', '2023-05-01'],
            'value': [100, 110, 200, 210],
        })
        data['filing_date'] = pd.to_datetime(data['filing_date'])

        # Filter for as_of_date = 2023-03-01
        as_of = pd.Timestamp('2023-03-01')
        pit_data = data[data['filing_date'] <= as_of]

        # Should only have AAPL Q1 and MSFT Q1
        self.assertEqual(len(pit_data), 2)
        self.assertIn('AAPL', pit_data['ticker'].values)
        self.assertIn('MSFT', pit_data['ticker'].values)


if __name__ == '__main__':
    unittest.main()
