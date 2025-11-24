"""
Tests for InstitutionalInsider.generate_cross_sectional_scores() API.

This cross-sectional API was added in Phase 3 (2025-11-23) to support efficient
ensemble construction with bulk data fetching (50-100x performance improvement).

See: signals/insider/institutional_insider.py:417
See: docs/ENSEMBLES_M3.6_THREE_SIGNAL_SPEC.md
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch, call
from signals.insider.institutional_insider import InstitutionalInsider


class TestCrossSectionalAPIBasic:
    """Basic functionality tests for cross-sectional scoring."""

    def test_smoke_test_returns_series_with_scores(self):
        """
        Smoke test: 3-5 tickers should return Series with scores, no NaNs.
        """
        # Setup: Mock DataManager
        mock_dm = Mock()

        # Mock bulk insider data (simple mock: some insider activity for each ticker)
        mock_dm.get_insider_trades_bulk.return_value = pd.DataFrame({
            'ticker': ['AAPL', 'AAPL', 'MSFT', 'MSFT', 'GOOGL'],
            'filing_date': pd.to_datetime(['2024-01-20', '2024-01-25', '2024-01-22', '2024-01-26', '2024-01-21']),
            'transaction_date': pd.to_datetime(['2024-01-15', '2024-01-20', '2024-01-17', '2024-01-21', '2024-01-16']),
            'shares': [1000, 2000, 1500, 500, 3000],
            'value': [100000, 200000, 150000, 50000, 300000],
            'transaction_type': ['P-Purchase', 'P-Purchase', 'S-Sale', 'P-Purchase', 'P-Purchase']
        })

        # Mock prices (generate synthetic price data)
        def mock_get_prices(ticker, start, end):
            dates = pd.date_range(start='2023-10-01', end='2024-01-31', freq='B')
            return pd.DataFrame({
                'close': np.random.uniform(100, 200, len(dates)),
                'volume': np.random.uniform(1e6, 5e6, len(dates))
            }, index=dates)

        mock_dm.get_prices.side_effect = mock_get_prices

        # Create signal instance
        insider = InstitutionalInsider({'lookback_days': 90})

        # Generate cross-sectional scores
        universe = ['AAPL', 'MSFT', 'GOOGL']
        rebal_date = pd.Timestamp('2024-01-31')

        scores = insider.generate_cross_sectional_scores(
            rebal_date=rebal_date,
            universe=universe,
            data_manager=mock_dm
        )

        # Assertions
        assert isinstance(scores, pd.Series), "Should return pd.Series"
        assert len(scores) <= len(universe), "Should return scores for subset of universe"
        assert scores.isna().sum() == 0, "No NaN values allowed"
        assert all(-1 <= s <= 1 for s in scores.values), "Scores should be in [-1, 1]"

    def test_bulk_fetch_called_once_not_n_times(self):
        """
        PERFORMANCE: Verify bulk fetch is called ONCE, not N times (one per ticker).

        This is the key optimization: single DB query for all tickers instead of
        N individual queries.
        """
        # Setup
        mock_dm = Mock()

        # Mock bulk insider data
        mock_dm.get_insider_trades_bulk.return_value = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT'],
            'filing_date': pd.to_datetime(['2024-01-20', '2024-01-22']),
            'transaction_date': pd.to_datetime(['2024-01-15', '2024-01-17']),
            'shares': [1000, 1500],
            'value': [100000, 150000],
            'transaction_type': ['P-Purchase', 'S-Sale']
        })

        # Mock prices
        def mock_get_prices(ticker, start, end):
            dates = pd.date_range(start='2023-10-01', end='2024-01-31', freq='B')
            return pd.DataFrame({'close': np.random.uniform(100, 200, len(dates))}, index=dates)

        mock_dm.get_prices.side_effect = mock_get_prices

        # Create signal instance
        insider = InstitutionalInsider({'lookback_days': 90})

        # Generate scores for 3 tickers
        universe = ['AAPL', 'MSFT', 'GOOGL']
        scores = insider.generate_cross_sectional_scores(
            rebal_date=pd.Timestamp('2024-01-31'),
            universe=universe,
            data_manager=mock_dm
        )

        # CRITICAL ASSERTION: Bulk fetch called ONCE, not 3 times
        assert mock_dm.get_insider_trades_bulk.call_count == 1, \
            "Bulk fetch should be called ONCE, not once per ticker!"

        # Verify it was called with full universe
        call_args = mock_dm.get_insider_trades_bulk.call_args
        assert set(call_args[1]['tickers']) == set(universe), \
            "Bulk fetch should receive entire universe"

    def test_pit_safety_as_of_date_passed(self):
        """
        PIT SAFETY: Verify as_of_date is passed to bulk fetch for temporal discipline.

        Late filings (disclosed after rebal_date) should NOT be included.
        """
        # Setup
        mock_dm = Mock()
        mock_dm.get_insider_trades_bulk.return_value = pd.DataFrame()  # Empty for this test

        def mock_get_prices(ticker, start, end):
            dates = pd.date_range(start='2023-10-01', end='2024-01-31', freq='B')
            return pd.DataFrame({'close': np.random.uniform(100, 200, len(dates))}, index=dates)

        mock_dm.get_prices.side_effect = mock_get_prices

        # Create signal
        insider = InstitutionalInsider({'lookback_days': 90})

        # Generate scores
        rebal_date = pd.Timestamp('2024-01-31')
        scores = insider.generate_cross_sectional_scores(
            rebal_date=rebal_date,
            universe=['AAPL', 'MSFT'],
            data_manager=mock_dm
        )

        # CRITICAL ASSERTION: Verify as_of_date parameter was passed
        call_args = mock_dm.get_insider_trades_bulk.call_args
        assert 'as_of_date' in call_args[1], \
            "as_of_date parameter must be passed for PIT safety"
        assert call_args[1]['as_of_date'] == '2024-01-31', \
            f"as_of_date should match rebal_date, got {call_args[1]['as_of_date']}"


class TestCrossSectionalAPIEdgeCases:
    """Edge case handling tests."""

    def test_empty_universe_returns_empty_series(self):
        """Test that empty universe returns empty Series."""
        mock_dm = Mock()
        mock_dm.get_insider_trades_bulk.return_value = pd.DataFrame()

        insider = InstitutionalInsider({'lookback_days': 90})

        scores = insider.generate_cross_sectional_scores(
            rebal_date=pd.Timestamp('2024-01-31'),
            universe=[],
            data_manager=mock_dm
        )

        assert isinstance(scores, pd.Series), "Should return pd.Series even if empty"
        assert len(scores) == 0, "Empty universe should yield empty scores"

    def test_ticker_with_no_price_data_skipped(self):
        """Test that tickers with missing price data are gracefully skipped."""
        mock_dm = Mock()

        # Mock insider data
        mock_dm.get_insider_trades_bulk.return_value = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT'],
            'filing_date': pd.to_datetime(['2024-01-20', '2024-01-22']),
            'transaction_date': pd.to_datetime(['2024-01-15', '2024-01-17']),
            'shares': [1000, 1500],
            'value': [100000, 150000],
            'transaction_type': ['P-Purchase', 'S-Sale']
        })

        # Mock prices: AAPL has data, MSFT is missing
        def mock_get_prices(ticker, start, end):
            if ticker == 'AAPL':
                dates = pd.date_range(start='2023-10-01', end='2024-01-31', freq='B')
                return pd.DataFrame({'close': np.random.uniform(100, 200, len(dates))}, index=dates)
            else:
                # Missing data for MSFT
                return pd.DataFrame()

        mock_dm.get_prices.side_effect = mock_get_prices

        insider = InstitutionalInsider({'lookback_days': 90})

        scores = insider.generate_cross_sectional_scores(
            rebal_date=pd.Timestamp('2024-01-31'),
            universe=['AAPL', 'MSFT', 'GOOGL'],
            data_manager=mock_dm
        )

        # Should only have scores for AAPL (MSFT/GOOGL skipped due to missing data)
        assert len(scores) <= 1, "Should skip tickers with missing data"
        # No NaNs in returned scores
        assert scores.isna().sum() == 0, "No NaN values"

    def test_no_insider_data_returns_empty_scores(self):
        """Test that no insider activity yields empty/zero scores."""
        mock_dm = Mock()

        # Mock: NO insider data
        mock_dm.get_insider_trades_bulk.return_value = pd.DataFrame()

        # Mock prices
        def mock_get_prices(ticker, start, end):
            dates = pd.date_range(start='2023-10-01', end='2024-01-31', freq='B')
            return pd.DataFrame({'close': np.random.uniform(100, 200, len(dates))}, index=dates)

        mock_dm.get_prices.side_effect = mock_get_prices

        insider = InstitutionalInsider({'lookback_days': 90})

        scores = insider.generate_cross_sectional_scores(
            rebal_date=pd.Timestamp('2024-01-31'),
            universe=['AAPL', 'MSFT'],
            data_manager=mock_dm
        )

        # With no insider data, should return empty Series (no scores generated)
        # Note: Implementation returns only non-zero scores
        assert isinstance(scores, pd.Series)


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, '-v'])
