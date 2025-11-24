"""
Integration tests for CrossSectionalQuality backtest wiring.

Tests that CrossSectionalQuality flows through the unified backtest engine
via make_signal_fn adapter without breaking invariants.

Phase 3 Milestone 2: Quality signal backtest integration smoke tests.
"""

import pytest
import pandas as pd
from datetime import datetime
from typing import List

from data.data_manager import DataManager
from core.universe_manager import UniverseManager
from signals.quality.cross_sectional_quality import CrossSectionalQuality
from core.backtest_engine import BacktestConfig, run_backtest
from core.signal_adapters import make_signal_fn


class TestQualityIntegration:
    """Smoke tests for CrossSectionalQuality backtest integration."""

    @pytest.fixture(scope="class")
    def setup(self):
        """Setup test fixtures."""
        dm = DataManager()
        um = UniverseManager(dm)
        return {'dm': dm, 'um': um}

    @pytest.mark.requires_full_db
    def test_cross_sectional_quality_smoke(self, setup):
        """
        Smoke test: CrossSectionalQuality flows through run_backtest via adapter.

        Validates:
        - Signal generation works end-to-end
        - Adapter wiring is correct
        - Backtest produces valid equity curve
        - No exceptions or data integrity issues

        Does NOT validate:
        - Performance metrics (use acceptance tests for that)
        - Tight tolerances or equivalence
        """
        dm = setup['dm']
        um = setup['um']

        # Small universe for smoke test speed
        def universe_fn(rebal_date: str) -> List[str]:
            """Get small S&P 500 sample for smoke test."""
            universe = um.get_universe(
                universe_type='sp500_actual',
                as_of_date=rebal_date,
                min_price=5.0
            )
            if isinstance(universe, pd.Series):
                return universe.tolist()[:50]  # First 50 for speed
            elif isinstance(universe, pd.DataFrame):
                return universe.index.tolist()[:50]
            else:
                return list(universe)[:50]

        # CrossSectionalQuality with default production parameters
        quality_params = {
            'w_profitability': 0.4,
            'w_growth': 0.3,
            'w_safety': 0.3,
            'winsorize_pct': [5, 95],
            'quintiles': True,
            'min_coverage': 0.3,  # Lower for small test universe
        }

        quality = CrossSectionalQuality(quality_params, data_manager=dm)

        # Create signal function via adapter
        signal_fn = make_signal_fn(quality, dm)

        # Short backtest window (6 months for smoke test)
        config = BacktestConfig(
            start_date='2022-01-31',
            end_date='2022-06-30',
            initial_capital=100000.0,
            rebalance_schedule='M',
            long_only=True,
            equal_weight=True,
            track_daily_equity=False,
            data_manager=dm
        )

        # Run backtest
        result = run_backtest(universe_fn, signal_fn, config)

        # Sanity checks (no tight bounds - just ensure it runs)
        assert len(result.equity_curve) > 0, "Equity curve should not be empty"
        assert result.num_rebalances > 0, "Should have at least one rebalance"
        assert result.final_equity > 0, "Final equity should be positive"
        assert not result.equity_curve.isna().all(), "Equity curve should not be all NaN"

        # Basic structure checks
        assert hasattr(result, 'total_return'), "Should have total_return metric"
        assert hasattr(result, 'sharpe'), "Should have sharpe metric"
        assert hasattr(result, 'max_drawdown'), "Should have max_drawdown metric"

        # Loose bounds (just to catch blow-ups, not performance validation)
        assert -0.9 <= result.total_return <= 5.0, (
            f"Total return {result.total_return:.2%} outside sanity bounds"
        )
        assert result.max_drawdown <= 0, "Max drawdown should be negative or zero"
        assert result.max_drawdown >= -0.95, (
            f"Max drawdown {result.max_drawdown:.2%} worse than -95% (catastrophic)"
        )

    def test_quality_adapter_signature(self, setup):
        """
        Test that make_signal_fn produces correct signature for quality signal.

        Validates adapter contract without running full backtest.
        """
        dm = setup['dm']

        quality = CrossSectionalQuality(
            {'w_profitability': 0.4, 'w_growth': 0.3, 'w_safety': 0.3},
            data_manager=dm
        )

        signal_fn = make_signal_fn(quality, dm)

        # Test that signal_fn is callable
        assert callable(signal_fn), "make_signal_fn should return a callable"

        # Test signature: signal_fn(rebal_date: str, tickers: List[str]) -> pd.Series
        test_tickers = ['AAPL', 'MSFT', 'GOOGL']
        test_date = '2022-01-31'

        # This should not raise
        result = signal_fn(test_date, test_tickers)

        # Check return type
        assert isinstance(result, pd.Series), "signal_fn should return pd.Series"

        # Result should be empty or contain subset of tickers (fixture DB limitation)
        assert len(result) <= len(test_tickers), (
            "Result should not have more tickers than input"
        )

        # If any scores returned, they should be floats
        if len(result) > 0:
            assert result.dtype in [float, 'float64'], "Scores should be float type"
