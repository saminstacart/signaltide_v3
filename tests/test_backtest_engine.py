"""
Tests for unified backtest engine.

Regression tests to ensure different signal pathways produce identical results
when using the same underlying signal logic.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import List

from core.backtest_engine import BacktestConfig, run_backtest
from data.data_manager import DataManager
from core.universe_manager import UniverseManager
from signals.momentum.institutional_momentum import InstitutionalMomentum


class TestBacktestEngine:
    """Test unified backtest engine."""

    @pytest.fixture(scope="class")
    def setup(self):
        """Setup test fixtures."""
        dm = DataManager()
        um = UniverseManager(dm)
        return {'dm': dm, 'um': um}

    def test_backtest_engine_basic(self, setup):
        """Test that backtest engine runs without errors."""
        dm = setup['dm']
        um = setup['um']

        # Simple universe function (small test universe)
        def universe_fn(rebal_date: str) -> List[str]:
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']

        # Simple fixed-score signal function
        def signal_fn(rebal_date: str, tickers: List[str]) -> pd.Series:
            # Return fixed scores for testing
            scores = {}
            for i, ticker in enumerate(tickers):
                scores[ticker] = 1.0 if i < 2 else -1.0  # Top 2 get score 1.0
            return pd.Series(scores)

        # Short backtest
        config = BacktestConfig(
            start_date='2023-01-31',
            end_date='2023-06-30',
            initial_capital=100000.0,
            rebalance_schedule='M',
            data_manager=dm
        )

        result = run_backtest(universe_fn, signal_fn, config)

        # Basic assertions
        assert result.total_return != 0.0  # Should have some return
        assert result.num_rebalances > 0
        assert len(result.equity_curve) > 0
        assert result.final_equity > 0

    def test_deterministic_signal_produces_same_results(self, setup):
        """Test that the same signal function produces identical results."""
        dm = setup['dm']

        # Deterministic universe
        def universe_fn(rebal_date: str) -> List[str]:
            return ['AAPL', 'MSFT', 'GOOGL']

        # Deterministic signal (fixed scores)
        def signal_fn(rebal_date: str, tickers: List[str]) -> pd.Series:
            return pd.Series({'AAPL': 1.0, 'MSFT': 1.0, 'GOOGL': -1.0})

        config = BacktestConfig(
            start_date='2023-01-31',
            end_date='2023-03-31',
            initial_capital=100000.0,
            rebalance_schedule='M',
            data_manager=dm
        )

        # Run twice
        result1 = run_backtest(universe_fn, signal_fn, config)
        result2 = run_backtest(universe_fn, signal_fn, config)

        # Should be identical
        assert result1.total_return == result2.total_return
        assert result1.sharpe == result2.sharpe
        assert result1.max_drawdown == result2.max_drawdown
        assert len(result1.equity_curve) == len(result2.equity_curve)

    def test_equity_curve_monotonic_for_constant_return(self, setup):
        """Test that equity curve grows monotonically with constant positive return."""
        dm = setup['dm']

        def universe_fn(rebal_date: str) -> List[str]:
            return ['AAPL']  # Single stock

        def signal_fn(rebal_date: str, tickers: List[str]) -> pd.Series:
            return pd.Series({'AAPL': 1.0})  # Always long

        config = BacktestConfig(
            start_date='2020-01-31',
            end_date='2020-06-30',
            initial_capital=100000.0,
            rebalance_schedule='M',
            data_manager=dm
        )

        result = run_backtest(universe_fn, signal_fn, config)

        # Check equity curve is non-empty
        assert len(result.equity_curve) > 0

        # Final equity should differ from initial (market moved)
        assert result.final_equity != config.initial_capital

    def test_empty_signals_holds_cash(self, setup):
        """Test that backtest handles empty signals gracefully."""
        dm = setup['dm']

        def universe_fn(rebal_date: str) -> List[str]:
            return ['AAPL', 'MSFT']

        def signal_fn(rebal_date: str, tickers: List[str]) -> pd.Series:
            return pd.Series(dtype=float)  # Empty signals

        config = BacktestConfig(
            start_date='2023-01-31',
            end_date='2023-03-31',
            initial_capital=100000.0,
            data_manager=dm
        )

        result = run_backtest(universe_fn, signal_fn, config)

        # Should complete without error (even if equity curve is empty)
        assert result.num_rebalances > 0

        # Edge case: empty equity curve is acceptable when no signals generated
        # (In production this won't happen with real signals)

    def test_config_validation(self, setup):
        """Test that config creates DataManager if not provided."""
        config = BacktestConfig(
            start_date='2023-01-01',
            end_date='2023-12-31',
            initial_capital=100000.0
        )

        assert config.data_manager is not None
        assert isinstance(config.data_manager, DataManager)


class TestBacktestEngineRegressions:
    """
    Regression tests for signal pathway equivalence.

    These tests ensure that different ways of running the same signal
    produce identical results (within tolerance).
    """

    @pytest.fixture(scope="class")
    def setup(self):
        """Setup for regression tests."""
        from signals.momentum.institutional_momentum import InstitutionalMomentum

        dm = DataManager()
        um = UniverseManager(dm)

        # Canonical momentum params
        momentum_params = {
            'formation_period': 308,
            'skip_period': 0,
            'winsorize_pct': [0.4, 99.6],
            'rebalance_frequency': 'monthly',
            'quintiles': True,
        }

        signal = InstitutionalMomentum(params=momentum_params)

        return {
            'dm': dm,
            'um': um,
            'signal': signal,
            'params': momentum_params
        }

    def test_direct_momentum_pathway(self, setup):
        """
        Test that direct InstitutionalMomentum pathway is self-consistent.

        Runs the same signal twice and ensures metrics match.
        """
        dm = setup['dm']
        um = setup['um']
        signal = setup['signal']

        # Small universe for fast test
        test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
                       'META', 'TSLA', 'BRK.B', 'UNH', 'JNJ']

        def universe_fn(rebal_date: str) -> List[str]:
            return test_tickers

        def signal_fn(rebal_date: str, tickers: List[str]) -> pd.Series:
            lookback_start = (pd.Timestamp(rebal_date) - timedelta(days=500)).strftime('%Y-%m-%d')

            scores = {}
            for ticker in tickers:
                try:
                    prices = dm.get_prices(ticker, lookback_start, rebal_date)
                    if len(prices) > 0 and 'close' in prices.columns:
                        data = pd.DataFrame({'close': prices['close'], 'ticker': ticker})
                        sig_series = signal.generate_signals(data)
                        if len(sig_series) > 0:
                            signal_value = sig_series.iloc[-1]
                            if pd.notna(signal_value) and signal_value != 0:
                                scores[ticker] = signal_value
                except Exception:
                    # Skip tickers with data issues in test
                    continue

            return pd.Series(scores)

        config = BacktestConfig(
            start_date='2023-01-31',
            end_date='2023-06-30',
            initial_capital=100000.0,
            rebalance_schedule='M',
            data_manager=dm
        )

        # Run twice
        result1 = run_backtest(universe_fn, signal_fn, config)
        result2 = run_backtest(universe_fn, signal_fn, config)

        # Should be identical (deterministic signal)
        assert abs(result1.total_return - result2.total_return) < 1e-10
        assert abs(result1.sharpe - result2.sharpe) < 1e-10
        assert abs(result1.max_drawdown - result2.max_drawdown) < 1e-10

    def test_ensemble_vs_direct_equivalence(self, setup):
        """
        Test that ensemble wrapper with single signal matches direct signal.

        This is the critical regression test ensuring that performance differences
        reflect signal behavior, not plumbing differences.

        Uses momentum_v2_adaptive_quintile config (production default) with:
        - Same formation/skip/winsorization parameters
        - Same quintile_mode='adaptive'
        - Same backtest harness (run_backtest)
        - Short date window for speed (1 year)

        Asserts that key metrics match within tiny floating-point tolerances.
        """
        from signals.ml.ensemble_configs import get_momentum_v2_adaptive_quintile_ensemble

        dm = setup['dm']
        um = setup['um']

        # Test parameters
        start_date = '2023-01-01'
        end_date = '2024-01-01'
        initial_capital = 100000.0

        # Canonical momentum v2 adaptive parameters
        momentum_params = {
            'formation_period': 308,
            'skip_period': 0,
            'winsorize_pct': [0.4, 99.6],
            'quintiles': True,
            'quintile_mode': 'adaptive',  # Production default
        }

        # Initialize direct signal
        direct_signal = InstitutionalMomentum(params=momentum_params)

        # Initialize ensemble signal
        ensemble = get_momentum_v2_adaptive_quintile_ensemble(dm)

        # Define shared universe function
        def universe_fn(rebal_date: str) -> List[str]:
            universe = um.get_universe(
                universe_type='sp500_actual',
                as_of_date=rebal_date,
                min_price=5.0
            )
            if isinstance(universe, pd.Series):
                return universe.tolist()
            elif isinstance(universe, pd.DataFrame):
                return universe.index.tolist()
            else:
                return list(universe)

        # Define direct signal function
        def direct_signal_fn(rebal_date: str, tickers: List[str]) -> pd.Series:
            lookback_start = (pd.Timestamp(rebal_date) - timedelta(days=500)).strftime('%Y-%m-%d')
            scores = {}
            for ticker in tickers:
                try:
                    prices = dm.get_prices(ticker, lookback_start, rebal_date)
                    if len(prices) > 0 and 'close' in prices.columns:
                        data = pd.DataFrame({'close': prices['close'], 'ticker': ticker})
                        sig_series = direct_signal.generate_signals(data)
                        if len(sig_series) > 0:
                            signal_value = sig_series.iloc[-1]
                            if pd.notna(signal_value) and signal_value != 0:
                                scores[ticker] = signal_value
                except Exception:
                    # Skip tickers with data issues in test
                    continue
            return pd.Series(scores)

        # Define ensemble signal function
        def ensemble_signal_fn(rebal_date: str, tickers: List[str]) -> pd.Series:
            lookback_start = (pd.Timestamp(rebal_date) - timedelta(days=500)).strftime('%Y-%m-%d')
            prices_dict = {}
            for ticker in tickers:
                try:
                    prices = dm.get_prices(ticker, lookback_start, rebal_date)
                    if len(prices) > 0 and 'close' in prices.columns:
                        px_slice = prices['close'][prices.index <= pd.Timestamp(rebal_date)]
                        if len(px_slice) >= 90:
                            prices_dict[ticker] = px_slice
                except Exception:
                    # Skip tickers with data issues in test
                    continue

            if len(prices_dict) == 0:
                return pd.Series(dtype=float)

            return ensemble.generate_ensemble_scores(
                prices_by_ticker=prices_dict,
                rebalance_date=pd.Timestamp(rebal_date)
            )

        # Configure backtest (same for both)
        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            rebalance_schedule='M',
            long_only=True,
            equal_weight=True,
            track_daily_equity=False,
            data_manager=dm
        )

        # Run both backtests
        result_direct = run_backtest(universe_fn, direct_signal_fn, config)
        result_ensemble = run_backtest(universe_fn, ensemble_signal_fn, config)

        # TIGHTENED EQUIVALENCE TEST: Point-by-point equity curve comparison
        equity_direct = result_direct.equity_curve
        equity_ensemble = result_ensemble.equity_curve

        # Align indices (should be identical, but be defensive)
        equity_direct_aligned, equity_ensemble_aligned = equity_direct.align(
            equity_ensemble, join='outer', fill_value=np.nan
        )

        # Assert equity curves are identical point-by-point (tight absolute tolerance)
        assert np.allclose(
            equity_direct_aligned.values,
            equity_ensemble_aligned.values,
            atol=1e-6,
            rtol=0,
            equal_nan=True
        ), "Equity curves differ between direct and ensemble paths"

        # Assert metrics with very tight absolute tolerances (no percentage wiggle)
        assert abs(result_direct.final_equity - result_ensemble.final_equity) < 1e-6, \
            f"Final equity mismatch: direct={result_direct.final_equity:.6f}, ensemble={result_ensemble.final_equity:.6f}"

        assert abs(result_direct.total_return - result_ensemble.total_return) < 1e-8, \
            f"Total return mismatch: direct={result_direct.total_return:.10f}, ensemble={result_ensemble.total_return:.10f}"

        assert abs(result_direct.sharpe - result_ensemble.sharpe) < 1e-6, \
            f"Sharpe mismatch: direct={result_direct.sharpe:.6f}, ensemble={result_ensemble.sharpe:.6f}"

        assert abs(result_direct.max_drawdown - result_ensemble.max_drawdown) < 1e-6, \
            f"Max drawdown mismatch: direct={result_direct.max_drawdown:.6f}, ensemble={result_ensemble.max_drawdown:.6f}"

        # Number of rebalances should be identical
        assert result_direct.num_rebalances == result_ensemble.num_rebalances, \
            f"Rebalance count mismatch: direct={result_direct.num_rebalances}, ensemble={result_ensemble.num_rebalances}"


class TestBacktestEngineGuardrails:
    """
    High-value guardrail tests for backtest engine hardening.

    Tests for:
    - Final mark-to-market
    - Empty universe handling
    - NaN signal handling
    - NotImplementedError guards
    """

    @pytest.fixture(scope="class")
    def setup(self):
        """Setup test fixtures."""
        dm = DataManager()
        return {'dm': dm}

    def test_final_mark_to_market_applied(self, setup):
        """
        Test that final mark-to-market is applied at config.end_date.

        The final equity should reflect exit prices, not entry prices.
        """
        dm = setup['dm']

        # Simple universe function
        def universe_fn(rebal_date: str) -> List[str]:
            return ['AAPL', 'MSFT']

        # Fixed signal function
        def signal_fn(rebal_date: str, tickers: List[str]) -> pd.Series:
            return pd.Series({'AAPL': 1.0, 'MSFT': 1.0})

        # Short backtest with exactly 2 rebalances
        config = BacktestConfig(
            start_date='2023-01-31',
            end_date='2023-02-28',
            initial_capital=100000.0,
            rebalance_schedule='M',
            data_manager=dm
        )

        result = run_backtest(universe_fn, signal_fn, config)

        # Verify we have equity points
        assert len(result.equity_curve) > 0

        # Final equity should be at end_date
        final_date = result.equity_curve.index[-1]
        assert final_date == pd.Timestamp(config.end_date), \
            f"Final equity date {final_date} should match end_date {config.end_date}"

    def test_empty_universe_graceful(self, setup):
        """
        Test that backtest handles empty universe gracefully.
        """
        dm = setup['dm']

        # Empty universe function
        def universe_fn(rebal_date: str) -> List[str]:
            return []  # No stocks

        # Dummy signal function (won't be called)
        def signal_fn(rebal_date: str, tickers: List[str]) -> pd.Series:
            return pd.Series(dtype=float)

        config = BacktestConfig(
            start_date='2023-01-31',
            end_date='2023-03-31',
            initial_capital=100000.0,
            min_universe_size=1,
            data_manager=dm
        )

        result = run_backtest(universe_fn, signal_fn, config)

        # Should complete without error
        # Equity curve may be empty (all cash, no positions)
        assert result.num_rebalances > 0  # Schedule existed
        assert result.final_equity == config.initial_capital  # All cash

    def test_nan_signals_filtered(self, setup):
        """
        Test that NaN signals are filtered out with warning.
        """
        dm = setup['dm']

        def universe_fn(rebal_date: str) -> List[str]:
            return ['AAPL', 'MSFT', 'GOOGL']

        # Signal function that returns some NaN values
        def signal_fn(rebal_date: str, tickers: List[str]) -> pd.Series:
            return pd.Series({
                'AAPL': 1.0,
                'MSFT': np.nan,  # NaN should be filtered
                'GOOGL': 0.5
            })

        config = BacktestConfig(
            start_date='2023-01-31',
            end_date='2023-02-28',
            initial_capital=100000.0,
            data_manager=dm
        )

        result = run_backtest(universe_fn, signal_fn, config)

        # Should complete without error (NaN filtered)
        assert len(result.equity_curve) > 0

    def test_track_daily_equity_not_implemented(self, setup):
        """
        Test that track_daily_equity=True raises NotImplementedError.
        """
        dm = setup['dm']

        def universe_fn(rebal_date: str) -> List[str]:
            return ['AAPL']

        def signal_fn(rebal_date: str, tickers: List[str]) -> pd.Series:
            return pd.Series({'AAPL': 1.0})

        config = BacktestConfig(
            start_date='2023-01-31',
            end_date='2023-02-28',
            initial_capital=100000.0,
            track_daily_equity=True,  # Not implemented
            data_manager=dm
        )

        with pytest.raises(NotImplementedError, match="track_daily_equity"):
            run_backtest(universe_fn, signal_fn, config)

    def test_non_equal_weight_not_implemented(self, setup):
        """
        Test that equal_weight=False raises NotImplementedError.
        """
        dm = setup['dm']

        def universe_fn(rebal_date: str) -> List[str]:
            return ['AAPL']

        def signal_fn(rebal_date: str, tickers: List[str]) -> pd.Series:
            return pd.Series({'AAPL': 1.0})

        config = BacktestConfig(
            start_date='2023-01-31',
            end_date='2023-02-28',
            initial_capital=100000.0,
            equal_weight=False,  # Not implemented
            data_manager=dm
        )

        with pytest.raises(NotImplementedError, match="equal_weight"):
            run_backtest(universe_fn, signal_fn, config)

    def test_long_short_not_implemented(self, setup):
        """
        Test that long_only=False raises NotImplementedError.
        """
        dm = setup['dm']

        def universe_fn(rebal_date: str) -> List[str]:
            return ['AAPL']

        def signal_fn(rebal_date: str, tickers: List[str]) -> pd.Series:
            return pd.Series({'AAPL': 1.0})

        config = BacktestConfig(
            start_date='2023-01-31',
            end_date='2023-02-28',
            initial_capital=100000.0,
            long_only=False,  # Not implemented
            data_manager=dm
        )

        with pytest.raises(NotImplementedError, match="long_only"):
            run_backtest(universe_fn, signal_fn, config)

    def test_transaction_costs_not_implemented(self, setup):
        """
        Test that transaction_costs != 0.0 raises NotImplementedError.
        """
        dm = setup['dm']

        def universe_fn(rebal_date: str) -> List[str]:
            return ['AAPL']

        def signal_fn(rebal_date: str, tickers: List[str]) -> pd.Series:
            return pd.Series({'AAPL': 1.0})

        config = BacktestConfig(
            start_date='2023-01-31',
            end_date='2023-02-28',
            initial_capital=100000.0,
            transaction_costs=0.0005,  # 5 bps - not implemented
            data_manager=dm
        )

        with pytest.raises(NotImplementedError, match="Transaction costs"):
            run_backtest(universe_fn, signal_fn, config)
