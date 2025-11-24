"""
Regression tests for backtest metrics annualization bug fix.

Background:
    Prior to 2025-11-24, _calculate_metrics() used a heuristic to infer
    annualization factor from equity curve length:
        periods_per_year = 252 if len(equity_curve) > 100 else 12

    This caused ~4.8x inflation of Sharpe/volatility for monthly backtests
    with >100 periods (e.g., 10 years = 120 months).

    Fix: Added explicit rebalance_schedule parameter.

See: docs/notes/m3_5_sharpe_discrepancy_resolution.md
See: docs/core/ERROR_PREVENTION_ARCHITECTURE.md (Annualization Heuristic Bug)
"""

import pytest
import pandas as pd
import numpy as np
from core.backtest_engine import _get_periods_per_year, _calculate_metrics


class TestGetPeriodsPerYear:
    """Test schedule-to-periods mapping."""

    def test_standard_schedules(self):
        """Test standard frequency mappings."""
        assert _get_periods_per_year('D') == 252  # Daily
        assert _get_periods_per_year('W') == 52   # Weekly
        assert _get_periods_per_year('M') == 12   # Monthly
        assert _get_periods_per_year('Q') == 4    # Quarterly
        assert _get_periods_per_year('Y') == 1    # Yearly
        assert _get_periods_per_year('A') == 1    # Annual (alias)

    def test_pandas_frequency_aliases(self):
        """Test pandas frequency string aliases."""
        # Month aliases
        assert _get_periods_per_year('ME') == 12   # Month end
        assert _get_periods_per_year('MS') == 12   # Month start

        # Week aliases
        assert _get_periods_per_year('W-MON') == 52  # Weekly Monday

        # Business day
        assert _get_periods_per_year('B') == 252

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        assert _get_periods_per_year('m') == 12
        assert _get_periods_per_year('M') == 12
        assert _get_periods_per_year('monthly') == 12  # Starts with 'M'

    def test_unknown_schedule_returns_none(self):
        """Test unknown schedule returns None."""
        assert _get_periods_per_year('UNKNOWN') is None
        assert _get_periods_per_year('XYZ') is None


class TestMonthlyAnnualizationRegression:
    """
    Regression tests for the annualization bug.

    The bug: Monthly backtests with >100 periods incorrectly used 252 (daily)
    instead of 12 (monthly), causing ~4.8x metric inflation.
    """

    def test_monthly_120_periods_uses_12_not_252(self):
        """
        REGRESSION: Monthly equity curve (120 points) should use 12 periods/year.

        This is the core bug case: 10 years of monthly data = 120 points.
        Old heuristic: len > 100 → 252 (WRONG!)
        New behavior: explicit schedule → 12 (CORRECT)
        """
        # Create synthetic monthly equity curve (10 years = 120 months)
        dates = pd.date_range('2015-01-31', periods=120, freq='ME')
        np.random.seed(42)
        monthly_returns = np.random.normal(0.0064, 0.0433, 120)  # ~8% annual return, 15% vol
        equity_values = 100000 * np.cumprod(1 + monthly_returns)
        equity_curve = pd.Series(equity_values, index=dates)

        # Calculate metrics WITH explicit rebalance_schedule
        metrics = _calculate_metrics(
            equity_curve,
            initial_capital=100000.0,
            rebalance_schedule='M'  # Explicit monthly
        )

        # ASSERT: Uses correct annualization
        assert metrics['periods_per_year'] == 12, \
            "Monthly curves must use 12 periods/year, not 252!"

        # Verify Sharpe is reasonable (not inflated 4.8x)
        # With ~8% return and 15% vol, Sharpe should be ~0.5, not ~2.4
        assert metrics['sharpe'] < 1.0, \
            "Sharpe should be reasonable, not inflated by wrong annualization"

    def test_monthly_short_curve_uses_12(self):
        """Test monthly curve with <100 points also uses 12."""
        # 5 years = 60 months (under old heuristic threshold)
        dates = pd.date_range('2020-01-31', periods=60, freq='ME')
        np.random.seed(43)
        monthly_returns = np.random.normal(0.006, 0.04, 60)
        equity_values = 100000 * np.cumprod(1 + monthly_returns)
        equity_curve = pd.Series(equity_values, index=dates)

        metrics = _calculate_metrics(
            equity_curve,
            initial_capital=100000.0,
            rebalance_schedule='M'
        )

        assert metrics['periods_per_year'] == 12

    def test_daily_500_periods_uses_252(self):
        """Test daily equity curve uses 252 periods/year."""
        # ~2 years of daily data
        dates = pd.date_range('2023-01-03', periods=500, freq='B')  # Business days
        np.random.seed(44)
        daily_returns = np.random.normal(0.0003, 0.01, 500)
        equity_values = 100000 * np.cumprod(1 + daily_returns)
        equity_curve = pd.Series(equity_values, index=dates)

        metrics = _calculate_metrics(
            equity_curve,
            initial_capital=100000.0,
            rebalance_schedule='D'  # Explicit daily
        )

        assert metrics['periods_per_year'] == 252

    def test_fallback_heuristic_still_works_with_warning(self, caplog):
        """
        Test backward compatibility: fallback heuristic still works but warns.

        When rebalance_schedule is None, the old heuristic is used but a
        deprecation warning is logged.
        """
        # 120-month curve (would trigger bug)
        dates = pd.date_range('2015-01-31', periods=120, freq='ME')
        np.random.seed(45)
        monthly_returns = np.random.normal(0.005, 0.04, 120)
        equity_values = 100000 * np.cumprod(1 + monthly_returns)
        equity_curve = pd.Series(equity_values, index=dates)

        # Call WITHOUT rebalance_schedule (trigger fallback)
        import logging
        with caplog.at_level(logging.WARNING):
            metrics = _calculate_metrics(
                equity_curve,
                initial_capital=100000.0,
                rebalance_schedule=None  # Fallback path
            )

        # Verify deprecation warning was logged
        assert "DEPRECATED" in caplog.text, \
            "Fallback should emit deprecation warning"
        assert "heuristic" in caplog.text, \
            "Warning should mention heuristic"

        # Fallback uses heuristic: len=120 > 100 → 252 (wrong, but preserved for compatibility)
        assert metrics['periods_per_year'] == 252, \
            "Fallback heuristic should still use old logic (for backward compat)"


class TestMetricsCalculationCorrectness:
    """Test that metrics calculations are correct with proper annualization."""

    def test_sharpe_calculation_with_explicit_schedule(self):
        """Test Sharpe ratio calculation with explicit annualization."""
        # Create deterministic monthly returns
        dates = pd.date_range('2020-01-31', periods=24, freq='ME')
        # Fixed 1% monthly return, 2% monthly std
        equity_values = np.array([100000 * (1.01 ** i) for i in range(24)])
        equity_curve = pd.Series(equity_values, index=dates)

        metrics = _calculate_metrics(
            equity_curve,
            initial_capital=100000.0,
            rebalance_schedule='M'
        )

        # Verify periods_per_year is correct
        assert metrics['periods_per_year'] == 12

        # Sharpe = (mean_return / std_return) * sqrt(periods_per_year)
        # With deterministic 1% monthly growth, std should be near zero
        # So Sharpe should be very high (or inf if std=0)
        assert metrics['sharpe'] > 10 or np.isinf(metrics['sharpe']), \
            "Deterministic returns should yield very high Sharpe"

    def test_volatility_annualization(self):
        """Test that volatility is properly annualized."""
        # Create returns with known volatility
        np.random.seed(100)
        monthly_std = 0.04  # 4% monthly std
        returns = np.random.normal(0, monthly_std, 120)
        equity_values = 100000 * np.cumprod(1 + returns)
        dates = pd.date_range('2015-01-31', periods=120, freq='ME')
        equity_curve = pd.Series(equity_values, index=dates)

        metrics = _calculate_metrics(
            equity_curve,
            initial_capital=100000.0,
            rebalance_schedule='M'
        )

        # Annual vol = monthly_std * sqrt(12)
        expected_annual_vol = monthly_std * np.sqrt(12)

        # Allow some tolerance due to randomness
        assert abs(metrics['volatility'] - expected_annual_vol) < 0.02, \
            f"Expected vol ~{expected_annual_vol:.2%}, got {metrics['volatility']:.2%}"


class TestRegressionAgainstKnownValues:
    """
    Test against known M+Q baseline values from Phase 3.

    The canonical M+Q 25/75 baseline (2015-2024) should have:
    - Sharpe ~0.63 (not 2.876!)
    - This test ensures we don't regress back to inflated values.
    """

    def test_baseline_shape_produces_reasonable_sharpe(self):
        """
        Test that a backtest shape similar to M+Q baseline yields reasonable Sharpe.

        M+Q baseline: 117 months, Sharpe should be ~0.6-0.7 (not 2.8+).
        """
        # Simulate M+Q-like performance:
        # 10-year CAGR ~7%, annual vol ~13%
        # This should give Sharpe ~0.5-0.7
        np.random.seed(999)
        n_months = 117
        monthly_mean = 0.0056  # ~7% annual
        monthly_std = 0.038    # ~13% annual vol

        returns = np.random.normal(monthly_mean, monthly_std, n_months)
        equity_values = 100000 * np.cumprod(1 + returns)
        dates = pd.date_range('2015-04-30', periods=n_months, freq='ME')
        equity_curve = pd.Series(equity_values, index=dates)

        metrics = _calculate_metrics(
            equity_curve,
            initial_capital=100000.0,
            rebalance_schedule='M'
        )

        # REGRESSION CHECK: Sharpe should be reasonable, not inflated
        assert metrics['sharpe'] < 1.5, \
            f"Sharpe {metrics['sharpe']:.2f} is too high - possible annualization bug regression!"

        assert metrics['sharpe'] > 0.2, \
            f"Sharpe {metrics['sharpe']:.2f} is too low - check calculation"

        # Verify correct annualization was used
        assert metrics['periods_per_year'] == 12


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, '-v'])
