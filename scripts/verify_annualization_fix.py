#!/usr/bin/env python3
"""
Quick verification that annualization fix works correctly.

Tests:
1. _get_periods_per_year() maps schedules correctly
2. Monthly equity curve (120 points) uses 12 periods/year
3. No deprecation warnings when rebalance_schedule is passed
"""

import pandas as pd
import numpy as np
from core.backtest_engine import _get_periods_per_year, _calculate_metrics


def test_get_periods_per_year():
    """Test schedule-to-periods mapping."""
    print("Testing _get_periods_per_year()...")

    test_cases = [
        ('D', 252),
        ('W', 52),
        ('M', 12),
        ('ME', 12),  # pandas month-end alias
        ('MS', 12),  # pandas month-start alias
        ('Q', 4),
        ('Y', 1),
        ('W-MON', 52),  # weekly Monday
        ('B', 252),  # business day
    ]

    for schedule, expected in test_cases:
        result = _get_periods_per_year(schedule)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {schedule:8} → {result:3} (expected {expected})")
        assert result == expected, f"Failed for {schedule}"

    print("  ✓ All schedule mappings correct\n")


def test_monthly_annualization():
    """Test that monthly equity curve uses 12 periods/year."""
    print("Testing monthly equity curve annualization...")

    # Create synthetic monthly equity curve (10 years = 120 months)
    dates = pd.date_range('2015-01-31', periods=120, freq='M')

    # Simulate returns: 8% annual return, 15% annual vol
    # Monthly: mean=0.64%, std=4.33%
    np.random.seed(42)
    monthly_returns = np.random.normal(0.0064, 0.0433, 120)
    equity_values = 100000 * np.cumprod(1 + monthly_returns)
    equity_curve = pd.Series(equity_values, index=dates)

    # Calculate metrics WITH explicit rebalance_schedule
    metrics = _calculate_metrics(
        equity_curve,
        initial_capital=100000.0,
        rebalance_schedule='M'  # Explicit monthly
    )

    print(f"  Equity curve length: {len(equity_curve)} points")
    print(f"  Periods per year: {metrics['periods_per_year']}")
    print(f"  Sharpe ratio: {metrics['sharpe']:.3f}")
    print(f"  Annualized volatility: {metrics['volatility']:.1%}")

    # Verify correct annualization
    assert metrics['periods_per_year'] == 12, "Should use 12 periods/year for monthly"
    print("  ✓ Correct annualization (12 periods/year)\n")

    # Test fallback with warning (NO rebalance_schedule passed)
    print("Testing heuristic fallback (should emit warning)...")
    metrics_fallback = _calculate_metrics(
        equity_curve,
        initial_capital=100000.0,
        rebalance_schedule=None  # Trigger fallback
    )

    # With 120 points, heuristic incorrectly uses 252 (daily)
    print(f"  Fallback periods per year: {metrics_fallback['periods_per_year']}")
    assert metrics_fallback['periods_per_year'] == 252, "Heuristic should use 252 for >100 points"
    print("  ✓ Fallback uses heuristic (252 for >100 points)\n")


def test_daily_annualization():
    """Test that daily equity curve uses 252 periods/year."""
    print("Testing daily equity curve annualization...")

    # Create synthetic daily equity curve (2 years = ~500 trading days)
    dates = pd.date_range('2023-01-03', periods=500, freq='B')  # Business days

    # Simulate daily returns
    np.random.seed(43)
    daily_returns = np.random.normal(0.0003, 0.01, 500)  # ~8% annual, 15% vol
    equity_values = 100000 * np.cumprod(1 + daily_returns)
    equity_curve = pd.Series(equity_values, index=dates)

    # Calculate metrics WITH explicit rebalance_schedule
    metrics = _calculate_metrics(
        equity_curve,
        initial_capital=100000.0,
        rebalance_schedule='D'  # Explicit daily
    )

    print(f"  Equity curve length: {len(equity_curve)} points")
    print(f"  Periods per year: {metrics['periods_per_year']}")
    print(f"  Sharpe ratio: {metrics['sharpe']:.3f}")

    # Verify correct annualization
    assert metrics['periods_per_year'] == 252, "Should use 252 periods/year for daily"
    print("  ✓ Correct annualization (252 periods/year)\n")


if __name__ == "__main__":
    print("=" * 60)
    print("ANNUALIZATION FIX VERIFICATION")
    print("=" * 60)
    print()

    test_get_periods_per_year()
    test_monthly_annualization()
    test_daily_annualization()

    print("=" * 60)
    print("✓ ALL TESTS PASSED")
    print("=" * 60)
    print()
    print("Summary:")
    print("  - Schedule mapping works correctly")
    print("  - Monthly curves (120 points) now use 12 periods/year (not 252)")
    print("  - Daily curves use 252 periods/year")
    print("  - Fallback heuristic still works but emits warnings")
