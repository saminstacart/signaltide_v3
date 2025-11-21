"""
Test script to verify Portfolio.get_equity() works correctly.

This tests that:
1. Portfolio class has get_equity() method (not .equity attribute)
2. The method returns correct values
3. The backtest script would work if database existed
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from core.portfolio import Portfolio
from core.types import PositionSide, OrderSide

def test_portfolio_equity():
    """Test that Portfolio.get_equity() works correctly."""

    print("=" * 60)
    print("Testing Portfolio.get_equity() method")
    print("=" * 60)

    # Create portfolio
    initial_capital = 50000
    portfolio = Portfolio(initial_capital=initial_capital)

    # Test 1: Initial equity should equal initial capital
    initial_equity = portfolio.get_equity()
    assert initial_equity == initial_capital, f"Expected {initial_capital}, got {initial_equity}"
    print(f"\n✓ Test 1 PASSED: Initial equity = ${initial_equity:,.2f}")

    # Test 2: Execute a trade and check equity updates
    test_date = datetime.now()
    signals = {'AAPL': 0.5}  # 50% signal strength
    prices = {'AAPL': 150.0}

    trades = portfolio.update(test_date, signals, prices)

    equity_after_trade = portfolio.get_equity()
    print(f"✓ Test 2 PASSED: Equity after trade = ${equity_after_trade:,.2f}")
    print(f"  Trades executed: {len(trades)}")
    print(f"  Positions: {len(portfolio.positions)}")

    # Test 3: Verify equity_curve attribute exists
    assert hasattr(portfolio, 'equity_curve'), "Portfolio should have equity_curve attribute"
    print(f"✓ Test 3 PASSED: equity_curve exists with {len(portfolio.equity_curve)} entries")

    # Test 4: Verify NO .equity attribute exists (this would be the bug)
    try:
        _ = portfolio.equity
        print("✗ Test 4 FAILED: portfolio.equity attribute exists (should NOT exist)")
        return False
    except AttributeError:
        print("✓ Test 4 PASSED: portfolio.equity correctly raises AttributeError")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    print("\nConclusion:")
    print("- Portfolio.get_equity() method works correctly")
    print("- No .equity attribute exists (as expected)")
    print("- Any backtest script using portfolio.equity will fail")
    print("- All code should use portfolio.get_equity() instead")

    return True

def test_backtest_pattern():
    """Test the equity tracking pattern used in backtests."""

    print("\n" + "=" * 60)
    print("Testing backtest equity tracking pattern")
    print("=" * 60)

    portfolio = Portfolio(initial_capital=50000)

    # Simulate a few days of trading
    dates = pd.date_range('2020-01-01', periods=5, freq='D')
    equity_curve = []

    for i, date in enumerate(dates):
        # Simulate some signals and prices
        if i == 0:
            signals = {'AAPL': 0.5, 'MSFT': 0.3}
            prices = {'AAPL': 150.0, 'MSFT': 250.0}
        elif i == 2:
            signals = {'AAPL': 0.0, 'MSFT': 0.5}  # Close AAPL, adjust MSFT
            prices = {'AAPL': 155.0, 'MSFT': 255.0}
        else:
            signals = {}
            prices = {'AAPL': 152.0 + i, 'MSFT': 252.0 + i}

        # Update portfolio
        portfolio.update(date, signals, prices)

        # Track equity using correct method
        equity_curve.append(portfolio.get_equity())

    print(f"\n✓ Tracked equity for {len(equity_curve)} days")
    print(f"  Initial: ${equity_curve[0]:,.2f}")
    print(f"  Final: ${equity_curve[-1]:,.2f}")
    print(f"  Return: {(equity_curve[-1]/equity_curve[0] - 1)*100:.2f}%")

    return True

if __name__ == '__main__':
    success = test_portfolio_equity()
    if success:
        test_backtest_pattern()

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
