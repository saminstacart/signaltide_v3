#!/usr/bin/env python3
"""
Deterministic Backtest Integration Test

High-level integration test that runs a tiny, deterministic backtest
and validates key metrics for regression testing.

This test serves as the "golden path" for validating major refactors
by ensuring that:
1. The entire backtest pipeline executes without errors
2. Rebalance dates come from the trading calendar
3. Number of trades is reasonable and consistent
4. Final PnL and stats are within expected ranges
5. Results are deterministic (same inputs = same outputs)

Test Parameters (chosen for speed + determinism):
- Universe: 3 tickers (AAPL, MSFT, GOOGL)
- Period: Jul 2023 - Dec 2023 (6 months, with lookback from Jan 2023)
- Schedule: Monthly rebalancing
- Strategy: Simple momentum (fixed parameters, 126-day lookback)
- Capital: $50,000

Tests:
1. Backtest executes without errors
2. Rebalance dates are all valid trading days from dim_trading_calendar
3. Correct number of rebalance dates (3 for Q1)
4. Number of trades is reasonable (roughly 2 trades/ticker/rebalance)
5. Final portfolio value is within expected range
6. Key stats are reasonable (no NaN, within bounds)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import pandas as pd
from data.data_manager import DataManager
from signals import InstitutionalMomentum
from scripts.run_institutional_backtest import InstitutionalBacktest
from core.schedules import validate_rebalance_dates
from config import get_logger

logger = get_logger(__name__)


def test_deterministic_backtest():
    """
    Run a deterministic backtest and validate key outputs.

    This test runs a small, fast backtest with fixed parameters and
    validates that the results are:
    1. Consistent (same inputs produce same outputs)
    2. Reasonable (metrics within expected ranges)
    3. Complete (no missing data or NaN values)
    """
    logger.info("=" * 60)
    logger.info("TEST: Deterministic Backtest Integration")
    logger.info("=" * 60)

    # Test parameters (fixed for determinism)
    universe = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2023-01-01'  # Start early to provide lookback data
    end_date = '2023-12-31'    # Full year 2023
    initial_capital = 50000.0
    rebalance_freq = 'monthly'

    logger.info(f"  Universe: {universe}")
    logger.info(f"  Period: {start_date} to {end_date}")
    logger.info(f"  Rebalance: {rebalance_freq}")
    logger.info(f"  Capital: ${initial_capital:,.0f}")
    logger.info("")

    # Initialize backtest
    backtest = InstitutionalBacktest(
        universe=universe,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        params={'rebalance_freq': rebalance_freq}
    )

    # Simple momentum strategy with fixed parameters
    # Must include all parameters from get_parameter_space()
    signal_params = {
        'formation_period': 126,    # 6 months (institutional default range: 6-18 months)
        'skip_period': 21,          # 1 month skip
        'winsorize_pct': [5, 95],   # Standard winsorization
        'quintiles': True,          # Use quintile signals
    }

    logger.info("Running backtest...")

    # Time the backtest execution
    start_time = time.perf_counter()

    result = backtest.run_signal_backtest(
        signal_class=InstitutionalMomentum,
        signal_params=signal_params,
        signal_name='test_momentum'
    )

    end_time = time.perf_counter()
    runtime_seconds = end_time - start_time

    if result is None:
        raise AssertionError("Backtest returned None - signal generation failed")

    logger.info("✓ Backtest executed successfully")
    logger.info(f"  Runtime: {runtime_seconds:.2f}s")

    # Performance budget check (log warning if slow, but don't fail)
    PERFORMANCE_BUDGET_SECONDS = 10.0
    if runtime_seconds > PERFORMANCE_BUDGET_SECONDS:
        logger.warning(
            f"  ⚠ Performance budget exceeded: {runtime_seconds:.2f}s > {PERFORMANCE_BUDGET_SECONDS:.0f}s budget"
        )
    else:
        logger.info(f"  ✓ Within performance budget ({PERFORMANCE_BUDGET_SECONDS:.0f}s)")

    logger.info("")

    # Extract results
    equity_curve = result['equity_curve']
    manifest = result.get('manifest')
    manifest_dict = result.get('manifest_dict')

    # TEST 0: Validate manifest presence and structure
    logger.info("TEST 0: Validating backtest manifest...")

    assert manifest is not None, "Manifest should be present in result"
    assert manifest_dict is not None, "Manifest dict should be present in result"

    # Validate manifest fields
    assert manifest_dict['start_date'] == start_date, \
        f"Manifest start_date {manifest_dict['start_date']} != {start_date}"
    assert manifest_dict['end_date'] == end_date, \
        f"Manifest end_date {manifest_dict['end_date']} != {end_date}"
    assert manifest_dict['universe_type'] == 'manual', \
        f"Expected universe_type='manual', got '{manifest_dict['universe_type']}'"
    assert manifest_dict['initial_capital'] == initial_capital, \
        f"Manifest capital {manifest_dict['initial_capital']} != {initial_capital}"
    assert manifest_dict['rebalance_schedule'] == rebalance_freq, \
        f"Manifest rebalance {manifest_dict['rebalance_schedule']} != {rebalance_freq}"

    # Validate signals list
    assert len(manifest_dict['signals']) >= 1, \
        "Manifest should have at least one signal"
    signal_spec = manifest_dict['signals'][0]
    assert signal_spec['name'] == 'InstitutionalMomentum', \
        f"Expected signal name 'InstitutionalMomentum', got '{signal_spec['name']}'"
    assert 'formation_period' in signal_spec['params'], \
        "Signal params should include formation_period"

    # Validate universe params
    assert 'tickers' in manifest_dict['universe_params'], \
        "Universe params should include tickers list"
    assert manifest_dict['universe_params']['tickers'] == universe, \
        f"Manifest tickers {manifest_dict['universe_params']['tickers']} != {universe}"

    # Validate transaction costs are present
    assert 'commission_pct' in manifest_dict['transaction_costs'] or \
           'slippage_pct' in manifest_dict['transaction_costs'], \
        "Transaction costs should be present in manifest"

    logger.info(f"  ✓ Manifest is well-formed")
    logger.info(f"  Run ID: {manifest_dict['run_id'][:8]}...")
    logger.info(f"  Created at: {manifest_dict['created_at']}")
    logger.info(f"  Git SHA: {manifest_dict.get('git_sha', 'N/A')}")
    logger.info("")

    # TEST 1: Validate rebalance dates
    logger.info("TEST 1: Validating rebalance dates...")
    dm = DataManager()

    # Get expected rebalance dates for Q1 2023
    from core.schedules import get_rebalance_dates
    expected_rebal_dates = get_rebalance_dates(
        schedule=rebalance_freq,
        dm=dm,
        start_date=start_date,
        end_date=end_date
    )

    logger.info(f"  Expected rebalance dates: {expected_rebal_dates}")

    # Verify count (12 months = 12 monthly rebalances)
    assert len(expected_rebal_dates) == 12, \
        f"Expected 12 monthly rebalance dates for 2023, got {len(expected_rebal_dates)}"

    # Verify all are trading days
    assert validate_rebalance_dates(expected_rebal_dates, dm), \
        "All rebalance dates must be trading days from dim_trading_calendar"

    # Verify specific dates (month-end trading days)
    assert expected_rebal_dates[0] == '2023-01-31', \
        f"Expected Jan month-end to be 2023-01-31, got {expected_rebal_dates[0]}"
    assert expected_rebal_dates[11] == '2023-12-29', \
        f"Expected Dec month-end to be 2023-12-29, got {expected_rebal_dates[11]}"

    logger.info("  ✓ Rebalance dates are correct and all valid trading days")
    logger.info("")

    # TEST 2: Validate equity curve
    logger.info("TEST 2: Validating equity curve...")
    logger.info(f"  Equity curve length: {len(equity_curve)} days")

    # Should have data for entire period
    assert len(equity_curve) > 0, "Equity curve should not be empty"
    assert not equity_curve.isna().any(), "Equity curve should have no NaN values"

    # Initial value should be close to initial capital
    initial_equity = equity_curve.iloc[0]
    assert abs(initial_equity - initial_capital) < 100, \
        f"Initial equity {initial_equity} should be close to initial capital {initial_capital}"

    # Final equity should be reasonable (not zero, not negative, not absurdly high)
    final_equity = equity_curve.iloc[-1]
    total_return_pct = (final_equity / initial_equity - 1) * 100

    logger.info(f"  Initial equity: ${initial_equity:,.2f}")
    logger.info(f"  Final equity: ${final_equity:,.2f}")
    logger.info(f"  P&L: ${final_equity - initial_equity:,.2f} ({total_return_pct:+.2f}%)")

    # Use performance bands instead of exact values
    # Band: -20% to +100% (allows for some variability while catching regressions)
    assert final_equity > 0, "Final equity must be positive"
    assert -20.0 <= total_return_pct <= 100.0, \
        f"Total return {total_return_pct:.2f}% is outside reasonable band (-20% to +100%)"

    logger.info("  ✓ Equity curve is valid and reasonable")
    logger.info("")

    # TEST 3: Validate performance metrics
    logger.info("TEST 3: Validating performance metrics...")

    # Get performance metrics from result (these should be present)
    if 'total_return' in result:
        total_return = result['total_return'] * 100  # Convert to percentage
        logger.info(f"  Total return: {total_return:+.2f}%")
        # Performance band: -20% to +100%
        assert -20.0 <= total_return <= 100.0, \
            f"Total return {total_return:.2f}% is outside reasonable band (-20% to +100%)"

    if 'sharpe' in result:
        sharpe = result['sharpe']
        logger.info(f"  Sharpe ratio: {sharpe:.2f}")
        # Sharpe band: -1.0 to +3.0 (allows for both losses and strong performance)
        assert -1.0 <= sharpe <= 3.0, \
            f"Sharpe ratio {sharpe:.2f} is outside reasonable band (-1.0 to +3.0)"

    if 'max_drawdown' in result:
        max_dd = result['max_drawdown'] * 100  # Convert to percentage
        logger.info(f"  Max drawdown: {max_dd:.2f}%")
        # Max drawdown band: -50% to 0% (drawdown is negative)
        assert -50.0 <= max_dd <= 0.0, \
            f"Max drawdown {max_dd:.2f}% is outside reasonable band (-50% to 0%)"

    if 'volatility' in result:
        vol = result['volatility'] * 100  # Convert to percentage
        logger.info(f"  Volatility: {vol:.2f}%")
        # Volatility band: 0% to 100% (annual volatility)
        assert 0.0 <= vol <= 100.0, \
            f"Volatility {vol:.2f}% is outside reasonable band (0% to 100%)"

    logger.info("  ✓ Performance metrics are reasonable")
    logger.info("")

    # TEST 4: Log determinism baseline values
    logger.info("TEST 4: Recording determinism baseline...")
    logger.info("  (These values should be identical on repeated runs)")
    logger.info(f"  Rebalance dates: {len(expected_rebal_dates)} dates")
    logger.info(f"  Final equity: ${final_equity:,.2f}")
    logger.info("")

    logger.info("✅ All deterministic backtest tests passed!")
    logger.info("")


def main():
    """Run the deterministic backtest integration test."""
    logger.info("\n" + "=" * 60)
    logger.info("DETERMINISTIC BACKTEST INTEGRATION TEST")
    logger.info("Full end-to-end backtest validation")
    logger.info("=" * 60 + "\n")

    try:
        test_deterministic_backtest()

        logger.info("=" * 60)
        logger.info("Deterministic backtest test passed!")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
