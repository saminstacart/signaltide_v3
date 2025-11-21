#!/usr/bin/env python3
"""
Integration Test: Calendar + Schedules + Universe PIT Semantics

Tests that the three core market plumbing pieces work together:
1. Trading calendar (dim_trading_calendar)
2. Rebalance schedules (core.schedules)
3. Point-in-time universe membership (UniverseManager)

This test does NOT depend on strategy internals. It only verifies that:
- Rebalance dates are real trading days from dim_trading_calendar
- UniverseManager respects [start, end) PIT semantics
- They can be composed together over a real date range

Tests:
1. Monthly rebalance dates are all trading days
2. A ticker that exits a universe is never returned after its end_date
3. A ticker with NULL end_date is present far in the future
4. Rebalance dates + universe membership work together
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_manager import DataManager
from core.schedules import get_rebalance_dates, validate_rebalance_dates
from core.universe_manager import UniverseManager
from config import get_logger

logger = get_logger(__name__)


def test_rebalance_dates_are_trading_days():
    """Test that all rebalance dates come from trading calendar."""
    logger.info("=" * 60)
    logger.info("TEST 1: Rebalance Dates Are Trading Days")
    logger.info("=" * 60)

    dm = DataManager()

    # Get monthly rebalance dates for 2023
    monthly_dates = get_rebalance_dates('monthly', dm, '2023-01-01', '2023-12-31')

    logger.info(f"Monthly rebalance dates for 2023: {len(monthly_dates)} dates")
    logger.info(f"Dates: {monthly_dates}")

    # Verify all are trading days using validate_rebalance_dates
    assert validate_rebalance_dates(monthly_dates, dm), \
        "All monthly rebalance dates must be trading days"

    # Verify we got exactly 12 dates (one per month)
    assert len(monthly_dates) == 12, \
        f"Expected 12 monthly dates for 2023, got {len(monthly_dates)}"

    # Spot check: Last trading day of Jan 2023 should be 2023-01-31 (Tuesday)
    assert monthly_dates[0] == '2023-01-31', \
        f"Expected Jan 2023 month-end to be 2023-01-31, got {monthly_dates[0]}"

    logger.info("✅ All rebalance dates are valid trading days\n")


def test_universe_pit_semantics_exit():
    """Test that a ticker leaving a universe is never returned after end_date."""
    logger.info("=" * 60)
    logger.info("TEST 2: Universe PIT Semantics - Ticker Exit")
    logger.info("=" * 60)

    dm = DataManager()
    um = UniverseManager(dm)

    # Find a ticker that left sp500_actual (has a non-NULL end_date)
    # We'll query the database directly for test setup
    from core.db import get_connection
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT ticker, membership_end_date
        FROM dim_universe_membership
        WHERE universe_name = 'sp500_actual'
          AND membership_end_date IS NOT NULL
        ORDER BY membership_end_date DESC
        LIMIT 1
    """)

    result = cursor.fetchone()
    conn.close()

    if not result:
        logger.info("  ⊘ No tickers with end_date found, skipping test")
        return

    ticker, end_date = result
    logger.info(f"  Testing ticker: {ticker}")
    logger.info(f"  End date: {end_date}")

    # With [start, end) semantics:
    # - The ticker should be IN the universe one day before end_date
    # - The ticker should be OUT of the universe on end_date itself

    # Get the day before end_date
    from datetime import datetime, timedelta
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    day_before = (end_dt - timedelta(days=1)).strftime('%Y-%m-%d')

    # Get last trading date <= day_before (should be IN universe)
    last_in = dm.get_last_trading_date(day_before)
    logger.info(f"  Last day in universe: {last_in}")

    # Use end_date as first day OUT of universe
    first_out = end_date
    logger.info(f"  First day out of universe: {first_out}")

    # Test 1: Ticker is IN universe on last_in
    members_before = um.get_universe_tickers('sp500_actual', as_of_date=last_in)
    assert ticker in members_before, \
        f"{ticker} should be in sp500_actual on {last_in} (before end_date)"
    logger.info(f"  ✓ {ticker} is in universe on {last_in}")

    # Test 2: Ticker is OUT of universe on first_out (PIT [start, end) semantics)
    members_after = um.get_universe_tickers('sp500_actual', as_of_date=first_out)
    assert ticker not in members_after, \
        f"{ticker} should NOT be in sp500_actual on {first_out} (on/after end_date)"
    logger.info(f"  ✓ {ticker} is NOT in universe on {first_out}")

    logger.info("✅ Universe PIT semantics correctly exclude ticker after end_date\n")


def test_universe_pit_semantics_null_end():
    """Test that a ticker with NULL end_date is present far in the future."""
    logger.info("=" * 60)
    logger.info("TEST 3: Universe PIT Semantics - NULL end_date")
    logger.info("=" * 60)

    dm = DataManager()
    um = UniverseManager(dm)

    # Find a ticker in sp500_actual with NULL end_date (currently in universe)
    from core.db import get_connection
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT ticker, membership_start_date
        FROM dim_universe_membership
        WHERE universe_name = 'sp500_actual'
          AND membership_end_date IS NULL
        LIMIT 1
    """)

    result = cursor.fetchone()
    conn.close()

    if not result:
        logger.info("  ⊘ No tickers with NULL end_date found, skipping test")
        return

    ticker, start_date = result
    logger.info(f"  Testing ticker: {ticker}")
    logger.info(f"  Start date: {start_date}")

    # Test 1: Ticker is in universe on start_date
    members_start = um.get_universe_tickers('sp500_actual', as_of_date=start_date)
    assert ticker in members_start, \
        f"{ticker} should be in sp500_actual on {start_date} (start_date)"
    logger.info(f"  ✓ {ticker} is in universe on {start_date}")

    # Test 2: Ticker is still in universe far in the future (2030-01-01)
    future_date = '2030-01-01'
    members_future = um.get_universe_tickers('sp500_actual', as_of_date=future_date)
    assert ticker in members_future, \
        f"{ticker} should be in sp500_actual on {future_date} (NULL end_date = indefinite)"
    logger.info(f"  ✓ {ticker} is still in universe on {future_date}")

    logger.info("✅ NULL end_date correctly represents indefinite membership\n")


def test_rebalance_plus_universe_integration():
    """Test that rebalance dates and universe membership work together."""
    logger.info("=" * 60)
    logger.info("TEST 4: Rebalance + Universe Integration")
    logger.info("=" * 60)

    dm = DataManager()
    um = UniverseManager(dm)

    # Get monthly rebalance dates for Q1 2023
    rebal_dates = get_rebalance_dates('monthly', dm, '2023-01-01', '2023-03-31')
    logger.info(f"Q1 2023 monthly rebalance dates: {rebal_dates}")

    # For each rebalance date, get universe membership
    for rebal_date in rebal_dates:
        members = um.get_universe_tickers('sp500_actual', as_of_date=rebal_date)
        logger.info(f"  {rebal_date}: {len(members)} members in sp500_actual")

        # Sanity checks
        assert len(members) > 0, \
            f"Universe should not be empty on {rebal_date}"
        assert len(members) < 600, \
            f"S&P 500 should have < 600 members (got {len(members)} on {rebal_date})"

    # Verify all dates are trading days (redundant but shows composition)
    assert validate_rebalance_dates(rebal_dates, dm), \
        "All rebalance dates must be trading days"

    logger.info("✅ Rebalance dates and universe membership work together\n")


def main():
    """Run all integration tests."""
    logger.info("\n" + "=" * 60)
    logger.info("BACKTEST INTEGRATION TEST SUITE")
    logger.info("Calendar + Schedules + Universe PIT")
    logger.info("=" * 60 + "\n")

    try:
        test_rebalance_dates_are_trading_days()
        test_universe_pit_semantics_exit()
        test_universe_pit_semantics_null_end()
        test_rebalance_plus_universe_integration()

        logger.info("=" * 60)
        logger.info("All integration tests passed!")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
