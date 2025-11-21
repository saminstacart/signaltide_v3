#!/usr/bin/env python3
"""
Test Rebalance Schedules - Validate schedule presets

Tests that core.schedules module correctly maps high-level schedule names
to DataManager's trading calendar methods.

Tests:
1. Daily schedule returns all trading days
2. Weekly schedule returns fewer dates, all on Fridays (or fallback)
3. Monthly schedule returns one date per month
4. All returned dates are valid trading days
5. Unknown schedule raises ValueError
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_manager import DataManager
from core.schedules import get_rebalance_dates, validate_rebalance_dates
from config import get_logger

logger = get_logger(__name__)


def test_daily_schedule():
    """Test that daily schedule returns all trading days."""
    logger.info("=" * 60)
    logger.info("TEST 1: Daily Schedule")
    logger.info("=" * 60)

    dm = DataManager()

    # Get daily rebalance dates for Q1 2024
    dates = get_rebalance_dates('daily', dm, '2024-01-01', '2024-03-31')

    logger.info(f"Q1 2024 daily rebalance dates: {len(dates)} dates")
    logger.info(f"First few: {dates[:3]}")
    logger.info(f"Last few: {dates[-3:]}")

    # Should match get_trading_dates_between exactly
    expected = dm.get_trading_dates_between('2024-01-01', '2024-03-31')
    assert dates == expected, f"Daily schedule should match get_trading_dates_between"

    # Verify all are trading days
    assert validate_rebalance_dates(dates, dm), "All daily dates should be trading days"

    logger.info("✅ Daily schedule test passed\n")


def test_weekly_schedule():
    """Test that weekly schedule returns fewer dates, all trading days."""
    logger.info("=" * 60)
    logger.info("TEST 2: Weekly Schedule")
    logger.info("=" * 60)

    dm = DataManager()

    # Get weekly rebalance dates for Q1 2024
    weekly_dates = get_rebalance_dates('weekly', dm, '2024-01-01', '2024-03-31')
    daily_dates = get_rebalance_dates('daily', dm, '2024-01-01', '2024-03-31')

    logger.info(f"Q1 2024 weekly rebalance dates: {len(weekly_dates)} dates")
    logger.info(f"Q1 2024 daily trading dates: {len(daily_dates)} dates")
    logger.info(f"Weekly dates: {weekly_dates}")

    # Weekly should be a subset of daily
    assert len(weekly_dates) < len(daily_dates), "Weekly should have fewer dates than daily"

    # Should roughly match get_weekly_rebalance_dates
    expected = dm.get_weekly_rebalance_dates('2024-01-01', '2024-03-31', day_of_week='Friday')
    assert weekly_dates == expected, "Weekly schedule should match get_weekly_rebalance_dates"

    # Verify all are trading days
    assert validate_rebalance_dates(weekly_dates, dm), "All weekly dates should be trading days"

    logger.info("✅ Weekly schedule test passed\n")


def test_monthly_schedule():
    """Test that monthly schedule returns one date per month."""
    logger.info("=" * 60)
    logger.info("TEST 3: Monthly Schedule")
    logger.info("=" * 60)

    dm = DataManager()

    # Get monthly rebalance dates for Q1 2024
    monthly_dates = get_rebalance_dates('monthly', dm, '2024-01-01', '2024-03-31')
    daily_dates = get_rebalance_dates('daily', dm, '2024-01-01', '2024-03-31')

    logger.info(f"Q1 2024 monthly rebalance dates: {len(monthly_dates)} dates")
    logger.info(f"Q1 2024 daily trading dates: {len(daily_dates)} dates")
    logger.info(f"Monthly dates: {monthly_dates}")

    # Should have exactly 3 dates (one per month)
    assert len(monthly_dates) == 3, f"Q1 should have 3 month-end dates, got {len(monthly_dates)}"

    # Should be much fewer than daily
    assert len(monthly_dates) < len(daily_dates), "Monthly should have fewer dates than daily"

    # Should match get_month_end_rebalance_dates exactly
    expected = dm.get_month_end_rebalance_dates('2024-01-01', '2024-03-31')
    assert monthly_dates == expected, "Monthly schedule should match get_month_end_rebalance_dates"

    # Verify all are trading days
    assert validate_rebalance_dates(monthly_dates, dm), "All monthly dates should be trading days"

    logger.info("✅ Monthly schedule test passed\n")


def test_alternate_schedule_names():
    """Test that alternate schedule names work (D, W, M, ME)."""
    logger.info("=" * 60)
    logger.info("TEST 4: Alternate Schedule Names")
    logger.info("=" * 60)

    dm = DataManager()

    # Test 'D' for daily
    daily_D = get_rebalance_dates('D', dm, '2024-01-01', '2024-01-31')
    daily_full = get_rebalance_dates('daily', dm, '2024-01-01', '2024-01-31')
    assert daily_D == daily_full, "'D' should match 'daily'"
    logger.info(f"  ✓ 'D' matches 'daily': {len(daily_D)} dates")

    # Test 'W' for weekly
    weekly_W = get_rebalance_dates('W', dm, '2024-01-01', '2024-01-31')
    weekly_full = get_rebalance_dates('weekly', dm, '2024-01-01', '2024-01-31')
    assert weekly_W == weekly_full, "'W' should match 'weekly'"
    logger.info(f"  ✓ 'W' matches 'weekly': {len(weekly_W)} dates")

    # Test 'M' and 'ME' for monthly
    monthly_M = get_rebalance_dates('M', dm, '2024-01-01', '2024-03-31')
    monthly_ME = get_rebalance_dates('ME', dm, '2024-01-01', '2024-03-31')
    monthly_full = get_rebalance_dates('monthly', dm, '2024-01-01', '2024-03-31')
    assert monthly_M == monthly_full, "'M' should match 'monthly'"
    assert monthly_ME == monthly_full, "'ME' should match 'monthly'"
    logger.info(f"  ✓ 'M' and 'ME' match 'monthly': {len(monthly_M)} dates")

    logger.info("✅ Alternate schedule names test passed\n")


def test_invalid_schedule():
    """Test that invalid schedule raises ValueError."""
    logger.info("=" * 60)
    logger.info("TEST 5: Invalid Schedule")
    logger.info("=" * 60)

    dm = DataManager()

    try:
        dates = get_rebalance_dates('quarterly', dm, '2024-01-01', '2024-03-31')
        assert False, "Should have raised ValueError for invalid schedule"
    except ValueError as e:
        logger.info(f"  ✓ Correctly raised ValueError: {e}")

    logger.info("✅ Invalid schedule test passed\n")


def test_validate_rebalance_dates():
    """Test that validate_rebalance_dates correctly identifies non-trading days."""
    logger.info("=" * 60)
    logger.info("TEST 6: Validate Rebalance Dates")
    logger.info("=" * 60)

    dm = DataManager()

    # Test with all trading days (should pass)
    valid_dates = ['2024-01-02', '2024-01-03', '2024-01-04']  # All weekdays
    assert validate_rebalance_dates(valid_dates, dm), "Valid trading days should pass"
    logger.info(f"  ✓ Valid trading days passed validation")

    # Test with a weekend (should fail)
    invalid_dates = ['2024-01-02', '2024-01-06', '2024-01-08']  # Jan 6 is Saturday
    assert not validate_rebalance_dates(invalid_dates, dm), "Weekend should fail validation"
    logger.info(f"  ✓ Weekend correctly failed validation")

    # Test with a holiday (should fail)
    holiday_dates = ['2024-01-02', '2024-07-04', '2024-01-05']  # July 4 is holiday
    assert not validate_rebalance_dates(holiday_dates, dm), "Holiday should fail validation"
    logger.info(f"  ✓ Holiday correctly failed validation")

    logger.info("✅ Validate rebalance dates test passed\n")


def main():
    """Run all tests."""
    logger.info("\n" + "=" * 60)
    logger.info("REBALANCE SCHEDULES TEST SUITE")
    logger.info("=" * 60 + "\n")

    try:
        test_daily_schedule()
        test_weekly_schedule()
        test_monthly_schedule()
        test_alternate_schedule_names()
        test_invalid_schedule()
        test_validate_rebalance_dates()

        logger.info("=" * 60)
        logger.info("All rebalance schedule tests passed!")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
