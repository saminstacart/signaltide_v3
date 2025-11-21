#!/usr/bin/env python3
"""
Test Rebalance Helpers - Validate DataManager rebalance date methods

Tests:
1. Month-end rebalance dates handle month boundaries correctly
2. Month-end rebalance dates skip holidays
3. Weekly rebalance dates return correct day of week
4. Weekly rebalance dates handle holidays (fallback to prior trading day)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_manager import DataManager
from config import get_logger

logger = get_logger(__name__)


def test_month_end_rebalance_dates():
    """Test that month-end rebalance dates return last trading day of each month."""
    logger.info("="*60)
    logger.info("TEST 1: Month-End Rebalance Dates")
    logger.info("="*60)

    dm = DataManager()

    # Test Q1 2024 (Jan-Mar)
    dates = dm.get_month_end_rebalance_dates('2024-01-01', '2024-03-31')

    logger.info(f"Q1 2024 month-end rebalance dates: {dates}")
    logger.info(f"Count: {len(dates)}")

    # Should return exactly 3 dates (one per month)
    assert len(dates) == 3, f"Expected 3 month-end dates, got {len(dates)}"

    # Check that dates are last trading days of each month
    # Jan 2024 ends on Wed 31st (trading day)
    # Feb 2024 ends on Thu 29th (trading day)
    # Mar 2024 ends on Sun 31st, so last trading day is Fri 28th
    expected = ['2024-01-31', '2024-02-29', '2024-03-28']

    assert dates == expected, f"Expected {expected}, got {dates}"

    logger.info("✅ Month-end rebalance dates test passed\n")


def test_month_end_handles_holidays():
    """Test month-end rebalance when month ends on holiday."""
    logger.info("="*60)
    logger.info("TEST 2: Month-End with Holiday")
    logger.info("="*60)

    dm = DataManager()

    # Test Dec 2023 - Christmas is on Monday Dec 25
    # Dec 31, 2023 is Sunday, so last trading day is Friday Dec 29
    dates = dm.get_month_end_rebalance_dates('2023-12-01', '2023-12-31')

    logger.info(f"December 2023 month-end: {dates}")

    # Should return exactly 1 date
    assert len(dates) == 1, f"Expected 1 month-end date, got {len(dates)}"

    # Should be Friday Dec 29 (last trading day before year-end weekend)
    expected = '2023-12-29'
    assert dates[0] == expected, f"Expected {expected}, got {dates[0]}"

    logger.info("✅ Month-end with holiday test passed\n")


def test_weekly_rebalance_fridays():
    """Test weekly rebalance on Fridays."""
    logger.info("="*60)
    logger.info("TEST 3: Weekly Rebalance (Fridays)")
    logger.info("="*60)

    dm = DataManager()

    # Test complete weeks in March 2024 (starts Fri, ends Sun)
    dates = dm.get_weekly_rebalance_dates('2024-03-01', '2024-03-29', day_of_week='Friday')

    logger.info(f"March 2024 weekly rebalance dates (Fridays): {dates}")
    logger.info(f"Count: {len(dates)}")

    # March 2024 Fridays: 1, 8, 15, 22, 29
    assert len(dates) >= 4, f"Expected at least 4 Fridays, got {len(dates)}"

    # Verify most are Fridays (some may be fallback to earlier days in partial weeks)
    from datetime import datetime
    friday_count = 0
    for date in dates:
        dt = datetime.strptime(date, '%Y-%m-%d')
        if dt.weekday() == 4:
            friday_count += 1

    assert friday_count >= 4, f"Expected at least 4 Fridays, got {friday_count}"

    logger.info("✅ Weekly rebalance (Fridays) test passed\n")


def test_weekly_rebalance_handles_holiday():
    """Test weekly rebalance when target day is a holiday."""
    logger.info("="*60)
    logger.info("TEST 4: Weekly Rebalance with Holiday")
    logger.info("="*60)

    dm = DataManager()

    # Test week with July 4, 2024 (Thursday)
    # July 1-5: Mon, Tue, Wed, Thu (holiday), Fri
    # If we target Friday, we should get all Fridays
    # If we target Thursday, the July 4 week should fallback to Wednesday July 3
    dates_friday = dm.get_weekly_rebalance_dates('2024-07-01', '2024-07-05', day_of_week='Friday')
    dates_thursday = dm.get_weekly_rebalance_dates('2024-07-01', '2024-07-05', day_of_week='Thursday')

    logger.info(f"July 1-5, 2024 (Friday target): {dates_friday}")
    logger.info(f"July 1-5, 2024 (Thursday target): {dates_thursday}")

    # Friday target: Should get Friday July 5
    assert '2024-07-05' in dates_friday, "Should include Friday July 5"

    # Thursday target: July 4 is holiday, should fallback to Wed July 3
    assert '2024-07-03' in dates_thursday, "Should fallback to Wednesday July 3 (July 4 is holiday)"
    assert '2024-07-04' not in dates_thursday, "Should NOT include Thursday July 4 (holiday)"

    logger.info("✅ Weekly rebalance with holiday test passed\n")


def test_multiple_months():
    """Test month-end rebalance over a longer period."""
    logger.info("="*60)
    logger.info("TEST 5: Multiple Months")
    logger.info("="*60)

    dm = DataManager()

    # Test full year 2023
    dates = dm.get_month_end_rebalance_dates('2023-01-01', '2023-12-31')

    logger.info(f"2023 month-end rebalance dates: {len(dates)} dates")
    logger.info(f"First few: {dates[:3]}")
    logger.info(f"Last few: {dates[-3:]}")

    # Should return exactly 12 dates (one per month)
    assert len(dates) == 12, f"Expected 12 month-end dates, got {len(dates)}"

    # All dates should be trading days
    for date in dates:
        last_trading = dm.get_last_trading_date(date)
        assert date == last_trading, f"{date} should be a trading day"

    logger.info("✅ Multiple months test passed\n")


def main():
    """Run all tests."""
    logger.info("\n" + "="*60)
    logger.info("REBALANCE HELPERS TEST SUITE")
    logger.info("="*60 + "\n")

    try:
        test_month_end_rebalance_dates()
        test_month_end_handles_holidays()
        test_weekly_rebalance_fridays()
        test_weekly_rebalance_handles_holiday()
        test_multiple_months()

        logger.info("="*60)
        logger.info("All rebalance helper tests passed!")
        logger.info("="*60)

        return 0

    except Exception as e:
        logger.error(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
