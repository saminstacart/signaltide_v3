#!/usr/bin/env python3
"""
Test Trading Calendar Helpers - Validate DataManager trading date methods

Tests:
1. get_last_trading_date handles weekends correctly
2. get_next_trading_date handles weekends correctly
3. get_trading_dates_between skips weekends and holidays
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_manager import DataManager
from config import get_logger

logger = get_logger(__name__)


def test_last_trading_date_weekend():
    """Test that get_last_trading_date returns Friday for a Saturday."""
    logger.info("="*60)
    logger.info("TEST 1: Last Trading Date (Weekend)")
    logger.info("="*60)

    dm = DataManager()

    # 2024-06-15 is a Saturday
    saturday = '2024-06-15'
    last_trading = dm.get_last_trading_date(saturday)

    logger.info(f"Input (Saturday): {saturday}")
    logger.info(f"Last trading date: {last_trading}")

    # Should return Friday 2024-06-14
    expected = '2024-06-14'
    assert last_trading == expected, f"Expected {expected}, got {last_trading}"

    logger.info(f"✅ Correctly returned Friday: {last_trading}\n")


def test_next_trading_date_weekend():
    """Test that get_next_trading_date returns Monday for a Friday."""
    logger.info("="*60)
    logger.info("TEST 2: Next Trading Date (Weekend)")
    logger.info("="*60)

    dm = DataManager()

    # 2024-06-14 is a Friday
    friday = '2024-06-14'
    next_trading = dm.get_next_trading_date(friday)

    logger.info(f"Input (Friday): {friday}")
    logger.info(f"Next trading date: {next_trading}")

    # Should return Monday 2024-06-17
    expected = '2024-06-17'
    assert next_trading == expected, f"Expected {expected}, got {next_trading}"

    logger.info(f"✅ Correctly returned Monday: {next_trading}\n")


def test_trading_dates_between_skips_weekends():
    """Test that get_trading_dates_between skips weekends."""
    logger.info("="*60)
    logger.info("TEST 3: Trading Dates Between (Weekend Span)")
    logger.info("="*60)

    dm = DataManager()

    # Range from Friday to Monday (spans weekend)
    start = '2024-06-14'  # Friday
    end = '2024-06-17'    # Monday

    trading_dates = dm.get_trading_dates_between(start, end)

    logger.info(f"Date range: {start} to {end}")
    logger.info(f"Trading dates: {trading_dates}")
    logger.info(f"Count: {len(trading_dates)}")

    # Should return only 2 days: Friday and Monday (not Sat/Sun)
    assert len(trading_dates) == 2, f"Expected 2 trading days, got {len(trading_dates)}"
    assert trading_dates[0] == start, f"First date should be {start}"
    assert trading_dates[1] == end, f"Last date should be {end}"

    logger.info("✅ Correctly excluded weekend\n")


def test_trading_dates_between_skips_holiday():
    """Test that trading dates exclude holidays."""
    logger.info("="*60)
    logger.info("TEST 4: Trading Dates Exclude Holidays")
    logger.info("="*60)

    dm = DataManager()

    # Range around July 4, 2024 (Thursday)
    # July 1-5, 2024: Mon, Tue, Wed, Thu(holiday), Fri
    start = '2024-07-01'  # Monday
    end = '2024-07-05'    # Friday

    trading_dates = dm.get_trading_dates_between(start, end)

    logger.info(f"Date range: {start} to {end} (includes July 4th holiday)")
    logger.info(f"Trading dates: {trading_dates}")
    logger.info(f"Count: {len(trading_dates)}")

    # Should return 4 days: Mon, Tue, Wed, Fri (not Thu July 4th)
    assert len(trading_dates) == 4, f"Expected 4 trading days, got {len(trading_dates)}"
    assert '2024-07-04' not in trading_dates, "July 4th should not be a trading day"

    logger.info("✅ Correctly excluded Independence Day holiday\n")


def test_last_trading_date_holiday():
    """Test last trading date when as_of is a holiday."""
    logger.info("="*60)
    logger.info("TEST 5: Last Trading Date (Holiday)")
    logger.info("="*60)

    dm = DataManager()

    # July 4, 2024 is a Thursday holiday
    holiday = '2024-07-04'
    last_trading = dm.get_last_trading_date(holiday)

    logger.info(f"Input (July 4th holiday): {holiday}")
    logger.info(f"Last trading date: {last_trading}")

    # Should return Wednesday July 3rd
    expected = '2024-07-03'
    assert last_trading == expected, f"Expected {expected}, got {last_trading}"

    logger.info(f"✅ Correctly returned day before holiday: {last_trading}\n")


def main():
    """Run all tests."""
    logger.info("SignalTide v3 - Trading Calendar Helper Tests")
    logger.info("="*60)
    logger.info("")

    try:
        test_last_trading_date_weekend()
        test_next_trading_date_weekend()
        test_trading_dates_between_skips_weekends()
        test_trading_dates_between_skips_holiday()
        test_last_trading_date_holiday()

        logger.info("="*60)
        logger.info("All trading calendar tests passed!")
        logger.info("="*60)

        return 0

    except Exception as e:
        logger.error(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
