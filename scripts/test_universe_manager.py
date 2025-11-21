#!/usr/bin/env python3
"""
Test UniverseManager - Validate expanded universe system

Tests:
1. Manual universe (backward compatibility)
2. Top N universe by market cap
3. S&P 500 proxy (top 500)
4. Market cap range filtering
5. Sector filtering
6. Point-in-time correctness
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.universe_manager import UniverseManager
from config import get_logger

logger = get_logger(__name__)


def test_manual_universe():
    """Test manual universe selection."""
    logger.info("="*60)
    logger.info("TEST 1: Manual Universe")
    logger.info("="*60)

    um = UniverseManager()

    # Test with known tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    universe = um.get_universe(
        universe_type='manual',
        as_of_date='2023-01-01',
        manual_tickers=tickers
    )

    logger.info(f"Input: {tickers}")
    logger.info(f"Output: {universe}")
    logger.info(f"Status: {'PASS' if set(universe) == set(tickers) else 'FAIL'}")

    assert set(universe) == set(tickers), "Manual universe should return exact tickers"

    logger.info("✅ Manual universe test passed\n")


def test_top_n_universe():
    """Test top N by market cap."""
    logger.info("="*60)
    logger.info("TEST 2: Top N Universe")
    logger.info("="*60)

    um = UniverseManager()

    # Test top 50
    universe = um.get_universe(
        universe_type='top_N',
        as_of_date='2023-01-01',
        top_n=50,
        min_price=5.0
    )

    logger.info(f"Requested: Top 50")
    logger.info(f"Received: {len(universe)} stocks")
    logger.info(f"Sample: {universe[:10]}")
    logger.info(f"Status: {'PASS' if len(universe) == 50 else 'FAIL'}")

    assert len(universe) == 50, f"Expected 50 stocks, got {len(universe)}"

    # Get info
    info = um.get_universe_info(universe, '2023-01-01')
    logger.info(f"Market cap range: ${info['marketcap'].min()/1e9:.1f}B - ${info['marketcap'].max()/1e9:.1f}B")

    logger.info("✅ Top N test passed\n")


def test_sp500_proxy():
    """Test S&P 500 proxy (top 500 by market cap)."""
    logger.info("="*60)
    logger.info("TEST 3: S&P 500 Proxy")
    logger.info("="*60)

    um = UniverseManager()

    universe = um.get_universe(
        universe_type='sp500_proxy',
        as_of_date='2023-01-01',
        min_price=5.0
    )

    logger.info(f"Requested: S&P 500 proxy")
    logger.info(f"Received: {len(universe)} stocks")
    logger.info(f"Sample: {universe[:10]}")
    logger.info(f"Status: {'PASS' if len(universe) == 500 else 'FAIL'}")

    assert len(universe) == 500, f"Expected 500 stocks, got {len(universe)}"

    # Get info
    info = um.get_universe_info(universe, '2023-01-01')
    logger.info(f"Market cap range: ${info['marketcap'].min()/1e9:.1f}B - ${info['marketcap'].max()/1e9:.1f}B")
    logger.info(f"Sectors: {dict(list(info['sector'].value_counts().items())[:5])}")

    logger.info("✅ S&P 500 proxy test passed\n")


def test_market_cap_range():
    """Test market cap range filtering."""
    logger.info("="*60)
    logger.info("TEST 4: Market Cap Range")
    logger.info("="*60)

    um = UniverseManager()

    # Test large cap (>$10B)
    min_cap = 10e9
    universe = um.get_universe(
        universe_type='market_cap_range',
        as_of_date='2023-01-01',
        min_market_cap=min_cap,
        min_price=5.0
    )

    logger.info(f"Requested: Market cap > ${min_cap/1e9:.0f}B")
    logger.info(f"Received: {len(universe)} stocks")
    logger.info(f"Sample: {universe[:10]}")

    # Verify all stocks meet minimum
    info = um.get_universe_info(universe, '2023-01-01')
    min_actual = info['marketcap'].min()
    logger.info(f"Actual min market cap: ${min_actual/1e9:.1f}B")
    logger.info(f"Status: {'PASS' if min_actual >= min_cap else 'FAIL'}")

    assert min_actual >= min_cap, f"Minimum market cap violated: ${min_actual/1e9:.1f}B < ${min_cap/1e9:.0f}B"

    logger.info("✅ Market cap range test passed\n")


def test_sector_filtering():
    """Test sector filtering."""
    logger.info("="*60)
    logger.info("TEST 5: Sector Filtering")
    logger.info("="*60)

    um = UniverseManager()

    # Test Technology sector
    universe = um.get_universe(
        universe_type='sector',
        as_of_date='2023-01-01',
        sectors=['Technology'],
        min_price=5.0
    )

    logger.info(f"Requested: Technology sector")
    logger.info(f"Received: {len(universe)} stocks")
    logger.info(f"Sample: {universe[:10]}")

    # Verify all stocks are Technology
    info = um.get_universe_info(universe, '2023-01-01')
    sectors = info['sector'].unique()
    logger.info(f"Unique sectors: {sectors}")
    logger.info(f"Status: {'PASS' if len(sectors) == 1 and sectors[0] == 'Technology' else 'FAIL'}")

    assert len(sectors) == 1 and sectors[0] == 'Technology', f"Expected only Technology sector, got {sectors}"

    logger.info("✅ Sector filtering test passed\n")


def test_point_in_time():
    """Test point-in-time correctness."""
    logger.info("="*60)
    logger.info("TEST 6: Point-in-Time Correctness")
    logger.info("="*60)

    um = UniverseManager()

    # Test with a date before COIN IPO (2021-04-14)
    universe_before = um.get_universe(
        universe_type='manual',
        as_of_date='2021-04-01',  # Before COIN IPO
        manual_tickers=['AAPL', 'COIN']  # COIN shouldn't be included
    )

    # Test with a date after COIN IPO
    universe_after = um.get_universe(
        universe_type='manual',
        as_of_date='2021-04-20',  # After COIN IPO
        manual_tickers=['AAPL', 'COIN']  # Both should be included
    )

    logger.info(f"Before COIN IPO (2021-04-01): {universe_before}")
    logger.info(f"After COIN IPO (2021-04-20): {universe_after}")

    before_test = 'COIN' not in universe_before and 'AAPL' in universe_before
    after_test = 'COIN' in universe_after and 'AAPL' in universe_after

    logger.info(f"Before IPO excludes COIN: {'PASS' if before_test else 'FAIL'}")
    logger.info(f"After IPO includes COIN: {'PASS' if after_test else 'FAIL'}")

    assert before_test, "COIN should not be in universe before IPO"
    assert after_test, "COIN should be in universe after IPO"

    logger.info("✅ Point-in-time test passed\n")


def test_sp500_actual():
    """Test actual S&P 500 membership from dim_universe_membership."""
    logger.info("="*60)
    logger.info("TEST 7: S&P 500 Actual Membership")
    logger.info("="*60)

    um = UniverseManager()

    # Test before TSLA joined S&P 500 (Dec 21, 2020)
    universe_before = um.get_universe(
        universe_type='sp500_actual',
        as_of_date='2020-12-01'
    )

    # Test after TSLA joined
    universe_after = um.get_universe(
        universe_type='sp500_actual',
        as_of_date='2021-01-01'
    )

    logger.info(f"S&P 500 on 2020-12-01: {len(universe_before)} members")
    logger.info(f"S&P 500 on 2021-01-01: {len(universe_after)} members")

    # AAPL should be in both (long-time member)
    aapl_test = 'AAPL' in universe_before and 'AAPL' in universe_after

    # TSLA should NOT be in before, but SHOULD be in after
    tsla_test = 'TSLA' not in universe_before and 'TSLA' in universe_after

    logger.info(f"AAPL in both periods: {'PASS' if aapl_test else 'FAIL'}")
    logger.info(f"TSLA joined Dec 2020: {'PASS' if tsla_test else 'FAIL'}")

    # Verify count is reasonable (~500)
    count_test = 450 <= len(universe_after) <= 550

    logger.info(f"Universe size reasonable: {'PASS' if count_test else 'FAIL'}")

    assert aapl_test, "AAPL should be in S&P 500 in both periods"
    assert tsla_test, "TSLA should join S&P 500 in Dec 2020"
    assert count_test, f"S&P 500 should have ~500 members, got {len(universe_after)}"

    logger.info("✅ S&P 500 actual membership test passed\n")


def test_sp500_former_members():
    """Test that former S&P 500 members are excluded after exit."""
    logger.info("="*60)
    logger.info("TEST 8: S&P 500 Former Members")
    logger.info("="*60)

    from pathlib import Path
    from core.db import get_read_only_connection

    um = UniverseManager()

    # Find a ticker that exited S&P 500
    db_path = Path("data/databases/market_data.db")
    conn = get_read_only_connection(db_path=db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT ticker, membership_end_date
        FROM dim_universe_membership
        WHERE universe_name = 'sp500_actual'
          AND membership_end_date IS NOT NULL
        ORDER BY membership_end_date DESC
        LIMIT 1;
    """)

    result = cursor.fetchone()
    conn.close()

    if result:
        ticker, exit_date = result
        logger.info(f"Testing former member: {ticker} (exited {exit_date})")

        # Get universe after exit
        universe_after_exit = um.get_universe(
            universe_type='sp500_actual',
            as_of_date=exit_date
        )

        excluded_test = ticker not in universe_after_exit

        logger.info(f"{ticker} excluded after exit: {'PASS' if excluded_test else 'FAIL'}")

        assert excluded_test, f"{ticker} should be excluded after {exit_date}"

        logger.info("✅ Former members test passed\n")
    else:
        logger.info("⚠️  No former members found in database, skipping test\n")


def test_pit_boundary_semantics():
    """Test precise PIT semantics on membership boundaries.

    INVARIANT being tested:
    - membership_start_date is INCLUSIVE (first date in universe)
    - membership_end_date is EXCLUSIVE (first date NOT in universe)

    This test ensures the [start, end) interval semantics are preserved.
    If this test fails, someone has changed the PIT query logic.
    """
    logger.info("="*60)
    logger.info("TEST 9: PIT Boundary Semantics")
    logger.info("="*60)

    from pathlib import Path
    from core.db import get_read_only_connection
    from datetime import datetime, timedelta

    um = UniverseManager()

    # Find a ticker with non-null end_date
    db_path = Path("data/databases/market_data.db")
    conn = get_read_only_connection(db_path=db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT ticker, membership_start_date, membership_end_date
        FROM dim_universe_membership
        WHERE universe_name = 'sp500_actual'
          AND membership_end_date IS NOT NULL
        ORDER BY membership_end_date DESC
        LIMIT 1;
    """)

    result = cursor.fetchone()
    conn.close()

    if not result:
        logger.info("⚠️  No tickers with end_date found, skipping test\n")
        return

    ticker, start_date, end_date = result
    logger.info(f"Testing ticker: {ticker}")
    logger.info(f"  membership_start_date: {start_date}")
    logger.info(f"  membership_end_date: {end_date}")

    # Convert to datetime for date arithmetic
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    # Calculate boundary dates
    before_start = (start_dt - timedelta(days=1)).strftime('%Y-%m-%d')
    on_start = start_date
    day_before_end = (end_dt - timedelta(days=1)).strftime('%Y-%m-%d')
    on_end = end_date
    after_end = (end_dt + timedelta(days=1)).strftime('%Y-%m-%d')

    logger.info(f"\nBoundary dates:")
    logger.info(f"  before_start: {before_start}")
    logger.info(f"  on_start:     {on_start}")
    logger.info(f"  before_end:   {day_before_end}")
    logger.info(f"  on_end:       {on_end}")
    logger.info(f"  after_end:    {after_end}")

    # Test 1: Day before start_date should EXCLUDE ticker
    universe_before_start = um.get_universe(
        universe_type='sp500_actual',
        as_of_date=before_start
    )
    test1 = ticker not in universe_before_start
    logger.info(f"\nTest 1 - Before start ({before_start}): {'PASS' if test1 else 'FAIL'}")
    logger.info(f"  Expected: EXCLUDED, Got: {'EXCLUDED' if test1 else 'INCLUDED'}")

    # Test 2: On start_date should INCLUDE ticker (start is inclusive)
    universe_on_start = um.get_universe(
        universe_type='sp500_actual',
        as_of_date=on_start
    )
    test2 = ticker in universe_on_start
    logger.info(f"\nTest 2 - On start ({on_start}): {'PASS' if test2 else 'FAIL'}")
    logger.info(f"  Expected: INCLUDED, Got: {'INCLUDED' if test2 else 'EXCLUDED'}")

    # Test 3: Day before end_date should INCLUDE ticker
    universe_before_end = um.get_universe(
        universe_type='sp500_actual',
        as_of_date=day_before_end
    )
    test3 = ticker in universe_before_end
    logger.info(f"\nTest 3 - Before end ({day_before_end}): {'PASS' if test3 else 'FAIL'}")
    logger.info(f"  Expected: INCLUDED, Got: {'INCLUDED' if test3 else 'EXCLUDED'}")

    # Test 4: On end_date should EXCLUDE ticker (end is exclusive)
    universe_on_end = um.get_universe(
        universe_type='sp500_actual',
        as_of_date=on_end
    )
    test4 = ticker not in universe_on_end
    logger.info(f"\nTest 4 - On end ({on_end}): {'PASS' if test4 else 'FAIL'}")
    logger.info(f"  Expected: EXCLUDED, Got: {'EXCLUDED' if test4 else 'INCLUDED'}")

    # Test 5: Day after end_date should EXCLUDE ticker
    universe_after_end = um.get_universe(
        universe_type='sp500_actual',
        as_of_date=after_end
    )
    test5 = ticker not in universe_after_end
    logger.info(f"\nTest 5 - After end ({after_end}): {'PASS' if test5 else 'FAIL'}")
    logger.info(f"  Expected: EXCLUDED, Got: {'EXCLUDED' if test5 else 'INCLUDED'}")

    # All tests must pass
    assert test1, f"Before start: {ticker} should be EXCLUDED on {before_start}"
    assert test2, f"On start: {ticker} should be INCLUDED on {on_start} (start is inclusive)"
    assert test3, f"Before end: {ticker} should be INCLUDED on {day_before_end}"
    assert test4, f"On end: {ticker} should be EXCLUDED on {on_end} (end is exclusive)"
    assert test5, f"After end: {ticker} should be EXCLUDED on {after_end}"

    logger.info("\n✅ PIT boundary semantics test passed")
    logger.info("   Confirmed: start is INCLUSIVE, end is EXCLUSIVE [start, end)\n")


def test_pit_null_end_date():
    """Test that NULL end_date means membership continues indefinitely."""
    logger.info("="*60)
    logger.info("TEST 10: PIT NULL End Date")
    logger.info("="*60)

    from pathlib import Path
    from core.db import get_read_only_connection

    um = UniverseManager()

    # Find a ticker with NULL end_date (current member)
    db_path = Path("data/databases/market_data.db")
    conn = get_read_only_connection(db_path=db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT ticker, membership_start_date
        FROM dim_universe_membership
        WHERE universe_name = 'sp500_actual'
          AND membership_end_date IS NULL
        LIMIT 1;
    """)

    result = cursor.fetchone()
    conn.close()

    if not result:
        logger.info("⚠️  No current members found, skipping test\n")
        return

    ticker, start_date = result
    logger.info(f"Testing ticker: {ticker} (current member)")
    logger.info(f"  membership_start_date: {start_date}")
    logger.info(f"  membership_end_date: NULL (ongoing)")

    # Test with a far-future date (should still be included)
    future_date = '2035-12-31'

    universe_future = um.get_universe(
        universe_type='sp500_actual',
        as_of_date=future_date
    )

    test = ticker in universe_future
    logger.info(f"\nTest - Far future ({future_date}): {'PASS' if test else 'FAIL'}")
    logger.info(f"  Expected: INCLUDED, Got: {'INCLUDED' if test else 'EXCLUDED'}")

    assert test, f"NULL end_date: {ticker} should be INCLUDED on {future_date}"

    logger.info("✅ NULL end_date test passed")
    logger.info("   Confirmed: NULL end_date means membership continues indefinitely\n")


def main():
    """Run all tests."""
    logger.info("\n" + "="*60)
    logger.info("UNIVERSE MANAGER COMPREHENSIVE TEST")
    logger.info("="*60 + "\n")

    tests = [
        test_manual_universe,
        test_top_n_universe,
        test_sp500_proxy,
        test_market_cap_range,
        test_sector_filtering,
        test_point_in_time,
        test_sp500_actual,
        test_sp500_former_members,
        test_pit_boundary_semantics,
        test_pit_null_end_date,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            logger.error(f"❌ {test.__name__} FAILED: {e}")
            failed += 1

    logger.info("="*60)
    logger.info(f"RESULTS: {passed} passed, {failed} failed")
    logger.info("="*60)

    if failed == 0:
        logger.info("✅ ALL TESTS PASSED")
    else:
        logger.error(f"❌ {failed} TESTS FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()
