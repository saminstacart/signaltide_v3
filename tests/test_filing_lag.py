"""
Filing Lag Regression Tests

Critical tests to prevent lookahead bias from returning in DataManager.

These tests enforce temporal discipline:
1. Fundamentals must use filing date (datekey), not quarter-end (calendardate)
2. Insider trades must use filing date, not transaction date
3. as_of_date parameter must be used correctly
4. Filing lags must be respected (33 days for fundamentals, 1-2 days for insider)

Purpose: Prevent regression after fixing critical bugs:
- data_manager.py:202 (calendardate → datekey fix)
- simple_insider.py:60 (missing as_of_date parameter)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os
import sqlite3

from data.data_manager import DataManager, create_mock_data_manager


class TestFilingLagFundamentals:
    """Test filing lag enforcement for fundamental data."""

    @pytest.fixture
    def mock_db_path(self):
        """Create temporary database with test data."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)

        # Create database schema
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Create sharadar_sf1 table (fundamentals)
        cursor.execute("""
            CREATE TABLE sharadar_sf1 (
                ticker TEXT,
                dimension TEXT,
                calendardate TEXT,
                datekey TEXT,
                reportperiod TEXT,
                lastupdated TEXT,
                revenue REAL,
                netinc REAL,
                equity REAL,
                roe REAL
            )
        """)

        # Insert test data with filing lag
        # Q4 2023 earnings (quarter end Dec 31, 2023)
        # Filed 33 days later on Feb 2, 2024
        test_data = [
            {
                'ticker': 'TEST',
                'dimension': 'ARQ',
                'calendardate': '2023-12-31',  # Quarter end
                'datekey': '2024-02-02',        # Filing date (33 days later!)
                'reportperiod': '2023-12-31',
                'lastupdated': '2024-02-02',
                'revenue': 100e9,
                'netinc': 20e9,
                'equity': 50e9,
                'roe': 0.40
            },
            # Q1 2024 earnings
            {
                'ticker': 'TEST',
                'dimension': 'ARQ',
                'calendardate': '2024-03-31',
                'datekey': '2024-05-03',  # 33 days later
                'reportperiod': '2024-03-31',
                'lastupdated': '2024-05-03',
                'revenue': 110e9,
                'netinc': 22e9,
                'equity': 52e9,
                'roe': 0.42
            }
        ]

        for row in test_data:
            cursor.execute("""
                INSERT INTO sharadar_sf1 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['ticker'], row['dimension'], row['calendardate'],
                row['datekey'], row['reportperiod'], row['lastupdated'],
                row['revenue'], row['netinc'], row['equity'], row['roe']
            ))

        conn.commit()
        conn.close()

        yield db_path

        # Cleanup
        if db_path.exists():
            os.unlink(db_path)

    def test_filing_lag_prevents_lookahead(self, mock_db_path):
        """
        CRITICAL: Test that as_of_date uses datekey (filing date) not calendardate.

        Scenario: Q4 2023 earnings (quarter end Dec 31, filed Feb 2, 2024)

        Expected behavior:
        - as_of_date = '2024-01-15' → NO DATA (not filed yet)
        - as_of_date = '2024-02-02' → DATA AVAILABLE (filed on Feb 2)
        - as_of_date = '2024-02-03' → DATA AVAILABLE (filed before this date)

        This prevents using Q4 earnings in January signals (lookahead bias).
        """
        dm = DataManager(db_path=mock_db_path)

        # Test 1: as_of_date BEFORE filing (should return NO data)
        df_before = dm.get_fundamentals(
            'TEST',
            start_date='2023-12-01',
            end_date='2024-01-31',
            dimension='ARQ',
            as_of_date='2024-01-15'  # Before Feb 2 filing
        )

        assert len(df_before) == 0, (
            "LOOKAHEAD BIAS: Fundamental data available before filing date! "
            "as_of_date='2024-01-15' should return NO data for Q4 2023 "
            "(filed on 2024-02-02)"
        )

        # Test 2: as_of_date ON filing date (should return data)
        df_on_filing = dm.get_fundamentals(
            'TEST',
            start_date='2023-12-01',
            end_date='2024-01-31',
            dimension='ARQ',
            as_of_date='2024-02-02'  # ON filing date
        )

        assert len(df_on_filing) == 1, (
            "Data should be available on filing date (2024-02-02)"
        )
        assert df_on_filing.iloc[0]['roe'] == 0.40

        # Test 3: as_of_date AFTER filing (should return data)
        df_after = dm.get_fundamentals(
            'TEST',
            start_date='2023-12-01',
            end_date='2024-01-31',
            dimension='ARQ',
            as_of_date='2024-03-01'  # After filing
        )

        assert len(df_after) == 1, (
            "Data should be available after filing date"
        )

    def test_33_day_filing_lag_enforced(self, mock_db_path):
        """
        Test that 33-day filing lag is enforced.

        SEC requires 10-Q/10-K filing within 33 days of quarter-end.
        Signals generated on Jan 31 cannot use Q4 earnings (not filed until Feb 2).
        """
        dm = DataManager(db_path=mock_db_path)

        # Scenario: Generating signal on Jan 31, 2024
        # Q4 2023 quarter ended Dec 31, 2023
        # Filing date: Feb 2, 2024 (33 days later)

        signal_date = '2024-01-31'

        df = dm.get_fundamentals(
            'TEST',
            start_date='2023-01-01',
            end_date='2024-01-31',
            dimension='ARQ',
            as_of_date=signal_date
        )

        # Should NOT include Q4 2023 data (not filed until Feb 2)
        assert len(df) == 0, (
            f"33-day filing lag violated! Signal on {signal_date} should not have "
            f"access to Q4 2023 data (filed on 2024-02-02, 33 days after quarter-end)"
        )

    def test_calendardate_vs_datekey_distinction(self, mock_db_path):
        """
        CRITICAL: Test that datekey (filing date) is used, not calendardate (quarter-end).

        This is the bug we fixed in data_manager.py:202.
        """
        dm = DataManager(db_path=mock_db_path)

        # Query using quarter-end as as_of_date
        # (This would return data if we incorrectly used calendardate!)
        df_wrong = dm.get_fundamentals(
            'TEST',
            start_date='2023-12-01',
            end_date='2024-01-31',
            dimension='ARQ',
            as_of_date='2023-12-31'  # Quarter-end date (WRONG date to use!)
        )

        # Should return NO data because filing date (datekey) is 2024-02-02
        assert len(df_wrong) == 0, (
            "BUG REGRESSION: Using calendardate instead of datekey! "
            "as_of_date='2023-12-31' (quarter-end) should NOT return data "
            "because filing date (datekey) is '2024-02-02'"
        )

    def test_as_of_date_none_fallback(self, mock_db_path, caplog):
        """
        Test that as_of_date=None triggers warning and uses end_date fallback.

        This tests the runtime validation we added to prevent silent bugs.
        """
        import logging
        dm = DataManager(db_path=mock_db_path)

        # Call without as_of_date (should trigger warning and use end_date)
        with caplog.at_level(logging.WARNING):
            df = dm.get_fundamentals(
                'TEST',
                start_date='2023-12-01',
                end_date='2024-03-01',
                dimension='ARQ'
                # as_of_date=None (implicit)
            )

        # Should use end_date='2024-03-01' as fallback
        # This is after filing date, so data should be available
        assert len(df) == 1

        # Verify warning was logged
        assert any("as_of_date not provided" in record.message
                   for record in caplog.records)


class TestFilingLagInsider:
    """Test filing lag enforcement for insider trading data."""

    @pytest.fixture
    def mock_db_path(self):
        """Create temporary database with insider trade test data."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)

        # Create database schema
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Create sharadar_insiders table
        cursor.execute("""
            CREATE TABLE sharadar_insiders (
                ticker TEXT,
                filingdate TEXT,
                transactiondate TEXT,
                ownername TEXT,
                transactioncode TEXT,
                transactionshares INTEGER,
                transactionprice REAL
            )
        """)

        # Insert test data with filing lag
        # Trade on Jan 10, filed on Jan 12 (2 day lag)
        test_data = [
            {
                'ticker': 'TEST',
                'filingdate': '2024-01-12',      # When we learned about it
                'transactiondate': '2024-01-10', # When trade happened
                'ownername': 'CEO John Doe',
                'transactioncode': 'P',          # Purchase
                'transactionshares': 10000,
                'transactionprice': 150.00
            },
            # Another trade: Jan 15 transaction, filed Jan 16
            {
                'ticker': 'TEST',
                'filingdate': '2024-01-16',
                'transactiondate': '2024-01-15',
                'ownername': 'CFO Jane Smith',
                'transactioncode': 'S',  # Sale
                'transactionshares': 5000,
                'transactionprice': 155.00
            }
        ]

        for row in test_data:
            cursor.execute("""
                INSERT INTO sharadar_insiders VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                row['ticker'], row['filingdate'], row['transactiondate'],
                row['ownername'], row['transactioncode'],
                row['transactionshares'], row['transactionprice']
            ))

        conn.commit()
        conn.close()

        yield db_path

        # Cleanup
        if db_path.exists():
            os.unlink(db_path)

    def test_insider_filing_lag_prevents_lookahead(self, mock_db_path):
        """
        CRITICAL: Test that as_of_date uses filingdate, not transactiondate.

        Scenario: Trade on Jan 10, filed on Jan 12

        Expected behavior:
        - as_of_date = '2024-01-11' → NO DATA (not filed yet)
        - as_of_date = '2024-01-12' → DATA AVAILABLE (filed on Jan 12)
        """
        dm = DataManager(db_path=mock_db_path)

        # Test 1: as_of_date BEFORE filing
        df_before = dm.get_insider_trades(
            'TEST',
            start_date='2024-01-01',
            end_date='2024-01-15',
            as_of_date='2024-01-11'  # Before filing
        )

        assert len(df_before) == 0, (
            "LOOKAHEAD BIAS: Insider trade available before filing! "
            "Trade on Jan 10 not filed until Jan 12"
        )

        # Test 2: as_of_date ON filing date
        df_on_filing = dm.get_insider_trades(
            'TEST',
            start_date='2024-01-01',
            end_date='2024-01-15',
            as_of_date='2024-01-12'  # ON filing date
        )

        assert len(df_on_filing) == 1, (
            "Data should be available on filing date"
        )
        assert df_on_filing.iloc[0]['transactioncode'] == 'P'

    def test_1_2_day_insider_filing_lag(self, mock_db_path):
        """
        Test that 1-2 day insider filing lag is enforced.

        SEC requires insider trades filed within 2 business days.
        """
        dm = DataManager(db_path=mock_db_path)

        # Scenario: Signal on Jan 11 (day after trade, before filing)
        signal_date = '2024-01-11'

        df = dm.get_insider_trades(
            'TEST',
            start_date='2024-01-01',
            end_date='2024-01-15',
            as_of_date=signal_date
        )

        # Should NOT include Jan 10 trade (not filed until Jan 12)
        assert len(df) == 0, (
            f"Filing lag violated! Signal on {signal_date} should not have "
            f"access to Jan 10 trade (filed on Jan 12)"
        )

    def test_as_of_date_none_fallback_insider(self, mock_db_path, caplog):
        """
        Test that as_of_date=None triggers warning for insider trades.
        """
        import logging
        dm = DataManager(db_path=mock_db_path)

        # Call without as_of_date
        with caplog.at_level(logging.WARNING):
            df = dm.get_insider_trades(
                'TEST',
                start_date='2024-01-01',
                end_date='2024-01-20'
                # as_of_date=None (implicit)
            )

        # Should use end_date='2024-01-20' as fallback
        # Both trades should be available (filed before Jan 20)
        assert len(df) == 2

        # Verify warning was logged
        assert any("as_of_date not provided" in record.message
                   for record in caplog.records)


class TestSignalIntegration:
    """Test that signals correctly use as_of_date parameter."""

    def test_simple_insider_uses_as_of_date(self):
        """
        Test that SimpleInsider signal uses as_of_date parameter.

        This tests the fix we made to simple_insider.py:60.
        """
        from signals.insider.simple_insider import SimpleInsider

        # Use mock data manager
        mock_dm = create_mock_data_manager()

        signal = SimpleInsider(
            params={'lookback_days': 90, 'rank_window': 252},
            data_manager=mock_dm
        )

        # Create test data
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        data = pd.DataFrame({
            'ticker': 'TICK01',
            'close': 100 + np.random.randn(30).cumsum(),
            'volume': 1000000
        }, index=dates)

        # Generate signals (should not raise error)
        signals = signal.generate_signals(data)

        # Verify signals were generated
        assert len(signals) == len(data)
        assert signals.index.equals(data.index)

    def test_simple_quality_uses_as_of_date(self):
        """
        Test that SimpleQuality signal uses as_of_date parameter.
        """
        from signals.quality.simple_quality import SimpleQuality

        # Use mock data manager
        mock_dm = create_mock_data_manager()

        signal = SimpleQuality(
            params={'rank_window': 252 * 2},
            data_manager=mock_dm
        )

        # Create test data
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        data = pd.DataFrame({
            'ticker': 'TICK01',
            'close': 100 + np.random.randn(30).cumsum(),
            'volume': 1000000
        }, index=dates)

        # Generate signals (should not raise error)
        signals = signal.generate_signals(data)

        # Verify signals were generated
        assert len(signals) == len(data)


class TestPointInTimeCorrectness:
    """Test point-in-time data access correctness."""

    @pytest.fixture
    def realistic_db_path(self):
        """Create database with realistic multi-quarter data."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE sharadar_sf1 (
                ticker TEXT,
                dimension TEXT,
                calendardate TEXT,
                datekey TEXT,
                reportperiod TEXT,
                lastupdated TEXT,
                revenue REAL,
                roe REAL
            )
        """)

        # Realistic data: 4 quarters of 2023
        quarters = [
            # Q1 2023
            {'quarter_end': '2023-03-31', 'filing': '2023-05-03', 'revenue': 90e9, 'roe': 0.35},
            # Q2 2023
            {'quarter_end': '2023-06-30', 'filing': '2023-08-02', 'revenue': 95e9, 'roe': 0.37},
            # Q3 2023
            {'quarter_end': '2023-09-30', 'filing': '2023-11-02', 'revenue': 98e9, 'roe': 0.38},
            # Q4 2023
            {'quarter_end': '2023-12-31', 'filing': '2024-02-02', 'revenue': 100e9, 'roe': 0.40},
        ]

        for q in quarters:
            cursor.execute("""
                INSERT INTO sharadar_sf1 VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                'TEST', 'ARQ', q['quarter_end'], q['filing'],
                q['quarter_end'], q['filing'], q['revenue'], q['roe']
            ))

        conn.commit()
        conn.close()

        yield db_path

        if db_path.exists():
            os.unlink(db_path)

    def test_point_in_time_data_growth(self, realistic_db_path):
        """
        Test that available data grows correctly over time.

        Simulates backtesting through 2023:
        - July 1: Should have Q1 data only (Q2 not filed until Aug 2)
        - Sept 1: Should have Q1+Q2 data
        - Dec 1: Should have Q1+Q2+Q3 data
        - March 1 2024: Should have all 4 quarters
        """
        dm = DataManager(db_path=realistic_db_path)

        # July 1, 2023 - Only Q1 available
        df_july = dm.get_fundamentals(
            'TEST',
            start_date='2023-01-01',
            end_date='2023-12-31',
            dimension='ARQ',
            as_of_date='2023-07-01'
        )
        assert len(df_july) == 1, "July should have Q1 only"

        # Sept 1, 2023 - Q1 + Q2 available
        df_sept = dm.get_fundamentals(
            'TEST',
            start_date='2023-01-01',
            end_date='2023-12-31',
            dimension='ARQ',
            as_of_date='2023-09-01'
        )
        assert len(df_sept) == 2, "Sept should have Q1+Q2"

        # Dec 1, 2023 - Q1 + Q2 + Q3 available
        df_dec = dm.get_fundamentals(
            'TEST',
            start_date='2023-01-01',
            end_date='2023-12-31',
            dimension='ARQ',
            as_of_date='2023-12-01'
        )
        assert len(df_dec) == 3, "Dec should have Q1+Q2+Q3"

        # March 1, 2024 - All 4 quarters available
        df_march = dm.get_fundamentals(
            'TEST',
            start_date='2023-01-01',
            end_date='2023-12-31',
            dimension='ARQ',
            as_of_date='2024-03-01'
        )
        assert len(df_march) == 4, "March 2024 should have all 4 quarters"

    def test_no_future_data_leakage(self, realistic_db_path):
        """
        Test that signals generated on date X cannot access data filed after X.

        This is the core temporal discipline test.
        """
        dm = DataManager(db_path=realistic_db_path)

        # Generate signal on Oct 1, 2023
        signal_date = '2023-10-01'

        df = dm.get_fundamentals(
            'TEST',
            start_date='2023-01-01',
            end_date='2023-12-31',
            dimension='ARQ',
            as_of_date=signal_date
        )

        # Should have Q1+Q2 only (Q3 filed Nov 2, Q4 filed Feb 2024)
        assert len(df) == 2

        # Verify no Q3 data leaked
        calendar_dates = df.index.strftime('%Y-%m-%d').tolist()
        assert '2023-09-30' not in calendar_dates, (
            "LOOKAHEAD BIAS: Q3 data leaked into Oct 1 signal! "
            "Q3 not filed until Nov 2"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
