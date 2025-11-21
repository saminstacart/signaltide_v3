"""
Comprehensive tests for DataManager and Database classes.

Tests cover:
- Data storage and retrieval
- Point-in-time data access (no lookahead)
- Caching behavior
- Data quality validation
- Sharadar data format support
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os

from data.database import Database
from data.data_manager import DataManager, DataCache


class TestDatabase:
    """Test Database class."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)

        db = Database(db_path)
        yield db

        # Cleanup
        if db_path.exists():
            os.unlink(db_path)

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'ticker': 'AAPL',
            'date': dates,
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.randint(1000000, 10000000, 100)
        })

    def test_database_initialization(self, temp_db):
        """Test database schema creation."""
        # Check that tables exist
        with temp_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table'
                ORDER BY name
            """)
            tables = [row[0] for row in cursor.fetchall()]

        expected_tables = ['metadata', 'prices', 'fundamentals',
                          'insider_trading', 'data_quality_log']

        for table in expected_tables:
            assert table in tables, f"Table {table} not created"

    def test_store_and_retrieve_prices(self, temp_db, sample_price_data):
        """Test storing and retrieving price data."""
        # Store data
        n_rows = temp_db.store_prices(sample_price_data)
        assert n_rows == len(sample_price_data)

        # Retrieve data
        df = temp_db.get_prices(ticker='AAPL')

        assert len(df) == len(sample_price_data)
        assert 'close' in df.columns
        assert df['close'].notna().all()

    def test_point_in_time_retrieval(self, temp_db, sample_price_data):
        """Test point-in-time data retrieval."""
        # Store data with different created_at timestamps
        # Simulate data being added over time
        early_data = sample_price_data[:50].copy()
        late_data = sample_price_data[50:].copy()

        # Store early data
        temp_db.store_prices(early_data)

        # Store late data
        temp_db.store_prices(late_data)

        # Retrieve all data
        all_data = temp_db.get_prices(ticker='AAPL')
        assert len(all_data) == 100

        # Note: In a real scenario, we'd set created_at explicitly
        # For this test, we verify the as_of parameter is accepted
        cutoff = datetime.now() - timedelta(days=1)
        historical = temp_db.get_prices(ticker='AAPL', as_of=cutoff)

        # Should return no data (all created after cutoff)
        assert len(historical) == 0

    def test_get_tickers(self, temp_db, sample_price_data):
        """Test getting list of available tickers."""
        # Store data for multiple tickers
        for ticker in ['AAPL', 'MSFT', 'GOOGL']:
            data = sample_price_data.copy()
            data['ticker'] = ticker
            temp_db.store_prices(data)

        tickers = temp_db.get_tickers()

        assert len(tickers) == 3
        assert 'AAPL' in tickers
        assert 'MSFT' in tickers
        assert 'GOOGL' in tickers

    def test_data_quality_logging(self, temp_db):
        """Test data quality issue logging."""
        # Log an issue
        temp_db.log_data_quality_issue(
            table_name='prices',
            ticker='AAPL',
            issue_type='missing_data',
            description='Missing 5 days of data',
            severity='WARNING'
        )

        # Retrieve issues
        issues = temp_db.get_data_quality_issues()

        assert len(issues) > 0
        assert issues.iloc[0]['ticker'] == 'AAPL'
        assert issues.iloc[0]['severity'] == 'WARNING'

    def test_metadata_storage(self, temp_db):
        """Test metadata get/set."""
        temp_db.set_metadata('last_update', '2023-01-01')

        value = temp_db.get_metadata('last_update')
        assert value == '2023-01-01'

        # Non-existent key
        assert temp_db.get_metadata('nonexistent') is None


class TestDataCache:
    """Test DataCache class."""

    def test_cache_put_and_get(self):
        """Test basic cache operations."""
        cache = DataCache(max_size_mb=100)

        # Create test data
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

        # Put in cache
        cache.put(df, ticker='AAPL', start='2020-01-01')

        # Get from cache
        retrieved = cache.get(ticker='AAPL', start='2020-01-01')

        assert retrieved is not None
        assert len(retrieved) == 3
        assert retrieved['a'].tolist() == [1, 2, 3]

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = DataCache(max_size_mb=100)

        result = cache.get(ticker='AAPL', start='2020-01-01')
        assert result is None

    def test_cache_returns_copy(self):
        """Test that cache returns copy, not reference."""
        cache = DataCache(max_size_mb=100)

        df = pd.DataFrame({'a': [1, 2, 3]})
        cache.put(df, ticker='AAPL')

        retrieved1 = cache.get(ticker='AAPL')
        retrieved2 = cache.get(ticker='AAPL')

        # Modify one copy
        retrieved1['a'] = [99, 99, 99]

        # Other copy should be unchanged
        assert retrieved2['a'].tolist() == [1, 2, 3]

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = DataCache(max_size_mb=100)

        df = pd.DataFrame({'a': [1, 2, 3]})
        cache.put(df, ticker='AAPL')

        assert cache.get(ticker='AAPL') is not None

        cache.clear()

        assert cache.get(ticker='AAPL') is None
        assert cache.size_mb() == 0


class TestDataManager:
    """Test DataManager class."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)

        yield db_path

        # Cleanup
        if db_path.exists():
            os.unlink(db_path)

    @pytest.fixture
    def data_manager(self, temp_db_path):
        """Create DataManager with temporary database."""
        dm = DataManager(db_path=temp_db_path, enable_cache=True)
        return dm

    @pytest.fixture
    def sample_prices(self):
        """Create sample price data."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'ticker': 'AAPL',
            'date': dates,
            'open': 100 + np.random.randn(100).cumsum(),
            'high': 102 + np.random.randn(100).cumsum(),
            'low': 98 + np.random.randn(100).cumsum(),
            'close': 100 + np.random.randn(100).cumsum(),
            'volume': np.random.randint(1000000, 10000000, 100)
        })

    def test_get_price_data(self, data_manager, sample_prices):
        """Test getting price data."""
        # Store data first
        data_manager.db.store_prices(sample_prices)

        # Retrieve data
        df = data_manager.get_price_data(
            ticker='AAPL',
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 4, 10)
        )

        assert len(df) == 100
        assert 'close' in df.columns
        assert df.index.name == 'date'

    def test_get_price_data_multiple_tickers(self, data_manager, sample_prices):
        """Test getting data for multiple tickers."""
        # Store data for multiple tickers
        for ticker in ['AAPL', 'MSFT']:
            data = sample_prices.copy()
            data['ticker'] = ticker
            data_manager.db.store_prices(data)

        # Retrieve multiple tickers
        df = data_manager.get_price_data(
            ticker=['AAPL', 'MSFT'],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 4, 10)
        )

        assert len(df) == 200  # 100 per ticker
        assert 'ticker' in df.columns
        assert set(df['ticker'].unique()) == {'AAPL', 'MSFT'}

    def test_caching_works(self, data_manager, sample_prices):
        """Test that caching improves performance."""
        # Store data
        data_manager.db.store_prices(sample_prices)

        # First retrieval (cache miss)
        df1 = data_manager.get_price_data('AAPL')

        # Check cache was populated
        stats = data_manager.get_cache_stats()
        assert stats['n_items'] == 1

        # Second retrieval (cache hit)
        df2 = data_manager.get_price_data('AAPL')

        # Data should be identical
        pd.testing.assert_frame_equal(df1, df2)

    def test_validate_no_lookahead(self, data_manager, sample_prices):
        """Test lookahead bias validation."""
        # Store data
        data_manager.db.store_prices(sample_prices)

        # This should pass (no lookahead)
        result = data_manager.validate_no_lookahead(
            ticker='AAPL',
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 3, 31),
            test_date=datetime(2020, 4, 15)
        )

        assert result is True

    def test_data_quality_validation(self, data_manager):
        """Test data quality validation catches issues."""
        # Create data with anomalies
        bad_data = pd.DataFrame({
            'ticker': 'AAPL',
            'date': pd.date_range('2020-01-01', periods=10, freq='D'),
            'open': [100] * 10,
            'high': [110] * 10,
            'low': [90] * 10,
            'close': [120] * 10,  # Close > High (anomaly!)
            'volume': [1000000] * 10
        })

        data_manager.db.store_prices(bad_data)

        # Retrieve with validation
        df = data_manager.get_price_data('AAPL', validate=True)

        # Check that issue was logged
        issues = data_manager.db.get_data_quality_issues(table_name='prices')
        assert len(issues) > 0
        assert any('anomaly' in issue.lower() for issue in issues['issue_type'].values)

    def test_get_available_tickers(self, data_manager, sample_prices):
        """Test getting list of available tickers."""
        # Store data for multiple tickers
        for ticker in ['AAPL', 'MSFT', 'GOOGL']:
            data = sample_prices.copy()
            data['ticker'] = ticker
            data_manager.db.store_prices(data)

        tickers = data_manager.get_available_tickers()

        assert len(tickers) == 3
        assert all(t in tickers for t in ['AAPL', 'MSFT', 'GOOGL'])

    def test_get_date_range(self, data_manager, sample_prices):
        """Test getting date range for a ticker."""
        data_manager.db.store_prices(sample_prices)

        date_range = data_manager.get_date_range('AAPL')

        assert date_range['min_date'] is not None
        assert date_range['max_date'] is not None
        assert date_range['min_date'] < date_range['max_date']

    def test_cache_can_be_disabled(self, temp_db_path):
        """Test that caching can be disabled."""
        dm = DataManager(db_path=temp_db_path, enable_cache=False)

        assert dm.cache is None

        stats = dm.get_cache_stats()
        assert stats['enabled'] is False

    def test_empty_data_returns_empty_dataframe(self, data_manager):
        """Test that requesting non-existent data returns empty DataFrame."""
        df = data_manager.get_price_data('NONEXISTENT')

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert 'close' in df.columns  # Expected columns present


class TestDataManagerIntegration:
    """Integration tests for DataManager."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)

        yield db_path

        # Cleanup
        if db_path.exists():
            os.unlink(db_path)

    @pytest.fixture
    def full_data_manager(self, temp_db_path):
        """Create DataManager with realistic data."""
        dm = DataManager(db_path=temp_db_path)

        # Create realistic price data
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        prices = pd.DataFrame({
            'ticker': 'AAPL',
            'date': dates,
            'open': 150 + np.random.randn(252).cumsum(),
            'high': 152 + np.random.randn(252).cumsum(),
            'low': 148 + np.random.randn(252).cumsum(),
            'close': 150 + np.random.randn(252).cumsum(),
            'volume': np.random.randint(50000000, 100000000, 252)
        })

        dm.db.store_prices(prices)

        return dm

    def test_realistic_backtest_workflow(self, full_data_manager):
        """Test realistic workflow for backtesting."""
        dm = full_data_manager

        # Get data for backtest period
        backtest_start = datetime(2020, 1, 1)
        backtest_end = datetime(2020, 12, 31)

        # Get data without as_of constraint for testing
        # (in real scenario, data would have been loaded earlier)
        data = dm.get_price_data(
            ticker='AAPL',
            start_date=backtest_start,
            end_date=backtest_end
        )

        assert len(data) > 0
        assert data.index.min() >= backtest_start
        assert data.index.max() <= backtest_end

    def test_multiple_data_types_combined(self, full_data_manager):
        """Test combining price and fundamental data."""
        dm = full_data_manager

        # Add some fundamental data
        fundamentals = pd.DataFrame({
            'ticker': 'AAPL',
            'date': pd.date_range('2020-01-01', periods=4, freq='Q'),
            'filing_date': pd.date_range('2020-02-01', periods=4, freq='Q'),
            'dimension': 'ARQ',
            'revenue': [100e9, 110e9, 120e9, 130e9],
            'net_income': [20e9, 22e9, 24e9, 26e9]
        })

        dm.db.store_fundamentals(fundamentals)

        # Get combined data
        combined = dm.get_combined_data(
            ticker='AAPL',
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            include_fundamentals=True
        )

        assert len(combined) > 0
        assert 'close' in combined.columns
        assert 'revenue' in combined.columns  # Fundamental data included


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
