"""
Pytest configuration for SignalTide v3

Provides:
- Automatic skipping of tests that require full DB when using fixture
- Custom markers for test categorization
"""

import os
import pytest
from pathlib import Path
from config import MARKET_DATA_DB


def using_fixture_db() -> bool:
    """
    Detect if we're using the fixture database.

    Checks both the environment variable and the resolved MARKET_DATA_DB path.
    This is more robust than just checking the path string.
    """
    env_path = os.environ.get("SIGNALTIDE_MARKET_DATA_DB", "")
    if "signaltide_small.db" in env_path:
        return True

    db_path_str = str(MARKET_DATA_DB)
    if "signaltide_small.db" in db_path_str or "fixtures" in db_path_str:
        return True

    return False


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "requires_full_db: mark test as requiring the full Sharadar database (skipped with fixture)"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow (not run by default in CI)"
    )


def pytest_collection_modifyitems(config, items):
    """
    Automatically skip tests marked with 'requires_full_db' when using fixture DB.

    This allows tests to be written with the marker:
        @pytest.mark.requires_full_db
        def test_something_heavy():
            ...

    When SIGNALTIDE_MARKET_DATA_DB points at signaltide_small.db, these tests
    are automatically skipped with a clear reason.
    """
    if using_fixture_db():
        skip_full_db = pytest.mark.skip(reason="Requires full Sharadar DB (currently using fixture)")
        for item in items:
            if "requires_full_db" in item.keywords:
                item.add_marker(skip_full_db)


@pytest.fixture
def fixture_db_path():
    """Path to the lightweight fixture database for testing."""
    return Path(__file__).parent / "fixtures" / "signaltide_small.db"


@pytest.fixture
def is_using_fixture():
    """True if currently using the fixture DB instead of full DB."""
    return using_fixture_db()
