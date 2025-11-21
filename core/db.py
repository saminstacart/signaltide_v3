#!/usr/bin/env python3
"""
Database Helper - Central database connection management

Ensures all database connections have proper configuration:
- Foreign keys enabled (PRAGMA foreign_keys = ON)
- WAL journal mode for concurrency
- Normal synchronous for balance of safety/speed

See docs/DATA_ARCHITECTURE.md for architecture details.
"""

import sqlite3
from pathlib import Path
from typing import Optional

# Default database path
DEFAULT_DB_PATH = Path("data/databases/market_data.db")


def get_connection(
    db_path: Optional[Path] = None,
    read_only: bool = False
) -> sqlite3.Connection:
    """
    Get database connection with proper configuration.

    Args:
        db_path: Path to database file (defaults to market_data.db)
        read_only: If True, open in read-only mode (default False)

    Returns:
        Configured SQLite connection with:
        - Foreign keys enabled
        - WAL journal mode (unless read-only)
        - Normal synchronous mode

    Raises:
        FileNotFoundError: If database file doesn't exist

    Example:
        >>> from core.db import get_connection
        >>> conn = get_connection()
        >>> cursor = conn.cursor()
        >>> cursor.execute("SELECT COUNT(*) FROM dim_trading_calendar")
        >>> count = cursor.fetchone()[0]
        >>> conn.close()
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found: {db_path}\n"
            f"Expected location: {db_path.absolute()}"
        )

    # Open connection (read-only or read-write)
    if read_only:
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    else:
        conn = sqlite3.connect(db_path)

    # CRITICAL: Enable foreign keys
    # SQLite disables foreign keys by default for backwards compatibility.
    # All SignalTide v3 code must have foreign keys enabled to ensure
    # data integrity in dimensional tables.
    conn.execute("PRAGMA foreign_keys = ON;")

    # Set journal mode and synchronous level (only for read-write connections)
    if not read_only:
        # WAL mode allows concurrent reads during writes
        conn.execute("PRAGMA journal_mode = WAL;")

        # NORMAL synchronous is a good balance between safety and speed
        # FULL is safer but slower, OFF is faster but risks corruption
        conn.execute("PRAGMA synchronous = NORMAL;")

    return conn


def get_read_only_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Get read-only database connection.

    Convenience wrapper for get_connection(read_only=True).
    Use this for queries that don't modify data.

    Args:
        db_path: Path to database file (defaults to market_data.db)

    Returns:
        Read-only SQLite connection with foreign keys enabled

    Example:
        >>> from core.db import get_read_only_connection
        >>> conn = get_read_only_connection()
        >>> # Safe to use for queries, will error on writes
        >>> conn.close()
    """
    return get_connection(db_path=db_path, read_only=True)


def verify_foreign_keys_enabled(conn: sqlite3.Connection) -> bool:
    """
    Verify that foreign keys are enabled on a connection.

    Args:
        conn: SQLite connection to check

    Returns:
        True if foreign keys are enabled, False otherwise

    Example:
        >>> from core.db import get_connection, verify_foreign_keys_enabled
        >>> conn = get_connection()
        >>> assert verify_foreign_keys_enabled(conn)
        >>> conn.close()
    """
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys;")
    result = cursor.fetchone()
    return result[0] == 1 if result else False
