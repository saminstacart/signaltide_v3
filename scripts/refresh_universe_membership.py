#!/usr/bin/env python3
"""
Refresh Universe Membership - Keep dim_universe_membership in sync with Sharadar

Rebuilds dim_universe_membership for sp500_actual from sp500_universe_2015_2025.
Run this after Sharadar data ingests to keep universe membership current.

This is idempotent and safe to run daily.

Usage:
    python3 scripts/refresh_universe_membership.py

See docs/DATA_ARCHITECTURE.md for architecture details.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sqlite3
from datetime import datetime
from core.db import get_connection

DB_PATH = Path("data/databases/market_data.db")


def refresh_sp500_actual_universe(conn: sqlite3.Connection) -> None:
    """
    Rebuild dim_universe_membership for 'sp500_actual' universe.

    Deletes existing rows and re-inserts from sp500_universe_2015_2025.
    This ensures the dim table stays in sync with Sharadar updates.
    """
    cursor = conn.cursor()

    print("Refreshing S&P 500 universe membership...")

    # Start transaction
    cursor.execute("BEGIN;")

    try:
        # Delete existing sp500_actual membership
        cursor.execute("""
            DELETE FROM dim_universe_membership
            WHERE universe_name = 'sp500_actual';
        """)
        deleted_count = cursor.rowcount
        print(f"  Cleared {deleted_count} existing sp500_actual records")

        # Rebuild from source
        cursor.execute("""
            INSERT INTO dim_universe_membership (
                universe_name,
                ticker,
                membership_start_date,
                membership_end_date,
                source,
                source_reference,
                removed_reason
            )
            SELECT
                'sp500_actual' AS universe_name,
                ticker,
                first_entry_date AS membership_start_date,
                CASE
                    WHEN is_current_member = 1 THEN NULL
                    WHEN last_exit_date <= first_entry_date THEN NULL
                    ELSE last_exit_date
                END AS membership_end_date,
                'sharadar' AS source,
                'sp500_universe_2015_2025 table' AS source_reference,
                CASE
                    WHEN is_current_member = 1 THEN NULL
                    WHEN last_exit_date <= first_entry_date THEN 'data_quality_skip'
                    ELSE 'index_rebalance'
                END AS removed_reason
            FROM sp500_universe_2015_2025
            WHERE ticker IS NOT NULL
              AND ticker != ''
              AND (is_current_member = 1 OR last_exit_date > first_entry_date);
        """)
        inserted_count = cursor.rowcount
        print(f"  Inserted {inserted_count} sp500_actual records")

        # Count current vs former members
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN membership_end_date IS NULL THEN 1 ELSE 0 END) as current,
                SUM(CASE WHEN membership_end_date IS NOT NULL THEN 1 ELSE 0 END) as former
            FROM dim_universe_membership
            WHERE universe_name = 'sp500_actual';
        """)
        total, current, former = cursor.fetchone()
        print(f"  Total: {total} members ({current} current, {former} former)")

        # Update meta table
        refreshed_at = datetime.utcnow().isoformat(timespec="seconds")
        cursor.execute("""
            INSERT INTO meta (key, value, updated_at)
            VALUES ('universe_sp500_actual_last_refreshed', ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = CURRENT_TIMESTAMP;
        """, (refreshed_at,))

        conn.commit()
        print(f"  ✓ Refresh complete at {refreshed_at}")

    except Exception as exc:
        conn.rollback()
        print(f"  ✗ Refresh failed: {exc}")
        raise


def main():
    """Main entry point."""
    print("=" * 60)
    print("SignalTide v3 - Refresh Universe Membership")
    print("=" * 60)
    print()

    try:
        conn = get_connection(db_path=DB_PATH)
        refresh_sp500_actual_universe(conn)
        conn.close()

        print()
        print("=" * 60)
        print("Universe membership refreshed successfully!")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
