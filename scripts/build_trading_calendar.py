#!/usr/bin/env python3
"""
Build Trading Calendar - Populate dim_trading_calendar

Generates a complete NYSE trading calendar from 2000-01-01 to 2035-12-31
and populates the dim_trading_calendar dimension table.

This is a one-time setup script. The calendar is static and does not need
daily refreshes. Re-run only if:
- The date range needs to be extended
- NYSE holiday schedule changes retroactively (rare)
- Schema changes require rebuilding

See docs/DATA_ARCHITECTURE.md for architecture details.

Usage:
    python3 scripts/build_trading_calendar.py

Dependencies:
    pip install pandas pandas_market_calendars
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sqlite3
from datetime import datetime, date
import pandas as pd
import pandas_market_calendars as mcal
from core.db import get_connection

# Database path
DB_PATH = Path("data/databases/market_data.db")

# Date range for calendar (generous runway for backtesting)
START_DATE = "2000-01-01"
END_DATE = "2035-12-31"


def build_calendar_dataframe() -> pd.DataFrame:
    """
    Build complete trading calendar DataFrame using NYSE schedule.

    Returns:
        DataFrame with all columns required for dim_trading_calendar
    """
    print(f"Building trading calendar: {START_DATE} to {END_DATE}")

    # 1. Create full calendar range
    all_dates = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
    df = pd.DataFrame({'calendar_date': all_dates})

    # 2. Get NYSE trading schedule
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=START_DATE, end_date=END_DATE)
    trading_dates = pd.to_datetime(schedule.index.date)

    # 3. Mark trading days
    df['is_trading_day'] = df['calendar_date'].isin(trading_dates).astype(int)

    # 4. Extract date components
    df['calendar_year'] = df['calendar_date'].dt.year
    df['calendar_month'] = df['calendar_date'].dt.month
    df['day_of_week'] = df['calendar_date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_of_month'] = df['calendar_date'].dt.day
    df['day_of_year'] = df['calendar_date'].dt.dayofyear

    # 5. Compute next/previous trading dates
    print("Computing next/previous trading dates...")
    trading_df = df[df['is_trading_day'] == 1].copy()
    trading_df = trading_df.sort_values('calendar_date').reset_index(drop=True)

    # Create lookup dictionaries for fast navigation
    next_trading = {}
    prev_trading = {}

    for i in range(len(trading_df)):
        current_date = trading_df.iloc[i]['calendar_date']

        # Next trading date
        if i < len(trading_df) - 1:
            next_trading[current_date] = trading_df.iloc[i + 1]['calendar_date']
        else:
            next_trading[current_date] = None

        # Previous trading date
        if i > 0:
            prev_trading[current_date] = trading_df.iloc[i - 1]['calendar_date']
        else:
            prev_trading[current_date] = None

    # For non-trading days, find next/previous trading day
    for idx, row in df.iterrows():
        current_date = row['calendar_date']

        if row['is_trading_day']:
            df.at[idx, 'next_trading_date'] = next_trading.get(current_date)
            df.at[idx, 'previous_trading_date'] = prev_trading.get(current_date)
        else:
            # For non-trading days, find next trading day
            future_trading = trading_df[trading_df['calendar_date'] > current_date]
            if not future_trading.empty:
                df.at[idx, 'next_trading_date'] = future_trading.iloc[0]['calendar_date']
            else:
                df.at[idx, 'next_trading_date'] = None

            # Find previous trading day
            past_trading = trading_df[trading_df['calendar_date'] < current_date]
            if not past_trading.empty:
                df.at[idx, 'previous_trading_date'] = past_trading.iloc[-1]['calendar_date']
            else:
                df.at[idx, 'previous_trading_date'] = None

    # 6. Compute period-end flags
    print("Computing period-end flags...")

    # Month-end: last trading day of each month
    df['year_month'] = df['calendar_date'].dt.to_period('M')
    month_end_dates = (
        df[df['is_trading_day'] == 1]
        .groupby('year_month')['calendar_date']
        .max()
        .values
    )
    df['is_month_end'] = df['calendar_date'].isin(month_end_dates).astype(int)

    # Quarter-end: last trading day of each quarter
    df['year_quarter'] = df['calendar_date'].dt.to_period('Q')
    quarter_end_dates = (
        df[df['is_trading_day'] == 1]
        .groupby('year_quarter')['calendar_date']
        .max()
        .values
    )
    df['is_quarter_end'] = df['calendar_date'].isin(quarter_end_dates).astype(int)

    # Year-end: last trading day of each year
    year_end_dates = (
        df[df['is_trading_day'] == 1]
        .groupby('calendar_year')['calendar_date']
        .max()
        .values
    )
    df['is_year_end'] = df['calendar_date'].isin(year_end_dates).astype(int)

    # Drop temporary columns
    df = df.drop(columns=['year_month', 'year_quarter'])

    # 7. Market close type and holiday names - pattern-based detection
    # We use dim_trading_calendar's is_trading_day (from NYSE schedule) as the source of truth.
    # Holiday names are for reporting/debugging only, not for trading logic.
    print("Computing market holidays using pattern-based detection...")

    def get_nth_weekday(year, month, weekday, n):
        """Get the nth occurrence of weekday in given month. n=1 is first, n=-1 is last."""
        from calendar import monthcalendar
        cal = monthcalendar(year, month)
        if n > 0:
            # Get nth occurrence
            weeks = [week for week in cal if week[weekday] != 0]
            return date(year, month, weeks[n-1][weekday]) if n <= len(weeks) else None
        else:
            # Get last occurrence
            weeks = [week for week in cal if week[weekday] != 0]
            return date(year, month, weeks[-1][weekday]) if weeks else None

    def detect_holiday_name(date_obj):
        """Determine holiday name for a non-trading weekday based on date patterns."""
        month = date_obj.month
        day = date_obj.day
        weekday = date_obj.weekday()  # 0=Mon, 6=Sun
        year = date_obj.year

        # Fixed-date holidays (may be observed on adjacent weekday)
        if month == 1 and day == 1:
            return "New Year's Day"
        if month == 7 and day == 4:
            return "Independence Day"
        if month == 12 and day == 25:
            return "Christmas"
        if month == 6 and day == 19 and year >= 2021:  # Juneteenth became federal holiday in 2021
            return "Juneteenth"

        # Nth weekday holidays
        # MLK Day: 3rd Monday in January
        mlk_day = get_nth_weekday(year, 1, 0, 3)
        if mlk_day and date_obj == mlk_day:
            return "Martin Luther King Jr. Day"

        # Presidents Day: 3rd Monday in February
        presidents_day = get_nth_weekday(year, 2, 0, 3)
        if presidents_day and date_obj == presidents_day:
            return "Presidents Day"

        # Memorial Day: Last Monday in May
        memorial_day = get_nth_weekday(year, 5, 0, -1)
        if memorial_day and date_obj == memorial_day:
            return "Memorial Day"

        # Labor Day: 1st Monday in September
        labor_day = get_nth_weekday(year, 9, 0, 1)
        if labor_day and date_obj == labor_day:
            return "Labor Day"

        # Thanksgiving: 4th Thursday in November
        thanksgiving = get_nth_weekday(year, 11, 3, 4)
        if thanksgiving and date_obj == thanksgiving:
            return "Thanksgiving"

        # Good Friday:
        # We approximate this as "any non-trading Friday in March or April".
        # In practice for the NYSE calendar, this matches the real Good Friday date
        # since it is the only non-trading Friday in those months.
        if weekday == 4 and month in (3, 4):  # Friday in March or April
            return "Good Friday"

        # Default for unmatched non-trading weekdays
        return "Market Holiday"

    # Find all non-trading weekdays
    non_trading_weekdays = df[
        (df['is_trading_day'] == 0) &
        (df['day_of_week'].between(0, 4))  # Monday=0 to Friday=4
    ]

    # Build holiday mapping
    holiday_map = {}
    for idx in non_trading_weekdays.index:
        date_obj = df.at[idx, 'calendar_date'].date()
        holiday_map[date_obj] = detect_holiday_name(date_obj)

    print(f"  Identified {len(holiday_map)} non-trading weekdays")

    # Initialize columns
    df['market_close_type'] = 'full_day'
    df['holiday_name'] = None

    # Set market_close_type for all non-trading days
    df.loc[df['is_trading_day'] == 0, 'market_close_type'] = 'closed'

    # Apply holiday names only to non-trading weekdays
    # (weekends get market_close_type='closed' but holiday_name=None)
    for date_obj, name in holiday_map.items():
        mask = df['calendar_date'].dt.date == date_obj
        df.loc[mask, 'holiday_name'] = name

    # LIMITATION: Early close days (e.g., day before Thanksgiving, Christmas Eve) are not yet tracked.
    # These days are currently marked as normal trading days (market_close_type='normal').
    # For monthly rebalancing strategies, this has minimal impact since rebalancing typically occurs
    # at month-end. If daily/weekly rebalancing is used, be aware that some execution times may be
    # shortened on early-close days.
    # See docs/core/ERROR_PREVENTION_ARCHITECTURE.md (Open Gaps) for details.
    # Future implementation would set market_close_type='early_close' for these dates.

    # 8. Convert dates to string format for SQLite
    df['calendar_date'] = df['calendar_date'].dt.strftime('%Y-%m-%d')
    df['next_trading_date'] = df['next_trading_date'].apply(
        lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else None
    )
    df['previous_trading_date'] = df['previous_trading_date'].apply(
        lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else None
    )

    print(f"Calendar built: {len(df)} days, {df['is_trading_day'].sum()} trading days")
    return df


def populate_trading_calendar(df: pd.DataFrame, conn: sqlite3.Connection) -> None:
    """
    Populate dim_trading_calendar table (idempotent).

    Deletes existing data and re-inserts to ensure consistency.
    """
    cursor = conn.cursor()

    print("Clearing existing calendar data...")
    cursor.execute("DELETE FROM dim_trading_calendar;")

    print("Inserting calendar data...")
    df.to_sql(
        'dim_trading_calendar',
        conn,
        if_exists='append',
        index=False,
        method='multi',
        chunksize=1000
    )

    conn.commit()

    # Verify
    cursor.execute("SELECT COUNT(*) FROM dim_trading_calendar;")
    total_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM dim_trading_calendar WHERE is_trading_day = 1;")
    trading_count = cursor.fetchone()[0]

    print(f"✓ Inserted {total_count} calendar days ({trading_count} trading days)")

    # Update meta table
    cursor.execute("""
        INSERT INTO meta (key, value, updated_at)
        VALUES ('trading_calendar_last_built', ?, CURRENT_TIMESTAMP)
        ON CONFLICT(key) DO UPDATE SET
            value = excluded.value,
            updated_at = CURRENT_TIMESTAMP;
    """, (datetime.now().isoformat(),))

    cursor.execute("""
        INSERT INTO meta (key, value, updated_at)
        VALUES ('trading_calendar_date_range', ?, CURRENT_TIMESTAMP)
        ON CONFLICT(key) DO UPDATE SET
            value = excluded.value,
            updated_at = CURRENT_TIMESTAMP;
    """, (f"{START_DATE} to {END_DATE}",))

    conn.commit()
    print("✓ Updated meta table")


def validate_calendar(conn: sqlite3.Connection) -> None:
    """Run validation checks on populated calendar."""
    cursor = conn.cursor()

    print("\nValidating calendar...")

    # Check 1: Verify known dates
    test_cases = [
        ('2023-07-04', 0, 'Independence Day (Tuesday)'),  # Holiday
        ('2023-07-05', 1, 'Day after Independence Day'),  # Trading day
        ('2023-12-25', 0, 'Christmas (Monday)'),          # Holiday
        ('2023-12-29', 1, 'Last trading day of 2023'),    # Year-end
    ]

    for test_date, expected_trading, description in test_cases:
        cursor.execute(
            "SELECT is_trading_day, is_year_end FROM dim_trading_calendar WHERE calendar_date = ?",
            (test_date,)
        )
        result = cursor.fetchone()
        if result:
            is_trading, is_year_end = result
            status = "✓" if is_trading == expected_trading else "✗"
            print(f"  {status} {test_date} ({description}): is_trading_day={is_trading}")
        else:
            print(f"  ✗ {test_date} not found in calendar")

    # Check 2: Verify no gaps in next_trading_date chain
    cursor.execute("""
        SELECT COUNT(*)
        FROM dim_trading_calendar
        WHERE is_trading_day = 1
          AND next_trading_date IS NOT NULL
          AND next_trading_date NOT IN (
              SELECT calendar_date FROM dim_trading_calendar WHERE is_trading_day = 1
          );
    """)
    bad_next = cursor.fetchone()[0]
    if bad_next == 0:
        print(f"  ✓ All next_trading_date references are valid")
    else:
        print(f"  ✗ Found {bad_next} invalid next_trading_date references")

    # Check 3: Verify weekends are marked as non-trading
    cursor.execute("""
        SELECT COUNT(*)
        FROM dim_trading_calendar
        WHERE day_of_week IN (5, 6)  -- Saturday=5, Sunday=6
          AND is_trading_day = 1;
    """)
    weekend_trading = cursor.fetchone()[0]
    if weekend_trading == 0:
        print(f"  ✓ No weekends marked as trading days")
    else:
        print(f"  ✗ Found {weekend_trading} weekends marked as trading days")

    print("Validation complete!\n")


def main():
    """Main entry point."""
    print("=" * 60)
    print("SignalTide v3 - Build Trading Calendar")
    print("=" * 60)
    print()

    try:
        # Build calendar DataFrame
        df = build_calendar_dataframe()

        # Connect to database
        conn = get_connection(db_path=DB_PATH)

        # Populate table
        populate_trading_calendar(df, conn)

        # Validate
        validate_calendar(conn)

        conn.close()

        print("=" * 60)
        print("Trading calendar built successfully!")
        print(f"Date range: {START_DATE} to {END_DATE}")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
