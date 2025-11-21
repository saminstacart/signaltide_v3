"""
Rebalance Schedule Utilities

Maps high-level rebalance schedule names to actual trading dates using
the trading calendar from DataManager.

All schedules respect:
- Trading calendar (no weekends/holidays)
- Month-end adjustments (last trading day of month)
- Weekly adjustments (fallback to prior trading day if target day is holiday)

Supported schedules:
- 'daily' or 'D': Every trading day
- 'weekly' or 'W': Weekly on Fridays (with holiday fallback)
- 'monthly' or 'M': Last trading day of each month
"""

from typing import List
from data.data_manager import DataManager


def get_rebalance_dates(
    schedule: str,
    dm: DataManager,
    start_date: str,
    end_date: str,
    day_of_week: str = 'Friday'
) -> List[str]:
    """
    Get rebalance dates for a given schedule.

    Args:
        schedule: Rebalance frequency. Accepted values (case-insensitive):
            - Daily: 'd', 'daily'
            - Weekly: 'w', 'weekly'
            - Monthly: 'm', 'me', 'monthly' (ME = Month End, pandas convention)
        dm: DataManager instance with trading calendar access
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        day_of_week: Target day for weekly rebalancing (default: 'Friday')

    Returns:
        List of rebalance dates (YYYY-MM-DD format), all guaranteed to be trading days

    Raises:
        ValueError: If schedule is not recognized

    Examples:
        >>> dm = DataManager()
        >>> dates = get_rebalance_dates('monthly', dm, '2024-01-01', '2024-03-31')
        >>> # Returns: ['2024-01-31', '2024-02-29', '2024-03-28']

        >>> dates = get_rebalance_dates('W', dm, '2024-01-01', '2024-01-31')
        >>> # Returns: ['2024-01-05', '2024-01-12', '2024-01-19', '2024-01-26']
    """
    # Normalize input: strip whitespace and convert to uppercase for case-insensitive matching
    schedule = schedule.strip().upper()

    if schedule in ('DAILY', 'D'):
        return dm.get_trading_dates_between(start_date, end_date)

    elif schedule in ('WEEKLY', 'W'):
        return dm.get_weekly_rebalance_dates(start_date, end_date, day_of_week=day_of_week)

    elif schedule in ('MONTHLY', 'M', 'ME'):  # ME = Month End (pandas convention)
        return dm.get_month_end_rebalance_dates(start_date, end_date)

    else:
        raise ValueError(
            f"Unknown rebalance schedule: {schedule}. "
            f"Supported: 'daily'/'D', 'weekly'/'W', 'monthly'/'M'"
        )


def validate_rebalance_dates(dates: List[str], dm: DataManager) -> bool:
    """
    Validate that all dates in list are trading days.

    Args:
        dates: List of dates to validate (YYYY-MM-DD format)
        dm: DataManager instance

    Returns:
        True if all dates are trading days, False otherwise

    Example:
        >>> dm = DataManager()
        >>> dates = ['2024-01-31', '2024-02-29', '2024-03-28']
        >>> valid = validate_rebalance_dates(dates, dm)
        >>> # Returns: True (all are trading days)
    """
    for date in dates:
        # Check if date equals its last trading date (i.e., is a trading day)
        last_trading = dm.get_last_trading_date(date)
        if date != last_trading:
            return False

    return True
