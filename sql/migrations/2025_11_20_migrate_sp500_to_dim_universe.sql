-- 2025-11-20: Migrate S&P 500 universe from sp500_universe_2015_2025
-- Populates dim_universe_membership with actual S&P 500 membership

PRAGMA foreign_keys = ON;

-- Populate dim_universe_membership with S&P 500 membership
-- Use universe_name = 'sp500_actual'

INSERT OR REPLACE INTO dim_universe_membership (
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
        -- For same-day entry/exit (data quality issue), skip the record
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
  -- Skip records where exit date is before or same as entry date (bad data)
  AND (is_current_member = 1 OR last_exit_date > first_entry_date);

-- Verify migration
SELECT
    'S&P 500 universe migration complete: ' ||
    COUNT(*) || ' total entries, ' ||
    SUM(CASE WHEN membership_end_date IS NULL THEN 1 ELSE 0 END) || ' current members'
AS status
FROM dim_universe_membership
WHERE universe_name = 'sp500_actual';
