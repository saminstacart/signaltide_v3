-- 2025-11-20: Index for point-in-time universe lookups
-- Optimizes the exact access pattern used by UniverseManager.get_universe_tickers
--
-- Query pattern:
--   SELECT ticker
--   FROM dim_universe_membership
--   WHERE universe_name = ?
--     AND membership_start_date <= ?
--     AND (membership_end_date IS NULL OR membership_end_date > ?)
--
-- This index covers all three filter columns in the optimal order.

PRAGMA foreign_keys = ON;

CREATE INDEX IF NOT EXISTS idx_universe_membership_lookup
ON dim_universe_membership (
    universe_name,
    membership_start_date,
    membership_end_date
);
