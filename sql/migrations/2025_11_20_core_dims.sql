-- 2025-11-20: Core dimension tables for SignalTide v3
-- Creates Tier 2 warehouse dimension tables for trading calendar and universe membership
-- See docs/DATA_ARCHITECTURE.md for architecture overview

PRAGMA foreign_keys = ON;

-- =============================================================================
-- Trading Calendar Dimension
-- =============================================================================
-- Precomputed NYSE trading calendar from 2010-2035
-- Enables fast date arithmetic and rebalancing logic

CREATE TABLE IF NOT EXISTS dim_trading_calendar (
    calendar_date DATE PRIMARY KEY,
    is_trading_day BOOLEAN NOT NULL,

    -- Date components for filtering and grouping
    calendar_year INTEGER NOT NULL,
    calendar_month INTEGER NOT NULL,
    day_of_week INTEGER NOT NULL CHECK(day_of_week BETWEEN 0 AND 6),  -- 0=Monday, 6=Sunday
    day_of_month INTEGER NOT NULL,
    day_of_year INTEGER NOT NULL,

    -- Precomputed navigation for fast lookups
    next_trading_date DATE,
    previous_trading_date DATE,

    -- Period-end flags for rebalancing logic
    is_month_end BOOLEAN NOT NULL DEFAULT 0,
    is_quarter_end BOOLEAN NOT NULL DEFAULT 0,
    is_year_end BOOLEAN NOT NULL DEFAULT 0,

    -- Market hours (for future half-day/early close support)
    market_close_type TEXT NOT NULL DEFAULT 'full_day'
        CHECK(market_close_type IN ('full_day', 'half_day', 'closed')),
    holiday_name TEXT,

    -- Ensure consistency: if not a trading day, market must be closed
    CHECK(NOT is_trading_day OR market_close_type != 'closed')
) WITHOUT ROWID;

-- Index for fast trading day lookups
CREATE INDEX IF NOT EXISTS idx_trading_calendar_trading
    ON dim_trading_calendar(calendar_date)
    WHERE is_trading_day = 1;

-- Index for next trading date navigation
CREATE INDEX IF NOT EXISTS idx_trading_calendar_next
    ON dim_trading_calendar(next_trading_date);

-- =============================================================================
-- Universe Membership Dimension (Slowly Changing Dimension Type 2)
-- =============================================================================
-- Tracks point-in-time membership of stocks in various universes (S&P 500, etc.)
-- Supports survivorship bias-free backtesting

CREATE TABLE IF NOT EXISTS dim_universe_membership (
    universe_name TEXT NOT NULL,      -- e.g. 'sp500_actual', 'nasdaq_actual', 'russell1000_actual'
    ticker TEXT NOT NULL,              -- Stock symbol
    membership_start_date DATE NOT NULL,
    membership_end_date DATE,          -- NULL = still a member

    -- Provenance tracking
    source TEXT NOT NULL,              -- 'sharadar', 'manual', 'computed'
    source_reference TEXT,             -- e.g. 'sp500_universe_2015_2025 table', 'manual CSV'

    -- Metadata
    added_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    removed_reason TEXT,               -- 'delisted', 'index_rebalance', 'merger', etc.

    -- Composite primary key: multiple membership periods for same ticker allowed
    PRIMARY KEY (universe_name, ticker, membership_start_date),

    -- Constraints
    CHECK (membership_end_date IS NULL OR membership_end_date > membership_start_date),
    CHECK (LENGTH(ticker) BETWEEN 1 AND 10)
) WITHOUT ROWID;

-- Index for active membership queries (most common query pattern)
CREATE INDEX IF NOT EXISTS idx_universe_membership_universe_active
    ON dim_universe_membership(universe_name, membership_end_date)
    WHERE membership_end_date IS NULL;

-- Index for ticker-based lookups
CREATE INDEX IF NOT EXISTS idx_universe_membership_ticker
    ON dim_universe_membership(ticker);

-- Index for point-in-time queries (critical for backtesting)
CREATE INDEX IF NOT EXISTS idx_universe_membership_pit
    ON dim_universe_membership(universe_name, membership_start_date, membership_end_date);

-- =============================================================================
-- Meta Table (Schema versioning and housekeeping)
-- =============================================================================
-- Tracks schema version, last refresh dates, and system metadata

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Insert initial schema version
INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', '1.0.0');
INSERT OR IGNORE INTO meta (key, value) VALUES ('created_at', CURRENT_TIMESTAMP);

-- =============================================================================
-- End of migration
-- =============================================================================

-- Verify tables were created
SELECT
    'Migration complete: ' || COUNT(*) || ' dimension tables created' AS status
FROM sqlite_master
WHERE type = 'table'
  AND name IN ('dim_trading_calendar', 'dim_universe_membership', 'meta');
