# SignalTide v3 - Data Architecture

**Version:** 1.0.0
**Last Updated:** 2025-11-20
**Status:** SOURCE OF TRUTH - All database design follows this architecture

---

## Philosophy

**Separation of Concerns > Convenience**

SignalTide uses a three-tier data architecture that separates vendor data, curated warehouse tables, and research artifacts. This ensures:

1. **Data Integrity** - Raw vendor data never changes
2. **Clear Lineage** - Every derived table has an obvious source
3. **Maintainability** - Research experiments don't pollute core logic
4. **Reproducibility** - Point-in-time queries work correctly

---

## Three-Tier Architecture

### Tier 1: Raw Vendor Data (Immutable)

**Purpose:** Store Sharadar data exactly as received, without transformation.

**Tables:**
- `sharadar_prices` - Daily OHLCV price data
- `sharadar_sf1` - Fundamental metrics (quarterly/annual)
- `sharadar_insiders` - Insider transaction filings
- `sharadar_tickers` - Ticker metadata and categories
- `sharadar_sp500` - S&P 500 add/remove event log
- `sharadar_daily` - Daily fundamental metrics
- `sharadar_sf3a` - Balance sheet data (if used)
- `sharadar_sfp` - Price-derived fundamentals (if used)

**Naming Convention:** `sharadar_*`

**Rules:**
- ✅ Read-only in production code
- ✅ Schema follows Sharadar conventions (`calendardate`, `datekey`, etc.)
- ✅ Updated only by ingest/refresh scripts
- ❌ NEVER join directly to Tier 3 tables
- ❌ NEVER modify schema or add indexes beyond what Sharadar provides

**Access Pattern:**
```python
# CORRECT: Read from Tier 1, transform into Tier 2
raw_data = dm.get_fundamentals(ticker, start_date, end_date, as_of_date)
# Process and cache in dim/fact tables

# WRONG: Join Tier 1 directly to backtest results
```

---

### Tier 2: Core SignalTide Warehouse (Curated)

**Purpose:** Provide clean, normalized dimensions and facts for backtesting and signal generation.

**Dimension Tables** (`dim_*`):

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `dim_trading_calendar` | NYSE trading days 2010-2035 | `calendar_date`, `is_trading_day`, `next_trading_date` |
| `dim_universe_membership` | Point-in-time universe constituents | `universe_name`, `ticker`, `membership_start_date`, `membership_end_date` |
| `dim_signal_definitions` (future) | Signal metadata and parameters | `signal_name`, `version`, `parameter_schema` |

**Fact Tables** (`fact_*`):

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `fact_signals_panel` (future) | Pre-computed signal values over time | `trading_date`, `ticker`, `signal_name`, `signal_value` |
| `fact_positions` (future) | Historical portfolio positions | `trading_date`, `ticker`, `quantity`, `market_value` |

**Naming Convention:** `dim_*` for dimensions, `fact_*` for facts

**Rules:**
- ✅ All new SignalTide code should use Tier 2 tables
- ✅ Date columns use explicit names (`calendar_date`, `trading_date`, `membership_start_date`)
- ✅ Foreign keys enforced with `PRAGMA foreign_keys = ON;`
- ✅ Rebuilt deterministically from Tier 1 via refresh scripts
- ✅ Indexed for point-in-time queries
- ❌ NEVER manually edited in production
- ❌ NEVER use bare `date` as column name

**Access Pattern:**
```python
# CORRECT: Use UniverseManager which queries dim_universe_membership
universe = um.get_universe_tickers("sp500_actual", as_of_date="2023-01-01")

# CORRECT: Use trading calendar for date arithmetic
next_rebalance = calendar.get_next_trading_date(current_date)

# WRONG: Query sp500_universe_2015_2025 directly (legacy table)
```

**Refresh Cadence:**
- `dim_trading_calendar` - Static, pre-computed to 2035
- `dim_universe_membership` - Daily, via `refresh_universe_membership.py`
- `dim_signal_definitions` - On-demand, when signal parameters change
- `fact_signals_panel` - Daily or on-demand, depending on signal type

---

### Tier 3: Research and Experiments (Transient)

**Purpose:** Store backtest results, optimization runs, and research artifacts that can be rebuilt.

**Tables:**

| Table | Purpose | Can Delete? |
|-------|---------|-------------|
| `backtest_results` | Backtest performance metrics | ✅ Yes - can rebuild |
| `backtests` | Backtest configuration metadata | ✅ Yes - can rebuild |
| `experiment_registry` | Cross-validation and optimization runs | ✅ Yes - can rebuild |
| `signals` | Generated signal values (cache) | ✅ Yes - can rebuild from Tier 1 |
| `cv_fold_results`, `cv_results` | Cross-validation artifacts | ✅ Yes - experimental |
| `sec_*` tables | NLP extractions and sentiment | ⚠️ Maybe - depends on OpenAI costs |

**Legacy Universe Table (DEPRECATED):**

| Table | Status | Replacement | Notes |
|-------|--------|-------------|-------|
| `sp500_universe_2015_2025` | ⚠️ DEPRECATED | `dim_universe_membership` with `universe_name = 'sp500_actual'` | Keep as read-only reference table. DO NOT query directly in production code. Migrated to Tier 2 on 2025-11-20. |

**Production Code Migration:**
- ❌ OLD: `SELECT * FROM sp500_universe_2015_2025 WHERE is_current_member = 1`
- ✅ NEW: `UniverseManager.get_universe('sp500_actual', as_of_date)`

This table remains in the database for:
1. Manual verification of migration correctness
2. Future re-migration if needed
3. Backwards compatibility with old notebooks/scripts (not recommended)

**Other Legacy Tables:**
- `narratives`, `sentiment_cache` - Move to separate research database (future)
- `bulk_data_ingestion` - Archive if no longer used

**Naming Convention:** `backtest_*`, `run_*`, or descriptive names

**Rules:**
- ⚠️ OK to be messy, but document what each table is for
- ✅ Can be dropped and rebuilt without breaking Tier 2
- ✅ Can have experimental schemas
- ❌ NEVER referenced by core signal logic or UniverseManager
- ❌ Don't let SEC/NLP tables creep into production backtest code

**Access Pattern:**
```python
# CORRECT: Backtest writes to Tier 3
results_df.to_sql("backtest_results", conn, if_exists="append")

# WRONG: Signal generation reads from backtest_results
# (This would create a circular dependency)
```

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Tier 1: Raw Vendor Data (sharadar_*)                        │
│ - Sharadar bulk download / API refresh                      │
│ - Immutable source of truth                                 │
└────────────────┬────────────────────────────────────────────┘
                 │
                 │ Daily Refresh Scripts
                 │ (deterministic transforms)
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Tier 2: SignalTide Warehouse (dim_*, fact_*)                │
│ - dim_trading_calendar (static)                             │
│ - dim_universe_membership (daily refresh)                   │
│ - fact_signals_panel (on-demand or daily)                   │
└────────────────┬────────────────────────────────────────────┘
                 │
                 │ UniverseManager, Signals, Backtests
                 │ (production code)
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Tier 3: Research & Experiments (backtest_*, run_*)          │
│ - Backtest results and configurations                       │
│ - Cross-validation folds                                    │
│ - SEC NLP extractions (maybe separate DB later)             │
└─────────────────────────────────────────────────────────────┘
```

---

## Daily Refresh Process

### Current Implementation

**After Sharadar data is ingested:**

1. **Sharadar Ingest** (manual or scheduled)
   - Download latest Sharadar bulk files
   - Populate `sharadar_*` tables (Tier 1)

2. **Universe Membership Refresh** (run after each ingest)
   ```bash
   python3 scripts/refresh_universe_membership.py
   ```
   - Reads `sp500_universe_2015_2025` (Tier 1 table)
   - Rebuilds `dim_universe_membership` for `sp500_actual` universe (Tier 2)
   - Updates `meta` table with last refresh timestamp
   - Idempotent and safe to run daily

3. **Trading Calendar** (one-time setup, no daily refresh)
   - Pre-computed to 2035 using `scripts/build_trading_calendar.py`
   - Only re-run if:
     - Date range needs extension beyond 2035
     - NYSE holiday schedule changes retroactively (rare)
     - Schema changes require rebuild

**Generated by Trading Calendar Script:**
- The `dim_trading_calendar` table is populated by `scripts/build_trading_calendar.py`
- Uses `pandas_market_calendars` with NYSE schedule
- Covers 2000-01-01 to 2035-12-31 (13,149 days, 9,049 trading days)
- Do not manually edit this table - re-run the script if changes are needed

### Future Enhancements

- Automate Sharadar ingest with cron or GitHub Actions
- Add `dim_universe_membership` refresh for other universes (NASDAQ, Russell)
- Populate `fact_signals_panel` daily for pre-computed signals
- Separate SEC NLP data into its own database

---

## Point-in-Time Correctness

### The Challenge

Backtests must only use data that would have been available at the time:
- Fundamentals have 30-45 day filing lag after quarter end
- Insider trades have 2-day SEC filing requirement
- Universe membership changes happen on specific dates
- Delisted stocks disappear from Sharadar after delisting

### The Solution

Tier 2 tables enforce point-in-time semantics:

**Trading Calendar:**
```sql
-- Get next trading day after a signal generation date
SELECT next_trading_date
FROM dim_trading_calendar
WHERE calendar_date = '2023-06-15';  -- Returns 2023-06-16 if that's a trading day
```

**Universe Membership:**
```sql
-- Get S&P 500 constituents as of 2020-01-15
SELECT ticker
FROM dim_universe_membership
WHERE universe_name = 'sp500_actual'
  AND membership_start_date <= '2020-01-15'
  AND (membership_end_date IS NULL OR membership_end_date >= '2020-01-15');
```

**Fundamentals (Tier 1, via DataManager):**
```sql
-- Get fundamentals filed by as_of_date (respects filing lag)
SELECT *
FROM sharadar_sf1
WHERE ticker = 'AAPL'
  AND dimension = 'ARQ'
  AND datekey <= '2023-06-30'  -- Filed by this date
ORDER BY datekey DESC;
```

---

## Migration Strategy

### Completed
- ✅ `dim_trading_calendar` created and populated
- ✅ `dim_universe_membership` created
- ✅ S&P 500 membership migrated from `sp500_universe_2015_2025`
- ✅ UniverseManager refactored to use `dim_universe_membership`

### In Progress
- 🔄 Document legacy tables for archival
- 🔄 Update all backtest scripts to use Tier 2 tables

### Future
- ⏳ Create `dim_signal_definitions` for signal metadata
- ⏳ Create `fact_signals_panel` for pre-computed signals
- ⏳ Move SEC NLP tables to separate research database
- ⏳ Archive `sp500_universe_2015_2025` after migration complete
- ⏳ Automate daily refresh pipeline

---

## Database Hygiene Rules

### For All Developers

1. **Never modify Tier 1 tables** - Treat `sharadar_*` as read-only
2. **Always enable foreign keys** - `PRAGMA foreign_keys = ON;` in all connections
3. **Use explicit date column names** - No bare `date` in Tier 2/3 tables
4. **Query Tier 2 for production logic** - Backtest and signal code uses `dim_*` and `fact_*`
5. **Keep Tier 3 clean** - Delete old experiment tables when done

### For Schema Changes

1. **Create migration SQL file** - `sql/migrations/YYYY_MM_DD_description.sql`
2. **Update this document** - Add new tables to appropriate tier
3. **Add to NAMING_CONVENTIONS.md** - Document any new naming patterns
4. **Test point-in-time correctness** - Verify queries respect as-of-date filtering
5. **Update refresh scripts** - If new Tier 2 tables need daily updates

---

## Files and Directories

**Documentation:**
- `docs/core/DATA_ARCHITECTURE.md` ← This file
- `docs/core/NAMING_CONVENTIONS.md` - Naming rules for tables and columns
- `docs/core/ARCHITECTURE.md` - Overall system design

**Migration SQL:**
- `sql/migrations/2025_11_20_core_dims.sql` - Trading calendar and universe membership
- Future migrations follow `YYYY_MM_DD_description.sql` naming

**Refresh Scripts:**
- `scripts/build_trading_calendar.py` - One-time trading calendar population
- `scripts/refresh_universe_membership.py` - Daily S&P 500 membership refresh
- Future: `scripts/refresh_signal_panel.py` - Daily signal computation

**Database:**
- `data/databases/market_data.db` - Single SQLite database for all tiers

---

## FAQ

**Q: Why not just use Sharadar tables directly?**
A: Sharadar uses vendor conventions (`calendardate`, `datekey`) that don't match our naming standards. Also, universe membership requires complex point-in-time queries that should be pre-computed in `dim_universe_membership`.

**Q: Can I create a new table for my experiment?**
A: Yes, but put it in Tier 3. Use `experiment_*` or descriptive names. Document it in this file under Tier 3. Delete it when the experiment is done.

**Q: What if I need data that spans Tier 1 and Tier 3?**
A: Join through Tier 2. For example: Tier 1 → Tier 2 (`dim_universe_membership`) → Tier 3 (`backtest_results`). Never join Tier 1 directly to Tier 3.

**Q: How do I add a new universe like NASDAQ or Russell 1000?**
A:
1. Add rows to `dim_universe_membership` with `universe_name = 'nasdaq_actual'` or `'russell1000_actual'`
2. Update `scripts/refresh_universe_membership.py` to rebuild from source data
3. Document the source in the `source_reference` column

**Q: What happens when Sharadar adds a new column?**
A: Tier 1 tables update automatically on next ingest. If Tier 2 needs the new column, create a migration and update refresh scripts.

---

**Last Updated:** 2025-11-20
**Next Review:** 2026-02-20 (Quarterly)
**Maintainer:** See CLAUDE.md for governance process
