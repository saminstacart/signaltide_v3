# SignalTide v3 - Current State

Project status tracking for institutional-grade systematic trading system.

**Last Updated:** 2025-11-20
**Current Phase:** Phase 3 (Backtest Orchestration & Reproducibility - Complete)
**Overall Status:** A+++ Architecture & Infrastructure Ready

---

## Phase 1 – Core Infrastructure & Signal Implementation ✅

**Goal:** Build institutional-grade data access, signal framework, and universe management.

**Status:** COMPLETE (Nov 10-20, 2025)

### Key Achievements

- **Data Layer (A+++ Grade):**
  - Point-in-time data access with `as_of_date` filtering
  - Proper filing lag handling (33-day fundamentals, 1-2 day insider)
  - Read-only database access for safety
  - LRU caching for performance
  - Fixed critical 33-day lookahead bias in quality signal (2025-11-20)

- **Signal Framework:**
  - 6 signals implemented (3 institutional-grade, 3 simple variants)
  - Institutional Momentum (Jegadeesh-Titman 12-1)
  - Institutional Quality (Asness QMJ)
  - Institutional Insider (Cohen-Malloy-Pomorski)
  - BaseSignal and InstitutionalSignal abstract classes
  - Parameter validation and optimization hooks

- **Universe Management:**
  - `core/universe_manager.py` with point-in-time market cap filtering
  - Multiple universe types: manual, top_N, sp500_proxy, sp1000_proxy, nasdaq_proxy, market_cap_range, sector
  - IPO/delisting boundary enforcement
  - Can handle 500-1000+ stocks for robust optimization

- **Testing Infrastructure:**
  - 60+ unit tests
  - 11 filing lag tests (100% passing)
  - Integration tests and mock data generators

### Legacy Status Reports

Earlier phase tracking documents have been archived:
- `archive/docs/STATUS_legacy.md` – Phase 1.6 completion snapshot
- `archive/docs/PROJECT_STATUS_legacy.md` – Comprehensive Phase 0-1.2 audit

---

## Phase 2 – Market Plumbing & Backtest Integration ✅

**Goal:** Harden the market "plumbing" layer (calendar, universes, schedules) and wire it cleanly into the institutional backtest engine with full tests and docs.

### What's Done

- **Trading calendar (`dim_trading_calendar`):**
  - Pattern-based holiday detection with named holidays (MLK, Presidents Day, Memorial Day, Labor Day, Thanksgiving, Christmas, Good Friday, etc.).
  - All non-trading weekdays are labeled with `holiday_name` and `market_close_type`.
  - `DataManager` exposes:
    - `get_last_trading_date(as_of_date)`
    - `get_next_trading_date(as_of_date)`
    - `get_trading_dates_between(start_date, end_date)`
  - In-memory calendar cache (`_trading_calendar_cache`, `_trading_days_only`) with proper `clear_cache()` behavior.

- **Universe PIT semantics (`dim_universe_membership` + `UniverseManager`):**
  - Invariant documented: `membership_start_date` is **inclusive**, `membership_end_date` is **exclusive** → `[start, end)` semantics.
  - All production PIT queries must go through `UniverseManager` (enforced via docstring invariant).
  - Boundary tests cover:
    - Day before start → excluded
    - On start → included
    - Day before end → included
    - On/after end → excluded
    - `NULL` end date → membership continues indefinitely.

- **Rebalance helpers & schedules:**
  - `DataManager`:
    - `get_month_end_rebalance_dates(start_date, end_date)`
    - `get_weekly_rebalance_dates(start_date, end_date, day_of_week='Friday')`
  - `core.schedules.get_rebalance_dates(schedule, dm, start_date, end_date)`:
    - Accepts `D/d/daily`, `W/w/weekly`, `M/ME/m/monthly` (month-end).
    - Returns only **trading days**, backed by `dim_trading_calendar`.

- **Backtest engine integration (`scripts/run_institutional_backtest.py`):**
  - Rebalance logic now calls `get_rebalance_dates(...)` instead of `DataFrame.resample`.
  - Signals are sampled on rebalance dates and forward-filled to daily.
  - Explicit comment: all trading date logic must go through `DataManager` + calendar helpers (no raw `pd.date_range` / index hacks).
  - CLI `--rebalance` help text documents all accepted values and notes NYSE calendar enforcement.

- **Integration test (`scripts/test_backtest_integration.py`):**
  - Proves that:
    - Rebalance dates are all valid trading days.
    - Universe exit follows `[start, end)` semantics.
    - `NULL` end dates behave as "indefinite membership."
    - Rebalance dates + SP500 universe compose cleanly over a real date range.

### Test Suite (Plumbing & Backtest Integrity)

All of these are currently **green** (30/30 tests passing):

```bash
# Trading calendar (NYSE holidays, weekend handling)
python3 scripts/test_trading_calendar.py

# Universe PIT semantics ([start, end) intervals)
python3 scripts/test_universe_manager.py

# Rebalance helper methods (month-end, weekly)
python3 scripts/test_rebalance_helpers.py

# Schedule presets (daily/weekly/monthly mapping)
python3 scripts/test_rebalance_schedules.py

# Integration test (calendar + schedules + universes)
python3 scripts/test_backtest_integration.py
```

**Rule:** All of the above must pass before committing any changes to:

* `dim_trading_calendar`
* `dim_universe_membership`
* `DataManager` calendar or universe helpers
* `core.schedules`
* `UniverseManager`
* Backtest date / rebalance wiring.

### Key Files

**Core Modules:**
- `data/data_manager.py` - Calendar helpers, PIT data access
- `core/universe_manager.py` - PIT universe membership (INVARIANT: [start, end) semantics)
- `core/schedules.py` - Rebalance schedule presets (daily/weekly/monthly)
- `scripts/run_institutional_backtest.py` - Main backtest driver with calendar integration

**Tests:**
- `scripts/test_trading_calendar.py` - 5 tests (weekends, holidays, date ranges)
- `scripts/test_universe_manager.py` - 10 tests (PIT semantics, boundary conditions)
- `scripts/test_rebalance_helpers.py` - 5 tests (month-end, weekly rebalance dates)
- `scripts/test_rebalance_schedules.py` - 6 tests (schedule presets, input validation)
- `scripts/test_backtest_integration.py` - 4 tests (calendar + schedules + universes composition)

**Setup Scripts:**
- `scripts/build_trading_calendar.py` - Populates `dim_trading_calendar` (one-time setup)

### Documentation

- **README.md** - Testing section documents all market plumbing tests
- **docs/DATA_ARCHITECTURE.md** - Trading calendar and universe architecture
- All modules have comprehensive docstrings with invariants documented inline
- CLI help text (`--rebalance`) matches API behavior

---

## Phase 3 – Backtest Orchestration & Reproducibility ✅

**Goal:** Add structured reproducibility to backtest runs through manifests and validation.

### What's Done

- **Backtest Manifest (`core/manifest.py`):**
  - Structured dataclass capturing all backtest parameters for full reproducibility.
  - Fields include: run ID, timestamps, date ranges, universe definition, signal specifications, execution parameters, transaction costs, and code versioning (git SHA).
  - `to_dict()` method for JSON-safe serialization.
  - `from_context()` classmethod for easy construction from backtest parameters.
  - Safe git SHA probing (returns None if git unavailable).

- **Backtest Engine Integration:**
  - `scripts/run_institutional_backtest.py` now creates a manifest for every backtest run.
  - Manifest and manifest dict included in all backtest results.
  - Manifest summary logged with every run (run_id, period, universe, signals).

- **Deterministic Integration Test Enhancement (`scripts/test_deterministic_backtest.py`):**
  - TEST 0: Validates manifest presence, structure, and all key fields.
  - TEST 1: Rebalance dates from trading calendar.
  - TEST 2: Equity curve with performance bands (-20% to +100%).
  - TEST 3: Performance metrics with bands (Sharpe: -1.0 to +3.0, drawdown: -50% to 0%, volatility: 0-100%).
  - TEST 4: Determinism baseline logging.
  - Performance budget: Logs runtime and warns if >10s (currently runs in ~0.09s).

### Test Suite (31 tests passing)

```bash
# Market plumbing tests (30 tests)
python3 scripts/test_trading_calendar.py          # 5 tests
python3 scripts/test_universe_manager.py           # 10 tests
python3 scripts/test_rebalance_helpers.py          # 5 tests
python3 scripts/test_rebalance_schedules.py        # 6 tests
python3 scripts/test_backtest_integration.py       # 4 tests

# Orchestration & reproducibility test (1 test with 5 subtests)
python3 scripts/test_deterministic_backtest.py     # 1 test (5 validations)
```

### Key Files

**New Modules:**
- `core/manifest.py` - BacktestManifest dataclass for reproducibility

**Modified:**
- `scripts/run_institutional_backtest.py` - Creates and returns manifests
- `scripts/test_deterministic_backtest.py` - Validates manifests, uses performance bands

---

## Phase 3 – Signal Panel & Backtest Engine Hardening (Remaining)

Focus: Move from "market plumbing is rock solid" to "signals + backtests are institutional-grade and reproducible."

### Track A – Signal Panel & Data Surfaces (Planned)

- [ ] Design and implement a `fact_signals_panel`-style table:
  - PIT-correct, versioned signal snapshots (momentum, quality, insider, etc.).
  - Clear schema for:
    - `as_of_date`, `ticker`, `signal_name`, `signal_value`, `signal_version`.
  - Compute-once, reuse in many backtests.

- [ ] Add tests to guarantee:
  - No look-ahead in signal generation (joins honor trading calendar + PIT universes).
  - Signals are stable for a given `(as_of_date, ticker, signal_version)`.

### Track B – Backtest Engine API & Config Hygiene (Partially Complete)

- [x] Add a "backtest manifest" concept:
  - Single struct / dict describing a run:
    - universe, signal set, rebalance schedule, cost model, constraints, data ranges.
  - Log manifests alongside results for reproducibility.
  - **Done:** `core/manifest.py` with full integration into backtest engine.

- [ ] Tighten `scripts/run_institutional_backtest.py` config surface:
  - Centralize parameters (rebalance, universe, signals, costs, constraints).
  - Add validation + helpful error messages for invalid configs.

### Track C – Performance & Profiling

- [ ] Profile:
  - Calendar + universe lookups.
  - Signal access patterns in a full backtest run.
- [ ] Introduce targeted caching where it materially improves runtime without hurting PIT semantics:
  - e.g. caching `get_universe_tickers(universe, as_of_date)` results for a run.

### Track D – Extended Integration Tests ✅ Complete

- [x] Add a higher-level integration test that:
  - Runs a tiny, deterministic backtest (few tickers, short date range).
  - Asserts:
    - Manifest structure and content
    - Rebalance dates
    - Final PnL and basic stats (within performance bands).
  - **Done:** `scripts/test_deterministic_backtest.py` with 5 validation tests.

- [x] Wire this test into the "golden path" for validating major refactors.
  - **Done:** Test runs in ~0.09s, includes performance budget monitoring.

---

## Development Standards (A+++ Hygiene)

For all work (Phase 2 completed, Phase 3 upcoming):

### 1. Keep top-level docs in sync
When changing behavior in trading calendar, universes, schedules, or backtest wiring:
- Update relevant docs (README.md, docs/DATA_ARCHITECTURE.md)
- Add pointers to relevant tests
- Keep high-level and skimmable

### 2. Tight, accurate docstrings & invariants
- Public methods have accurate docstrings (inputs, outputs, assumptions)
- Critical invariants are documented close to code
- Example: PIT semantics comment right above query in UniverseManager

### 3. Testing docs & discoverability
- README.md Testing section lists all plumbing tests
- Module docstrings match what tests actually cover
- Integration tests state what is being composed

### 4. Hygiene sweep after each chunk
- Run `python3 -m compileall .`
- Check for stray TODOs: `rg "TODO" scripts core data | head -40`
- Verify CLI help matches behavior
- Remove debug prints

### 5. All plumbing tests must pass
Before committing changes to market infrastructure:
```bash
python3 scripts/test_trading_calendar.py && \
python3 scripts/test_universe_manager.py && \
python3 scripts/test_rebalance_helpers.py && \
python3 scripts/test_rebalance_schedules.py && \
python3 scripts/test_backtest_integration.py
```

---

## Next Actions

**Immediate (if continuing to Phase 3):**
1. Review Phase 3 tracks and prioritize
2. Start with Track A (Signal Panel) or Track D (Deterministic Integration Test)
3. Maintain Phase 2 test suite as baseline - never let it regress

**Maintenance:**
- Keep all 30 plumbing tests green
- Update CURRENT_STATE.md as Phase 3 progresses
- Apply A+++ hygiene standards to all new code
