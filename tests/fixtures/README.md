# Test Fixtures for SignalTide v3

## Overview

This directory contains lightweight test fixtures for CI and fast local testing.

## Files

### `signaltide_small.db`
A minimal SQLite database mimicking the Sharadar schema with:
- **3 tickers**: AAPL, MSFT, GOOGL
- **Time range**: 2020 Q1 (Jan-Mar) + specific dates for edge case testing
- **Tables**: All core tables with only essential columns
  - `sharadar_prices`: OHLCV price data
  - `sharadar_sf1`: Fundamental metrics (quarterly)
  - `sharadar_insiders`: Insider trading records
  - `sharadar_tickers`: Ticker metadata
  - `dim_trading_calendar`: Trading days, weekends, holidays
  - `dim_universe_membership`: Universe tracking

### `mock_sharadar_schema.sql`
SQL script to rebuild the fixture database.

## Usage

### In Tests
Set the environment variable to point at the fixture:

```bash
export SIGNALTIDE_MARKET_DATA_DB=tests/fixtures/signaltide_small.db
python scripts/test_trading_calendar.py
```

### In CI
Use the `test-ci` Makefile target:

```bash
make test-ci
```

This automatically sets `SIGNALTIDE_MARKET_DATA_DB` to use the fixture.

### Rebuilding the Fixture
If you need to modify the fixture data:

1. Edit `mock_sharadar_schema.sql`
2. Rebuild the database:
   ```bash
   make build-fixture-db
   # or manually:
   sqlite3 tests/fixtures/signaltide_small.db < tests/fixtures/mock_sharadar_schema.sql
   ```

## Limitations

This fixture is designed for:
- ✅ Trading calendar tests
- ✅ Universe manager tests
- ✅ Data manager API tests
- ✅ Point-in-time correctness tests
- ✅ Plumbing and infrastructure tests

It is **NOT** suitable for:
- ❌ Full backtests (insufficient data range)
- ❌ Signal validation (too few tickers)
- ❌ Performance benchmarking (unrealistic dataset size)
- ❌ Statistical significance testing

For production research and backtesting, use the full 7.6GB Sharadar database at:
`data/databases/signaltide.db`

## Size Comparison

- **Fixture DB**: ~100 KB (this fixture)
- **Full DB**: 7.6 GB (production Sharadar database)
- **Speed improvement**: ~1000x faster for plumbing tests

## CI/CD Integration

This fixture enables fast, self-contained CI pipelines without requiring:
- External database servers
- API keys
- Large data downloads
- Long test setup times

Perfect for GitHub Actions, pre-commit hooks, and developer machines.
