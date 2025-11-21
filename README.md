# SignalTide v3: Institutional-Grade Quantitative Trading System

## Project Philosophy

SignalTide v3 is a research-driven quantitative trading system built with institutional best practices and academic rigor. This system is designed for a $50K AUM strategy but implements the methodology and validation framework you would find in a $10M+ fund.

**Core Principles:**

1. **Correctness Over Speed**: We prioritize methodological soundness over execution speed
2. **No Premature Optimization**: Let Optuna explore the full parameter space
3. **Rigorous Validation**: Every signal must pass comprehensive validation before deployment
4. **Academic Standards**: All methodology is documented with academic-grade explanations
5. **Simplicity**: Keep the codebase under 50 production files with single-responsibility modules
6. **No Lookahead Bias**: Strict temporal discipline in all data operations
7. **Statistical Rigor**: Use deflated Sharpe ratios, purged K-fold CV, and Monte Carlo validation

## System Overview

SignalTide v3 is a multi-signal portfolio optimization system that:

- Aggregates multiple technical and fundamental signals
- Uses regime detection to adapt to market conditions
- Employs Optuna for hyperparameter optimization with parallel execution
- Implements institutional-grade validation to prevent overfitting
- Manages risk through dynamic position sizing and stop losses
- Operates on cryptocurrency markets with plans for multi-asset expansion

## Architecture

```
signaltide_v3/
├── core/           # Base classes and portfolio management
├── signals/        # Individual trading signals
├── validation/     # Validation framework (purged K-fold, Monte Carlo, etc.)
├── optimization/   # Optuna-based hyperparameter optimization
├── data/           # Data management and storage
├── backtest/       # Backtesting engine
├── tests/          # Comprehensive test suite
└── docs/           # Detailed methodology documentation
```

## Key Features

**Signal Framework:**
- Extensible BaseSignal class for easy signal development
- Automatic validation pipeline for all signals
- Vectorized operations for performance
- Built-in bias detection

**Validation Framework:**
- Purged K-Fold Cross-Validation (prevents temporal leakage)
- Monte Carlo Permutation Tests
- Deflated Sharpe Ratio calculations
- Statistical significance testing
- Walk-forward analysis

**Optimization:**
- Optuna-based hyperparameter optimization
- Parallel trial execution
- Wide parameter search spaces (no premature filtering)
- Automatic overfitting detection
- Study persistence and resumability

**Risk Management:**
- Dynamic position sizing
- Multiple stop-loss strategies
- Drawdown monitoring
- Regime-aware risk adjustment

## Getting Started

### 0. First-time setup

After cloning the repo, create the standard data/logs/results directories (these are gitignored):

```bash
make setup-dirs
```

### 1. Install dependencies and run tests

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
make test

# Run validation framework
make validate

# Run hyperparameter optimization
make optimize

# Run backtest
make backtest
```

## Testing

### Running Tests

**Quick test (unit tests only):**
```bash
make test
```

**CI test suite (uses lightweight fixture DB):**
```bash
make test-ci
```

Runs tests against a small SQLite fixture (~100KB) instead of the full 7.6GB database.
Perfect for fast iteration and CI pipelines.

**Full plumbing tests (31 tests):**
```bash
make test-plumbing
```

Requires the full Sharadar database.

**Complete pre-commit suite (unit + plumbing):**
```bash
make test-all
```

This is the **canonical test suite** to run before committing.

### Market Plumbing & Backtest Integrity Tests

These tests verify the core trading infrastructure (calendar, universes, schedules):

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

# Deterministic end-to-end backtest test (validates manifest & performance bands)
python3 scripts/test_deterministic_backtest.py

# Run all plumbing and orchestration tests
python3 scripts/test_trading_calendar.py && \
python3 scripts/test_universe_manager.py && \
python3 scripts/test_rebalance_helpers.py && \
python3 scripts/test_rebalance_schedules.py && \
python3 scripts/test_backtest_integration.py && \
python3 scripts/test_deterministic_backtest.py
```

All tests must pass before committing changes to market plumbing or backtest engine.

**Note:** The deterministic backtest test serves as the "golden path" for validating major refactors. It validates:
- Complete backtest manifest structure (run ID, parameters, git SHA, etc.)
- Performance metrics within expected bands (prevents regressions)
- Performance budget monitoring (~0.09s runtime)

**Makefile Shortcuts:**
```bash
make test-plumbing    # Run all 31 plumbing + orchestration tests
make test             # Full test suite (currently aliases test-plumbing)
make lint             # Run ruff linter
make fmt              # Run ruff formatter
make clean            # Clean cache and temp files
```

## Documentation

- **CURRENT_STATE.md**: Current project status and progress tracking
- **DOCUMENTATION_MAP.md**: Complete navigation guide to all docs

**Core Design Docs** (docs/core/):
- **ARCHITECTURE.md**: System design and data flow
- **DATA_ARCHITECTURE.md**: Market DB schema, trading calendar, universes
- **INSTITUTIONAL_METHODS.md**: Institutional-grade signal methodologies
- **METHODOLOGY.md**: Academic explanation of methods
- **HYPERPARAMETERS.md**: All tunable parameters and ranges
- **ANTI_OVERFITTING.md**: Validation approach
- **OPTUNA_GUIDE.md**: Optimization strategy
- **PRODUCTION_READY.md**: Production deployment checklist

## Development Guidelines

1. **Every signal must pass validation** before being included in the portfolio
2. **Optuna controls all parameters** - no manual overrides
3. **Document all assumptions** in code comments
4. **Use vectorized operations** wherever possible
5. **Single responsibility** per module
6. **Write tests first** for new functionality

## Status

See CURRENT_STATE.md for current development status and progress tracking.

## License

Proprietary - All rights reserved.
