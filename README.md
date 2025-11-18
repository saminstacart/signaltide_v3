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

## Documentation

- **CURRENT_STATE.md**: Track project progress and status
- **NEXT_STEPS.md**: Prioritized task list
- **HYPERPARAMETERS.md**: All tunable parameters and their ranges
- **ARCHITECTURE.md**: System design and data flow
- **docs/METHODOLOGY.md**: Academic explanation of methods
- **docs/ANTI_OVERFITTING.md**: Validation approach
- **docs/OPTUNA_GUIDE.md**: Optimization strategy

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
