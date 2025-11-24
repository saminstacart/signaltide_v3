# Architecture - SignalTide v3

**Last Updated:** 2025-11-18

This document explains the system design and data flow for SignalTide v3.

---

## Design Principles

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **No Lookahead Bias**: Strict temporal discipline enforced at the architecture level
3. **Vectorized Operations**: Use pandas/numpy vectorization for performance
4. **Testability**: All components designed for easy unit testing
5. **Extensibility**: Easy to add new signals, validators, or optimizers
6. **Minimal Files**: Keep total production files under 50
7. **Immutability**: Data flows through system without mutation where possible

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Optuna Optimization                      │
│  Controls all hyperparameters, runs parallel trials          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ├─→ Parameter Space Definition
                       │   (from HYPERPARAMETERS.md)
                       │
                       └─→ Objective Function
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Validation Framework                      │
│  Purged K-Fold CV, Monte Carlo, Statistical Tests           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       └─→ For each fold:
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Backtest Engine                         │
│  Simulates trading with realistic costs and constraints      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ├─→ Portfolio Manager
                       │   (Position sizing, risk management)
                       │
                       └─→ Signal Aggregator
                            │
                            ▼
                   ┌────────────────┐
                   │  Individual    │
                   │    Signals     │
                   └────────┬───────┘
                            │
                            └─→ Data Manager
                                 │
                                 ▼
                         ┌──────────────┐
                         │   Database   │
                         │  (SQLite)    │
                         └──────────────┘
```

---

## Core Modules

### 1. Data Layer (`data/`)

**Responsibility**: All data access, storage, and quality assurance

**Components**:
- `data_manager.py`: Main interface for all data operations
- `database.py`: SQLite schema and queries
- `data_quality.py`: Validation and cleaning
- `data_cache.py`: In-memory caching for performance

**Key Principles**:
- Single source of truth for all market data
- Enforce temporal ordering - no future data leaks
- Automatic data quality checks on ingestion
- Versioned data for reproducibility

**Data Flow**:
```
External Source → DataManager.ingest() → Quality Checks → Database
                                                              ↓
Portfolio/Signals → DataManager.get_data(start, end) ← Cache ← Database
```

**Anti-Lookahead Protection**:
```python
class DataManager:
    def get_data(self, symbols, start, end, as_of=None):
        """
        as_of: Ensures we only get data that would have been
               available at that timestamp (point-in-time data)
        """
        # Implementation ensures no future data leaks
```

---

### 2. Core Framework (`core/`)

**Responsibility**: Base classes and fundamental abstractions

**Components**:
- `base_signal.py`: Abstract base class for all signals
- `portfolio.py`: Portfolio state and position management
- `types.py`: Common type definitions and enums

#### BaseSignal Interface

```python
from abc import ABC, abstractmethod
import pandas as pd

class BaseSignal(ABC):
    """
    Abstract base class that all signals must implement.

    Enforces consistent interface and prevents lookahead bias.
    """

    def __init__(self, params: dict, data_manager: DataManager):
        self.params = params
        self.data_manager = data_manager
        self.name = self.__class__.__name__

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from data.

        Args:
            data: OHLCV data with DatetimeIndex

        Returns:
            Series with same index as data, values in [-1, 1]
            -1 = strong sell, 0 = neutral, 1 = strong buy

        Must not use any future data - only data up to each timestamp.
        """
        pass

    @abstractmethod
    def get_parameter_space(self) -> dict:
        """
        Define the Optuna search space for this signal's parameters.

        Returns:
            Dict mapping parameter names to Optuna suggestions
        """
        pass

    def validate_no_lookahead(self, data: pd.DataFrame, signals: pd.Series) -> bool:
        """
        Verify that signals don't use future data.
        Base implementation provided, can be overridden.
        """
        # Check that signal at time t only uses data up to time t
        pass
```

#### Portfolio Manager

```python
class Portfolio:
    """
    Manages portfolio state, positions, and executions.

    Handles:
    - Position sizing based on signal strength and risk parameters
    - Order execution with realistic slippage and commissions
    - Risk management (stops, position limits, drawdown)
    - Performance tracking
    """

    def __init__(self, initial_capital: float, params: dict):
        self.capital = initial_capital
        self.params = params
        self.positions = {}
        self.trades = []
        self.equity_curve = []

    def update(self, timestamp: pd.Timestamp, signals: dict,
               prices: dict) -> dict:
        """
        Update portfolio based on signals and current prices.

        Returns:
            Dict of executed trades
        """
        pass

    def get_metrics(self) -> dict:
        """
        Calculate performance metrics.

        Returns:
            Dict with Sharpe, drawdown, win rate, etc.
        """
        pass
```

---

### 3. Signals (`signals/`)

**Responsibility**: Individual trading signal implementations

**Structure**:
```
signals/
├── __init__.py
├── momentum/
│   ├── __init__.py
│   ├── price_momentum.py
│   └── volume_momentum.py
├── mean_reversion/
│   ├── __init__.py
│   └── zscore_reversion.py
├── volatility/
│   ├── __init__.py
│   └── vol_breakout.py
└── ml/
    ├── __init__.py
    └── ensemble_signal.py
```

**Each Signal**:
- Inherits from BaseSignal
- Implements generate_signals() with no lookahead
- Defines its parameter space
- Includes inline documentation of methodology
- Has corresponding test file

**Example**:
```python
# signals/momentum/price_momentum.py
class PriceMomentumSignal(BaseSignal):
    """
    Simple price momentum signal.

    Methodology:
    - Calculate rolling returns over lookback period
    - Normalize to [-1, 1] range
    - Apply threshold filter

    Parameters:
    - lookback_period: Period for momentum calculation
    - threshold: Minimum momentum to generate signal
    """

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        lookback = self.params['lookback_period']
        threshold = self.params['threshold']

        # Calculate momentum (no lookahead - uses shift)
        returns = data['close'].pct_change(lookback)

        # Normalize to [-1, 1]
        signals = returns.clip(-1, 1)

        # Apply threshold
        signals = signals.where(signals.abs() > threshold, 0)

        return signals

    def get_parameter_space(self) -> dict:
        return {
            'lookback_period': ('int', 5, 200),
            'threshold': ('float', 0.0, 0.5),
        }
```

---

### 4. Validation Framework (`validation/`)

**Responsibility**: Prevent overfitting through rigorous validation

**Components**:
- `purged_kfold.py`: Purged K-Fold cross-validation
- `monte_carlo.py`: Monte Carlo permutation testing
- `statistical_tests.py`: Statistical significance tests
- `deflated_sharpe.py`: Deflated Sharpe ratio calculation
- `walkforward.py`: Walk-forward analysis

#### Purged K-Fold CV

```python
class PurgedKFold:
    """
    K-Fold CV with purging and embargo to prevent leakage.

    Based on "Advances in Financial Machine Learning" by Marcos López de Prado.

    - Purging: Remove samples from training set that overlap with test set
    - Embargo: Add gap after training period to account for non-instantaneous info decay
    """

    def __init__(self, n_splits=5, purge_pct=0.05, embargo_pct=0.01):
        self.n_splits = n_splits
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct

    def split(self, data: pd.DataFrame):
        """
        Generate train/test splits with purging and embargo.

        Yields:
            (train_indices, test_indices) tuples
        """
        pass
```

#### Monte Carlo Validation

```python
class MonteCarloValidator:
    """
    Permutation testing to verify signal skill vs luck.

    Randomly permutes signals and compares performance to actual.
    If actual performance is not significantly better than permuted,
    signal likely has no real edge.
    """

    def validate(self, signal: BaseSignal, data: pd.DataFrame,
                 n_trials=1000) -> dict:
        """
        Run Monte Carlo validation.

        Returns:
            Dict with p-value, percentile, and confidence metrics
        """
        pass
```

---

### 5. Optimization (`optimization/`)

**Responsibility**: Hyperparameter optimization via Optuna

**Components**:
- `optuna_manager.py`: Manages Optuna studies
- `parameter_space.py`: Parses HYPERPARAMETERS.md into Optuna space
- `objective.py`: Objective function for optimization

#### Optuna Manager

```python
class OptunaManager:
    """
    Manages hyperparameter optimization studies.

    Features:
    - Parallel trial execution
    - Study persistence to SQLite
    - Automatic overfitting detection
    - Progress tracking and visualization
    """

    def __init__(self, study_name: str, storage_path: str):
        self.study_name = study_name
        self.storage = f'sqlite:///{storage_path}'

    def optimize(self, objective_fn, n_trials=100, n_jobs=-1):
        """
        Run optimization with parallel trials.

        Args:
            objective_fn: Function to maximize (e.g., validated Sharpe)
            n_trials: Number of trials to run
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction='maximize',
            load_if_exists=True
        )

        study.optimize(objective_fn, n_trials=n_trials, n_jobs=n_jobs)

        return study

    def get_best_params(self) -> dict:
        """Get best parameters from completed study."""
        pass

    def check_overfitting(self, train_score, test_score, threshold=0.3):
        """
        Check if optimization has overfit.

        If test score is significantly worse than train score,
        likely overfit to training data.
        """
        pass
```

---

### 6. Backtest Orchestration

**Responsibility**: Realistic trading simulation

**Current Implementation**:
Backtesting is orchestrated via `scripts/run_institutional_backtest.py` using core modules:

- `core/portfolio.py`: Portfolio accounting, position sizing, transaction costs
- `core/execution.py`: TransactionCostModel (commission, slippage, spread)
- `core/manifest.py`: BacktestManifest for reproducibility tracking
- `core/schedules.py`: Rebalance schedule helpers (monthly, weekly, daily)
- `core/universe_manager.py`: Point-in-time universe construction

**Key Features**:
- Monthly rebalancing (96-98% turnover reduction vs daily)
- Realistic transaction costs (~5 bps for $50K Schwab account)
- Point-in-time data discipline (no lookahead bias)
- Deterministic results with manifest tracking

**Example Usage**:
```bash
# Via main backtest script
python scripts/run_institutional_backtest.py \
    --universe manual \
    --tickers AAPL,MSFT,GOOGL \
    --period 2020-01-01,2024-12-31
```

---

## Data Flow Examples

### Example 1: Running a Backtest

```
1. User initiates backtest
2. DataManager loads historical data (with point-in-time constraints)
3. For each timestamp:
   a. Each signal generates signal value (using only data up to timestamp)
   b. Portfolio aggregates signals
   c. Portfolio calculates position sizes
   d. Portfolio executes trades (with slippage/commissions)
   e. Portfolio updates equity curve
4. Metrics calculated from equity curve
5. Results returned
```

### Example 2: Hyperparameter Optimization

```
1. OptunaManager creates/loads study
2. For each trial (in parallel):
   a. ParameterSpace samples parameters
   b. Validation framework creates train/test splits
   c. For each split:
      i. Backtest on training data
      ii. Backtest on test data
   d. Calculate validated Sharpe ratio
   e. Return to Optuna
3. Optuna suggests next parameters (TPE sampler)
4. Repeat until n_trials completed
5. Return best parameters
```

### Example 3: Adding a New Signal

```
1. Create new file in signals/ directory
2. Inherit from BaseSignal
3. Implement generate_signals() method
4. Implement get_parameter_space() method
5. Add parameters to HYPERPARAMETERS.md
6. Create test file in tests/signals/
7. Run validation framework
8. If validation passes, include in portfolio
```

---

## File Structure

```
signaltide_v3/
│
├── config.py                    # System configuration
├── requirements.txt             # Python dependencies
├── Makefile                     # Common commands
│
├── core/                        # Core framework
│   ├── __init__.py
│   ├── base_signal.py          # Abstract signal class
│   ├── portfolio.py            # Portfolio manager
│   └── types.py                # Common types
│
├── data/                        # Data layer
│   ├── __init__.py
│   ├── data_manager.py         # Main data interface
│   ├── database.py             # SQLite operations
│   ├── data_quality.py         # Quality checks
│   └── cache.py                # Caching layer
│
├── signals/                     # Trading signals
│   ├── __init__.py
│   ├── momentum/               # Momentum signals
│   ├── mean_reversion/         # Mean reversion signals
│   ├── volatility/             # Volatility signals
│   └── ml/                     # ML-based signals
│
├── validation/                  # Validation framework
│   ├── __init__.py
│   ├── purged_kfold.py        # Purged K-Fold CV
│   ├── monte_carlo.py         # Monte Carlo validation
│   ├── statistical_tests.py   # Statistical tests
│   └── deflated_sharpe.py     # Deflated Sharpe
│
├── core/                        # Core infrastructure
│   ├── base_signal.py         # Signal base class
│   ├── institutional_base.py  # Institutional signal base
│   ├── db.py                  # Read-only DB connections
│   ├── universe_manager.py    # Point-in-time universes
│   ├── portfolio.py           # Portfolio accounting
│   ├── execution.py           # Transaction cost model
│   ├── manifest.py            # Backtest manifests
│   ├── schedules.py           # Rebalance schedules
│   └── types.py               # Type definitions
│
├── signals/                     # Trading signals
│   ├── momentum/
│   │   ├── institutional_momentum.py  # Jegadeesh-Titman momentum
│   │   └── simple_momentum.py         # Simple momentum baseline
│   ├── quality/
│   │   ├── institutional_quality.py   # Asness QMJ quality
│   │   └── simple_quality.py          # Simple quality baseline
│   └── insider/
│       ├── institutional_insider.py   # Cohen-Malloy-Pomorski insider
│       └── simple_insider.py          # Simple insider baseline
│
├── data/                        # Data management
│   ├── data_manager.py        # Sharadar data interface
│   └── mock_generator.py      # Test data generation
│
├── tests/                       # Unit tests
│   └── test_institutional_signals.py
│
├── docs/core/                   # Design documentation
│   ├── ARCHITECTURE.md        # This file
│   ├── DATA_ARCHITECTURE.md   # Database schema
│   ├── METHODOLOGY.md         # Academic methods
│   ├── INSTITUTIONAL_METHODS.md
│   └── PRODUCTION_READY.md
│
└── scripts/                     # Orchestration scripts
    ├── run_institutional_backtest.py  # Main backtest driver
    ├── build_trading_calendar.py      # Calendar construction
    ├── validate_sharadar_data.py      # Data validation
    ├── test_trading_calendar.py       # Calendar tests
    ├── test_universe_manager.py       # Universe tests
    ├── test_rebalance_schedules.py    # Schedule tests
    └── test_deterministic_backtest.py # E2E backtest test
```

---

## Key Design Decisions

### Why SQLite?
- Self-contained, no separate database server
- ACID compliance for data integrity
- Fast enough for our scale ($50K AUM)
- Easy backup and versioning
- Can upgrade to PostgreSQL later if needed

### Why Vectorized Operations?
- 10-100x faster than iterative approaches
- Leverages optimized numpy/pandas code
- Forces proper data structure design
- Easier to verify no lookahead bias

### Why Strict Abstract Base Classes?
- Enforces consistent interface across signals
- Makes it impossible to forget critical methods
- Self-documenting code
- Easy to add new signals

### Why Purged K-Fold?
- Standard K-Fold leaks information in time series
- Purging removes overlapping samples
- Embargo accounts for non-instantaneous information decay
- Industry standard for time series ML

### Why Monte Carlo Validation?
- Verifies edge is real vs luck
- No parametric assumptions needed
- Intuitive interpretation (p-value, percentile)
- Catches data snooping bias

---

## Performance Considerations

- **Caching**: DataManager caches frequently accessed data in memory
- **Vectorization**: All operations use pandas/numpy vectorization
- **Lazy Loading**: Data loaded only when needed
- **Parallel Execution**: Optuna trials run in parallel
- **Database Indexing**: Proper indices on timestamp columns
- **Minimal Copies**: Pass views instead of copies where safe

---

## Security Considerations

- **No Hardcoded Credentials**: Use .env for API keys
- **Input Validation**: Validate all external data
- **SQL Injection**: Use parameterized queries only
- **File Permissions**: Restrict database file permissions
- **Logging**: Never log sensitive information (API keys, credentials)

---

## Extensibility Points

Easy to extend:
- Add new signals (inherit from BaseSignal)
- Add new validators (implement validation interface)
- Add new portfolio methods (modify Portfolio class)
- Add new data sources (extend DataManager)
- Add new optimization samplers (configure Optuna)

---

## Testing Strategy

- **Unit Tests**: Test each component in isolation
- **Integration Tests**: Test components working together
- **Data Integrity Tests**: Verify no lookahead bias
- **Performance Tests**: Ensure acceptable runtime
- **Statistical Tests**: Verify validation math is correct

---

## Monitoring & Logging

- Log all data access with timestamps
- Log all trades with execution details
- Log validation results
- Log optimization progress
- Use structured logging (JSON format)
- Separate log levels: DEBUG, INFO, WARNING, ERROR

---

## Future Considerations

- Multi-asset support (stocks, futures, options)
- Real-time data integration
- Live trading interface
- Web dashboard for monitoring
- Cloud deployment (AWS/GCP)
- Distributed backtesting
- Advanced regime detection
- Alternative data integration
