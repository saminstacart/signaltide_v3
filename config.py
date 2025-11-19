"""
Configuration for SignalTide v3.

This module centralizes all system configuration.
Settings can be overridden via environment variables.
"""

import os
from pathlib import Path

# Load environment variables from .env file (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, will use environment variables only
    pass

# ============================================================================
# Project Paths
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DB_DIR = DATA_DIR / "databases"
CACHE_DIR = DATA_DIR / "cache"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for directory in [DATA_DIR, DB_DIR, CACHE_DIR, LOGS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Database Configuration
# ============================================================================

# Main database for market data (Sharadar data)
# Supports environment variable override
MARKET_DATA_DB = Path(os.getenv(
    "SIGNALTIDE_DB_PATH",
    str(DB_DIR / "market_data.db")
))

# Fallback to v2 database if SIGNALTIDE_DB_PATH not set and file doesn't exist
if not MARKET_DATA_DB.exists() and "SIGNALTIDE_DB_PATH" not in os.environ:
    v2_db_path = Path("/Users/samuelksherman/signaltide/data/signaltide.db")
    if v2_db_path.exists():
        MARKET_DATA_DB = v2_db_path

# Database for Optuna studies
OPTUNA_DB = DB_DIR / "optuna_studies.db"
OPTUNA_STORAGE = f"sqlite:///{OPTUNA_DB}"

# Cache settings for DataManager
DB_CACHE_SIZE = int(os.getenv("SIGNALTIDE_CACHE_SIZE", "100"))

# ============================================================================
# Data Configuration
# ============================================================================

# Default data source
DEFAULT_EXCHANGE = os.getenv("DEFAULT_EXCHANGE", "binance")

# Default trading pair
DEFAULT_SYMBOL = os.getenv("DEFAULT_SYMBOL", "BTC/USDT")

# Default timeframe
DEFAULT_TIMEFRAME = os.getenv("DEFAULT_TIMEFRAME", "1h")

# Data quality checks
REQUIRE_COMPLETE_DATA = True  # Fail if data has gaps
MAX_MISSING_PCT = 0.01  # Maximum 1% missing data allowed

# ============================================================================
# Portfolio Configuration
# ============================================================================

# Initial capital
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "50000"))

# Default portfolio parameters
DEFAULT_PORTFOLIO_PARAMS = {
    'initial_capital': INITIAL_CAPITAL,
    'max_positions': 5,
    'position_sizing_method': 'volatility_scaled',
    'rebalance_frequency': '1D',
}

# ============================================================================
# Risk Management
# ============================================================================

DEFAULT_RISK_PARAMS = {
    'max_position_size': 0.20,  # 20% of portfolio
    'stop_loss_pct': 0.05,  # 5% stop loss
    'take_profit_pct': 0.15,  # 15% take profit
    'max_portfolio_drawdown': 0.25,  # 25% max drawdown
    'drawdown_scale_factor': 0.5,  # 50% reduction in exposure during drawdown
}

# ============================================================================
# Transaction Costs
# ============================================================================

DEFAULT_TRANSACTION_COSTS = {
    'commission_pct': 0.001,  # 0.1% commission (10 bps)
    'slippage_pct': 0.001,  # 0.1% slippage
    'spread_pct': 0.0005,  # 0.05% spread (5 bps)
}

# ============================================================================
# Validation Configuration
# ============================================================================

VALIDATION_PARAMS = {
    # Purged K-Fold
    'n_splits': 5,
    'purge_pct': 0.05,  # 5% purge
    'embargo_pct': 0.01,  # 1% embargo

    # Sample size
    'min_sample_size': 252,  # Minimum 1 year of daily data

    # Monte Carlo
    'monte_carlo_n_trials': 1000,

    # Statistical significance
    'significance_level': 0.05,  # p-value threshold
}

# ============================================================================
# Optimization Configuration
# ============================================================================

OPTUNA_PARAMS = {
    'n_trials': int(os.getenv("OPTUNA_N_TRIALS", "100")),
    'n_jobs': int(os.getenv("OPTUNA_N_JOBS", "-1")),  # -1 = all cores
    'timeout': None,  # No timeout
    'sampler': 'TPE',  # Tree-structured Parzen Estimator
    'n_startup_trials': 20,  # Random trials before TPE
    'pruner': 'Median',
}

# ============================================================================
# Logging Configuration
# ============================================================================

# Environment-based log level
SIGNALTIDE_ENV = os.getenv("SIGNALTIDE_ENV", "development")
if SIGNALTIDE_ENV == "production":
    LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING")
elif SIGNALTIDE_ENV == "staging":
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
else:  # development
    LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")

# Standard Python logging format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

LOG_CONFIG = {
    'level': LOG_LEVEL,
    'format': LOG_FORMAT,
    'date_format': LOG_DATE_FORMAT,
    'log_file': LOGS_DIR / f'signaltide_{SIGNALTIDE_ENV}.log',
    'console_output': SIGNALTIDE_ENV != 'production',
}

import logging

def setup_logging():
    """Setup logging configuration."""
    handlers = []

    # Console handler
    if LOG_CONFIG['console_output']:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, LOG_CONFIG['level']))
        console_formatter = logging.Formatter(
            LOG_CONFIG['format'],
            datefmt=LOG_CONFIG['date_format']
        )
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)

    # File handler
    file_handler = logging.FileHandler(LOG_CONFIG['log_file'])
    file_handler.setLevel(getattr(logging, LOG_CONFIG['level']))
    file_formatter = logging.Formatter(
        LOG_CONFIG['format'],
        datefmt=LOG_CONFIG['date_format']
    )
    file_handler.setFormatter(file_formatter)
    handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, LOG_CONFIG['level']),
        format=LOG_CONFIG['format'],
        datefmt=LOG_CONFIG['date_format'],
        handlers=handlers
    )

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module."""
    return logging.getLogger(name)

# Setup logging on import
setup_logging()

# ============================================================================
# Performance Configuration
# ============================================================================

# Cache settings
ENABLE_CACHE = True
CACHE_SIZE_MB = 500  # Maximum cache size in MB

# Parallel execution
MAX_WORKERS = os.cpu_count()  # Use all CPU cores

# Numba JIT compilation
ENABLE_NUMBA = True

# ============================================================================
# API Configuration (for data fetching)
# ============================================================================

# Exchange API keys (loaded from .env)
EXCHANGE_API_KEY = os.getenv("EXCHANGE_API_KEY", "")
EXCHANGE_API_SECRET = os.getenv("EXCHANGE_API_SECRET", "")

# Rate limiting
API_RATE_LIMIT = 1000  # Max requests per minute

# ============================================================================
# Testing Configuration
# ============================================================================

# Use smaller datasets for testing
TEST_MODE = os.getenv("TEST_MODE", "false").lower() == "true"

if TEST_MODE:
    # Override settings for faster testing
    VALIDATION_PARAMS['n_splits'] = 3
    VALIDATION_PARAMS['monte_carlo_n_trials'] = 100
    OPTUNA_PARAMS['n_trials'] = 10

# ============================================================================
# Feature Flags
# ============================================================================

FEATURES = {
    'regime_detection': True,
    'ml_signals': True,
    'advanced_risk_management': True,
    'real_time_data': False,  # Not implemented yet
    'live_trading': False,  # Not implemented yet
}

# ============================================================================
# Constants
# ============================================================================

# Trading constants
TRADING_DAYS_PER_YEAR = 252
HOURS_PER_YEAR = 8760
MINUTES_PER_YEAR = 525600

# Annualization factors
ANNUALIZATION_FACTORS = {
    '1m': 525600,
    '5m': 105120,
    '15m': 35040,
    '1h': 8760,
    '4h': 2190,
    '1d': 252,
    '1w': 52,
}

# ============================================================================
# Validation
# ============================================================================

def validate_config():
    """Validate configuration settings."""
    errors = []

    # Check required directories exist
    if not PROJECT_ROOT.exists():
        errors.append(f"Project root does not exist: {PROJECT_ROOT}")

    # Check required environment variables for live trading
    if FEATURES['live_trading']:
        if not EXCHANGE_API_KEY:
            errors.append("EXCHANGE_API_KEY not set (required for live trading)")
        if not EXCHANGE_API_SECRET:
            errors.append("EXCHANGE_API_SECRET not set (required for live trading)")

    # Validate numeric ranges
    if not 0 < DEFAULT_RISK_PARAMS['max_position_size'] <= 1:
        errors.append(f"Invalid max_position_size: {DEFAULT_RISK_PARAMS['max_position_size']}")

    if not 0 < DEFAULT_RISK_PARAMS['stop_loss_pct'] < 1:
        errors.append(f"Invalid stop_loss_pct: {DEFAULT_RISK_PARAMS['stop_loss_pct']}")

    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

# Run validation on import
validate_config()

# ============================================================================
# Helper Functions
# ============================================================================

def get_db_path(db_name: str) -> Path:
    """Get full path to a database file."""
    return DB_DIR / f"{db_name}.db"

def get_annualization_factor(timeframe: str) -> int:
    """Get annualization factor for a given timeframe."""
    if timeframe not in ANNUALIZATION_FACTORS:
        raise ValueError(f"Unknown timeframe: {timeframe}. Valid: {list(ANNUALIZATION_FACTORS.keys())}")
    return ANNUALIZATION_FACTORS[timeframe]

def is_feature_enabled(feature: str) -> bool:
    """Check if a feature is enabled."""
    return FEATURES.get(feature, False)

# ============================================================================
# Display Configuration (for debugging)
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SignalTide v3 Configuration")
    print("=" * 80)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Database Directory: {DB_DIR}")
    print(f"Logs Directory: {LOGS_DIR}")
    print(f"\nInitial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Default Exchange: {DEFAULT_EXCHANGE}")
    print(f"Default Symbol: {DEFAULT_SYMBOL}")
    print(f"Default Timeframe: {DEFAULT_TIMEFRAME}")
    print(f"\nOptuna Trials: {OPTUNA_PARAMS['n_trials']}")
    print(f"Parallel Jobs: {OPTUNA_PARAMS['n_jobs']}")
    print(f"\nValidation Splits: {VALIDATION_PARAMS['n_splits']}")
    print(f"Monte Carlo Trials: {VALIDATION_PARAMS['monte_carlo_n_trials']}")
    print(f"\nEnabled Features:")
    for feature, enabled in FEATURES.items():
        status = "✓" if enabled else "✗"
        print(f"  {status} {feature}")
    print(f"\nTest Mode: {'ON' if TEST_MODE else 'OFF'}")
    print("=" * 80)
