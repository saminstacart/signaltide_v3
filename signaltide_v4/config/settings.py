"""
Global settings for SignalTide v4.

All configurable parameters with academic citations where applicable.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    """Immutable global settings for SignalTide v4."""

    # Database
    db_path: str = field(default_factory=lambda: os.environ.get(
        'SIGNALTIDE_DB_PATH',
        '/Users/samuelksherman/signaltide/data/signaltide.db'
    ))

    # Trading parameters
    initial_capital: float = 50_000.0
    max_position_pct: float = 0.10  # 10% max per position
    max_sector_pct: float = 0.40  # 40% max sector concentration
    transaction_cost_bps: float = 5.0  # 5 bps default
    rebalance_frequency: str = 'M'  # Monthly

    # Universe
    universe_type: str = 'sp500_actual'
    min_price: float = 5.0
    min_market_cap: float = 100_000_000  # $100M minimum

    # Signal parameters (with academic citations)
    # Residual Momentum - Blitz (2011)
    momentum_lookback_days: int = 252
    momentum_skip_days: int = 21  # Skip most recent month

    # Quality - Ball et al. (2016)
    quality_lookback_quarters: int = 4  # TTM for CbOP

    # Insider - Cohen, Malloy & Pomorski (2012)
    insider_lookback_days: int = 365
    insider_min_transaction_value: float = 10_000.0

    # Filing lag for PIT compliance
    filing_lag_days: int = 45  # Conservative SEC filing lag

    # Walk-forward validation
    # Note: With 9-year backtest (2015-07 to 2024-06), we need train + 5*test < 108 months
    # 36-month training allows 5 proper OOS folds while maintaining statistical power
    walk_forward_train_months: int = 36  # 3 years training
    walk_forward_test_months: int = 12   # 1 year test
    walk_forward_min_folds: int = 5      # Minimum OOS periods

    # Deflated Sharpe Ratio - Bailey & López de Prado (2014)
    dsr_min_confidence: float = 0.95  # 95% confidence required
    dsr_trials_adjustment: int = 100  # Assume 100 strategy trials

    # Portfolio construction
    top_n_positions: int = 25
    use_inverse_vol_weighting: bool = True
    hysteresis_threshold: float = 0.20  # 20% rank change to rebalance

    # Phase 4 Stabilization Parameters
    # Entry/Exit thresholds (percentile-based hysteresis)
    entry_percentile: int = 10     # Must be top 10% to ENTER
    exit_percentile: int = 50      # Only EXIT if below top 50%
    min_holding_months: int = 2    # Minimum holding period

    # Signal smoothing
    signal_smoothing_window: int = 3  # 3-month EMA for signal smoothing

    # Coverage requirements
    min_signals_required: int = 2     # Need at least 2 signals with data
    min_coverage_per_signal: float = 0.30  # Each signal needs 30% coverage

    # Warmup period
    warmup_months: int = 6            # Skip first 6 months for signal ramp-up
    min_universe_size: int = 50       # Need 50+ tickers minimum

    # Hard sector cap (override 40% with redistribution)
    hard_sector_cap: float = 0.35     # 35% hard cap with redistribution

    # Logging
    log_level: str = 'INFO'
    log_dir: Path = field(default_factory=lambda: Path('logs'))

    # Results
    results_dir: Path = field(default_factory=lambda: Path('results'))

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            'db_path': self.db_path,
            'initial_capital': self.initial_capital,
            'max_position_pct': self.max_position_pct,
            'max_sector_pct': self.max_sector_pct,
            'transaction_cost_bps': self.transaction_cost_bps,
            'rebalance_frequency': self.rebalance_frequency,
            'universe_type': self.universe_type,
            'min_price': self.min_price,
            'momentum_lookback_days': self.momentum_lookback_days,
            'quality_lookback_quarters': self.quality_lookback_quarters,
            'insider_lookback_days': self.insider_lookback_days,
            'filing_lag_days': self.filing_lag_days,
            'walk_forward_train_months': self.walk_forward_train_months,
            'walk_forward_test_months': self.walk_forward_test_months,
            'dsr_min_confidence': self.dsr_min_confidence,
            'top_n_positions': self.top_n_positions,
            'use_inverse_vol_weighting': self.use_inverse_vol_weighting,
            'hysteresis_threshold': self.hysteresis_threshold,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get singleton settings instance."""
    return Settings()
