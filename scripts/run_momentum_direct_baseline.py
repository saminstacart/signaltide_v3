"""
Run baseline backtest using InstitutionalMomentum directly (no ensemble wrapper).

This script tests whether the ensemble layer introduces any differences vs direct signal usage.

Comparison targets:
- Trial 11 diagnostic: 89.53% return, 0.245 Sharpe, -44.76% max DD
- Ensemble baseline: 129.89% return, 0.601 Sharpe, -30.20% max DD

If direct matches ensemble baseline, the gap is due to signal implementation differences.
If direct matches Trial 11, the ensemble layer is introducing the gap.

REFACTORED: Now uses core/backtest_engine.py for consistent execution.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from datetime import timedelta
from typing import List

from data.data_manager import DataManager
from core.universe_manager import UniverseManager
from signals.momentum.institutional_momentum import InstitutionalMomentum
from core.backtest_engine import BacktestConfig, run_backtest
from core.signal_adapters import make_signal_fn
from config import get_logger

logger = get_logger(__name__)


def main():
    """Run direct momentum baseline using unified backtest harness."""

    # Initialize components
    dm = DataManager()
    um = UniverseManager(dm)

    # Canonical Momentum v2 parameters (Trial 11)
    momentum_params = {
        'formation_period': 308,
        'skip_period': 0,
        'winsorize_pct': [0.4, 99.6],
        'rebalance_frequency': 'monthly',
        'quintiles': True,
    }

    # Initialize signal
    signal = InstitutionalMomentum(params=momentum_params)

    logger.info("=" * 80)
    logger.info("DIRECT MOMENTUM BASELINE (via unified harness)")
    logger.info("=" * 80)
    logger.info(f"Momentum params: {momentum_params}")
    logger.info("")

    # Define universe function
    def universe_fn(rebal_date: str) -> List[str]:
        """Get S&P 500 PIT universe at rebalance date."""
        universe = um.get_universe(
            universe_type='sp500_actual',
            as_of_date=rebal_date,
            min_price=5.0
        )

        if isinstance(universe, pd.Series):
            return universe.tolist()
        elif isinstance(universe, pd.DataFrame):
            return universe.index.tolist()
        else:
            return list(universe)

    # Use adapter to create signal function
    signal_fn = make_signal_fn(signal, dm)

    # Configure backtest
    config = BacktestConfig(
        start_date='2015-04-01',
        end_date='2024-12-31',
        initial_capital=100000.0,
        rebalance_schedule='M',
        long_only=True,
        equal_weight=True,
        track_daily_equity=False,  # Rebalance-point only for now
        data_manager=dm
    )

    # Run backtest
    result = run_backtest(universe_fn, signal_fn, config)

    # Additional logging
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON TO TARGETS")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"{'Metric':<20} {'Trial 11':<15} {'Direct':<15} {'Ensemble':<15}")
    logger.info("-" * 65)
    logger.info(f"{'Total Return':<20} {'89.53%':<15} {f'{result.total_return:.2%}':<15} {'129.89%':<15}")
    logger.info(f"{'Sharpe Ratio':<20} {'0.245':<15} {f'{result.sharpe:.3f}':<15} {'0.601':<15}")
    logger.info(f"{'Max Drawdown':<20} {'-44.76%':<15} {f'{result.max_drawdown:.2%}':<15} {'-30.20%':<15}")
    logger.info("")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
