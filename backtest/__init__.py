"""
Backtesting Stub Package

NOTE: This package is currently a placeholder. Actual backtesting is orchestrated via:
  - scripts/run_institutional_backtest.py (main backtest driver)
  - core/portfolio.py (portfolio accounting and position sizing)
  - core/execution.py (transaction cost modeling)
  - core/manifest.py (backtest reproducibility tracking)
  - core/schedules.py (rebalancing schedule helpers)

This stub exists to maintain a clean package structure for future consolidation.
If you need to run backtests, use:
  python scripts/run_institutional_backtest.py --help

See docs/core/ARCHITECTURE.md (Section 6: Backtest Orchestration) for details.
See docs/core/ERROR_PREVENTION_ARCHITECTURE.md (Open Gaps) for implementation status.
"""

__all__ = []
