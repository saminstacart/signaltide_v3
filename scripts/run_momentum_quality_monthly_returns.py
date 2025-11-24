"""
Momentum + Quality Monthly Returns Materialization

Generates aligned monthly return series for:
1. Momentum-only ensemble (v2)
2. Momentum + Quality v1 ensemble (50/50 weights)

These monthly returns are used as inputs for Optuna-based weight optimization,
allowing efficient weight tuning without re-running expensive backtests.

Outputs:
    results/ensemble_baselines/momentum_v2_monthly_returns.csv
    results/ensemble_baselines/momentum_quality_v1_monthly_returns.csv

Schema: date,return (month-end dates with monthly returns)

Usage:
    python3 scripts/run_momentum_quality_monthly_returns.py

    # Shorter period for testing
    python3 scripts/run_momentum_quality_monthly_returns.py --start 2020-01-01
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List
import warnings
warnings.filterwarnings('ignore')

from data.data_manager import DataManager
from core.universe_manager import UniverseManager
from core.backtest_engine import BacktestConfig, run_backtest
from core.signal_adapters import make_ensemble_signal_fn, make_multisignal_ensemble_fn
from signals.ml.ensemble_configs import get_momentum_v2_ensemble, get_momentum_quality_v1_ensemble
from config import get_logger

logger = get_logger(__name__)


class MonthlyReturnsGenerator:
    """
    Generates monthly return series for momentum and momentum+quality ensembles.
    """

    def __init__(self,
                 start_date: str = '2015-04-01',
                 end_date: str = '2024-12-31',
                 initial_capital: float = 100000.0):
        """
        Initialize monthly returns generator.

        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital (used for consistency with baselines)
        """
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital

        self.dm = DataManager()
        self.um = UniverseManager(self.dm)

        logger.info("=" * 80)
        logger.info("MONTHLY RETURNS GENERATION - Momentum & Momentum+Quality")
        logger.info("=" * 80)
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Capital: ${initial_capital:,.0f}")
        logger.info(f"Universe: sp500_actual (min_price=5.0)")
        logger.info("=" * 80)
        logger.info("")

    def generate_monthly_returns(self):
        """
        Run backtests and extract monthly returns.

        Returns aligned monthly return series for both strategies.
        """
        # Define shared universe function
        def universe_fn(rebal_date: str) -> List[str]:
            """Get S&P 500 PIT universe at rebalance date."""
            universe = self.um.get_universe(
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

        # Shared backtest config
        config = BacktestConfig(
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.initial_capital,
            rebalance_schedule='M',  # Monthly rebalancing
            long_only=True,
            equal_weight=True,
            track_daily_equity=False,
            data_manager=self.dm
        )

        # 1. Run momentum-only ensemble
        logger.info("Running momentum_v2 backtest...")
        momentum_ensemble = get_momentum_v2_ensemble(self.dm)
        momentum_signal_fn = make_ensemble_signal_fn(momentum_ensemble, self.dm, lookback_days=500)
        momentum_result = run_backtest(universe_fn, momentum_signal_fn, config)

        logger.info(f"  Momentum v2: Total Return={momentum_result.total_return:.2%}, "
                   f"Sharpe={momentum_result.sharpe:.3f}, Rebalances={momentum_result.num_rebalances}")
        logger.info("")

        # 2. Run momentum + quality ensemble
        logger.info("Running momentum_quality_v1 backtest...")
        mq_ensemble = get_momentum_quality_v1_ensemble(self.dm)
        mq_signal_fn = make_multisignal_ensemble_fn(mq_ensemble, self.dm)
        mq_result = run_backtest(universe_fn, mq_signal_fn, config)

        logger.info(f"  Momentum+Quality v1: Total Return={mq_result.total_return:.2%}, "
                   f"Sharpe={mq_result.sharpe:.3f}, Rebalances={mq_result.num_rebalances}")
        logger.info("")

        # 3. Extract monthly returns from equity curves
        logger.info("Extracting monthly returns...")

        # Get month-end dates and returns
        momentum_monthly = self._extract_monthly_returns(
            momentum_result.equity_curve,
            "momentum_v2"
        )
        mq_monthly = self._extract_monthly_returns(
            mq_result.equity_curve,
            "momentum_quality_v1"
        )

        # 4. Sanity check alignment
        self._verify_alignment(momentum_monthly, mq_monthly)

        # 5. Save to CSV
        self._save_monthly_returns(momentum_monthly, "momentum_v2")
        self._save_monthly_returns(mq_monthly, "momentum_quality_v1")

        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ MONTHLY RETURNS GENERATION COMPLETE")
        logger.info("=" * 80)

        return {
            'momentum_v2': momentum_monthly,
            'momentum_quality_v1': mq_monthly
        }

    def _extract_monthly_returns(self, equity_curve: pd.Series, label: str) -> pd.DataFrame:
        """
        Extract monthly returns from daily equity curve.

        Args:
            equity_curve: Daily equity series (index=dates, values=equity)
            label: Strategy label for logging

        Returns:
            DataFrame with columns: date (month-end), return (monthly return)
        """
        # Resample to month-end, taking last value of each month
        monthly_equity = equity_curve.resample('M').last()

        # Calculate monthly returns
        monthly_returns = monthly_equity.pct_change().dropna()

        # Create DataFrame with proper schema
        df = pd.DataFrame({
            'date': monthly_returns.index,
            'return': monthly_returns.values
        })

        logger.info(f"  {label}: Extracted {len(df)} monthly returns")
        logger.info(f"    Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"    Mean monthly return: {df['return'].mean():.4%}")
        logger.info(f"    Std monthly return: {df['return'].std():.4%}")

        return df

    def _verify_alignment(self, df1: pd.DataFrame, df2: pd.DataFrame):
        """
        Verify that two monthly return series are properly aligned.

        Args:
            df1: First monthly returns DataFrame
            df2: Second monthly returns DataFrame
        """
        # Check same number of months
        if len(df1) != len(df2):
            logger.warning(f"⚠️  Length mismatch: {len(df1)} vs {len(df2)} months")
        else:
            logger.info(f"✅ Both series have {len(df1)} months")

        # Check date alignment
        date_match = (df1['date'] == df2['date']).all()
        if not date_match:
            logger.warning("⚠️  Date indices not perfectly aligned")
            # Show first mismatch
            mismatches = df1['date'] != df2['date']
            if mismatches.any():
                first_mismatch_idx = mismatches.idxmax()
                logger.warning(f"    First mismatch at index {first_mismatch_idx}: "
                             f"{df1.loc[first_mismatch_idx, 'date']} vs "
                             f"{df2.loc[first_mismatch_idx, 'date']}")
        else:
            logger.info("✅ Date indices perfectly aligned")

        # Correlation check (sanity)
        corr = df1['return'].corr(df2['return'])
        logger.info(f"✅ Return correlation: {corr:.4f}")

    def _save_monthly_returns(self, df: pd.DataFrame, label: str):
        """
        Save monthly returns to CSV.

        Args:
            df: Monthly returns DataFrame
            label: Strategy label (used for filename)
        """
        output_dir = Path('results/ensemble_baselines')
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / f'{label}_monthly_returns.csv'

        # Save with date formatted as string
        df_to_save = df.copy()
        df_to_save['date'] = df_to_save['date'].dt.strftime('%Y-%m-%d')

        df_to_save.to_csv(csv_path, index=False, float_format='%.8f')

        logger.info(f"  Saved: {csv_path}")
        logger.info(f"    Rows: {len(df_to_save)}")


def main():
    """Generate monthly returns for momentum and momentum+quality."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate monthly returns for momentum and momentum+quality ensembles',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--start', default='2015-04-01',
                       help='Start date (default: 2015-04-01)')
    parser.add_argument('--end', default='2024-12-31',
                       help='End date (default: 2024-12-31)')
    parser.add_argument('--capital', type=float, default=100000,
                       help='Initial capital (default: $100,000)')

    args = parser.parse_args()

    # Generate monthly returns
    generator = MonthlyReturnsGenerator(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital
    )

    results = generator.generate_monthly_returns()

    logger.info("")
    logger.info("Files generated:")
    logger.info("  - results/ensemble_baselines/momentum_v2_monthly_returns.csv")
    logger.info("  - results/ensemble_baselines/momentum_quality_v1_monthly_returns.csv")
    logger.info("")
    logger.info("These files are now ready for Optuna-based weight optimization.")
    logger.info("")


if __name__ == '__main__':
    main()
