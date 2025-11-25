"""
Phase B2: Quality Winsorization & Coverage Parameter Sweep.

Grid search over:
- winsorize_pct: [1,99], [3,97], [5,95], [7,93], [10,90]
- min_coverage: 0.3, 0.5, 0.7

Evaluates each configuration and identifies robust parameter regions.
Uses DSR for statistical significance testing.

Usage:
    python3 scripts/sweep_quality_winsorization.py
    python3 scripts/sweep_quality_winsorization.py --quick  # Shorter IS period
"""

import sys
sys.path.insert(0, '.')

import argparse
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from itertools import product

from data.data_manager import DataManager
from core.universe_manager import UniverseManager
from core.backtest_engine import BacktestConfig, run_backtest
from signals.quality.cross_sectional_quality import CrossSectionalQuality
from signals.momentum.institutional_momentum import InstitutionalMomentum
from signals.ml.ensemble_signal import EnsembleSignal, EnsembleMember
from core.signal_adapters import make_multisignal_ensemble_fn
from core.deflated_sharpe import compute_deflated_sharpe

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Grid search parameters
WINSORIZE_OPTIONS = [
    [1, 99],
    [3, 97],
    [5, 95],  # Current default
    [7, 93],
    [10, 90],
]

COVERAGE_OPTIONS = [0.3, 0.5, 0.7]


def create_mq_ensemble(
    dm: DataManager,
    winsorize_pct: List[float],
    min_coverage: float,
    w_profitability: float = 0.4,
    w_growth: float = 0.3,
    w_safety: float = 0.3,
) -> EnsembleSignal:
    """
    Create M+Q ensemble with specified Quality preprocessing params.
    """
    # Momentum params (Phase 2 optimized, fixed)
    momentum_params = {
        'formation_period': 308,
        'skip_period': 0,
        'winsorize_pct': [0.4, 99.6],
        'quintiles': True,
        'adaptive_quintiles': True,
    }

    # Quality params (variable preprocessing)
    quality_params = {
        'w_profitability': w_profitability,
        'w_growth': w_growth,
        'w_safety': w_safety,
        'winsorize_pct': winsorize_pct,
        'quintiles': True,
        'min_coverage': min_coverage,
    }

    members = [
        EnsembleMember(
            signal_name="InstitutionalMomentum",
            version="v2",
            weight=0.25,  # Fixed from Phase 3.1
            normalize="none",
            params=momentum_params,
        ),
        EnsembleMember(
            signal_name="CrossSectionalQuality",
            version="v1",
            weight=0.75,  # Fixed from Phase 3.1
            normalize="none",
            params=quality_params,
        ),
    ]

    return EnsembleSignal(
        members=members,
        data_manager=dm,
        enforce_go_only=False,
    )


def run_backtest_with_config(
    dm: DataManager,
    winsorize_pct: List[float],
    min_coverage: float,
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """
    Run backtest with specified Quality preprocessing configuration.
    """
    um = UniverseManager(dm)

    ensemble = create_mq_ensemble(dm, winsorize_pct, min_coverage)
    signal_fn = make_multisignal_ensemble_fn(ensemble, dm)

    def universe_fn(rebal_date: str) -> List[str]:
        universe = um.get_universe(
            universe_type='sp500_actual',
            as_of_date=rebal_date,
            min_price=5.0
        )
        if isinstance(universe, pd.Series):
            return universe.tolist()
        elif isinstance(universe, pd.DataFrame):
            return universe.index.tolist()
        return list(universe)

    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=100000.0,
        rebalance_schedule='M',
        long_only=True,
        equal_weight=True,
        track_daily_equity=False,
        data_manager=dm,
    )

    result = run_backtest(universe_fn, signal_fn, config)

    # Extract monthly returns for DSR
    monthly_returns = []
    if hasattr(result, 'equity_curve') and len(result.equity_curve) > 1:
        returns = result.equity_curve.pct_change().dropna()
        monthly_returns = returns.values.tolist()

    if len(monthly_returns) < 2:
        monthly_returns = [0.01, 0.01]

    return {
        'sharpe': result.sharpe,
        'total_return': result.total_return,
        'max_drawdown': result.max_drawdown,
        'num_rebalances': result.num_rebalances,
        'monthly_returns': np.array(monthly_returns),
    }


def run_grid_search(
    start_date: str = '2015-04-01',
    end_date: str = '2021-12-31',
    output_dir: Path = Path('results/optimization'),
) -> pd.DataFrame:
    """
    Run grid search over winsorization and coverage parameters.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("QUALITY PREPROCESSING PARAMETER SWEEP (Phase B2)")
    logger.info("=" * 60)
    logger.info(f"Period: {start_date} to {end_date} (IS)")
    logger.info(f"Winsorize options: {WINSORIZE_OPTIONS}")
    logger.info(f"Coverage options: {COVERAGE_OPTIONS}")
    logger.info(f"Total configurations: {len(WINSORIZE_OPTIONS) * len(COVERAGE_OPTIONS)}")
    logger.info("=" * 60)

    dm = DataManager()
    results = []

    # Generate all parameter combinations
    param_grid = list(product(WINSORIZE_OPTIONS, COVERAGE_OPTIONS))
    total_configs = len(param_grid)

    for idx, (winsorize, coverage) in enumerate(param_grid, 1):
        logger.info(f"\n[{idx}/{total_configs}] Testing: winsorize={winsorize}, coverage={coverage}")

        try:
            metrics = run_backtest_with_config(
                dm, winsorize, coverage, start_date, end_date
            )

            results.append({
                'winsorize_low': winsorize[0],
                'winsorize_high': winsorize[1],
                'min_coverage': coverage,
                'sharpe': metrics['sharpe'],
                'total_return': metrics['total_return'],
                'max_drawdown': metrics['max_drawdown'],
                'num_rebalances': metrics['num_rebalances'],
            })

            logger.info(f"  Sharpe: {metrics['sharpe']:.4f}, Return: {metrics['total_return']:.2%}")

        except Exception as e:
            logger.error(f"  Failed: {e}")
            results.append({
                'winsorize_low': winsorize[0],
                'winsorize_high': winsorize[1],
                'min_coverage': coverage,
                'sharpe': None,
                'total_return': None,
                'max_drawdown': None,
                'num_rebalances': None,
            })

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Find best configuration
    valid_results = results_df.dropna(subset=['sharpe'])
    if len(valid_results) > 0:
        best_idx = valid_results['sharpe'].idxmax()
        best_config = valid_results.loc[best_idx]

        logger.info("\n" + "=" * 60)
        logger.info("GRID SEARCH COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Best configuration:")
        logger.info(f"  Winsorize: [{best_config['winsorize_low']}, {best_config['winsorize_high']}]")
        logger.info(f"  Min Coverage: {best_config['min_coverage']}")
        logger.info(f"  Sharpe: {best_config['sharpe']:.4f}")
        logger.info(f"  Return: {best_config['total_return']:.2%}")

        # Compare to default [5, 95], 0.5
        default_row = results_df[
            (results_df['winsorize_low'] == 5) &
            (results_df['winsorize_high'] == 95) &
            (results_df['min_coverage'] == 0.5)
        ]
        if len(default_row) > 0:
            default_sharpe = default_row['sharpe'].iloc[0]
            improvement = (best_config['sharpe'] - default_sharpe) / abs(default_sharpe) * 100
            logger.info(f"\nDefault [5,95]/0.5 Sharpe: {default_sharpe:.4f}")
            logger.info(f"Improvement: {improvement:+.1f}%")

    # Apply DSR correction (treating each config as a "trial")
    n_trials = len(valid_results)
    if n_trials > 0 and best_config is not None:
        # Get the best config's monthly returns
        best_result = run_backtest_with_config(
            dm,
            [best_config['winsorize_low'], best_config['winsorize_high']],
            best_config['min_coverage'],
            start_date, end_date
        )
        returns = best_result['monthly_returns']
        T = len(returns)
        skewness = float(pd.Series(returns).skew()) if T > 2 else 0
        kurtosis = float(pd.Series(returns).kurtosis()) if T > 3 else 0

        dsr, pval = compute_deflated_sharpe(
            observed_sharpe=best_config['sharpe'],
            num_trials=n_trials,
            returns_skewness=skewness,
            returns_kurtosis=kurtosis,
            T=T
        )

        logger.info("\n" + "-" * 40)
        logger.info("DSR ANALYSIS")
        logger.info("-" * 40)
        logger.info(f"Deflated Sharpe: {dsr:.4f}")
        logger.info(f"P-value: {pval:.4f}")
        logger.info(f"Significant (p<0.05): {pval < 0.05}")

        results_df['dsr'] = dsr
        results_df['dsr_pvalue'] = pval

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_df.to_csv(output_dir / f'quality_preprocessing_sweep_{timestamp}.csv', index=False)
    logger.info(f"\nResults saved to {output_dir}")

    # Recommendation
    logger.info("\n" + "=" * 60)
    logger.info("RECOMMENDATION")
    logger.info("=" * 60)

    # Analyze sensitivity - check if best is significantly better than neighbors
    if len(valid_results) > 0:
        sharpe_range = valid_results['sharpe'].max() - valid_results['sharpe'].min()
        sharpe_std = valid_results['sharpe'].std()

        if sharpe_range < 0.1:
            logger.info("Parameters are ROBUST - minimal sensitivity to preprocessing choices")
            logger.info("Recommendation: KEEP DEFAULTS ([5,95], coverage=0.5)")
        elif sharpe_std < 0.05:
            logger.info("Parameters show LOW sensitivity - modest variation across configs")
            logger.info(f"Recommendation: Consider best config but defaults are acceptable")
        else:
            logger.info("Parameters show MODERATE sensitivity - tuning may help")
            logger.info(f"Recommendation: Use best config if DSR significant")

    return results_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quality Preprocessing Parameter Sweep')
    parser.add_argument('--is-start', type=str, default='2015-04-01',
                        help='IS period start date')
    parser.add_argument('--is-end', type=str, default='2021-12-31',
                        help='IS period end date')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with shorter IS period (2018-2021)')

    args = parser.parse_args()

    if args.quick:
        args.is_start = '2018-01-01'
        args.is_end = '2021-12-31'
        logger.info("Quick mode: IS=2018-2021")

    results = run_grid_search(
        start_date=args.is_start,
        end_date=args.is_end,
    )

    # Print summary table
    print("\n" + "=" * 80)
    print("PARAMETER SWEEP RESULTS")
    print("=" * 80)
    print(results.sort_values('sharpe', ascending=False).to_string(index=False))
