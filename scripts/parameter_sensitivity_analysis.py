"""
Parameter Sensitivity Analysis for SignalTide v3.

Tests robustness of optimized parameters by perturbing them ±10%, ±20%
and measuring performance degradation.

A robust parameter should show graceful degradation, not cliff edges.
Cliff edges (sudden >30% drops) indicate overfitting to specific values.

Usage:
    python3 scripts/parameter_sensitivity_analysis.py --param formation_period
    python3 scripts/parameter_sensitivity_analysis.py --param momentum_weight
    python3 scripts/parameter_sensitivity_analysis.py --all
"""

import sys
sys.path.insert(0, '.')

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Callable, Any, Tuple
from datetime import datetime
import logging

from data.data_manager import DataManager
from core.universe_manager import UniverseManager
from core.backtest_engine import BacktestConfig, run_backtest
from signals.ml.ensemble_configs import get_momentum_quality_v1_ensemble
from core.signal_adapters import make_multisignal_ensemble_fn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Production parameter values (baseline)
PRODUCTION_PARAMS = {
    # Momentum parameters (from Phase 2 Optuna)
    'formation_period': 308,
    'skip_period': 0,
    'momentum_winsorize_low': 0.4,
    'momentum_winsorize_high': 99.6,

    # Quality parameters (academic defaults)
    'w_profitability': 0.4,
    'w_growth': 0.3,
    'w_safety': 0.3,
    'quality_winsorize_low': 5,
    'quality_winsorize_high': 95,

    # Ensemble weights (Phase 3.1 calibrated)
    'momentum_weight': 0.25,
    'quality_weight': 0.75,
}

# Perturbation levels to test
PERTURBATIONS = [-0.20, -0.10, 0.0, 0.10, 0.20]


def run_sensitivity_backtest(
    dm: DataManager,
    momentum_weight: float,
    quality_weight: float,
    start_date: str = '2015-04-01',
    end_date: str = '2024-12-31'
) -> Dict:
    """
    Run a backtest with specified ensemble weights.

    Returns dict with sharpe, total_return, max_drawdown.
    """
    um = UniverseManager(dm)

    # Get ensemble and modify weights if needed
    ensemble = get_momentum_quality_v1_ensemble(dm)

    # Note: Currently ensemble weights are set at config creation
    # For weight sensitivity, we'll modify members directly
    for member in ensemble.members:
        if member.signal_name == 'InstitutionalMomentum':
            member.weight = momentum_weight
        elif member.signal_name == 'CrossSectionalQuality':
            member.weight = quality_weight

    # Create signal function
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
        data_manager=dm
    )

    result = run_backtest(universe_fn, signal_fn, config)

    return {
        'sharpe': result.sharpe,
        'total_return': result.total_return,
        'max_drawdown': result.max_drawdown,
        'num_rebalances': result.num_rebalances
    }


def run_weight_sensitivity(
    dm: DataManager,
    base_momentum_weight: float = 0.25,
    base_quality_weight: float = 0.75,
    perturbations: List[float] = None,
    start_date: str = '2015-04-01',
    end_date: str = '2024-12-31'
) -> pd.DataFrame:
    """
    Run sensitivity analysis on ensemble weights.

    Note: Weights must sum to 1.0, so perturbing momentum also changes quality.
    """
    if perturbations is None:
        perturbations = PERTURBATIONS

    results = []

    for pct in perturbations:
        # Perturb momentum weight, adjust quality to keep sum = 1.0
        test_mom_weight = base_momentum_weight * (1 + pct)
        test_quality_weight = 1.0 - test_mom_weight

        # Skip if weights go out of bounds
        if test_mom_weight < 0.05 or test_mom_weight > 0.95:
            logger.warning(f"Skipping pct={pct}: momentum weight {test_mom_weight} out of bounds")
            continue

        logger.info(f"Testing M={test_mom_weight:.3f}, Q={test_quality_weight:.3f} ({pct:+.0%} from base)")

        try:
            metrics = run_sensitivity_backtest(
                dm, test_mom_weight, test_quality_weight, start_date, end_date
            )

            results.append({
                'parameter': 'momentum_weight',
                'base_value': base_momentum_weight,
                'perturbation_pct': pct,
                'test_value': test_mom_weight,
                'quality_weight': test_quality_weight,
                'sharpe': metrics['sharpe'],
                'total_return': metrics['total_return'],
                'max_drawdown': metrics['max_drawdown']
            })
        except Exception as e:
            logger.error(f"Failed at pct={pct}: {e}")
            continue

    return pd.DataFrame(results)


def assess_robustness(sensitivity_df: pd.DataFrame, metric: str = 'sharpe') -> Dict:
    """
    Assess parameter robustness from sensitivity results.

    Criteria:
    - Graceful: Max degradation < 20% at ±20% perturbation
    - Stable: Monotonic or near-monotonic response
    - No cliffs: No sudden >30% drops between adjacent perturbations
    """
    if len(sensitivity_df) == 0:
        return {'is_robust': False, 'error': 'No data'}

    base_row = sensitivity_df[sensitivity_df['perturbation_pct'] == 0]
    if len(base_row) == 0:
        return {'is_robust': False, 'error': 'No baseline (0% perturbation)'}

    base_value = base_row[metric].iloc[0]

    # Compute degradation at each perturbation
    sensitivity_df = sensitivity_df.copy()
    sensitivity_df['degradation_pct'] = (base_value - sensitivity_df[metric]) / abs(base_value) if base_value != 0 else 0

    max_degradation = sensitivity_df['degradation_pct'].max()

    # Check for cliff edges (sudden drops between adjacent perturbations)
    sorted_df = sensitivity_df.sort_values('perturbation_pct')
    diffs = sorted_df['degradation_pct'].diff().abs()
    has_cliff = (diffs > 0.30).any()

    # Check if response is monotonic (parameter has clear directional effect)
    is_monotonic = (
        sorted_df[metric].is_monotonic_increasing or
        sorted_df[metric].is_monotonic_decreasing
    )

    return {
        'parameter': sensitivity_df['parameter'].iloc[0],
        'base_value': sensitivity_df['base_value'].iloc[0],
        'base_sharpe': base_value,
        'max_degradation_pct': max_degradation,
        'min_sharpe': sensitivity_df[metric].min(),
        'max_sharpe': sensitivity_df[metric].max(),
        'has_cliff_edge': has_cliff,
        'is_monotonic': is_monotonic,
        'is_robust': max_degradation < 0.20 and not has_cliff,
        'recommendation': (
            'KEEP - Parameter is robust' if max_degradation < 0.20 and not has_cliff else
            'INVESTIGATE - Cliff edge detected' if has_cliff else
            'INVESTIGATE - High degradation'
        ),
        'sensitivity_data': sensitivity_df.to_dict('records')
    }


def generate_sensitivity_report(assessment: Dict) -> str:
    """Generate human-readable sensitivity report."""
    param = assessment.get('parameter', 'Unknown')
    base = assessment.get('base_value', 'N/A')

    report = f"""
================================================================================
PARAMETER SENSITIVITY REPORT: {param}
================================================================================

BASE VALUE: {base}
BASE SHARPE: {assessment.get('base_sharpe', 'N/A'):.4f}

PERTURBATION RESULTS:
"""

    if 'sensitivity_data' in assessment:
        for row in assessment['sensitivity_data']:
            pct = row['perturbation_pct']
            val = row['test_value']
            sharpe = row['sharpe']
            deg = row.get('degradation_pct', 0)
            report += f"  {pct:+6.0%}: value={val:.3f}, sharpe={sharpe:.4f}, degradation={deg:+.1%}\n"

    report += f"""
ROBUSTNESS METRICS:
  Max Degradation: {assessment.get('max_degradation_pct', 0):.1%}
  Has Cliff Edge:  {assessment.get('has_cliff_edge', 'N/A')}
  Is Monotonic:    {assessment.get('is_monotonic', 'N/A')}
  IS ROBUST:       {assessment.get('is_robust', 'N/A')}

RECOMMENDATION: {assessment.get('recommendation', 'N/A')}
================================================================================
"""
    return report


def run_full_sensitivity_analysis(
    start_date: str = '2018-01-01',
    end_date: str = '2024-12-31'
) -> Dict:
    """
    Run full sensitivity analysis on key parameters.

    Uses shorter period (2018-2024) for faster iteration.
    """
    dm = DataManager()
    all_results = {}

    logger.info("="*60)
    logger.info("PARAMETER SENSITIVITY ANALYSIS")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info("="*60)

    # 1. Ensemble weight sensitivity
    logger.info("\n--- Ensemble Weight Sensitivity ---")
    weight_df = run_weight_sensitivity(
        dm,
        base_momentum_weight=0.25,
        base_quality_weight=0.75,
        start_date=start_date,
        end_date=end_date
    )

    if len(weight_df) > 0:
        assessment = assess_robustness(weight_df)
        all_results['momentum_weight'] = assessment
        print(generate_sensitivity_report(assessment))

    # Save results
    output_dir = Path('results/optimization')
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(weight_df) > 0:
        weight_df.to_csv(output_dir / 'sensitivity_momentum_weight.csv', index=False)

    # Summary
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'period': f"{start_date} to {end_date}",
        'parameters_tested': list(all_results.keys()),
        'all_robust': all(r.get('is_robust', False) for r in all_results.values()),
        'results': {k: {
            'is_robust': v.get('is_robust'),
            'max_degradation': v.get('max_degradation_pct'),
            'recommendation': v.get('recommendation')
        } for k, v in all_results.items()}
    }

    pd.DataFrame([summary]).to_csv(output_dir / 'sensitivity_summary.csv', index=False)
    logger.info(f"\nResults saved to {output_dir}")

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Sensitivity Analysis')
    parser.add_argument('--param', type=str, default='momentum_weight',
                        help='Parameter to analyze')
    parser.add_argument('--all', action='store_true',
                        help='Run full analysis on all parameters')
    parser.add_argument('--start', type=str, default='2018-01-01',
                        help='Backtest start date')
    parser.add_argument('--end', type=str, default='2024-12-31',
                        help='Backtest end date')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with 3-year window')

    args = parser.parse_args()

    if args.quick:
        args.start = '2022-01-01'
        args.end = '2024-12-31'

    if args.all:
        results = run_full_sensitivity_analysis(args.start, args.end)
    else:
        # Single parameter analysis
        dm = DataManager()

        if args.param == 'momentum_weight':
            logger.info(f"Analyzing momentum_weight sensitivity ({args.start} to {args.end})")
            df = run_weight_sensitivity(
                dm,
                start_date=args.start,
                end_date=args.end
            )
            if len(df) > 0:
                assessment = assess_robustness(df)
                print(generate_sensitivity_report(assessment))

                output_dir = Path('results/optimization')
                output_dir.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_dir / f'sensitivity_{args.param}.csv', index=False)
        else:
            logger.warning(f"Parameter '{args.param}' not yet implemented")
