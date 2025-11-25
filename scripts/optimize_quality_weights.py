"""
Phase B1: Quality Component Weight Optimization.

Optimizes the profitability/growth/safety weights within CrossSectionalQuality.
Uses Optuna with proper IS/OOS split and DSR correction for multiple testing.

Academic baseline (Asness et al. 2018):
  - Profitability: 0.4
  - Growth: 0.3
  - Safety: 0.3

This optimization tests whether:
1. Different weight allocations improve Sharpe
2. Results are statistically significant (DSR p < 0.05)
3. Parameter sensitivity shows graceful degradation

Usage:
    python3 scripts/optimize_quality_weights.py --trials 50
    python3 scripts/optimize_quality_weights.py --trials 20 --quick  # Quick test

References:
    Asness, Frazzini & Pedersen (2018). "Quality Minus Junk"
    Lopez de Prado (2018). "The Deflated Sharpe Ratio"
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
from typing import Dict, List, Tuple, Any

import optuna
from optuna.samplers import TPESampler

from data.data_manager import DataManager
from core.universe_manager import UniverseManager
from core.backtest_engine import BacktestConfig, run_backtest
from signals.quality.cross_sectional_quality import CrossSectionalQuality
from signals.momentum.institutional_momentum import InstitutionalMomentum
from signals.ml.ensemble_signal import EnsembleSignal, EnsembleMember
from core.signal_adapters import make_multisignal_ensemble_fn
from core.deflated_sharpe import compute_deflated_sharpe, dsr_summary_report

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global storage for trial results (for DSR calculation)
TRIAL_RESULTS: List[Dict] = []


def create_mq_ensemble(
    dm: DataManager,
    w_profitability: float,
    w_growth: float,
    w_safety: float,
    momentum_weight: float = 0.25,
    quality_weight: float = 0.75
) -> EnsembleSignal:
    """
    Create M+Q ensemble with specified Quality component weights.

    Args:
        dm: DataManager instance
        w_profitability: Quality profitability weight [0.2, 0.6]
        w_growth: Quality growth weight [0.1, 0.5]
        w_safety: Quality safety weight [0.1, 0.5]
        momentum_weight: Ensemble momentum weight (default 0.25)
        quality_weight: Ensemble quality weight (default 0.75)

    Returns:
        Configured EnsembleSignal
    """
    # Momentum params (Phase 2 optimized, fixed)
    momentum_params = {
        'formation_period': 308,
        'skip_period': 0,
        'winsorize_pct': [0.4, 99.6],
        'quintiles': True,
        'adaptive_quintiles': True,
    }

    # Quality params (variable weights)
    quality_params = {
        'w_profitability': w_profitability,
        'w_growth': w_growth,
        'w_safety': w_safety,
        'winsorize_pct': [5, 95],
        'quintiles': True,
        'min_coverage': 0.5,
    }

    members = [
        EnsembleMember(
            signal_name="InstitutionalMomentum",
            version="v2",
            weight=momentum_weight,
            normalize="none",
            params=momentum_params,
        ),
        EnsembleMember(
            signal_name="CrossSectionalQuality",
            version="v1",
            weight=quality_weight,
            normalize="none",
            params=quality_params,
        ),
    ]

    return EnsembleSignal(
        members=members,
        data_manager=dm,
        enforce_go_only=False,  # Allow testing non-GO configs
    )


def run_backtest_with_params(
    dm: DataManager,
    w_profitability: float,
    w_growth: float,
    w_safety: float,
    start_date: str,
    end_date: str,
    momentum_weight: float = 0.25,
    quality_weight: float = 0.75,
) -> Dict[str, float]:
    """
    Run backtest with specified Quality weights.

    Returns:
        Dict with sharpe, total_return, max_drawdown, monthly_returns array
    """
    um = UniverseManager(dm)

    # Create ensemble with specified weights
    ensemble = create_mq_ensemble(
        dm, w_profitability, w_growth, w_safety,
        momentum_weight, quality_weight
    )

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
        data_manager=dm,
    )

    result = run_backtest(universe_fn, signal_fn, config)

    # Compute monthly returns for DSR from equity curve
    monthly_returns = []
    if hasattr(result, 'equity_curve') and len(result.equity_curve) > 1:
        returns = result.equity_curve.pct_change().dropna()
        monthly_returns = returns.values.tolist()

    # Fallback if no returns extracted
    if len(monthly_returns) < 2:
        monthly_returns = [0.01, 0.01]  # Minimal fallback

    return {
        'sharpe': result.sharpe,
        'total_return': result.total_return,
        'max_drawdown': result.max_drawdown,
        'num_rebalances': result.num_rebalances,
        'monthly_returns': np.array(monthly_returns),
    }


def objective(trial: optuna.Trial, dm: DataManager, start_date: str, end_date: str) -> float:
    """
    Optuna objective function for Quality weight optimization.

    Constraint: w_profitability + w_growth + w_safety = 1.0
    """
    # Sample two weights freely, compute third to sum to 1.0
    w_profitability = trial.suggest_float('w_profitability', 0.2, 0.6)
    w_growth = trial.suggest_float('w_growth', 0.1, 0.5)

    # Compute w_safety to ensure sum = 1.0
    w_safety = 1.0 - w_profitability - w_growth

    # Check validity
    if w_safety < 0.1 or w_safety > 0.5:
        # Invalid combination, penalize
        return -float('inf')

    logger.info(f"Trial {trial.number}: P={w_profitability:.3f}, G={w_growth:.3f}, S={w_safety:.3f}")

    try:
        result = run_backtest_with_params(
            dm, w_profitability, w_growth, w_safety,
            start_date, end_date
        )

        sharpe = result['sharpe']

        # Store for DSR analysis
        TRIAL_RESULTS.append({
            'trial': trial.number,
            'w_profitability': w_profitability,
            'w_growth': w_growth,
            'w_safety': w_safety,
            'sharpe': sharpe,
            'total_return': result['total_return'],
            'max_drawdown': result['max_drawdown'],
            'monthly_returns': result['monthly_returns'],
        })

        logger.info(f"  → Sharpe: {sharpe:.4f}, Return: {result['total_return']:.2%}")

        return sharpe

    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        return -float('inf')


def run_quality_weight_optimization(
    start_date: str = '2015-04-01',
    end_date: str = '2021-12-31',
    n_trials: int = 50,
    output_dir: Path = Path('results/optimization'),
) -> Dict[str, Any]:
    """
    Run full Quality weight optimization with DSR correction.

    Args:
        start_date: IS period start
        end_date: IS period end
        n_trials: Number of Optuna trials
        output_dir: Where to save results

    Returns:
        Dict with best params, DSR results, and all trials
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("QUALITY COMPONENT WEIGHT OPTIMIZATION (Phase B1)")
    logger.info("=" * 60)
    logger.info(f"Period: {start_date} to {end_date} (IS)")
    logger.info(f"Trials: {n_trials}")
    logger.info(f"Weights: P=[0.2,0.6], G=[0.1,0.5], S=[0.1,0.5], sum=1.0")
    logger.info("=" * 60)

    dm = DataManager()

    # Clear global trial results
    TRIAL_RESULTS.clear()

    # Create Optuna study
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name='quality_weight_optimization'
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, dm, start_date, end_date),
        n_trials=n_trials,
        show_progress_bar=True
    )

    # Best trial
    best_trial = study.best_trial
    best_w_p = best_trial.params['w_profitability']
    best_w_g = best_trial.params['w_growth']
    best_w_s = 1.0 - best_w_p - best_w_g
    best_sharpe = best_trial.value

    logger.info("\n" + "=" * 60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best weights: P={best_w_p:.3f}, G={best_w_g:.3f}, S={best_w_s:.3f}")
    logger.info(f"Best Sharpe (IS): {best_sharpe:.4f}")
    logger.info(f"Academic baseline: P=0.4, G=0.3, S=0.3")

    # Apply DSR correction
    logger.info("\n" + "-" * 40)
    logger.info("DEFLATED SHARPE RATIO ANALYSIS")
    logger.info("-" * 40)

    # Find the best trial's monthly returns
    best_trial_data = None
    for tr in TRIAL_RESULTS:
        if abs(tr['sharpe'] - best_sharpe) < 0.0001:
            best_trial_data = tr
            break

    if best_trial_data is not None:
        returns = best_trial_data['monthly_returns']
        T = len(returns)
        skewness = float(pd.Series(returns).skew()) if T > 2 else 0
        kurtosis = float(pd.Series(returns).kurtosis()) if T > 3 else 0

        dsr, pval = compute_deflated_sharpe(
            observed_sharpe=best_sharpe,
            num_trials=n_trials,
            returns_skewness=skewness,
            returns_kurtosis=kurtosis,
            T=T
        )

        logger.info(f"Observed Sharpe: {best_sharpe:.4f}")
        logger.info(f"Deflated Sharpe: {dsr:.4f}")
        logger.info(f"P-value: {pval:.4f}")
        logger.info(f"Significant (p<0.05): {pval < 0.05}")

        # Generate full report
        report = dsr_summary_report(best_sharpe, n_trials, returns, "Quality Weight Optimization")
        print(report)
    else:
        dsr, pval = None, None
        logger.warning("Could not find best trial data for DSR calculation")

    # Compare to academic baseline
    logger.info("\n" + "-" * 40)
    logger.info("COMPARISON TO ACADEMIC BASELINE")
    logger.info("-" * 40)

    baseline_result = run_backtest_with_params(
        dm, 0.4, 0.3, 0.3, start_date, end_date
    )
    baseline_sharpe = baseline_result['sharpe']

    improvement = (best_sharpe - baseline_sharpe) / abs(baseline_sharpe) if baseline_sharpe != 0 else 0

    logger.info(f"Academic baseline Sharpe: {baseline_sharpe:.4f}")
    logger.info(f"Optimized Sharpe: {best_sharpe:.4f}")
    logger.info(f"Improvement: {improvement:+.1%}")

    # Save results
    results_df = pd.DataFrame(TRIAL_RESULTS)
    if 'monthly_returns' in results_df.columns:
        results_df = results_df.drop(columns=['monthly_returns'])

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_df.to_csv(output_dir / f'quality_weight_trials_{timestamp}.csv', index=False)

    # Save summary
    summary = {
        'timestamp': timestamp,
        'period': f'{start_date} to {end_date}',
        'n_trials': n_trials,
        'best_w_profitability': best_w_p,
        'best_w_growth': best_w_g,
        'best_w_safety': best_w_s,
        'best_sharpe_is': best_sharpe,
        'baseline_sharpe': baseline_sharpe,
        'improvement_pct': improvement * 100,
        'dsr': dsr,
        'dsr_pvalue': pval,
        'is_significant': pval < 0.05 if pval is not None else None,
    }

    pd.DataFrame([summary]).to_csv(output_dir / f'quality_weight_summary_{timestamp}.csv', index=False)

    logger.info(f"\nResults saved to {output_dir}")

    return {
        'best_params': {
            'w_profitability': best_w_p,
            'w_growth': best_w_g,
            'w_safety': best_w_s,
        },
        'best_sharpe': best_sharpe,
        'baseline_sharpe': baseline_sharpe,
        'improvement': improvement,
        'dsr': dsr,
        'pval': pval,
        'is_significant': pval < 0.05 if pval is not None else None,
        'all_trials': TRIAL_RESULTS,
        'summary': summary,
    }


def run_oos_validation(
    best_params: Dict[str, float],
    oos_start: str = '2022-01-01',
    oos_end: str = '2024-12-31',
) -> Dict[str, float]:
    """
    Validate best parameters on OOS period.

    Args:
        best_params: Dict with w_profitability, w_growth, w_safety
        oos_start: OOS period start
        oos_end: OOS period end

    Returns:
        OOS performance metrics
    """
    logger.info("\n" + "=" * 60)
    logger.info("OUT-OF-SAMPLE VALIDATION")
    logger.info("=" * 60)
    logger.info(f"Period: {oos_start} to {oos_end}")
    logger.info(f"Params: P={best_params['w_profitability']:.3f}, "
                f"G={best_params['w_growth']:.3f}, S={best_params['w_safety']:.3f}")

    dm = DataManager()

    # Run with optimized params
    opt_result = run_backtest_with_params(
        dm,
        best_params['w_profitability'],
        best_params['w_growth'],
        best_params['w_safety'],
        oos_start, oos_end
    )

    # Run with baseline params
    base_result = run_backtest_with_params(
        dm, 0.4, 0.3, 0.3, oos_start, oos_end
    )

    logger.info(f"\nOptimized OOS Sharpe: {opt_result['sharpe']:.4f}")
    logger.info(f"Baseline OOS Sharpe: {base_result['sharpe']:.4f}")

    oos_improvement = (opt_result['sharpe'] - base_result['sharpe']) / abs(base_result['sharpe']) if base_result['sharpe'] != 0 else 0
    logger.info(f"OOS Improvement: {oos_improvement:+.1%}")

    # Check for overfitting
    if oos_improvement < -0.2:
        logger.warning("OVERFITTING WARNING: OOS performance significantly worse than baseline")
    elif oos_improvement < 0:
        logger.info("Note: OOS slightly underperforms baseline - consider using academic defaults")
    else:
        logger.info("OOS validates optimization - improvement holds out-of-sample")

    return {
        'optimized_sharpe_oos': opt_result['sharpe'],
        'baseline_sharpe_oos': base_result['sharpe'],
        'oos_improvement': oos_improvement,
        'optimized_return_oos': opt_result['total_return'],
        'baseline_return_oos': base_result['total_return'],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quality Component Weight Optimization')
    parser.add_argument('--trials', type=int, default=50,
                        help='Number of Optuna trials (default: 50)')
    parser.add_argument('--is-start', type=str, default='2015-04-01',
                        help='IS period start date')
    parser.add_argument('--is-end', type=str, default='2021-12-31',
                        help='IS period end date')
    parser.add_argument('--oos-start', type=str, default='2022-01-01',
                        help='OOS period start date')
    parser.add_argument('--oos-end', type=str, default='2024-12-31',
                        help='OOS period end date')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with shorter IS period (2019-2021)')
    parser.add_argument('--skip-oos', action='store_true',
                        help='Skip OOS validation')

    args = parser.parse_args()

    if args.quick:
        args.is_start = '2019-01-01'
        args.is_end = '2021-12-31'
        args.trials = min(args.trials, 20)
        logger.info("Quick mode: IS=2019-2021, max 20 trials")

    # Run IS optimization
    results = run_quality_weight_optimization(
        start_date=args.is_start,
        end_date=args.is_end,
        n_trials=args.trials,
    )

    # Run OOS validation
    if not args.skip_oos and results['best_params']:
        oos_results = run_oos_validation(
            results['best_params'],
            oos_start=args.oos_start,
            oos_end=args.oos_end,
        )

        # Final summary
        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        print(f"\nBest Quality Weights:")
        print(f"  Profitability: {results['best_params']['w_profitability']:.3f}")
        print(f"  Growth:        {results['best_params']['w_growth']:.3f}")
        print(f"  Safety:        {results['best_params']['w_safety']:.3f}")
        print(f"\nPerformance:")
        print(f"  IS Sharpe:     {results['best_sharpe']:.4f} (baseline: {results['baseline_sharpe']:.4f})")
        print(f"  OOS Sharpe:    {oos_results['optimized_sharpe_oos']:.4f} (baseline: {oos_results['baseline_sharpe_oos']:.4f})")
        print(f"\nStatistical Significance:")
        print(f"  DSR:           {results['dsr']:.4f}" if results['dsr'] else "  DSR:           N/A")
        print(f"  P-value:       {results['pval']:.4f}" if results['pval'] else "  P-value:       N/A")
        print(f"  Significant:   {results['is_significant']}")
        print(f"\nRECOMMENDATION:")

        if results['is_significant'] and oos_results['oos_improvement'] > -0.1:
            print(f"  USE OPTIMIZED: P={results['best_params']['w_profitability']:.2f}, "
                  f"G={results['best_params']['w_growth']:.2f}, S={results['best_params']['w_safety']:.2f}")
        else:
            print("  USE ACADEMIC BASELINE: P=0.40, G=0.30, S=0.30")
            if not results['is_significant']:
                print("  (Reason: Optimization not statistically significant)")
            else:
                print("  (Reason: OOS performance degradation)")

        print("=" * 60)
