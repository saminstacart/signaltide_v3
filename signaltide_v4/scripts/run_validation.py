#!/usr/bin/env python3
"""
Validation suite runner for SignalTide v4.

Runs comprehensive validation including:
- Deflated Sharpe Ratio
- Walk-Forward Validation
- Fama-French 5-Factor Attribution
- Monte Carlo simulation

Usage:
    python -m signaltide_v4.scripts.run_validation --help
    python -m signaltide_v4.scripts.run_validation --returns results/backtest_returns.csv
    python -m signaltide_v4.scripts.run_validation --backtest-file results/backtest_v4_results.json
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from signaltide_v4.validation.deflated_sharpe import DeflatedSharpeCalculator
from signaltide_v4.validation.walk_forward import WalkForwardValidator
from signaltide_v4.validation.factor_attribution import FactorAttributor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='SignalTide v4 Validation Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Validate from return series CSV
    python -m signaltide_v4.scripts.run_validation --returns returns.csv

    # Validate from backtest results JSON
    python -m signaltide_v4.scripts.run_validation --backtest-file results.json

    # Run with specific trials count for DSR
    python -m signaltide_v4.scripts.run_validation --returns returns.csv --n-trials 500
        """
    )

    parser.add_argument(
        '--returns',
        type=str,
        help='Path to CSV file with daily returns (date,return columns)'
    )
    parser.add_argument(
        '--backtest-file',
        type=str,
        help='Path to backtest results JSON file'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=100,
        help='Number of trials for DSR calculation (default: 100)'
    )
    parser.add_argument(
        '--train-months',
        type=int,
        default=60,
        help='Training period months for walk-forward (default: 60)'
    )
    parser.add_argument(
        '--test-months',
        type=int,
        default=12,
        help='Test period months for walk-forward (default: 12)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/validation_v4_results.json',
        help='Output file for validation results'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def load_returns(args) -> pd.Series:
    """Load returns from file."""
    if args.returns:
        df = pd.read_csv(args.returns, index_col=0, parse_dates=True)
        if 'return' in df.columns:
            returns = df['return']
        else:
            returns = df.iloc[:, 0]
        returns.name = 'returns'
        return returns

    elif args.backtest_file:
        with open(args.backtest_file, 'r') as f:
            data = json.load(f)

        if 'returns' in data:
            returns_data = data['returns']
            if isinstance(returns_data, dict):
                returns = pd.Series(returns_data)
                returns.index = pd.to_datetime(returns.index)
            else:
                returns = pd.Series(returns_data)
            return returns

    raise ValueError("Must provide --returns or --backtest-file")


def run_deflated_sharpe(returns: pd.Series, n_trials: int) -> dict:
    """Run Deflated Sharpe Ratio validation."""
    logger.info("Running Deflated Sharpe Ratio validation...")

    calculator = DeflatedSharpeCalculator(n_trials=n_trials)
    result = calculator.calculate(returns)

    output = {
        'observed_sharpe': result.observed_sharpe,
        'deflated_sharpe': result.deflated_sharpe,
        'expected_max_sharpe': result.expected_max_sharpe,
        'p_value': result.p_value,
        'confidence_level': result.confidence_level,
        'is_significant': result.is_significant,
        'n_trials': result.n_trials,
        'n_observations': result.n_observations,
    }

    logger.info(
        f"DSR Result: Observed SR={result.observed_sharpe:.3f}, "
        f"Deflated SR={result.deflated_sharpe:.3f}, "
        f"p-value={result.p_value:.4f}, "
        f"Significant={result.is_significant}"
    )

    return output


def run_walk_forward(
    returns: pd.Series,
    train_months: int,
    test_months: int,
) -> dict:
    """Run walk-forward validation."""
    logger.info("Running walk-forward validation...")

    validator = WalkForwardValidator(
        train_months=train_months,
        test_months=test_months,
    )
    result = validator.validate_returns(returns)

    output = {
        'n_folds': result.n_folds,
        'pct_positive': result.pct_positive,
        'mean_train_sharpe': result.mean_train_sharpe,
        'mean_test_sharpe': result.mean_test_sharpe,
        'std_test_sharpe': result.std_test_sharpe,
        'train_test_correlation': result.train_test_correlation,
        'is_valid': result.is_valid,
        'fold_details': [
            {
                'fold': i + 1,
                'train_sharpe': fd.train_sharpe,
                'test_sharpe': fd.test_sharpe,
                'train_period': f"{fd.train_start} to {fd.train_end}",
                'test_period': f"{fd.test_start} to {fd.test_end}",
            }
            for i, fd in enumerate(result.fold_details)
        ],
    }

    logger.info(
        f"Walk-Forward Result: {result.n_folds} folds, "
        f"{result.pct_positive:.0%} positive, "
        f"Mean test SR={result.mean_test_sharpe:.3f}, "
        f"Valid={result.is_valid}"
    )

    return output


def run_factor_attribution(returns: pd.Series) -> dict:
    """Run Fama-French factor attribution."""
    logger.info("Running Fama-French 5-factor attribution...")

    attributor = FactorAttributor()
    result = attributor.attribute(returns)

    interpretation = attributor.interpret(result)

    output = {
        'alpha': result.alpha,
        'alpha_annualized': result.alpha * 252,
        'alpha_t_stat': result.alpha_t_stat,
        'alpha_p_value': result.alpha_p_value,
        'alpha_significant': result.alpha_significant,
        'factor_betas': {
            'MKT-RF': result.mkt_rf_beta,
            'SMB': result.smb_beta,
            'HML': result.hml_beta,
            'RMW': result.rmw_beta,
            'CMA': result.cma_beta,
        },
        'r_squared': result.r_squared,
        'adj_r_squared': result.adj_r_squared,
        'interpretation': interpretation,
    }

    logger.info(
        f"Factor Attribution: Alpha={result.alpha*252:.2%} (annualized), "
        f"t={result.alpha_t_stat:.2f}, p={result.alpha_p_value:.4f}, "
        f"R²={result.r_squared:.3f}"
    )

    return output


def run_monte_carlo(returns: pd.Series, n_simulations: int = 1000) -> dict:
    """Run Monte Carlo simulation for p-value estimation."""
    logger.info(f"Running Monte Carlo simulation ({n_simulations} simulations)...")

    # Observed metrics
    observed_sharpe = (
        returns.mean() / returns.std() * np.sqrt(252)
        if returns.std() > 0 else 0
    )
    observed_total_return = (1 + returns).prod() - 1

    # Simulate random returns with same mean and std
    mean_ret = returns.mean()
    std_ret = returns.std()
    n = len(returns)

    simulated_sharpes = []
    simulated_returns = []

    np.random.seed(42)
    for _ in range(n_simulations):
        sim_returns = np.random.normal(mean_ret, std_ret, n)
        sim_sharpe = np.mean(sim_returns) / np.std(sim_returns) * np.sqrt(252)
        sim_total = np.prod(1 + sim_returns) - 1

        simulated_sharpes.append(sim_sharpe)
        simulated_returns.append(sim_total)

    # P-value: fraction of simulations with higher Sharpe
    p_value_sharpe = np.mean([s >= observed_sharpe for s in simulated_sharpes])
    p_value_return = np.mean([r >= observed_total_return for r in simulated_returns])

    output = {
        'observed_sharpe': observed_sharpe,
        'observed_total_return': observed_total_return,
        'n_simulations': n_simulations,
        'p_value_sharpe': p_value_sharpe,
        'p_value_total_return': p_value_return,
        'simulated_sharpe_mean': np.mean(simulated_sharpes),
        'simulated_sharpe_std': np.std(simulated_sharpes),
        'simulated_sharpe_percentiles': {
            '5th': np.percentile(simulated_sharpes, 5),
            '25th': np.percentile(simulated_sharpes, 25),
            '50th': np.percentile(simulated_sharpes, 50),
            '75th': np.percentile(simulated_sharpes, 75),
            '95th': np.percentile(simulated_sharpes, 95),
        },
    }

    logger.info(
        f"Monte Carlo: Observed SR={observed_sharpe:.3f}, "
        f"Simulated mean SR={np.mean(simulated_sharpes):.3f}, "
        f"p-value={p_value_sharpe:.4f}"
    )

    return output


def calculate_summary_grade(results: dict) -> dict:
    """Calculate overall validation grade."""
    scores = []
    details = []

    # DSR check
    if results.get('deflated_sharpe', {}).get('is_significant', False):
        scores.append(1.0)
        details.append("✓ DSR significant")
    else:
        scores.append(0.0)
        details.append("✗ DSR not significant")

    # Walk-forward check
    wf = results.get('walk_forward', {})
    if wf.get('is_valid', False):
        scores.append(1.0)
        details.append("✓ Walk-forward valid")
    elif wf.get('pct_positive', 0) >= 0.5:
        scores.append(0.5)
        details.append("~ Walk-forward marginal")
    else:
        scores.append(0.0)
        details.append("✗ Walk-forward failed")

    # Alpha check
    attr = results.get('factor_attribution', {})
    if attr.get('alpha_significant', False) and attr.get('alpha', 0) > 0:
        scores.append(1.0)
        details.append("✓ Positive significant alpha")
    elif attr.get('alpha', 0) > 0:
        scores.append(0.5)
        details.append("~ Positive but not significant alpha")
    else:
        scores.append(0.0)
        details.append("✗ No positive alpha")

    # Monte Carlo check
    mc = results.get('monte_carlo', {})
    if mc.get('p_value_sharpe', 1.0) < 0.05:
        scores.append(1.0)
        details.append("✓ Monte Carlo significant")
    elif mc.get('p_value_sharpe', 1.0) < 0.10:
        scores.append(0.5)
        details.append("~ Monte Carlo marginal")
    else:
        scores.append(0.0)
        details.append("✗ Monte Carlo not significant")

    avg_score = np.mean(scores) if scores else 0

    if avg_score >= 0.75:
        grade = 'A'
        recommendation = 'PRODUCTION READY'
    elif avg_score >= 0.5:
        grade = 'B'
        recommendation = 'PROCEED WITH CAUTION'
    elif avg_score >= 0.25:
        grade = 'C'
        recommendation = 'NEEDS IMPROVEMENT'
    else:
        grade = 'F'
        recommendation = 'DO NOT DEPLOY'

    return {
        'grade': grade,
        'score': avg_score,
        'recommendation': recommendation,
        'details': details,
    }


def main():
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("SignalTide v4 Validation Suite")
    logger.info("=" * 60)

    # Load returns
    try:
        returns = load_returns(args)
        logger.info(f"Loaded {len(returns)} daily returns")
    except Exception as e:
        logger.error(f"Failed to load returns: {e}")
        return 1

    # Run all validations
    results = {
        'metadata': {
            'run_timestamp': datetime.now().isoformat(),
            'n_observations': len(returns),
            'start_date': str(returns.index[0]) if hasattr(returns.index[0], 'strftime') else str(returns.index[0]),
            'end_date': str(returns.index[-1]) if hasattr(returns.index[-1], 'strftime') else str(returns.index[-1]),
        }
    }

    # Deflated Sharpe Ratio
    try:
        results['deflated_sharpe'] = run_deflated_sharpe(returns, args.n_trials)
    except Exception as e:
        logger.error(f"DSR validation failed: {e}")
        results['deflated_sharpe'] = {'error': str(e)}

    # Walk-Forward Validation
    try:
        results['walk_forward'] = run_walk_forward(
            returns, args.train_months, args.test_months
        )
    except Exception as e:
        logger.error(f"Walk-forward validation failed: {e}")
        results['walk_forward'] = {'error': str(e)}

    # Factor Attribution
    try:
        results['factor_attribution'] = run_factor_attribution(returns)
    except Exception as e:
        logger.error(f"Factor attribution failed: {e}")
        results['factor_attribution'] = {'error': str(e)}

    # Monte Carlo
    try:
        results['monte_carlo'] = run_monte_carlo(returns)
    except Exception as e:
        logger.error(f"Monte Carlo simulation failed: {e}")
        results['monte_carlo'] = {'error': str(e)}

    # Calculate summary grade
    results['summary'] = calculate_summary_grade(results)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)

    summary = results['summary']
    logger.info(f"Grade: {summary['grade']} (Score: {summary['score']:.2f})")
    logger.info(f"Recommendation: {summary['recommendation']}")
    logger.info("\nDetails:")
    for detail in summary['details']:
        logger.info(f"  {detail}")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {args.output}")
    logger.info("=" * 60)

    # Return code based on grade
    return 0 if summary['grade'] in ['A', 'B'] else 1


if __name__ == '__main__':
    sys.exit(main())
