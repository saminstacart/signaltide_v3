#!/usr/bin/env python3
"""
Main backtest runner for SignalTide v4.

Usage:
    python -m signaltide_v4.scripts.run_backtest --help
    python -m signaltide_v4.scripts.run_backtest --start 2015-01-01 --end 2024-12-31
    python -m signaltide_v4.scripts.run_backtest --universe SP500 --validate
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from signaltide_v4.config.settings import get_settings
from signaltide_v4.backtest.engine import BacktestEngine, run_backtest
from signaltide_v4.backtest.transaction_costs import TransactionCostModel
from signaltide_v4.validation.deflated_sharpe import DeflatedSharpeCalculator
from signaltide_v4.validation.walk_forward import WalkForwardValidator
from signaltide_v4.validation.factor_attribution import FactorAttributor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/backtest_v4.log'),
    ]
)
logger = logging.getLogger(__name__)


# SURVIVORSHIP-BIAS-FREE UNIVERSE
# Top 50 stocks by market cap as of Q2 2015 (actual large caps at backtest start)
# This fixes the survivorship bias from using today's winners retroactively.
# Note: Starts from 2015-07-01 to ensure Q2 2015 fundamentals are available.
DEFAULT_UNIVERSE = [
    'AAPL', 'GOOGL', 'MSFT', 'BRK.B', 'XOM', 'WFC', 'JNJ', 'META', 'GE', 'JPM',
    'AMZN', 'PFE', 'T', 'WMT', 'DIS', 'PG', 'BAC', 'VZ', 'ORCL', 'KO',
    'C', 'GILD', 'V', 'MRK', 'CVX', 'IBM', 'CMCSA', 'HD', 'PEP', 'INTC',
    'AMGN', 'CSCO', 'PM', 'AGN', 'CVS', 'UNH', 'BMY', 'ABBV', 'MA', 'SLB',
    'CELG', 'MO', 'QCOM', 'BA', 'MDT', 'WBA', 'NKE', 'KHC', 'MMM', 'MCD',
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='SignalTide v4 Backtest Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run basic backtest
    python -m signaltide_v4.scripts.run_backtest

    # Run with custom dates
    python -m signaltide_v4.scripts.run_backtest --start 2018-01-01 --end 2023-12-31

    # Run with validation
    python -m signaltide_v4.scripts.run_backtest --validate --attribution

    # Run with higher transaction costs (stress test)
    python -m signaltide_v4.scripts.run_backtest --cost-bps 20
        """
    )

    parser.add_argument(
        '--start',
        type=str,
        default='2015-07-01',
        help='Start date (YYYY-MM-DD) - default 2015-07-01 to ensure Q2 2015 fundamentals available'
    )
    parser.add_argument(
        '--end',
        type=str,
        default='2024-12-31',
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=50_000.0,
        help='Initial capital (default: $50,000)'
    )
    parser.add_argument(
        '--cost-bps',
        type=float,
        default=5.0,
        help='Transaction cost in basis points (default: 5)'
    )
    parser.add_argument(
        '--universe',
        type=str,
        default='default',
        choices=['default', 'SP500', 'SP100', 'custom'],
        help='Stock universe to use'
    )
    parser.add_argument(
        '--tickers',
        type=str,
        nargs='+',
        help='Custom tickers (use with --universe custom)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run DSR and walk-forward validation'
    )
    parser.add_argument(
        '--attribution',
        action='store_true',
        help='Run FF5 factor attribution'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/backtest_v4_results.json',
        help='Output file for results'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def get_universe(args):
    """Get stock universe based on arguments."""
    if args.universe == 'custom' and args.tickers:
        return args.tickers
    elif args.universe == 'SP100':
        # Top 100 by market cap (subset)
        return DEFAULT_UNIVERSE[:100] if len(DEFAULT_UNIVERSE) >= 100 else DEFAULT_UNIVERSE
    else:
        # Default universe
        return DEFAULT_UNIVERSE


def run_validation(result, args):
    """Run validation tests on backtest results."""
    validation_results = {}

    # Deflated Sharpe Ratio
    logger.info("Running Deflated Sharpe Ratio validation...")
    dsr_calc = DeflatedSharpeCalculator()
    dsr_result = dsr_calc.calculate(result.returns)

    validation_results['dsr'] = {
        'observed_sharpe': dsr_result.observed_sharpe,
        'deflated_sharpe': dsr_result.deflated_sharpe,
        'p_value': dsr_result.p_value,
        'confidence': dsr_result.confidence_level,
        'is_significant': dsr_result.is_significant,
        'n_trials_corrected': dsr_result.n_trials,
    }

    logger.info(
        f"DSR: Observed={dsr_result.observed_sharpe:.3f}, "
        f"Deflated={dsr_result.deflated_sharpe:.3f}, "
        f"p={dsr_result.p_value:.3f}, significant={dsr_result.is_significant}"
    )

    # Walk-Forward Validation
    logger.info("Running walk-forward validation...")
    wf_validator = WalkForwardValidator()
    wf_result = wf_validator.validate_returns(result.returns)

    validation_results['walk_forward'] = {
        'n_folds': wf_result.n_folds,
        'pct_positive': wf_result.pct_positive,
        'mean_test_sharpe': wf_result.mean_test_sharpe,
        'std_test_sharpe': wf_result.std_test_sharpe,
        'train_test_correlation': wf_result.train_test_correlation,
        'is_valid': wf_result.is_valid,
    }

    logger.info(
        f"Walk-Forward: {wf_result.n_folds} folds, "
        f"{wf_result.pct_positive:.0%} positive, "
        f"mean SR={wf_result.mean_test_sharpe:.3f}, "
        f"valid={wf_result.is_valid}"
    )

    return validation_results


def run_attribution(result, args):
    """Run factor attribution analysis."""
    logger.info("Running Fama-French 5-factor attribution...")

    attributor = FactorAttributor()
    attr_result = attributor.attribute(result.returns)

    interpretation = attributor.interpret(attr_result)

    attribution = {
        'alpha': attr_result.alpha,
        'alpha_t_stat': attr_result.alpha_t_stat,
        'alpha_p_value': attr_result.alpha_p_value,
        'alpha_significant': attr_result.alpha_significant,
        'factor_betas': {
            'MKT-RF': attr_result.mkt_rf_beta,
            'SMB': attr_result.smb_beta,
            'HML': attr_result.hml_beta,
            'RMW': attr_result.rmw_beta,
            'CMA': attr_result.cma_beta,
        },
        'r_squared': attr_result.r_squared,
        'interpretation': interpretation,
    }

    logger.info(f"FF5 Alpha: {attr_result.alpha:.2%} (p={attr_result.alpha_p_value:.3f})")
    logger.info(f"R-squared: {attr_result.r_squared:.3f}")

    return attribution


def main():
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path('logs').mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("SignalTide v4 Backtest")
    logger.info("=" * 60)

    # Get universe
    universe = get_universe(args)
    logger.info(f"Universe: {len(universe)} tickers")
    logger.info(f"Period: {args.start} to {args.end}")
    logger.info(f"Capital: ${args.capital:,.0f}")
    logger.info(f"Transaction cost: {args.cost_bps} bps")

    # Run backtest
    logger.info("\nRunning backtest...")
    start_time = datetime.now()

    result = run_backtest(
        universe=universe,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        transaction_cost_bps=args.cost_bps,
    )

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Backtest completed in {elapsed:.1f} seconds")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Return:     {result.total_return:>10.2%}")
    logger.info(f"CAGR:             {result.cagr:>10.2%}")
    logger.info(f"Sharpe Ratio:     {result.sharpe_ratio:>10.3f}")
    logger.info(f"Sortino Ratio:    {result.sortino_ratio:>10.3f}")
    logger.info(f"Max Drawdown:     {result.max_drawdown:>10.2%}")
    logger.info(f"Calmar Ratio:     {result.calmar_ratio:>10.3f}")
    logger.info(f"Volatility:       {result.volatility:>10.2%}")
    logger.info(f"Total Costs:      ${result.total_costs:>9,.0f}")
    logger.info(f"Avg Positions:    {result.avg_positions:>10.1f}")
    logger.info(f"Final Value:      ${result.final_value:>9,.0f}")

    # Build output
    output = {
        'metadata': {
            'start_date': args.start,
            'end_date': args.end,
            'initial_capital': args.capital,
            'transaction_cost_bps': args.cost_bps,
            'universe_size': len(universe),
            'run_timestamp': datetime.now().isoformat(),
        },
        'performance': result.summary(),
        'final_value': result.final_value,
    }

    # Run validation if requested
    if args.validate:
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION")
        logger.info("=" * 60)
        output['validation'] = run_validation(result, args)

    # Run attribution if requested
    if args.attribution:
        logger.info("\n" + "=" * 60)
        logger.info("FACTOR ATTRIBUTION")
        logger.info("=" * 60)
        output['attribution'] = run_attribution(result, args)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {args.output}")
    logger.info("=" * 60)

    return 0 if result.sharpe_ratio > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
