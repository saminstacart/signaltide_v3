#!/usr/bin/env python3
"""
SPY benchmark comparison for SignalTide v4.

Compares strategy performance against S&P 500 (SPY) benchmark.

Usage:
    python -m signaltide_v4.scripts.compare_spy --help
    python -m signaltide_v4.scripts.compare_spy --returns strategy_returns.csv
    python -m signaltide_v4.scripts.compare_spy --backtest-file results.json
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

from signaltide_v4.data.sharadar_adapter import SharadarAdapter
from signaltide_v4.backtest.metrics import MetricsCalculator, BenchmarkComparison

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='SignalTide v4 vs SPY Benchmark Comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare from return series CSV
    python -m signaltide_v4.scripts.compare_spy --returns returns.csv

    # Compare from backtest results JSON
    python -m signaltide_v4.scripts.compare_spy --backtest-file results.json

    # Compare with specific date range
    python -m signaltide_v4.scripts.compare_spy --returns returns.csv --start 2020-01-01 --end 2024-12-31
        """
    )

    parser.add_argument(
        '--returns',
        type=str,
        help='Path to CSV file with daily returns'
    )
    parser.add_argument(
        '--backtest-file',
        type=str,
        help='Path to backtest results JSON file'
    )
    parser.add_argument(
        '--start',
        type=str,
        help='Start date override (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end',
        type=str,
        help='End date override (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/spy_comparison_v4.json',
        help='Output file for comparison results'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def load_returns(args) -> pd.Series:
    """Load strategy returns from file."""
    if args.returns:
        df = pd.read_csv(args.returns, index_col=0, parse_dates=True)
        if 'return' in df.columns:
            returns = df['return']
        else:
            returns = df.iloc[:, 0]
        returns.name = 'strategy'
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
            returns.name = 'strategy'
            return returns

    raise ValueError("Must provide --returns or --backtest-file")


def get_spy_returns(start_date: str, end_date: str) -> pd.Series:
    """Get SPY returns from database."""
    adapter = SharadarAdapter()

    try:
        prices = adapter.get_prices(['SPY'], start_date, end_date)

        if prices.empty:
            logger.warning("No SPY data in database, using synthetic benchmark")
            return None

        spy_prices = prices['SPY'].dropna()
        spy_returns = spy_prices.pct_change().dropna()
        spy_returns.name = 'SPY'

        return spy_returns

    except Exception as e:
        logger.warning(f"Failed to get SPY data: {e}")
        return None

    finally:
        adapter.close()


def calculate_rolling_comparison(
    strategy: pd.Series,
    benchmark: pd.Series,
    window: int = 252,
) -> dict:
    """Calculate rolling comparison metrics."""
    # Align series
    combined = pd.DataFrame({
        'strategy': strategy,
        'benchmark': benchmark,
    }).dropna()

    if len(combined) < window:
        return {}

    strategy = combined['strategy']
    benchmark = combined['benchmark']

    # Rolling excess returns
    excess = strategy - benchmark
    rolling_excess = excess.rolling(window).mean() * 252  # Annualized

    # Rolling beta
    def calc_beta(x):
        strat = x['strategy']
        bench = x['benchmark']
        cov = np.cov(strat, bench)
        return cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 1.0

    rolling_beta = combined.rolling(window).apply(
        lambda x: np.cov(x['strategy'], x['benchmark'])[0, 1] / np.var(x['benchmark'])
        if np.var(x['benchmark']) > 0 else 1.0,
        raw=False
    )

    # Rolling correlation
    rolling_corr = strategy.rolling(window).corr(benchmark)

    # Convert to serializable format
    return {
        'rolling_excess_return': {
            str(k): v for k, v in rolling_excess.dropna().to_dict().items()
        },
        'rolling_correlation': {
            str(k): v for k, v in rolling_corr.dropna().to_dict().items()
        },
    }


def analyze_periods(
    strategy: pd.Series,
    benchmark: pd.Series,
) -> dict:
    """Analyze performance by time periods."""
    combined = pd.DataFrame({
        'strategy': strategy,
        'benchmark': benchmark,
    }).dropna()

    if len(combined) < 21:
        return {}

    results = {}

    # Monthly analysis
    monthly_strat = (1 + combined['strategy']).resample('ME').prod() - 1
    monthly_bench = (1 + combined['benchmark']).resample('ME').prod() - 1

    excess_monthly = monthly_strat - monthly_bench

    results['monthly'] = {
        'strategy_mean': float(monthly_strat.mean()),
        'benchmark_mean': float(monthly_bench.mean()),
        'excess_mean': float(excess_monthly.mean()),
        'win_rate': float((excess_monthly > 0).mean()),
        'n_months': len(excess_monthly),
    }

    # Annual analysis
    if len(combined) >= 252:
        annual_strat = (1 + combined['strategy']).resample('Y').prod() - 1
        annual_bench = (1 + combined['benchmark']).resample('Y').prod() - 1

        excess_annual = annual_strat - annual_bench

        results['annual'] = {
            'strategy_returns': {str(k.year): float(v) for k, v in annual_strat.items()},
            'benchmark_returns': {str(k.year): float(v) for k, v in annual_bench.items()},
            'excess_returns': {str(k.year): float(v) for k, v in excess_annual.items()},
            'win_rate': float((excess_annual > 0).mean()),
            'n_years': len(excess_annual),
        }

    return results


def analyze_drawdowns(
    strategy: pd.Series,
    benchmark: pd.Series,
) -> dict:
    """Compare drawdown characteristics."""
    def calc_drawdown_stats(returns):
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        return {
            'max_drawdown': float(drawdown.min()),
            'avg_drawdown': float(drawdown[drawdown < 0].mean()) if len(drawdown[drawdown < 0]) > 0 else 0,
            'current_drawdown': float(drawdown.iloc[-1]),
        }

    combined = pd.DataFrame({
        'strategy': strategy,
        'benchmark': benchmark,
    }).dropna()

    return {
        'strategy': calc_drawdown_stats(combined['strategy']),
        'benchmark': calc_drawdown_stats(combined['benchmark']),
    }


def print_comparison_table(
    strategy_metrics: dict,
    benchmark_metrics: dict,
    comparison: BenchmarkComparison,
):
    """Print formatted comparison table."""
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON: STRATEGY vs SPY")
    print("=" * 70)

    print(f"\n{'Metric':<30} {'Strategy':>15} {'SPY':>15} {'Diff':>10}")
    print("-" * 70)

    metrics = [
        ('Total Return', 'total_return', '.2%'),
        ('CAGR', 'cagr', '.2%'),
        ('Volatility', 'volatility', '.2%'),
        ('Sharpe Ratio', 'sharpe_ratio', '.3f'),
        ('Sortino Ratio', 'sortino_ratio', '.3f'),
        ('Max Drawdown', 'max_drawdown', '.2%'),
        ('Calmar Ratio', 'calmar_ratio', '.3f'),
    ]

    for name, key, fmt in metrics:
        strat_val = strategy_metrics.get(key, 0)
        bench_val = benchmark_metrics.get(key, 0)
        diff = strat_val - bench_val

        strat_str = f"{strat_val:{fmt}}"
        bench_str = f"{bench_val:{fmt}}"
        diff_str = f"{diff:+{fmt}}"

        print(f"{name:<30} {strat_str:>15} {bench_str:>15} {diff_str:>10}")

    print("\n" + "-" * 70)
    print("BENCHMARK COMPARISON METRICS")
    print("-" * 70)

    print(f"{'Alpha (annualized)':<30} {comparison.alpha:>15.2%}")
    print(f"{'Beta':<30} {comparison.beta:>15.3f}")
    print(f"{'Correlation':<30} {comparison.correlation:>15.3f}")
    print(f"{'Tracking Error':<30} {comparison.tracking_error:>15.2%}")
    print(f"{'Information Ratio':<30} {comparison.information_ratio:>15.3f}")
    print(f"{'Up Capture':<30} {comparison.up_capture:>15.1f}%")
    print(f"{'Down Capture':<30} {comparison.down_capture:>15.1f}%")
    print(f"{'Alpha Significant (p<0.05)':<30} {str(comparison.alpha_significant):>15}")

    print("=" * 70)


def main():
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("SignalTide v4 vs SPY Benchmark Comparison")
    logger.info("=" * 60)

    # Load strategy returns
    try:
        strategy_returns = load_returns(args)
        logger.info(f"Loaded {len(strategy_returns)} strategy returns")
    except Exception as e:
        logger.error(f"Failed to load strategy returns: {e}")
        return 1

    # Determine date range
    if hasattr(strategy_returns.index[0], 'strftime'):
        start_date = args.start or strategy_returns.index[0].strftime('%Y-%m-%d')
        end_date = args.end or strategy_returns.index[-1].strftime('%Y-%m-%d')
    else:
        start_date = args.start or str(strategy_returns.index[0])
        end_date = args.end or str(strategy_returns.index[-1])

    logger.info(f"Analysis period: {start_date} to {end_date}")

    # Get SPY returns
    spy_returns = get_spy_returns(start_date, end_date)

    if spy_returns is None:
        logger.error("Could not retrieve SPY benchmark data")
        return 1

    logger.info(f"Loaded {len(spy_returns)} SPY returns")

    # Initialize metrics calculator
    calculator = MetricsCalculator()

    # Calculate metrics for both
    strategy_metrics = calculator.calculate_metrics(strategy_returns)
    benchmark_metrics = calculator.calculate_metrics(spy_returns)

    # Compare to benchmark
    comparison = calculator.compare_to_benchmark(strategy_returns, spy_returns)

    # Print comparison table
    print_comparison_table(
        strategy_metrics.to_dict(),
        benchmark_metrics.to_dict(),
        comparison,
    )

    # Additional analyses
    period_analysis = analyze_periods(strategy_returns, spy_returns)
    drawdown_analysis = analyze_drawdowns(strategy_returns, spy_returns)

    # Build output
    results = {
        'metadata': {
            'run_timestamp': datetime.now().isoformat(),
            'start_date': start_date,
            'end_date': end_date,
            'strategy_observations': len(strategy_returns),
            'benchmark_observations': len(spy_returns),
        },
        'strategy_metrics': strategy_metrics.to_dict(),
        'benchmark_metrics': benchmark_metrics.to_dict(),
        'comparison': comparison.to_dict(),
        'period_analysis': period_analysis,
        'drawdown_analysis': drawdown_analysis,
        'verdict': {
            'outperforms': strategy_metrics.cagr > benchmark_metrics.cagr,
            'lower_risk': strategy_metrics.volatility < benchmark_metrics.volatility,
            'better_risk_adjusted': strategy_metrics.sharpe_ratio > benchmark_metrics.sharpe_ratio,
            'positive_alpha': comparison.alpha > 0,
            'alpha_significant': comparison.alpha_significant,
            'recommendation': 'DEPLOY' if (
                comparison.alpha > 0 and
                comparison.alpha_significant and
                strategy_metrics.sharpe_ratio > benchmark_metrics.sharpe_ratio
            ) else 'REVIEW',
        },
    }

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {args.output}")

    # Print verdict
    verdict = results['verdict']
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"Outperforms SPY: {verdict['outperforms']}")
    print(f"Lower Risk: {verdict['lower_risk']}")
    print(f"Better Risk-Adjusted: {verdict['better_risk_adjusted']}")
    print(f"Positive Alpha: {verdict['positive_alpha']}")
    print(f"Alpha Significant: {verdict['alpha_significant']}")
    print(f"\nRecommendation: {verdict['recommendation']}")
    print("=" * 70)

    return 0 if verdict['recommendation'] == 'DEPLOY' else 1


if __name__ == '__main__':
    sys.exit(main())
