"""
Ensemble Baseline Runner

Runs production ensemble configurations for validation and diagnostics.
Uses centralized ensemble configs from signals/ml/ensemble_configs.py.

This replaces direct signal calls with ensemble path to:
1. Validate ensemble matches standalone signal performance
2. Establish ensemble as canonical production path
3. Enable easy multi-signal expansion

Usage:
    # Momentum-only ensemble (full period)
    python3 scripts/run_ensemble_baseline.py --config momentum_v2

    # Shorter validation window
    python3 scripts/run_ensemble_baseline.py --config momentum_v2 --start 2022-01-01

    # Debug mode (compare ensemble vs direct for first 6 rebalances)
    python3 scripts/run_ensemble_baseline.py --config momentum_v2 --debug

Outputs:
    results/ensemble_baselines/{config}_diagnostic.md
    results/ensemble_baselines/{config}_monthly_returns.csv
    results/ensemble_baselines/{config}_equity_curve.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from data.data_manager import DataManager
from core.universe_manager import UniverseManager
from signals.ml.ensemble_configs import get_momentum_v2_ensemble, list_available_ensembles
from signals.momentum.institutional_momentum import InstitutionalMomentum
from core.schedules import get_rebalance_dates
from validation.simple_validation import (
    compute_basic_metrics,
    compute_monthly_returns,
    compute_regime_metrics
)
from config import get_logger

logger = get_logger(__name__)


class EnsembleBaselineRunner:
    """
    Runs production ensemble configurations through full diagnostic.

    Validates that ensemble path produces identical results to standalone signals.
    """

    def __init__(self,
                 config_name: str = 'momentum_v2',
                 start_date: str = '2015-04-01',
                 end_date: str = '2024-12-31',
                 initial_capital: float = 50000,
                 debug_compare: bool = False):
        """
        Initialize ensemble baseline runner.

        Args:
            config_name: Ensemble config to run ('momentum_v2', etc.)
            start_date: Backtest start
            end_date: Backtest end
            initial_capital: Starting capital
            debug_compare: If True, compare ensemble vs direct signal (first 6 rebalances)
        """
        self.config_name = config_name
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.debug_compare = debug_compare

        self.dm = DataManager()
        self.um = UniverseManager(self.dm)

        logger.info("=" * 80)
        logger.info(f"Ensemble Baseline: {config_name}")
        logger.info("=" * 80)
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Capital: ${initial_capital:,.0f}")
        logger.info(f"Debug compare: {debug_compare}")
        logger.info("=" * 80)

        # Initialize ensemble
        if config_name == 'momentum_v2':
            self.ensemble = get_momentum_v2_ensemble(self.dm)

            # For debug comparison
            if self.debug_compare:
                momentum_params = {
                    'formation_period': 308,
                    'skip_period': 0,
                    'winsorize_pct': [0.4, 99.6],
                    'rebalance_frequency': 'monthly',
                    'quintiles': True
                }
                self.direct_signal = InstitutionalMomentum(params=momentum_params)
        else:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(list_available_ensembles().keys())}")

    def run_baseline(self) -> Dict:
        """
        Run full ensemble baseline diagnostic.

        Returns:
            Dict with results
        """
        results = {
            'config': self.config_name,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_capital': self.initial_capital
        }

        # 1. Build universe
        logger.info("\n1. Building S&P 500 PIT universe...")
        universe = self.um.get_universe(
            universe_type='sp500_actual',
            as_of_date=self.start_date,
            min_price=5.0
        )
        logger.info(f"Universe: {len(universe)} stocks")

        # 2. Get rebalance dates
        logger.info("\n2. Getting monthly rebalance dates...")
        rebalance_dates = get_rebalance_dates(
            schedule='M',
            dm=self.dm,
            start_date=self.start_date,
            end_date=self.end_date
        )
        logger.info(f"Rebalance dates: {len(rebalance_dates)} month-ends")

        # 3. Fetch price data
        logger.info("\n3. Fetching price data...")
        lookback_buffer = timedelta(days=500)  # 308 formation + buffer
        start_dt = pd.Timestamp(self.start_date)
        price_start_date = (start_dt - lookback_buffer).strftime('%Y-%m-%d')

        prices_dict = {}
        for ticker in universe:
            try:
                prices = self.dm.get_prices(ticker, price_start_date, self.end_date)
                if len(prices) > 0 and 'close' in prices.columns:
                    prices_dict[ticker] = prices['close']
            except Exception as e:
                logger.debug(f"Failed to load {ticker}: {e}")

        logger.info(f"Price data loaded: {len(prices_dict)} stocks")

        # 4. Run backtest with ensemble
        logger.info("\n4. Running ensemble backtest...")
        equity_curve, monthly_returns = self._run_backtest(
            prices_dict=prices_dict,
            rebalance_dates=rebalance_dates
        )

        # 5. Compute metrics
        logger.info("\n5. Computing performance metrics...")

        # Basic metrics (pass equity curve, not returns!)
        basic_metrics = compute_basic_metrics(equity_curve)
        results['metrics'] = basic_metrics

        # Regime analysis
        regime_metrics = compute_regime_metrics(monthly_returns)
        results['regimes'] = regime_metrics

        # 6. Save results
        logger.info("\n6. Saving results...")
        self._save_results(results, equity_curve, monthly_returns)

        # 7. Print summary
        logger.info("\n" + "=" * 80)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Final Equity: ${equity_curve.iloc[-1]:,.0f}")
        logger.info(f"Total Return: {basic_metrics.get('total_return', 0.0):.2%}")
        logger.info(f"CAGR: {basic_metrics.get('annual_return', 0.0):.2%}")
        logger.info(f"Volatility: {basic_metrics.get('volatility', 0.0):.2%}")
        logger.info(f"Sharpe: {basic_metrics.get('sharpe', 0.0):.3f}")
        logger.info(f"Max Drawdown: {basic_metrics.get('max_drawdown', 0.0):.2%}")
        logger.info("=" * 80)

        # 8. Compare with Trial 11 (if momentum_v2 config)
        if self.config_name == 'momentum_v2':
            logger.info("\n8. Comparing with Trial 11 baseline...")
            self._print_trial11_comparison(basic_metrics)

        return results

    def _run_backtest(self,
                     prices_dict: Dict[str, pd.Series],
                     rebalance_dates: list) -> tuple:
        """
        Run ensemble backtest with monthly rebalancing.

        Args:
            prices_dict: Dict mapping ticker -> price series
            rebalance_dates: List of rebalance timestamps

        Returns:
            (equity_curve, monthly_returns)
        """
        portfolio_value = self.initial_capital
        current_holdings = {}

        equity_data = []
        monthly_returns_data = []

        # Debug comparison counters
        debug_limit = 6 if self.debug_compare else 0
        debug_count = 0

        for i, rebal_date in enumerate(pd.DatetimeIndex(rebalance_dates)):
            logger.info(f"\nRebalance {i+1}/{len(rebalance_dates)}: {rebal_date.date()}")

            # Build prices_by_ticker up to rebal_date
            prices_by_ticker = {}
            for ticker, px_series in prices_dict.items():
                px_slice = px_series[px_series.index <= rebal_date]
                if len(px_slice) >= 90:  # Minimum history
                    prices_by_ticker[ticker] = px_slice

            if len(prices_by_ticker) == 0:
                logger.warning("No tickers with sufficient history, skipping")
                continue

            # Generate ensemble scores
            ensemble_scores = self.ensemble.generate_ensemble_scores(
                prices_by_ticker=prices_by_ticker,
                rebalance_date=rebal_date
            )

            logger.info(f"  Ensemble scores: {len(ensemble_scores)} tickers, "
                       f"range=[{ensemble_scores.min():.3f}, {ensemble_scores.max():.3f}]")

            # Debug comparison (first N rebalances)
            if self.debug_compare and debug_count < debug_limit:
                self._debug_compare_scores(
                    ensemble_scores=ensemble_scores,
                    prices_by_ticker=prices_by_ticker,
                    rebal_date=rebal_date,
                    rebal_num=i+1
                )
                debug_count += 1

            # Filter to valid scores
            scores = ensemble_scores.dropna()
            scores = scores[scores != 0]

            if len(scores) < 20:
                logger.warning(f"Only {len(scores)} stocks with signals, skipping")
                continue

            # Take top quintile (highest scores)
            sorted_scores = scores.sort_values(ascending=False)
            quintile_size = len(sorted_scores) // 5
            top_quintile = sorted_scores.iloc[:quintile_size].index.tolist()

            logger.info(f"  Top quintile: {len(top_quintile)} stocks")

            # Equal-weight allocation
            weight_per_stock = 1.0 / len(top_quintile)

            # Get rebalance prices
            rebal_prices = {}
            for ticker in top_quintile:
                if ticker in prices_dict:
                    px = prices_dict[ticker]
                    px_at_rebal = px[px.index <= rebal_date]
                    if len(px_at_rebal) > 0:
                        rebal_prices[ticker] = px_at_rebal.iloc[-1]

            # Rebalance holdings
            current_holdings = {}
            for ticker in top_quintile:
                if ticker in rebal_prices:
                    current_holdings[ticker] = {
                        'shares': (portfolio_value * weight_per_stock) / rebal_prices[ticker],
                        'entry_price': rebal_prices[ticker]
                    }

            # Track equity until next rebalance
            if i + 1 < len(rebalance_dates):
                next_rebal = pd.Timestamp(rebalance_dates[i + 1])
            else:
                next_rebal = pd.Timestamp(self.end_date)

            # Get all dates in the holding period (including rebal_date)
            all_dates = sorted(set().union(*[
                set(prices_dict[t].index[(prices_dict[t].index >= rebal_date) &
                                         (prices_dict[t].index <= next_rebal)])
                for t in current_holdings.keys() if t in prices_dict
            ]))

            for date in all_dates:
                total_value = 0.0
                for ticker, holding in current_holdings.items():
                    if ticker in prices_dict:
                        px = prices_dict[ticker]
                        px_at_date = px[px.index <= date]
                        if len(px_at_date) > 0:
                            current_price = px_at_date.iloc[-1]
                            total_value += holding['shares'] * current_price

                equity_data.append({
                    'date': date,
                    'equity': total_value if total_value > 0 else portfolio_value
                })

            # Update portfolio value from last equity point
            if equity_data:
                portfolio_value = equity_data[-1]['equity']

        # Build equity curve
        if not equity_data:
            logger.warning("No equity data collected!")
            return pd.Series(dtype=float), pd.Series(dtype=float)

        equity_df = pd.DataFrame(equity_data)
        equity_curve = equity_df.set_index('date')['equity']

        # Compute monthly returns from equity curve
        # Align to rebalance dates
        monthly_returns_data = []
        for i in range(len(rebalance_dates)):
            rebal_date = pd.Timestamp(rebalance_dates[i])

            if i + 1 < len(rebalance_dates):
                next_rebal = pd.Timestamp(rebalance_dates[i + 1])
            else:
                # Last period: use end_date
                next_rebal = pd.Timestamp(self.end_date)

            # Get equity at start and end of period
            eq_start = equity_curve[equity_curve.index <= rebal_date]
            eq_end = equity_curve[(equity_curve.index > rebal_date) &
                                  (equity_curve.index <= next_rebal)]

            if len(eq_start) > 0 and len(eq_end) > 0:
                period_return = (eq_end.iloc[-1] / eq_start.iloc[-1]) - 1
                monthly_returns_data.append({
                    'date': rebal_date,
                    'return': period_return
                })

        monthly_df = pd.DataFrame(monthly_returns_data)
        if len(monthly_df) > 0:
            monthly_returns = monthly_df.set_index('date')['return']
        else:
            monthly_returns = pd.Series(dtype=float)

        return equity_curve, monthly_returns

    def _debug_compare_scores(self,
                             ensemble_scores: pd.Series,
                             prices_by_ticker: Dict[str, pd.Series],
                             rebal_date: pd.Timestamp,
                             rebal_num: int):
        """
        Debug comparison: ensemble vs direct signal.

        Args:
            ensemble_scores: Scores from ensemble
            prices_by_ticker: Price data
            rebal_date: Current rebalance date
            rebal_num: Rebalance number
        """
        logger.info(f"  DEBUG COMPARE (Rebalance {rebal_num}):")

        # Generate direct scores
        direct_scores = {}
        for ticker, px_series in prices_by_ticker.items():
            data = pd.DataFrame({'close': px_series, 'ticker': ticker})
            data = data[data.index <= rebal_date]

            try:
                sig_series = self.direct_signal.generate_signals(data)
                if len(sig_series) > 0:
                    signal_value = sig_series.iloc[-1]
                    if pd.notna(signal_value) and signal_value != 0:
                        direct_scores[ticker] = signal_value
            except:
                pass

        direct_scores_series = pd.Series(direct_scores)

        # Compare
        common = set(ensemble_scores.index) & set(direct_scores_series.index)
        diffs = []
        for ticker in common:
            diff = abs(ensemble_scores[ticker] - direct_scores_series[ticker])
            diffs.append(diff)

        max_diff = max(diffs) if diffs else 0.0
        mean_diff = np.mean(diffs) if diffs else 0.0

        logger.info(f"    Common tickers: {len(common)}")
        logger.info(f"    Max diff: {max_diff:.2e}")
        logger.info(f"    Mean diff: {mean_diff:.2e}")

        if max_diff > 1e-9:
            logger.warning(f"    ⚠️  Ensemble != Direct (max diff: {max_diff:.2e})")
        else:
            logger.info(f"    ✅ Ensemble matches Direct")

    def _save_results(self,
                     results: Dict,
                     equity_curve: pd.Series,
                     monthly_returns: pd.Series):
        """
        Save results to files.

        Args:
            results: Results dict
            equity_curve: Daily equity series
            monthly_returns: Monthly return series
        """
        output_dir = Path('results/ensemble_baselines')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save equity curve
        equity_df = pd.DataFrame({
            'date': equity_curve.index,
            'equity': equity_curve.values
        })
        equity_path = output_dir / f"{self.config_name}_equity_curve.csv"
        equity_df.to_csv(equity_path, index=False)
        logger.info(f"  Saved equity curve: {equity_path}")

        # Save monthly returns
        monthly_df = pd.DataFrame({
            'date': monthly_returns.index,
            'return': monthly_returns.values
        })
        monthly_path = output_dir / f"{self.config_name}_monthly_returns.csv"
        monthly_df.to_csv(monthly_path, index=False)
        logger.info(f"  Saved monthly returns: {monthly_path}")

        # Generate diagnostic report
        self._generate_report(results, output_dir)

    def _generate_report(self, results: Dict, output_dir: Path):
        """
        Generate markdown diagnostic report.

        Args:
            results: Results dict
            output_dir: Output directory
        """
        report_path = output_dir / f"{self.config_name}_diagnostic.md"

        metrics = results['metrics']
        regimes = results['regimes']

        with open(report_path, 'w') as f:
            f.write(f"# Ensemble Baseline Diagnostic: {self.config_name}\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Period:** {self.start_date} to {self.end_date}\n\n")
            f.write(f"**Universe:** S&P 500 PIT (sp500_actual)\n\n")
            f.write(f"**Rebalancing:** Monthly (month-end)\n\n")
            f.write(f"**Capital:** ${self.initial_capital:,.0f}\n\n")

            f.write("---\n\n")
            f.write("## Performance Metrics\n\n")
            f.write(f"- **Total Return:** {metrics.get('total_return', 0.0):.2%}\n")
            f.write(f"- **CAGR:** {metrics.get('annual_return', 0.0):.2%}\n")
            f.write(f"- **Volatility:** {metrics.get('volatility', 0.0):.2%}\n")
            f.write(f"- **Sharpe Ratio:** {metrics.get('sharpe', 0.0):.3f}\n")
            f.write(f"- **Max Drawdown:** {metrics.get('max_drawdown', 0.0):.2%}\n")
            f.write(f"- **Num Days:** {metrics.get('num_days', 0)}\n")
            f.write(f"- **Num Months:** {metrics.get('num_months', 0)}\n\n")

            if 'regimes' in results:
                f.write("## Regime Analysis\n\n")
                for regime_name, regime_data in regimes.items():
                    f.write(f"### {regime_name}\n\n")
                    f.write(f"- Mean Return: {regime_data.get('mean_return', 'N/A')}\n")
                    f.write(f"- Sharpe: {regime_data.get('sharpe', 'N/A')}\n")
                    f.write(f"- Months: {regime_data.get('num_months', 'N/A')}\n\n")

            f.write("---\n\n")
            f.write("**Report Generated:** {}\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

        logger.info(f"  Saved diagnostic report: {report_path}")

    def _print_trial11_comparison(self, basic_metrics: Dict):
        """
        Compare ensemble baseline metrics with Trial 11.

        Parses Trial 11 diagnostic report and prints side-by-side comparison.

        Args:
            basic_metrics: Ensemble baseline metrics
        """
        import re

        trial11_path = Path(__file__).parent.parent / "results" / "MOMENTUM_PHASE2_TRIAL11_DIAGNOSTIC.md"

        if not trial11_path.exists():
            logger.warning(f"Trial 11 report not found: {trial11_path}")
            return

        # Parse Trial 11 metrics
        trial11_metrics = {}
        try:
            with open(trial11_path, 'r') as f:
                content = f.read()

            # Extract full sample metrics (2015-04-01 to 2024-12-31)
            # Look for "Full Sample Performance" section
            full_sample_section = re.search(
                r'## Full Sample Performance.*?\n\n(.*?)\n\n##',
                content,
                re.DOTALL
            )

            if full_sample_section:
                metrics_text = full_sample_section.group(1)

                # Extract individual metrics
                total_return_match = re.search(r'\*\*Total Return:\*\*\s+([\d.]+)%', metrics_text)
                annual_return_match = re.search(r'\*\*Annual Return:\*\*\s+([\d.]+)%', metrics_text)
                volatility_match = re.search(r'\*\*Volatility:\*\*\s+([\d.]+)%', metrics_text)
                sharpe_match = re.search(r'\*\*Sharpe Ratio:\*\*\s+([\d.]+)', metrics_text)
                max_dd_match = re.search(r'\*\*Max Drawdown:\*\*\s+-([\d.]+)%', metrics_text)

                if total_return_match:
                    trial11_metrics['total_return'] = float(total_return_match.group(1)) / 100
                if annual_return_match:
                    trial11_metrics['annual_return'] = float(annual_return_match.group(1)) / 100
                if volatility_match:
                    trial11_metrics['volatility'] = float(volatility_match.group(1)) / 100
                if sharpe_match:
                    trial11_metrics['sharpe'] = float(sharpe_match.group(1))
                if max_dd_match:
                    trial11_metrics['max_drawdown'] = -float(max_dd_match.group(1)) / 100

        except Exception as e:
            logger.warning(f"Error parsing Trial 11 metrics: {e}")
            return

        if not trial11_metrics:
            logger.warning("Could not extract Trial 11 metrics from report")
            return

        # Print comparison
        logger.info("\n" + "=" * 80)
        logger.info("TRIAL 11 vs ENSEMBLE BASELINE COMPARISON")
        logger.info("=" * 80)
        logger.info("")
        logger.info(f"{'Metric':<20} {'Trial 11':<15} {'Ensemble':<15} {'Difference':<15}")
        logger.info("-" * 80)

        metrics_to_compare = [
            ('Total Return', 'total_return', '{:.2%}'),
            ('CAGR', 'annual_return', '{:.2%}'),
            ('Volatility', 'volatility', '{:.2%}'),
            ('Sharpe', 'sharpe', '{:.3f}'),
            ('Max Drawdown', 'max_drawdown', '{:.2%}')
        ]

        for label, key, fmt in metrics_to_compare:
            trial11_val = trial11_metrics.get(key, 0.0)
            ensemble_val = basic_metrics.get(key, 0.0)
            diff = ensemble_val - trial11_val

            trial11_str = fmt.format(trial11_val)
            ensemble_str = fmt.format(ensemble_val)

            # Format difference
            if '%' in fmt:
                diff_str = f"{diff:+.2%}"
            else:
                diff_str = f"{diff:+.3f}"

            logger.info(f"{label:<20} {trial11_str:<15} {ensemble_str:<15} {diff_str:<15}")

        logger.info("=" * 80)
        logger.info("")
        logger.info("NOTE: Differences expected due to:")
        logger.info("  - Signal generation implementation variations")
        logger.info("  - Price data alignment edge cases")
        logger.info("  - Floating point precision in calculations")
        logger.info("")
        logger.info("Key validation: Ensemble should be in same ballpark as Trial 11")
        logger.info("  (~90% total return, ~0.25 Sharpe, ~-45% max DD)")
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Run ensemble baseline diagnostics',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    configs = list(list_available_ensembles().keys())
    parser.add_argument('--config', default='momentum_v2',
                       choices=configs,
                       help=f'Ensemble config to run (default: momentum_v2)')
    parser.add_argument('--start', default='2015-04-01',
                       help='Start date (default: 2015-04-01)')
    parser.add_argument('--end', default='2024-12-31',
                       help='End date (default: 2024-12-31)')
    parser.add_argument('--capital', type=float, default=50000,
                       help='Initial capital (default: $50,000)')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode: compare ensemble vs direct for first 6 rebalances')

    args = parser.parse_args()

    # Run baseline
    runner = EnsembleBaselineRunner(
        config_name=args.config,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        debug_compare=args.debug
    )

    results = runner.run_baseline()

    logger.info("\n" + "=" * 80)
    logger.info("✅ ENSEMBLE BASELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results in: results/ensemble_baselines/")


if __name__ == '__main__':
    main()
