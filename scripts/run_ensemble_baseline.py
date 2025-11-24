"""
Ensemble Baseline Runner (REFACTORED - Step 1)

Runs production ensemble configurations using unified backtest engine.
Uses centralized ensemble configs from signals/ml/ensemble_configs.py.

REFACTORED: Now uses core/backtest_engine.py for consistent execution.

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
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from data.data_manager import DataManager
from core.universe_manager import UniverseManager
from core.backtest_engine import BacktestConfig, run_backtest
from core.signal_adapters import make_ensemble_signal_fn
from signals.ml.ensemble_configs import get_momentum_v2_ensemble, list_available_ensembles
from signals.momentum.institutional_momentum import InstitutionalMomentum
from validation.simple_validation import compute_regime_metrics
from config import get_logger

logger = get_logger(__name__)


class EnsembleBaselineRunner:
    """
    Runs production ensemble configurations through full diagnostic.

    REFACTORED: Uses unified backtest engine for core execution.
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
            debug_compare: If True, compare ensemble vs direct (first 6 rebalances)
        """
        self.config_name = config_name
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.debug_compare = debug_compare

        self.dm = DataManager()
        self.um = UniverseManager(self.dm)

        logger.info("=" * 80)
        logger.info(f"Ensemble Baseline (via unified harness): {config_name}")
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
                logger.info("Debug mode enabled: will compare ensemble vs direct signals")
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

        # Define universe function (S&P 500 PIT)
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

        # Define signal function (ensemble path)
        debug_counter = [0]  # Mutable counter for debug mode

        def signal_fn(rebal_date: str, tickers: List[str]) -> pd.Series:
            """Generate ensemble signals for tickers."""
            # Fetch price data with lookback
            lookback_days = 500
            lookback_start = (pd.Timestamp(rebal_date) - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

            prices_dict = {}
            for ticker in tickers:
                try:
                    prices = self.dm.get_prices(ticker, lookback_start, rebal_date)
                    if len(prices) > 0 and 'close' in prices.columns:
                        # Filter to data up to rebal_date
                        px_slice = prices['close'][prices.index <= pd.Timestamp(rebal_date)]
                        if len(px_slice) >= 90:  # Minimum history
                            prices_dict[ticker] = px_slice
                except Exception as e:
                    logger.debug(f"Could not fetch prices for {ticker}: {e}")

            if len(prices_dict) == 0:
                return pd.Series(dtype=float)

            # Generate ensemble scores
            ensemble_scores = self.ensemble.generate_ensemble_scores(
                prices_by_ticker=prices_dict,
                rebalance_date=pd.Timestamp(rebal_date)
            )

            # Debug comparison (first 6 rebalances only)
            if self.debug_compare and debug_counter[0] < 6:
                debug_counter[0] += 1
                self._debug_compare_signals(
                    rebal_date,
                    prices_dict,
                    ensemble_scores,
                    debug_counter[0]
                )

            return ensemble_scores

        # Configure backtest
        config = BacktestConfig(
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.initial_capital,
            rebalance_schedule='M',
            long_only=True,
            equal_weight=True,
            track_daily_equity=False,  # Rebalance-point only
            data_manager=self.dm
        )

        # Run backtest via unified engine
        backtest_result = run_backtest(universe_fn, signal_fn, config)

        # Package results
        results['metrics'] = {
            'total_return': backtest_result.total_return,
            'annual_return': backtest_result.cagr,
            'volatility': backtest_result.volatility,
            'sharpe': backtest_result.sharpe,
            'max_drawdown': backtest_result.max_drawdown
        }

        # Compute monthly returns
        equity_curve = backtest_result.equity_curve
        monthly_returns = equity_curve.pct_change().dropna()

        # Regime analysis
        regime_metrics = compute_regime_metrics(monthly_returns)
        results['regimes'] = regime_metrics

        # Save results
        logger.info("\nSaving results...")
        self._save_results(results, equity_curve, monthly_returns)

        # Compare with Trial 11 (if momentum_v2 config)
        if self.config_name == 'momentum_v2':
            logger.info("\nComparing with Trial 11 baseline...")
            self._print_trial11_comparison(results['metrics'])

        return results

    def _debug_compare_signals(self,
                               rebal_date: str,
                               prices_dict: Dict,
                               ensemble_scores: pd.Series,
                               iteration: int):
        """
        Compare ensemble vs direct signal generation (debug mode only).

        Args:
            rebal_date: Rebalance date
            prices_dict: Dict of ticker -> price series
            ensemble_scores: Scores from ensemble
            iteration: Debug iteration number
        """
        logger.info(f"\n  DEBUG COMPARISON #{iteration}: {rebal_date}")

        # Generate direct signals for comparison
        direct_scores = {}
        for ticker, px_series in prices_dict.items():
            try:
                data = pd.DataFrame({'close': px_series, 'ticker': ticker})
                sig_series = self.direct_signal.generate_signals(data)
                if len(sig_series) > 0:
                    signal_value = sig_series.iloc[-1]
                    if pd.notna(signal_value) and signal_value != 0:
                        direct_scores[ticker] = signal_value
            except Exception as e:
                logger.debug(f"Direct signal generation failed for {ticker}: {e}")

        direct_scores = pd.Series(direct_scores)

        # Compare
        common_tickers = set(ensemble_scores.index) & set(direct_scores.index)
        if len(common_tickers) > 0:
            ensemble_common = ensemble_scores.loc[list(common_tickers)]
            direct_common = direct_scores.loc[list(common_tickers)]

            correlation = ensemble_common.corr(direct_common)
            max_diff = (ensemble_common - direct_common).abs().max()

            logger.info(f"    Common tickers: {len(common_tickers)}")
            logger.info(f"    Correlation: {correlation:.4f}")
            logger.info(f"    Max difference: {max_diff:.6f}")
        else:
            logger.info(f"    No common tickers for comparison")

    def _save_results(self,
                     results: Dict,
                     equity_curve: pd.Series,
                     monthly_returns: pd.Series):
        """Save results to files."""
        output_dir = Path('results/ensemble_baselines')
        output_dir.mkdir(parents=True, exist_ok=True)

        config = results['config']

        # 1. Equity curve CSV
        equity_path = output_dir / f"{config}_equity_curve.csv"
        equity_curve.to_frame('equity').to_csv(equity_path)
        logger.info(f"  Saved equity curve: {equity_path}")

        # 2. Monthly returns CSV
        returns_path = output_dir / f"{config}_monthly_returns.csv"
        monthly_returns.to_frame('return').to_csv(returns_path)
        logger.info(f"  Saved monthly returns: {returns_path}")

        # 3. Diagnostic markdown
        diagnostic_path = output_dir / f"{config}_diagnostic.md"
        self._write_diagnostic_md(diagnostic_path, results)
        logger.info(f"  Saved diagnostic: {diagnostic_path}")

    def _write_diagnostic_md(self, path: Path, results: Dict):
        """Write diagnostic markdown report."""
        metrics = results['metrics']

        content = f"""# Ensemble Baseline Diagnostic

**Config:** {results['config']}
**Period:** {results['start_date']} to {results['end_date']}
**Capital:** ${results['initial_capital']:,.0f}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Performance Summary

- **Total Return:** {metrics['total_return']:.2%}
- **CAGR:** {metrics['annual_return']:.2%}
- **Volatility:** {metrics['volatility']:.2%}
- **Sharpe Ratio:** {metrics['sharpe']:.3f}
- **Max Drawdown:** {metrics['max_drawdown']:.2%}

---

## Regime Performance

"""
        # Add regime metrics if available
        if 'regimes' in results:
            for regime_name, regime_data in results['regimes'].items():
                if isinstance(regime_data, dict):
                    content += f"### {regime_name}\n"
                    content += f"- Mean Monthly Return: {regime_data.get('mean_return', 0.0):.2%}\n"
                    content += f"- Sharpe: {regime_data.get('sharpe', 0.0):.3f}\n"
                    content += f"- Months: {regime_data.get('num_months', 0)}\n\n"

        content += """
---

**NOTE:** Generated via unified backtest engine (Step 1 refactor)
"""

        path.write_text(content)

    def _print_trial11_comparison(self, basic_metrics: Dict):
        """
        Compare ensemble baseline with Trial 11 diagnostic.

        Parses Trial 11 report and prints side-by-side comparison.
        """
        trial11_path = Path('results/MOMENTUM_PHASE2_TRIAL11_DIAGNOSTIC.md')

        if not trial11_path.exists():
            logger.warning(f"Trial 11 report not found: {trial11_path}")
            return

        # Parse Trial 11 metrics
        content = trial11_path.read_text()
        trial11_metrics = {}

        try:
            # Extract metrics using regex
            total_return_match = re.search(r'\*\*Total Return:\*\*\s+([\d.]+)%', content)
            annual_return_match = re.search(r'\*\*Annual Return:\*\*\s+([\d.]+)%', content)
            volatility_match = re.search(r'\*\*Volatility:\*\*\s+([\d.]+)%', content)
            sharpe_match = re.search(r'\*\*Sharpe Ratio:\*\*\s+([\d.]+)', content)
            max_dd_match = re.search(r'\*\*Max Drawdown:\*\*\s+-([\d.]+)%', content)

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


def main():
    parser = argparse.ArgumentParser(
        description='Run ensemble baseline diagnostics (via unified harness)',
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
    logger.info("✅ ENSEMBLE BASELINE COMPLETE (unified harness)")
    logger.info("=" * 80)
    logger.info(f"Results in: results/ensemble_baselines/")


if __name__ == '__main__':
    main()
