"""
Compare monthly vs weekly rebalancing frequencies.

Runs the same signals with different rebalancing frequencies to identify
which produces better risk-adjusted returns.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.portfolio import Portfolio
from signals.momentum.institutional_momentum import InstitutionalMomentum
from signals.quality.institutional_quality import InstitutionalQuality
from signals.insider.institutional_insider import InstitutionalInsider
from data.data_manager import DataManager
from config import get_logger
from scripts.spy_benchmark_analysis import SPYBenchmarkAnalysis

logger = get_logger(__name__)


class RebalancingComparison:
    """Compare monthly vs weekly rebalancing frequencies."""

    def __init__(self, universe, start_date, end_date, initial_capital=50000):
        self.universe = universe
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.dm = DataManager()

    def run_backtest_with_rebalance(self, signal_class, signal_params, signal_name,
                                     rebalance_freq='M', needs_dm=False):
        """
        Run backtest with specified rebalancing frequency.

        Args:
            signal_class: Signal class to test
            signal_params: Signal parameters
            signal_name: Signal name for logging
            rebalance_freq: 'M' for monthly, 'W' for weekly
            needs_dm: Whether signal needs data_manager

        Returns:
            Dict with performance metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {signal_name} with {rebalance_freq} rebalancing")
        logger.info(f"{'='*60}")

        # Generate signals
        logger.info(f"Generating signals for {len(self.universe)} stocks...")

        if needs_dm:
            signal = signal_class(signal_params, data_manager=self.dm)
        else:
            signal = signal_class(signal_params)

        all_signals = {}
        for ticker in self.universe:
            try:
                sig = signal.generate(ticker, self.start_date, self.end_date)
                all_signals[ticker] = sig
            except Exception as e:
                logger.error(f"Error generating {signal_name} for {ticker}: {e}")
                continue

        if not all_signals:
            logger.error(f"No signals generated for {signal_name}")
            return None

        # Combine into DataFrame
        signals_df = pd.DataFrame(all_signals)

        # Get prices for all stocks
        prices = {}
        for ticker in self.universe:
            try:
                price_data = self.dm.get_prices([ticker], self.start_date, self.end_date)
                prices[ticker] = price_data[ticker]
            except Exception as e:
                logger.error(f"Error fetching prices for {ticker}: {e}")
                continue

        prices_df = pd.DataFrame(prices)

        # Align signals and prices
        common_index = signals_df.index.intersection(prices_df.index)
        signals_df = signals_df.loc[common_index]
        prices_df = prices_df.loc[common_index]

        # Apply rebalancing frequency
        if rebalance_freq == 'M':
            # Monthly: Use month-end signal, hold for entire next month
            rebal_signals = signals_df.resample('ME').last()
        elif rebalance_freq == 'W':
            # Weekly: Use week-end signal, hold for entire next week
            rebal_signals = signals_df.resample('W').last()
        else:
            raise ValueError(f"Unknown rebalance frequency: {rebalance_freq}")

        # Forward-fill to daily
        signals_rebal = rebal_signals.reindex(signals_df.index, method='ffill').fillna(0)

        # Remove duplicates if any
        if signals_rebal.index.duplicated().any():
            logger.warning(f"Found {signals_rebal.index.duplicated().sum()} duplicate dates in signals, removing...")
            signals_rebal = signals_rebal[~signals_rebal.index.duplicated(keep='last')]

        if prices_df.index.duplicated().any():
            logger.warning(f"Found {prices_df.index.duplicated().sum()} duplicate dates in prices, removing...")
            prices_df = prices_df[~prices_df.index.duplicated(keep='last')]

        # Initialize portfolio with optimal parameters
        portfolio_params = {
            'max_position_size': 1.0,
            'max_positions': len(self.universe),
            'stop_loss_pct': None,
            'take_profit_pct': None,
            'max_portfolio_drawdown': 0.25,
            'drawdown_scale_factor': 0.5,
            'commission_pct': 0.0,
            'slippage_pct': 0.0002,
            'spread_pct': 0.0003,
        }

        portfolio = Portfolio(
            initial_capital=self.initial_capital,
            params=portfolio_params
        )

        # Run backtest
        logger.info(f"Running portfolio backtest...")
        logger.info(f"Backtest period: {signals_rebal.index[0]} to {signals_rebal.index[-1]}")
        logger.info(f"Trading days: {len(signals_rebal)}")

        for date in signals_rebal.index:
            today_signals = signals_rebal.loc[date]
            today_prices = prices_df.loc[date]

            # Create signal and price dicts
            signals_dict = {}
            prices_dict = {}

            for ticker in today_signals.index:
                signal_value = today_signals[ticker]
                price_value = today_prices[ticker]

                if signal_value != 0 and pd.notna(price_value):
                    signals_dict[ticker] = float(signal_value)
                    prices_dict[ticker] = float(price_value)

            portfolio.update(date, signals_dict, prices_dict)

        # Calculate metrics
        metrics = portfolio.get_metrics()

        # Get equity curve for SPY comparison
        equity_df = pd.DataFrame(portfolio.equity_curve)
        equity_df.set_index('timestamp', inplace=True)

        return {
            'rebalance_freq': rebalance_freq,
            'total_return': metrics.total_return,
            'total_return_pct': metrics.total_return_pct,
            'sharpe_ratio': metrics.sharpe_ratio,
            'sortino_ratio': metrics.sortino_ratio,
            'max_drawdown_pct': metrics.max_drawdown_pct,
            'calmar_ratio': metrics.calmar_ratio,
            'win_rate': metrics.win_rate,
            'profit_factor': metrics.profit_factor,
            'n_trades': metrics.n_trades,
            'equity_curve': equity_df,
        }

    def compare_signal(self, signal_class, signal_params, signal_name, needs_dm=False):
        """
        Compare monthly vs weekly rebalancing for a signal.

        Returns:
            Dict with monthly and weekly results
        """
        logger.info(f"\n\n{'='*60}")
        logger.info(f"COMPARING REBALANCING FREQUENCIES: {signal_name}")
        logger.info(f"{'='*60}")

        # Test monthly
        monthly_results = self.run_backtest_with_rebalance(
            signal_class, signal_params, signal_name,
            rebalance_freq='M', needs_dm=needs_dm
        )

        # Test weekly
        weekly_results = self.run_backtest_with_rebalance(
            signal_class, signal_params, signal_name,
            rebalance_freq='W', needs_dm=needs_dm
        )

        # Compare results
        logger.info(f"\n{'='*60}")
        logger.info(f"COMPARISON RESULTS: {signal_name}")
        logger.info(f"{'='*60}")

        comparison = pd.DataFrame({
            'Monthly': {
                'Total Return': f"{monthly_results['total_return_pct']*100:.2f}%",
                'Sharpe Ratio': f"{monthly_results['sharpe_ratio']:.3f}",
                'Sortino Ratio': f"{monthly_results['sortino_ratio']:.3f}",
                'Max DD': f"{monthly_results['max_drawdown_pct']*100:.2f}%",
                'Calmar Ratio': f"{monthly_results['calmar_ratio']:.3f}",
                'Win Rate': f"{monthly_results['win_rate']*100:.1f}%",
                'Trades': monthly_results['n_trades'],
            },
            'Weekly': {
                'Total Return': f"{weekly_results['total_return_pct']*100:.2f}%",
                'Sharpe Ratio': f"{weekly_results['sharpe_ratio']:.3f}",
                'Sortino Ratio': f"{weekly_results['sortino_ratio']:.3f}",
                'Max DD': f"{weekly_results['max_drawdown_pct']*100:.2f}%",
                'Calmar Ratio': f"{weekly_results['calmar_ratio']:.3f}",
                'Win Rate': f"{weekly_results['win_rate']*100:.1f}%",
                'Trades': weekly_results['n_trades'],
            }
        })

        logger.info(f"\n{comparison.to_string()}")

        # Determine winner
        monthly_sharpe = monthly_results['sharpe_ratio']
        weekly_sharpe = weekly_results['sharpe_ratio']

        if monthly_sharpe > weekly_sharpe:
            logger.info(f"\n✅ MONTHLY WINS (Sharpe: {monthly_sharpe:.3f} vs {weekly_sharpe:.3f})")
            winner = 'monthly'
        else:
            logger.info(f"\n✅ WEEKLY WINS (Sharpe: {weekly_sharpe:.3f} vs {monthly_sharpe:.3f})")
            winner = 'weekly'

        return {
            'signal_name': signal_name,
            'monthly': monthly_results,
            'weekly': weekly_results,
            'winner': winner,
            'comparison_table': comparison,
        }


def main():
    """Run rebalancing frequency comparison."""

    # Configuration
    universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'JNJ', 'XOM']
    start_date = '2020-01-01'
    end_date = '2024-12-31'
    initial_capital = 50000

    logger.info(f"Rebalancing Frequency Comparison")
    logger.info(f"Universe: {len(universe)} stocks")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Capital: ${initial_capital:,}")

    comparison = RebalancingComparison(universe, start_date, end_date, initial_capital)

    # Compare signals
    results = {}

    # 1. Momentum
    logger.info("\n\n" + "="*80)
    logger.info("SIGNAL 1/3: MOMENTUM")
    logger.info("="*80)

    momentum_params = {
        'formation_period': 252,  # 12 months
        'skip_period': 21,        # 1 month
        'quintiles': True
    }

    results['momentum'] = comparison.compare_signal(
        InstitutionalMomentum,
        momentum_params,
        'InstitutionalMomentum',
        needs_dm=False
    )

    # 2. Insider (best performer from previous test)
    logger.info("\n\n" + "="*80)
    logger.info("SIGNAL 2/3: INSIDER")
    logger.info("="*80)

    insider_params = {
        'lookback_days': 90,
        'min_transaction_value': 10000,
        'cluster_window': 7,
        'cluster_min_insiders': 3
    }

    results['insider'] = comparison.compare_signal(
        InstitutionalInsider,
        insider_params,
        'InstitutionalInsider',
        needs_dm=True
    )

    # Summary
    logger.info("\n\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)

    for signal_name, result in results.items():
        logger.info(f"\n{result['signal_name']}:")
        logger.info(f"  Winner: {result['winner'].upper()}")
        monthly_sharpe = result['monthly']['sharpe_ratio']
        weekly_sharpe = result['weekly']['sharpe_ratio']
        logger.info(f"  Monthly Sharpe: {monthly_sharpe:.3f}")
        logger.info(f"  Weekly Sharpe: {weekly_sharpe:.3f}")
        logger.info(f"  Difference: {abs(monthly_sharpe - weekly_sharpe):.3f}")

    logger.info(f"\n{'='*80}")
    logger.info("✅ REBALANCING COMPARISON COMPLETE")
    logger.info(f"{'='*80}")


if __name__ == '__main__':
    main()
