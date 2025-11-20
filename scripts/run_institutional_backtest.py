"""
Institutional Signal Backtest Runner

Runs comprehensive backtest of institutional signals with SPY benchmark comparison.

Features:
- Flexible universe selection (manual list, S&P 500, market cap filter)
- Point-in-time data access (no lookahead bias)
- Realistic transaction costs for $50K Schwab account
- Full SPY benchmark analysis
- Comprehensive reporting

Usage:
    # Start with 10 stocks to validate
    python3 scripts/run_institutional_backtest.py \
        --universe manual \
        --tickers AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA,JPM,JNJ,XOM \
        --period 2020-01-01,2024-12-31 \
        --signals institutional

    # Full S&P 500 point-in-time universe (future)
    python3 scripts/run_institutional_backtest.py \
        --universe sp500 \
        --period 2020-01-01,2024-12-31 \
        --signals institutional

Monitor:
    # Terminal 1: Run backtest
    python3 scripts/run_institutional_backtest.py ...

    # Terminal 2: Monitor logs
    tail -f logs/signaltide_dev.log

    # Terminal 3: Watch progress
    watch -n 5 'ls -lh results/'
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import argparse
from tqdm import tqdm

from data.data_manager import DataManager
from signals import InstitutionalMomentum, InstitutionalQuality, InstitutionalInsider
from core.portfolio import Portfolio
from scripts.spy_benchmark_analysis import SPYBenchmarkAnalysis
from config import get_logger, DEFAULT_TRANSACTION_COSTS

logger = get_logger(__name__)


class UniverseSelector:
    """
    Handles universe selection with point-in-time logic.
    """

    def __init__(self, data_manager: DataManager):
        self.dm = data_manager

    def get_manual_universe(self, tickers: List[str]) -> List[str]:
        """
        Simple manual ticker list.

        Good for:
        - Initial validation
        - Testing specific stocks
        - Small liquid universe
        """
        logger.info(f"Manual universe: {len(tickers)} tickers")
        return tickers

    def get_sp500_universe(self, start_date: str, end_date: str) -> Dict[str, List[str]]:
        """
        S&P 500 point-in-time universe.

        Returns universe at each rebalance date (monthly).

        TODO: Implement point-in-time S&P 500 membership
        For now, uses top 500 by market cap as proxy.
        """
        logger.warning("S&P 500 point-in-time not yet implemented")
        logger.warning("Using market cap filter as proxy")

        return self.get_market_cap_universe(
            min_market_cap=2e9,  # $2B minimum
            max_stocks=500,
            start_date=start_date,
            end_date=end_date
        )

    def get_market_cap_universe(self,
                                min_market_cap: float,
                                max_stocks: int,
                                start_date: str,
                                end_date: str) -> Dict[str, List[str]]:
        """
        Select universe by market cap at each rebalance date.

        Point-in-time: Uses market cap as of rebalance date.

        TODO: Query Sharadar fundamentals for market cap
        For now, returns None (not yet implemented)
        """
        logger.warning("Market cap universe not yet implemented")
        logger.warning("Requires querying Sharadar fundamentals for market cap")

        return None


class InstitutionalBacktest:
    """
    Comprehensive backtest runner for institutional signals.
    """

    def __init__(self,
                 universe: List[str],
                 start_date: str,
                 end_date: str,
                 initial_capital: float = 50000,
                 transaction_costs: Optional[Dict] = None,
                 params: Optional[Dict] = None):
        """
        Initialize backtest.

        Args:
            universe: List of tickers to trade
            start_date: Backtest start (YYYY-MM-DD)
            end_date: Backtest end (YYYY-MM-DD)
            initial_capital: Starting capital (default $50k)
            transaction_costs: Cost model (default: Schwab $50k model)
            params: Additional parameters (e.g., rebalance_freq)
        """
        self.universe = universe
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.params = params or {}

        # Transaction costs (default to realistic Schwab model)
        if transaction_costs is None:
            self.transaction_costs = {
                'commission_pct': 0.0,      # $0 commission
                'slippage_pct': 0.0002,     # 2 bps
                'spread_pct': 0.0003,       # 3 bps
                # Total: 5 bps per trade
            }
        else:
            self.transaction_costs = transaction_costs

        logger.info(f"Backtest initialized:")
        logger.info(f"  Universe: {len(universe)} stocks")
        logger.info(f"  Period: {start_date} to {end_date}")
        logger.info(f"  Capital: ${initial_capital:,.0f}")
        logger.info(f"  Transaction costs: {sum(self.transaction_costs.values())*10000:.1f} bps")

        self.dm = DataManager()
        self.results = {}

    def run_signal_backtest(self, signal_class, signal_params: Dict, signal_name: str, needs_dm: bool = False):
        """
        Run backtest for a single signal.

        Args:
            signal_class: Signal class (e.g., InstitutionalMomentum)
            signal_params: Signal parameters
            signal_name: Name for logging
            needs_dm: Whether signal requires DataManager

        Returns:
            Dict with backtest results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Running backtest: {signal_name}")
        logger.info(f"{'='*60}")

        # Initialize signal
        if needs_dm:
            signal = signal_class(signal_params, data_manager=self.dm)
        else:
            signal = signal_class(signal_params)

        # Initialize portfolio with transaction costs in params dict
        portfolio_params = {
            # Position sizing - use full equity allocation
            'max_position_size': 1.0,  # Allow up to 100% per position (we'll manage via equal weight)
            'max_positions': len(self.universe),  # Number of stocks in universe

            # Risk management - DISABLE tight stops that kill performance
            'stop_loss_pct': None,  # No stop losses for monthly rebalanced portfolios
            'take_profit_pct': None,  # Let winners run

            # Drawdown management - only intervene at portfolio level
            'max_portfolio_drawdown': 0.25,  # 25% max portfolio drawdown
            'drawdown_scale_factor': 0.5,  # Reduce exposure by 50% if in drawdown
        }

        if self.transaction_costs:
            # Merge transaction costs into params
            portfolio_params.update(self.transaction_costs)

        portfolio = Portfolio(
            initial_capital=self.initial_capital,
            params=portfolio_params
        )

        # Generate signals for each stock
        all_signals = {}
        all_prices = {}

        logger.info(f"Generating signals for {len(self.universe)} stocks...")

        for ticker in tqdm(self.universe, desc="Stocks"):
            try:
                # Get price data
                prices = self.dm.get_prices(ticker, self.start_date, self.end_date)

                if len(prices) < 100:
                    logger.warning(f"  {ticker}: Insufficient data ({len(prices)} days)")
                    continue

                # Add ticker column for signals that need it
                prices = prices.copy()
                if 'ticker' not in prices.columns:
                    prices['ticker'] = ticker

                # Generate signals
                signals = signal.generate_signals(prices)

                if signals is not None and len(signals) > 0:
                    all_signals[ticker] = signals
                    all_prices[ticker] = prices['close']

                    non_zero = (signals != 0).sum()
                    logger.debug(f"  {ticker}: {len(signals)} days, {non_zero} non-zero signals")

            except Exception as e:
                logger.error(f"  {ticker}: Error - {str(e)}")
                continue

        logger.info(f"Successfully generated signals for {len(all_signals)} stocks")

        if len(all_signals) == 0:
            logger.error("No signals generated! Cannot run backtest.")
            return None

        # Combine signals and prices into DataFrames
        signals_df = pd.DataFrame(all_signals)
        prices_df = pd.DataFrame(all_prices)

        # Remove duplicate dates (keep last occurrence)
        if signals_df.index.duplicated().any():
            logger.warning(f"Found {signals_df.index.duplicated().sum()} duplicate dates in signals, removing...")
            signals_df = signals_df[~signals_df.index.duplicated(keep='last')]

        if prices_df.index.duplicated().any():
            logger.warning(f"Found {prices_df.index.duplicated().sum()} duplicate dates in prices, removing...")
            prices_df = prices_df[~prices_df.index.duplicated(keep='last')]

        # Apply rebalancing frequency if specified
        rebalance_freq = self.params.get('rebalance_freq', 'M')  # Default monthly
        if rebalance_freq != 'D':  # D = daily (no rebalancing)
            logger.info(f"Applying {rebalance_freq} rebalancing frequency...")
            if rebalance_freq == 'M':
                rebal_signals = signals_df.resample('ME').last()
            elif rebalance_freq == 'W':
                rebal_signals = signals_df.resample('W').last()
            else:
                raise ValueError(f"Unknown rebalance frequency: {rebalance_freq}")

            # Forward-fill to daily
            signals_df = rebal_signals.reindex(signals_df.index, method='ffill').fillna(0)

        # Align indices
        common_index = signals_df.index.intersection(prices_df.index)
        signals_df = signals_df.loc[common_index]
        prices_df = prices_df.loc[common_index]

        logger.info(f"Backtest period: {common_index.min().date()} to {common_index.max().date()}")
        logger.info(f"Trading days: {len(common_index)}")

        # Run backtest
        logger.info("Running portfolio backtest...")

        equity_curve = []
        daily_returns = []

        for date in tqdm(common_index, desc="Backtest"):
            # Get signals and prices for this date
            today_signals = signals_df.loc[date]
            today_prices = prices_df.loc[date]

            # Create signal and price dicts, filtering for valid entries
            signals_dict = {}
            prices_dict = {}

            for ticker in today_signals.index:
                signal_value = today_signals[ticker]
                price_value = today_prices[ticker]

                # Only include if signal is non-zero and price is valid
                if signal_value != 0 and pd.notna(price_value):
                    signals_dict[ticker] = float(signal_value)
                    prices_dict[ticker] = float(price_value)

            # Update portfolio with today's signals and prices
            portfolio.update(date, signals_dict, prices_dict)

            # Track equity
            equity_curve.append(portfolio.get_equity())

        # Calculate returns
        equity_series = pd.Series(equity_curve, index=common_index)
        returns = equity_series.pct_change().fillna(0)

        # Calculate metrics
        total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe = (annual_return - 0.02) / volatility if volatility > 0 else 0

        max_dd = (equity_series / equity_series.expanding().max() - 1).min()

        logger.info(f"\n{signal_name} Results:")
        logger.info(f"  Total Return: {total_return*100:.2f}%")
        logger.info(f"  Annual Return: {annual_return*100:.2f}%")
        logger.info(f"  Volatility: {volatility*100:.2f}%")
        logger.info(f"  Sharpe Ratio: {sharpe:.3f}")
        logger.info(f"  Max Drawdown: {max_dd*100:.2f}%")

        return {
            'signal_name': signal_name,
            'returns': returns,
            'equity_curve': equity_series,
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe': sharpe,
            'max_drawdown': max_dd
        }

    def run_all_signals(self):
        """
        Run backtest for all institutional signals.

        Returns:
            Dict with results for each signal
        """
        results = {}

        # Momentum
        logger.info("\n" + "="*60)
        logger.info("SIGNAL 1/3: INSTITUTIONAL MOMENTUM")
        logger.info("="*60)

        momentum_params = {
            'formation_period': 252,  # 12 months
            'skip_period': 21,        # 1 month
            'quintiles': True
        }

        momentum_results = self.run_signal_backtest(
            InstitutionalMomentum,
            momentum_params,
            "InstitutionalMomentum",
            needs_dm=False  # Momentum doesn't need DataManager
        )

        if momentum_results:
            results['momentum'] = momentum_results

        # Quality
        logger.info("\n" + "="*60)
        logger.info("SIGNAL 2/3: INSTITUTIONAL QUALITY")
        logger.info("="*60)

        quality_params = {
            'use_profitability': True,
            'use_growth': True,
            'use_safety': True
        }

        quality_results = self.run_signal_backtest(
            InstitutionalQuality,
            quality_params,
            "InstitutionalQuality",
            needs_dm=True  # Quality needs DataManager for fundamentals
        )

        if quality_results:
            results['quality'] = quality_results

        # Insider
        logger.info("\n" + "="*60)
        logger.info("SIGNAL 3/3: INSTITUTIONAL INSIDER")
        logger.info("="*60)

        insider_params = {
            'lookback_days': 90,
            'min_transaction_value': 10000,
            'cluster_window': 7,
            'cluster_min_insiders': 3
        }

        insider_results = self.run_signal_backtest(
            InstitutionalInsider,
            insider_params,
            "InstitutionalInsider",
            needs_dm=True  # Insider needs DataManager for transactions
        )

        if insider_results:
            results['insider'] = insider_results

        self.results = results
        return results

    def run_spy_comparison(self):
        """
        Run SPY benchmark analysis for each signal.
        """
        # Get SPY returns
        logger.info("\n" + "="*60)
        logger.info("FETCHING SPY BENCHMARK DATA")
        logger.info("="*60)

        spy_prices = self.dm.get_prices('SPY', self.start_date, self.end_date)
        spy_returns = spy_prices['close'].pct_change().dropna()

        logger.info(f"SPY data: {len(spy_returns)} days")

        # Analyze each signal vs SPY
        for signal_name, signal_results in self.results.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"SPY BENCHMARK: {signal_name.upper()}")
            logger.info(f"{'='*60}")

            analyzer = SPYBenchmarkAnalysis(
                signal_results['returns'],
                spy_returns,
                rf_rate=0.02
            )

            metrics = analyzer.run_full_analysis()

            # Store results
            signal_results['spy_metrics'] = metrics

    def generate_report(self, output_file: str = 'results/institutional_backtest_report.md'):
        """
        Generate comprehensive markdown report.

        Args:
            output_file: Path to save report
        """
        logger.info(f"\nGenerating report: {output_file}")

        report = []
        report.append("# Institutional Signal Backtest Report\n")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**Period:** {self.start_date} to {self.end_date}\n")
        report.append(f"**Universe:** {len(self.universe)} stocks\n")
        report.append(f"**Capital:** ${self.initial_capital:,.0f}\n")
        report.append(f"**Transaction Costs:** {sum(self.transaction_costs.values())*10000:.1f} bps\n")

        report.append("\n---\n\n")
        report.append("## Signal Performance Summary\n\n")

        report.append("| Signal | Return | Annual | Volatility | Sharpe | Max DD |\n")
        report.append("|--------|--------|--------|------------|--------|--------|\n")

        for signal_name, results in self.results.items():
            report.append(
                f"| {signal_name.title()} | "
                f"{results['total_return']*100:.2f}% | "
                f"{results['annual_return']*100:.2f}% | "
                f"{results['volatility']*100:.2f}% | "
                f"{results['sharpe']:.3f} | "
                f"{results['max_drawdown']*100:.2f}% |\n"
            )

        # SPY comparison
        report.append("\n## SPY Benchmark Comparison\n\n")

        for signal_name, results in self.results.items():
            if 'spy_metrics' in results:
                metrics = results['spy_metrics']

                report.append(f"\n### {signal_name.title()}\n\n")
                report.append(f"- **Information Ratio:** {metrics.get('information_ratio', 0):.3f}\n")
                report.append(f"- **Alpha:** {metrics.get('alpha', 0)*100:.2f}% (p={metrics.get('alpha_pvalue', 1):.3f})\n")
                report.append(f"- **Beta:** {metrics.get('beta', 0):.3f}\n")
                report.append(f"- **Win Rate:** {metrics.get('rolling_win_rate', 0)*100:.1f}%\n")

        # Save report
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(''.join(report))

        logger.info(f"✅ Report saved: {output_file}")

        return output_file


def main():
    parser = argparse.ArgumentParser(description='Institutional Signal Backtest')
    parser.add_argument('--universe', default='manual',
                       choices=['manual', 'sp500', 'market_cap'],
                       help='Universe selection method')
    parser.add_argument('--tickers', default='AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA,JPM,JNJ,XOM',
                       help='Comma-separated ticker list (for manual universe)')
    parser.add_argument('--period', default='2020-01-01,2024-12-31',
                       help='Backtest period (start,end)')
    parser.add_argument('--capital', type=float, default=50000,
                       help='Initial capital')
    parser.add_argument('--rebalance', default='M',
                       choices=['M', 'W', 'D'],
                       help='Rebalancing frequency: M=monthly, W=weekly, D=daily')

    args = parser.parse_args()

    # Parse dates
    start_date, end_date = args.period.split(',')

    # Get universe
    dm = DataManager()
    universe_selector = UniverseSelector(dm)

    if args.universe == 'manual':
        universe = universe_selector.get_manual_universe(args.tickers.split(','))
    elif args.universe == 'sp500':
        universe = universe_selector.get_sp500_universe(start_date, end_date)
        if universe is None:
            logger.error("S&P 500 universe not yet implemented")
            return
    else:
        logger.error(f"Universe type '{args.universe}' not yet implemented")
        return

    # Run backtest
    backtest = InstitutionalBacktest(
        universe=universe,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.capital,
        params={'rebalance_freq': args.rebalance}
    )

    # Run all signals
    results = backtest.run_all_signals()

    if not results:
        logger.error("No results generated!")
        return

    # SPY comparison
    backtest.run_spy_comparison()

    # Generate report
    backtest.generate_report()

    logger.info("\n" + "="*60)
    logger.info("✅ BACKTEST COMPLETE")
    logger.info("="*60)
    logger.info("See: results/institutional_backtest_report.md")


if __name__ == '__main__':
    main()
