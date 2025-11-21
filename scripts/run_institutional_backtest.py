"""
Institutional Signal Backtest Runner

Runs comprehensive backtest of institutional signals with SPY benchmark comparison.

Features:
- Flexible universe selection with point-in-time correctness:
  * manual: Explicit ticker list (good for testing)
  * sp500_proxy: Top 500 by market cap (S&P 500 approximation)
  * sp1000_proxy: Top 1000 by market cap (Russell 1000 approximation)
  * nasdaq_proxy: Technology + Communication Services sectors
  * top_N: Top N stocks by market cap
  * market_cap_range: Filter by market cap range (large/mid/small cap)
  * sector: Filter by GICS sector
- Point-in-time data access (no lookahead bias)
- Realistic transaction costs for $50K Schwab account
- Full SPY benchmark analysis
- Comprehensive reporting

Usage Examples:
    # Manual universe (10 stocks - good for testing)
    python3 scripts/run_institutional_backtest.py \
        --universe manual \
        --tickers AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA,JPM,JNJ,XOM

    # S&P 500 proxy (top 500 by market cap)
    python3 scripts/run_institutional_backtest.py \
        --universe sp500_proxy \
        --period 2020-01-01,2024-12-31

    # Top 100 stocks by market cap
    python3 scripts/run_institutional_backtest.py \
        --universe top_N \
        --top-n 100

    # Large cap only (>$10B)
    python3 scripts/run_institutional_backtest.py \
        --universe market_cap_range \
        --min-mcap 10000000000

    # Technology sector only
    python3 scripts/run_institutional_backtest.py \
        --universe sector \
        --sectors Technology

Monitor:
    # Terminal 1: Run backtest
    python3 scripts/run_institutional_backtest.py ...

    # Terminal 2: Monitor logs
    tail -f logs/signaltide_development.log

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
from core.universe_manager import UniverseManager
from core.schedules import get_rebalance_dates
from core.manifest import BacktestManifest
from scripts.spy_benchmark_analysis import SPYBenchmarkAnalysis
from config import get_logger, DEFAULT_TRANSACTION_COSTS

logger = get_logger(__name__)


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

    def _build_manifest(
        self,
        signal_class,
        signal_params: Dict,
        signal_name: str
    ) -> BacktestManifest:
        """
        Build a backtest manifest from current run context.

        Args:
            signal_class: Signal class being backtested
            signal_params: Parameters passed to the signal
            signal_name: Name of the signal

        Returns:
            BacktestManifest instance with full run metadata
        """
        # Extract universe type and params from self
        # For simplicity, we'll use 'manual' if we have a list of tickers
        universe_type = 'manual'
        universe_params = {'tickers': self.universe}

        # Build signal specification
        signals = [{
            'name': signal_class.__name__,
            'module': f"{signal_class.__module__}",
            'params': signal_params.copy()
        }]

        # Get rebalance schedule from params (default to monthly)
        rebalance_schedule = self.params.get('rebalance_freq', 'M')

        # Build manifest
        return BacktestManifest.from_context(
            dm=self.dm,
            start_date=self.start_date,
            end_date=self.end_date,
            universe_type=universe_type,
            universe_params=universe_params,
            signals=signals,
            initial_capital=self.initial_capital,
            rebalance_schedule=rebalance_schedule,
            transaction_costs=self.transaction_costs.copy()
        )

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

        # Apply rebalancing frequency if specified using trading calendar
        # All rebalancing dates come from dim_trading_calendar to ensure we never
        # rebalance on weekends/holidays and always use month-end trading days.
        rebalance_freq = self.params.get('rebalance_freq', 'M')  # Default monthly
        if rebalance_freq != 'D':  # D = daily (no rebalancing)
            logger.info(f"Applying {rebalance_freq} rebalancing frequency using trading calendar...")

            # Get rebalance dates from trading calendar (respects holidays/weekends)
            rebal_dates = get_rebalance_dates(
                schedule=rebalance_freq,
                dm=self.dm,
                start_date=self.start_date,
                end_date=self.end_date
            )

            # Convert to pandas datetime index
            rebal_index = pd.to_datetime(rebal_dates)

            # Sample signals on rebalance dates only
            rebal_signals = signals_df.reindex(rebal_index, method='ffill')

            # Forward-fill to daily (each rebalance signal holds until next rebalance)
            signals_df = rebal_signals.reindex(signals_df.index, method='ffill').fillna(0)

            logger.info(f"Using {len(rebal_dates)} rebalance dates from {rebal_dates[0]} to {rebal_dates[-1]}")

        # Align indices
        common_index = signals_df.index.intersection(prices_df.index)
        signals_df = signals_df.loc[common_index]
        prices_df = prices_df.loc[common_index]

        logger.info(f"Backtest period: {common_index.min().date()} to {common_index.max().date()}")
        logger.info(f"Trading days: {len(common_index)}")

        # Run backtest
        # IMPORTANT: All trading date logic must go through DataManager's trading calendar helpers.
        # - Rebalance dates come from get_rebalance_dates() which uses dim_trading_calendar
        # - Never derive "trading days" from price indices or pd.date_range directly
        # - This ensures consistency with market holidays and month-end calculations
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

        # Build backtest manifest for reproducibility
        manifest = self._build_manifest(
            signal_class=signal_class,
            signal_params=signal_params,
            signal_name=signal_name
        )

        # Log manifest summary
        logger.info(f"\nBacktest manifest: run_id={manifest.run_id[:8]}... "
                    f"period={manifest.start_date}..{manifest.end_date} "
                    f"universe={manifest.universe_type} "
                    f"signals={[s['name'] for s in manifest.to_dict()['signals']]}")

        return {
            'signal_name': signal_name,
            'returns': returns,
            'equity_curve': equity_series,
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'manifest': manifest,
            'manifest_dict': manifest.to_dict()
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
    parser = argparse.ArgumentParser(
        description='Institutional Signal Backtest',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Manual universe (10 stocks)
  %(prog)s --universe manual --tickers AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA,JPM,JNJ,XOM

  # Top 500 by market cap (S&P 500 proxy)
  %(prog)s --universe sp500_proxy --period 2020-01-01,2024-12-31

  # Top 1000 by market cap (Russell 1000 proxy)
  %(prog)s --universe sp1000_proxy

  # Top 100 by market cap
  %(prog)s --universe top_N --top-n 100

  # Technology sector only
  %(prog)s --universe sector --sectors Technology

  # Large cap stocks (>$10B)
  %(prog)s --universe market_cap_range --min-mcap 10000000000
        """
    )

    parser.add_argument('--universe', default='manual',
                       choices=['manual', 'top_N', 'market_cap_range', 'sector',
                               'sp500_proxy', 'sp1000_proxy', 'nasdaq_proxy'],
                       help='Universe selection method')
    parser.add_argument('--tickers', default='AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA,JPM,JNJ,XOM',
                       help='Comma-separated ticker list (for manual universe)')
    parser.add_argument('--top-n', type=int,
                       help='Number of top stocks by market cap (for top_N universe)')
    parser.add_argument('--min-mcap', type=float,
                       help='Minimum market cap in USD (for market_cap_range)')
    parser.add_argument('--max-mcap', type=float,
                       help='Maximum market cap in USD (for market_cap_range)')
    parser.add_argument('--sectors',
                       help='Comma-separated GICS sectors (for sector universe)')
    parser.add_argument('--min-price', type=float, default=5.0,
                       help='Minimum stock price to avoid penny stocks (default: $5)')
    parser.add_argument('--period', default='2020-01-01,2024-12-31',
                       help='Backtest period (start,end)')
    parser.add_argument('--capital', type=float, default=50000,
                       help='Initial capital')
    parser.add_argument('--rebalance', default='M',
                       help='Rebalancing frequency (case-insensitive). Accepted values: '\
                            'M/ME/monthly (month-end, default), '\
                            'W/weekly (Fridays), '\
                            'D/daily (every trading day). '\
                            'All dates respect NYSE trading calendar.')

    args = parser.parse_args()

    # Parse dates
    start_date, end_date = args.period.split(',')

    # Get universe
    dm = DataManager()
    universe_manager = UniverseManager(dm)

    logger.info(f"\n{'='*60}")
    logger.info(f"UNIVERSE CONSTRUCTION")
    logger.info(f"{'='*60}")
    logger.info(f"Type: {args.universe}")
    logger.info(f"Date: {start_date}")

    # Build universe based on type
    if args.universe == 'manual':
        tickers = args.tickers.split(',')
        universe = universe_manager.get_universe(
            universe_type='manual',
            as_of_date=start_date,
            manual_tickers=tickers
        )

    elif args.universe == 'top_N':
        if not args.top_n:
            logger.error("--top-n required for top_N universe")
            return
        universe = universe_manager.get_universe(
            universe_type='top_N',
            as_of_date=start_date,
            top_n=args.top_n,
            min_price=args.min_price
        )

    elif args.universe == 'market_cap_range':
        if args.min_mcap is None and args.max_mcap is None:
            logger.error("--min-mcap or --max-mcap required for market_cap_range universe")
            return
        universe = universe_manager.get_universe(
            universe_type='market_cap_range',
            as_of_date=start_date,
            min_market_cap=args.min_mcap,
            max_market_cap=args.max_mcap,
            min_price=args.min_price
        )

    elif args.universe == 'sector':
        if not args.sectors:
            logger.error("--sectors required for sector universe")
            return
        sectors = args.sectors.split(',')
        universe = universe_manager.get_universe(
            universe_type='sector',
            as_of_date=start_date,
            sectors=sectors,
            min_price=args.min_price
        )

    elif args.universe == 'sp500_proxy':
        logger.info("Using top 500 by market cap as S&P 500 proxy")
        universe = universe_manager.get_universe(
            universe_type='sp500_proxy',
            as_of_date=start_date,
            min_price=args.min_price
        )

    elif args.universe == 'sp1000_proxy':
        logger.info("Using top 1000 by market cap as Russell 1000 proxy")
        universe = universe_manager.get_universe(
            universe_type='sp1000_proxy',
            as_of_date=start_date,
            min_price=args.min_price
        )

    elif args.universe == 'nasdaq_proxy':
        logger.info("Using Technology + Communication Services as NASDAQ proxy")
        universe = universe_manager.get_universe(
            universe_type='nasdaq_proxy',
            as_of_date=start_date,
            min_price=args.min_price
        )

    else:
        logger.error(f"Unknown universe type: {args.universe}")
        return

    if not universe:
        logger.error("No stocks in universe!")
        return

    logger.info(f"Universe size: {len(universe)} stocks")

    # Show universe info
    universe_info = universe_manager.get_universe_info(universe, start_date)
    if len(universe_info) > 0:
        logger.info(f"Market cap range: ${universe_info['marketcap'].min()/1e9:.1f}B - ${universe_info['marketcap'].max()/1e9:.1f}B")
        logger.info(f"Sectors: {universe_info['sector'].value_counts().to_dict()}")

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
