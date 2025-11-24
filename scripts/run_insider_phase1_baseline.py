"""
Insider Trading Signal - Phase 1 Baseline Backtest

Run baseline performance test for InstitutionalInsider signal on S&P 500 PIT universe.

Tests two portfolio strategies:
1. Long-only top quintile (high insider buying)
2. Long-short factor (top decile vs bottom decile)

Usage:
    python3 scripts/run_insider_phase1_baseline.py

Outputs:
    results/INSIDER_PHASE1_BASELINE.md - Performance summary
    results/INSIDER_PHASE1_DECILES.csv - Decile-level returns
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from data.data_manager import DataManager
from core.universe_manager import UniverseManager
from signals.insider.institutional_insider import InstitutionalInsider
from core.schedules import get_rebalance_dates
from validation.simple_validation import compute_basic_metrics, compute_regime_metrics
from config import get_logger

logger = get_logger(__name__)


class InsiderPhase1Baseline:
    """
    Phase 1 baseline backtest for InstitutionalInsider signal.

    Builds decile portfolios and computes long-only and long-short performance.
    """

    def __init__(self,
                 start_date: str = '2015-04-01',
                 end_date: str = '2024-12-31',
                 initial_capital: float = 50000,
                 use_bulk_insiders: bool = True,
                 debug_compare_bulk: bool = False):
        """
        Initialize Insider Phase 1 baseline.

        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital in dollars
            use_bulk_insiders: If True, prefetch all insider data in bulk (50-100x faster)
            debug_compare_bulk: If True, compare bulk vs legacy signals for consistency
        """
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.use_bulk_insiders = use_bulk_insiders
        self.debug_compare_bulk = debug_compare_bulk

        self.dm = DataManager()
        self.um = UniverseManager(self.dm)

        # Initialize InstitutionalInsider with default Phase 1 parameters
        self.insider_params = {
            'lookback_days': 90,
            'min_transaction_value': 10000,
            'cluster_window': 7,
            'cluster_min_insiders': 3,
            'ceo_weight': 3.0,
            'cfo_weight': 2.5,
            'winsorize_pct': [5, 95],
            'rebalance_frequency': 'monthly'
        }
        self.insider = InstitutionalInsider(self.insider_params, data_manager=self.dm)

        logger.info("=" * 80)
        logger.info("Insider Phase 1 Baseline Backtest")
        logger.info("=" * 80)
        logger.info(f"Period: {self.start_date} to {self.end_date}")
        logger.info(f"Universe: S&P 500 PIT (sp500_actual)")
        logger.info(f"Capital: ${initial_capital:,.0f}")
        logger.info(f"Bulk insider mode: {self.use_bulk_insiders}")
        logger.info(f"Debug comparison: {self.debug_compare_bulk}")
        logger.info(f"Parameters: {self.insider_params}")
        logger.info("=" * 80)

    def run_baseline(self) -> Dict:
        """
        Run Phase 1 baseline backtest.

        Returns:
            Dict with baseline results
        """
        results = {
            'test_date': datetime.now().isoformat(),
            'period': f"{self.start_date} to {self.end_date}",
            'signal': 'InstitutionalInsider',
            'parameters': self.insider_params
        }

        # 1. Build decile portfolios
        logger.info("\n1. Building decile portfolios...")
        decile_data = self._build_decile_portfolios()
        results['decile_portfolios'] = decile_data

        # 2. Build long-only top quintile
        logger.info("\n2. Building long-only top quintile portfolio...")
        long_only_equity = self._build_long_only_portfolio(decile_data)
        results['long_only_equity'] = long_only_equity

        # 3. Build long-short factor portfolio
        logger.info("\n3. Building long-short factor portfolio...")
        long_short_equity = self._build_long_short_portfolio(decile_data)
        results['long_short_equity'] = long_short_equity

        # 4. Compute metrics
        logger.info("\n4. Computing performance metrics...")
        results['long_only_metrics'] = compute_basic_metrics(long_only_equity)
        results['long_short_metrics'] = compute_basic_metrics(long_short_equity)

        # 5. Regime metrics
        logger.info("\n5. Computing regime metrics...")
        results['long_short_regimes'] = compute_regime_metrics(long_short_equity)

        # 6. Decile summary
        logger.info("\n6. Computing decile summary...")
        results['decile_summary'] = self._compute_decile_summary(decile_data)

        # 7. Save results
        logger.info("\n7. Saving results...")
        self._save_baseline_results(results)

        logger.info("\n" + "=" * 80)
        logger.info("Phase 1 Baseline Complete")
        logger.info("=" * 80)

        return results

    def _build_decile_portfolios(self) -> Dict:
        """
        Build all 10 decile portfolios based on Insider scores.

        Returns:
            Dict with decile equity curves
        """
        # Get S&P 500 PIT universe
        logger.info(f"Building S&P 500 PIT universe as of {self.start_date}...")
        universe = self.um.get_universe(
            universe_type='sp500_actual',
            as_of_date=self.start_date,
            min_price=5.0
        )
        logger.info(f"Universe: {len(universe)} stocks")

        # Get monthly rebalance dates
        rebalance_dates = get_rebalance_dates(
            schedule='M',
            dm=self.dm,
            start_date=self.start_date,
            end_date=self.end_date
        )
        logger.info(f"Rebalance dates: {len(rebalance_dates)} month-ends")

        # Prefetch bulk insider data if enabled (50-100x speedup)
        bulk_insider_data = None
        if self.use_bulk_insiders:
            logger.info("Prefetching bulk insider data for entire universe...")
            universe_list = sorted(list(universe))

            # Extend date range by lookback to ensure we have enough data
            from datetime import datetime, timedelta
            start_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
            extended_start = (start_dt - timedelta(days=self.insider_params['lookback_days'])).strftime('%Y-%m-%d')

            bulk_insider_data = self.dm.get_insider_trades_bulk(
                tickers=universe_list,
                start_date=extended_start,
                end_date=self.end_date,
                as_of_date=self.end_date
            )
            logger.info(f"Prefetched {len(bulk_insider_data)} insider transactions across {len(universe_list)} tickers")

        # Get price data for all stocks
        logger.info("Fetching price data...")
        all_prices = {}
        for ticker in universe:
            try:
                prices = self.dm.get_prices(ticker, self.start_date, self.end_date)
                if len(prices) > 0:
                    all_prices[ticker] = prices['close']
            except:
                pass
        prices_df = pd.DataFrame(all_prices)
        logger.info(f"Price data: {prices_df.shape}")

        # Build decile equity curves
        decile_equity = {i: pd.Series(dtype=float) for i in range(1, 11)}

        # Track portfolio over time
        for i in range(len(rebalance_dates) - 1):
            rebal_date = pd.Timestamp(rebalance_dates[i])
            next_rebal_date = pd.Timestamp(rebalance_dates[i + 1])

            logger.info(f"Rebalance {i+1}/{len(rebalance_dates)-1}: {rebal_date.date()}")

            # Generate insider scores for this rebalance date
            scores = {}

            # Debug comparison: On first 3 rebalances, compare bulk vs legacy for first 5 tickers
            debug_this_rebalance = self.debug_compare_bulk and i < 3

            for ticker in universe:
                if ticker not in all_prices:
                    continue

                # Get price history up to rebalance date
                ticker_prices = prices_df[ticker][prices_df.index <= rebal_date].dropna()

                if len(ticker_prices) < 90:  # Need enough history
                    continue

                # Generate signal
                try:
                    data = pd.DataFrame({'close': ticker_prices, 'ticker': ticker})

                    # Generate signal with bulk data (if enabled)
                    signal = self.insider.generate_signals(data, bulk_insider_data=bulk_insider_data)

                    # Debug comparison: Compare bulk vs legacy on first few tickers
                    if debug_this_rebalance and len(scores) < 5:
                        # Generate legacy signal (without bulk data)
                        signal_legacy = self.insider.generate_signals(data, bulk_insider_data=None)

                        # Compare signals
                        if len(signal) > 0 and len(signal_legacy) > 0:
                            max_diff = (signal - signal_legacy).abs().max()
                            logger.info(f"  DEBUG: {ticker} signal comparison: max abs diff = {max_diff:.2e}")

                            if max_diff > 1e-9:
                                logger.warning(f"  WARNING: Bulk vs legacy signals differ for {ticker}!")
                            else:
                                logger.info(f"  DEBUG: {ticker} signals match ✓")

                    # Get signal value at rebalance date
                    if len(signal) > 0 and not signal.iloc[-1] == 0:
                        scores[ticker] = signal.iloc[-1]
                except Exception as e:
                    logger.debug(f"Error generating signal for {ticker}: {e}")
                    continue

            if len(scores) < 50:
                logger.warning(f"Only {len(scores)} stocks with signals, skipping rebalance")
                continue

            # Sort by score and form deciles
            ranked = pd.Series(scores).sort_values(ascending=False)
            decile_size = len(ranked) // 10

            deciles = {}
            for d in range(1, 11):
                start_idx = (d - 1) * decile_size
                end_idx = start_idx + decile_size if d < 10 else len(ranked)
                deciles[d] = ranked.iloc[start_idx:end_idx].index.tolist()

            logger.info(f"  Decile sizes: D1={len(deciles[1])}, D5={len(deciles[5])}, D10={len(deciles[10])}")

            # Track each decile's performance until next rebalance
            holding_period = prices_df[
                (prices_df.index > rebal_date) &
                (prices_df.index <= next_rebal_date)
            ]

            for decile in range(1, 11):
                decile_tickers = deciles[decile]
                decile_prices = holding_period[decile_tickers].dropna(axis=1, how='all')

                if decile_prices.empty:
                    continue

                # Equal-weight portfolio returns
                decile_returns = decile_prices.pct_change().mean(axis=1)

                # Compound returns
                if len(decile_equity[decile]) == 0:
                    # Initialize at rebalance date
                    decile_equity[decile] = pd.Series(
                        self.initial_capital,
                        index=[rebal_date]
                    )

                # Build equity curve over holding period
                for date, ret in decile_returns.items():
                    if pd.notna(ret):
                        last_value = decile_equity[decile].iloc[-1]
                        decile_equity[decile].loc[date] = last_value * (1 + ret)

        logger.info(f"Built {len(decile_equity)} decile portfolios")
        return decile_equity

    def _build_long_only_portfolio(self, decile_data: Dict) -> pd.Series:
        """
        Build long-only top quintile portfolio (top 20% = deciles 1-2).

        Args:
            decile_data: Dict with decile equity curves

        Returns:
            Equity curve Series
        """
        # Combine top 2 deciles (top quintile = top 20%)
        top_quintile = (decile_data[1] + decile_data[2]) / 2

        logger.info(f"Long-only portfolio: {len(top_quintile)} days")
        return top_quintile

    def _build_long_short_portfolio(self, decile_data: Dict) -> pd.Series:
        """
        Build long-short factor portfolio (long D1, short D10).

        Args:
            decile_data: Dict with decile equity curves

        Returns:
            Equity curve Series
        """
        # Long top decile, short bottom decile
        long_leg = decile_data[1]
        short_leg = decile_data[10]

        # Align dates
        common_dates = long_leg.index.intersection(short_leg.index)

        # Long-short returns
        long_returns = long_leg.loc[common_dates].pct_change()
        short_returns = short_leg.loc[common_dates].pct_change()

        # 50% long, 50% short (dollar-neutral)
        ls_returns = 0.5 * long_returns - 0.5 * short_returns

        # Build equity curve
        ls_equity = (1 + ls_returns).cumprod() * self.initial_capital
        ls_equity.iloc[0] = self.initial_capital

        logger.info(f"Long-short portfolio: {len(ls_equity)} days")
        return ls_equity

    def _compute_decile_summary(self, decile_data: Dict) -> pd.DataFrame:
        """
        Compute summary statistics for each decile.

        Args:
            decile_data: Dict with decile equity curves

        Returns:
            DataFrame with decile metrics
        """
        summary = []

        for decile in range(1, 11):
            equity = decile_data[decile]

            if len(equity) < 2:
                continue

            metrics = compute_basic_metrics(equity)

            summary.append({
                'decile': decile,
                'total_return': metrics['total_return'],
                'annual_return': metrics['annual_return'],
                'volatility': metrics['volatility'],
                'sharpe': metrics['sharpe'],
                'max_drawdown': metrics['max_drawdown']
            })

        return pd.DataFrame(summary)

    def _save_baseline_results(self, results: Dict):
        """
        Save baseline results to files.

        Args:
            results: Dict with all results
        """
        # Save decile summary CSV
        decile_df = results['decile_summary']
        decile_path = Path('results/INSIDER_PHASE1_DECILES.csv')
        decile_df.to_csv(decile_path, index=False)
        logger.info(f"Saved decile summary: {decile_path}")

        # Generate baseline markdown report
        md_path = Path('results/INSIDER_PHASE1_BASELINE.md')

        with open(md_path, 'w') as f:
            f.write("# Insider Phase 1 Baseline Results\n\n")
            f.write(f"**Date:** {results['test_date'][:10]}\n")
            f.write(f"**Period:** {results['period']}\n")
            f.write(f"**Signal:** {results['signal']}\n\n")
            f.write("---\n\n")

            # Parameters
            f.write("## Configuration\n\n")
            f.write("**Parameters:**\n")
            for key, val in results['parameters'].items():
                f.write(f"- {key}: {val}\n")
            f.write("\n")

            # Long-only metrics
            f.write("## Long-Only Top Quintile\n\n")
            lo_metrics = results['long_only_metrics']
            f.write(f"- **Total Return:** {lo_metrics['total_return']*100:.2f}%\n")
            f.write(f"- **Annual Return:** {lo_metrics['annual_return']*100:.2f}%\n")
            f.write(f"- **Volatility:** {lo_metrics['volatility']*100:.2f}%\n")
            f.write(f"- **Sharpe Ratio:** {lo_metrics['sharpe']:.3f}\n")
            f.write(f"- **Max Drawdown:** {lo_metrics['max_drawdown']*100:.2f}%\n")
            f.write(f"- **Days:** {lo_metrics['num_days']}\n\n")

            # Long-short metrics
            f.write("## Long-Short Factor Portfolio\n\n")
            ls_metrics = results['long_short_metrics']
            f.write(f"- **Total Return:** {ls_metrics['total_return']*100:.2f}%\n")
            f.write(f"- **Annual Return:** {ls_metrics['annual_return']*100:.2f}%\n")
            f.write(f"- **Volatility:** {ls_metrics['volatility']*100:.2f}%\n")
            f.write(f"- **Sharpe Ratio:** {ls_metrics['sharpe']:.3f}\n")
            f.write(f"- **Max Drawdown:** {ls_metrics['max_drawdown']*100:.2f}%\n")
            f.write(f"- **Days:** {ls_metrics['num_days']}\n\n")

            # Regime performance
            f.write("## Regime Performance (Long-Short)\n\n")
            f.write("| Regime | Mean Return | Sharpe | Num Months |\n")
            f.write("|--------|-------------|--------|------------|\n")
            for regime, metrics in results['long_short_regimes'].items():
                f.write(f"| {regime} | {metrics['mean_return']*100:.2f}% | "
                       f"{metrics['sharpe']:.3f} | {metrics['num_months']} |\n")
            f.write("\n")

            # Decile table
            f.write("## Decile Returns\n\n")
            f.write("| Decile | Annual Return | Sharpe | Max DD |\n")
            f.write("|--------|---------------|--------|--------|\n")
            for _, row in results['decile_summary'].iterrows():
                f.write(f"| D{int(row['decile'])} | {row['annual_return']*100:.2f}% | "
                       f"{row['sharpe']:.3f} | {row['max_drawdown']*100:.2f}% |\n")
            f.write("\n")

            f.write("---\n\n")
            f.write(f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        logger.info(f"Saved baseline report: {md_path}")


def main():
    """Run Insider Phase 1 baseline backtest."""
    baseline = InsiderPhase1Baseline(
        start_date='2015-04-01',
        end_date='2024-12-31',
        initial_capital=50000
    )

    results = baseline.run_baseline()

    # Print summary
    print("\n" + "=" * 80)
    print("INSIDER PHASE 1 BASELINE - SUMMARY")
    print("=" * 80)
    print(f"\nLong-Only Top Quintile:")
    print(f"  Sharpe: {results['long_only_metrics']['sharpe']:.3f}")
    print(f"  Annual Return: {results['long_only_metrics']['annual_return']*100:.2f}%")
    print(f"\nLong-Short Factor Portfolio:")
    print(f"  Sharpe: {results['long_short_metrics']['sharpe']:.3f}")
    print(f"  Annual Return: {results['long_short_metrics']['annual_return']*100:.2f}%")
    print(f"\nDecile Spread (D1 - D10):")
    d1_return = results['decile_summary'].iloc[0]['annual_return']
    d10_return = results['decile_summary'].iloc[-1]['annual_return']
    print(f"  {(d1_return - d10_return)*100:.2f}% per year")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
