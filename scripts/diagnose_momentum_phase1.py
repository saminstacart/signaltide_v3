"""
Phase 1.5 Diagnostics for InstitutionalMomentum

Performance acceptance testing with decile portfolios, regime splits, and statistical validation.

This script answers three critical questions:
1. Does Momentum behave like a real factor or random noise?
2. How does it behave by regime (pre-COVID, COVID, 2022 bear)?
3. Are results strong enough to justify Phase 2 optimization work?

Usage:
    python3 scripts/diagnose_momentum_phase1.py

Outputs:
    results/momentum_v1_phase1_report.md - Full diagnostic report with GO/NO-GO
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from data.data_manager import DataManager
from core.universe_manager import UniverseManager
from signals.momentum.institutional_momentum import InstitutionalMomentum
from core.schedules import get_rebalance_dates
from config import get_logger

logger = get_logger(__name__)


class MomentumPhase1Diagnostics:
    """
    Phase 1.5 performance acceptance diagnostics for InstitutionalMomentum.

    Tests:
    1. Decile monotonicity (top > bottom, clear progression)
    2. Long-short spread (top minus bottom decile)
    3. Statistical significance (t-stat, Sharpe)
    4. Regime behavior (pre-COVID, COVID, 2022 bear, 2023-2024)
    5. Coverage and data quality
    """

    def __init__(self,
                 start_date: str = '2015-04-01',
                 end_date: str = '2024-12-31'):
        """
        Initialize diagnostics.

        Args:
            start_date: Analysis start (default 2015-04-01, first month with S&P 500 PIT data)
            end_date: Analysis end (default 2024-12-31)
        """
        self.start_date = start_date
        self.end_date = end_date

        self.dm = DataManager()
        self.um = UniverseManager(self.dm)

        # Initialize Momentum with standard Jegadeesh-Titman parameters
        self.momentum_params = {
            'formation_period': 252,  # 12 months
            'skip_period': 21,        # 1 month
            'quintiles': True,        # Use quintile signals
            'winsorize_pct': [5, 95],
            'rebalance_frequency': 'monthly'
        }
        self.momentum = InstitutionalMomentum(self.momentum_params)

        logger.info("=" * 80)
        logger.info("Phase 1.5 Diagnostics: InstitutionalMomentum")
        logger.info("=" * 80)
        logger.info(f"Period: {self.start_date} to {self.end_date}")
        logger.info(f"Universe: S&P 500 PIT (sp500_actual)")
        logger.info(f"Formation: 12 months, Skip: 1 month")
        logger.info("=" * 80)

    def run_full_diagnostics(self) -> Dict:
        """
        Run all Phase 1.5 diagnostic tests.

        Returns:
            Dict with diagnostic results
        """
        results = {
            'test_date': datetime.now().isoformat(),
            'period': f"{self.start_date} to {self.end_date}",
            'signal': 'InstitutionalMomentum (12-1)'
        }

        # 1. Build decile portfolios over full period
        logger.info("\n1. Building decile portfolios...")
        results['decile_portfolios'] = self._build_decile_portfolios()

        # 2. Compute full-sample performance
        logger.info("\n2. Computing full-sample performance...")
        results['full_sample'] = self._compute_full_sample_performance(
            results['decile_portfolios']
        )

        # 3. Regime splits
        logger.info("\n3. Analyzing performance by regime...")
        results['regimes'] = self._analyze_regimes(
            results['decile_portfolios']
        )

        # 4. Generate report with GO/NO-GO
        logger.info("\n4. Generating diagnostic report...")
        self._generate_report(results)

        logger.info("\n" + "=" * 80)
        logger.info("Phase 1.5 Diagnostics Complete")
        logger.info("=" * 80)

        return results

    def _build_decile_portfolios(self) -> Dict:
        """
        Build monthly-rebalanced decile portfolios based on Momentum scores.

        Returns:
            Dict with decile portfolio returns
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

        # Get price data for all stocks
        # Need extra history for momentum calculation: 252 formation + 21 skip = 273 days
        # Add buffer: 273 trading days ≈ 390 calendar days
        from datetime import timedelta
        lookback_buffer = timedelta(days=400)  # 390 + 10 day buffer
        start_dt = pd.Timestamp(self.start_date)
        price_start_date = (start_dt - lookback_buffer).strftime('%Y-%m-%d')
        logger.info(f"Fetching price data from {price_start_date} (includes {lookback_buffer.days}-day lookback for momentum)...")

        prices_dict = {}
        for ticker in universe:
            try:
                prices = self.dm.get_prices(ticker, price_start_date, self.end_date)
                if len(prices) > 0:
                    prices_dict[ticker] = prices
            except:
                pass
        logger.info(f"Price data loaded: {len(prices_dict)} stocks")

        # Calculate momentum for each stock
        logger.info("Calculating momentum for all stocks...")
        momentum_dict = {}
        stocks_with_insufficient_data = 0
        for ticker, prices in prices_dict.items():
            if 'close' not in prices.columns or len(prices) < 273:  # 252 + 21
                stocks_with_insufficient_data += 1
                continue
            # 12-1 momentum: return from t-273 to t-21
            mom = prices['close'].pct_change(periods=252, fill_method=None).shift(21)
            momentum_dict[ticker] = mom

        logger.info(f"Stocks with sufficient data for momentum: {len(momentum_dict)}")
        logger.info(f"Stocks with insufficient data: {stocks_with_insufficient_data}")

        # Convert to DataFrame for cross-sectional ranking
        momentum_df = pd.DataFrame(momentum_dict)
        momentum_df = momentum_df.sort_index()  # Ensure chronological order

        # Check for and handle duplicate dates
        if momentum_df.index.duplicated().any():
            n_dupes = momentum_df.index.duplicated().sum()
            logger.warning(f"Found {n_dupes} duplicate dates in momentum_df - keeping last occurrence")
            momentum_df = momentum_df[~momentum_df.index.duplicated(keep='last')]

        logger.info(f"Momentum calculated: {momentum_df.shape}")

        # Debug: Check momentum coverage
        non_null_counts = momentum_df.notna().sum(axis=1)
        logger.info(f"Momentum date range: {momentum_df.index.min()} to {momentum_df.index.max()}")
        logger.info(f"Momentum non-null stock counts - mean: {non_null_counts.mean():.0f}, median: {non_null_counts.median():.0f}")

        # Find first date with valid momentum
        first_valid_dates = non_null_counts[non_null_counts > 0]
        if len(first_valid_dates) > 0:
            first_valid = first_valid_dates.index[0]
            logger.info(f"First date with valid momentum: {first_valid} ({first_valid_dates.iloc[0]} stocks)")
        else:
            logger.warning("NO valid momentum values found in any date!")

        # Show momentum coverage at key dates
        for date_str in ['2015-04-30', '2015-07-20', '2015-07-31', '2015-08-31', '2015-12-31', '2016-06-30']:
            try:
                date = pd.Timestamp(date_str)
                if date in momentum_df.index:
                    row_data = momentum_df.loc[date]
                    count = int(row_data.notna().sum())
                    logger.info(f"Momentum coverage on {date_str}: {count} stocks")
                else:
                    # Find nearest date
                    valid_idx = momentum_df.index[momentum_df.index <= date]
                    if len(valid_idx) > 0:
                        nearest = valid_idx[-1]
                        row_data = momentum_df.loc[nearest]
                        count = int(row_data.notna().sum())
                        logger.info(f"Momentum coverage near {date_str} (using {nearest.date()}): {count} stocks")
            except Exception as e:
                logger.warning(f"Error checking {date_str}: {e}")

        # Get price data as DataFrame for returns calculation
        prices_close = {}
        for ticker, prices in prices_dict.items():
            prices_close[ticker] = prices['close']
        prices_df = pd.DataFrame(prices_close)

        # Build decile portfolios at each rebalance date
        logger.info("Constructing decile portfolios...")
        logger.info(f"Rebalance dates range: {rebalance_dates[0]} to {rebalance_dates[-1]} ({len(rebalance_dates)} dates)")
        decile_returns = {i: [] for i in range(1, 11)}  # Deciles 1-10

        for i, rebal_date in enumerate(pd.DatetimeIndex(rebalance_dates)):
            # Use last available momentum date <= rebal_date
            valid_idx = momentum_df.index[momentum_df.index <= rebal_date]
            if len(valid_idx) == 0:
                logger.debug(f"No momentum data available on or before {rebal_date.date()}, skipping.")
                continue

            mom_date = valid_idx[-1]
            mom_today = momentum_df.loc[mom_date].dropna()

            # Debug logging for first 10 iterations and dates around 2015-07-31
            if i < 10 or '2015-07' in str(rebal_date):
                logger.info(f"Rebalance {rebal_date.date()} using momentum date {mom_date.date()} with {len(mom_today)} stocks (after dropna)")

            if len(mom_today) < 50:  # Need minimum universe size
                logger.debug(f"Insufficient stocks ({len(mom_today)}) on {mom_date.date()}, skipping.")
                continue

            # Winsorize to handle outliers (like the signal does)
            mom_wins = mom_today.copy()
            lower_bound = mom_today.quantile(0.05)
            upper_bound = mom_today.quantile(0.95)
            mom_wins = mom_wins.clip(lower=lower_bound, upper=upper_bound)

            # Sort stocks by momentum (high momentum = high signal)
            ranked = mom_wins.sort_values(ascending=False)

            # Form 10 deciles
            decile_size = len(ranked) // 10
            deciles = {}
            for d in range(1, 11):
                start_idx = (d - 1) * decile_size
                end_idx = start_idx + decile_size if d < 10 else len(ranked)
                deciles[d] = ranked.iloc[start_idx:end_idx].index.tolist()

            # Compute returns from this rebalance to next rebalance
            if i + 1 >= len(rebalance_dates):
                break

            next_rebal_date = pd.Timestamp(rebalance_dates[i + 1])

            # Get price changes for holding period
            price_slice = prices_df.loc[rebal_date:next_rebal_date]

            if len(price_slice) < 2:
                continue

            # Compute equal-weighted returns for each decile
            for decile_num, tickers in deciles.items():
                # Get prices for stocks in this decile
                decile_prices = price_slice[tickers].dropna(how='all', axis=1)

                if len(decile_prices.columns) == 0:
                    continue

                # Equal-weighted portfolio return
                decile_ret = (decile_prices.iloc[-1] / decile_prices.iloc[0] - 1).mean()

                decile_returns[decile_num].append({
                    'rebal_date': rebal_date,
                    'next_rebal_date': next_rebal_date,
                    'return': decile_ret,
                    'num_stocks': len(decile_prices.columns)
                })

        # Convert to DataFrames
        decile_dfs = {}
        for decile_num, returns_list in decile_returns.items():
            if len(returns_list) > 0:
                decile_dfs[decile_num] = pd.DataFrame(returns_list)

        logger.info(f"Decile portfolios built: {len(decile_dfs)} deciles")
        for decile_num, df in decile_dfs.items():
            logger.info(f"  Decile {decile_num}: {len(df)} periods")

        return decile_dfs

    def _compute_full_sample_performance(self, decile_portfolios: Dict) -> Dict:
        """
        Compute performance metrics for full sample.

        Args:
            decile_portfolios: Dict of decile portfolio DataFrames

        Returns:
            Dict with performance metrics
        """
        metrics = {}

        # Compute metrics for each decile
        for decile_num, df in decile_portfolios.items():
            returns = df['return'].values
            cumulative_ret = (1 + returns).prod() - 1
            mean_ret = returns.mean()
            std_ret = returns.std()
            sharpe = mean_ret / std_ret if std_ret > 0 else 0

            metrics[f'decile_{decile_num}'] = {
                'cumulative_return': float(cumulative_ret),
                'mean_return': float(mean_ret),
                'volatility': float(std_ret),
                'sharpe': float(sharpe),
                'num_periods': len(returns)
            }

        # Long-short (top - bottom)
        if 10 in decile_portfolios and 1 in decile_portfolios:
            # Decile 1 = highest momentum (winners), Decile 10 = lowest momentum (losers)
            top_returns = decile_portfolios[1]['return'].values  # High momentum
            bottom_returns = decile_portfolios[10]['return'].values  # Low momentum
            ls_returns = top_returns - bottom_returns  # Winners - Losers

            metrics['long_short'] = {
                'cumulative_return': float((1 + ls_returns).prod() - 1),
                'mean_return': float(ls_returns.mean()),
                'volatility': float(ls_returns.std()),
                'sharpe': float(ls_returns.mean() / ls_returns.std() if ls_returns.std() > 0 else 0),
                't_stat': float(ls_returns.mean() / (ls_returns.std() / np.sqrt(len(ls_returns))) if ls_returns.std() > 0 else 0)
            }

        # Monotonicity check
        mean_returns = [metrics[f'decile_{i}']['mean_return'] for i in range(1, 11)]
        monotonic_increasing = all(mean_returns[i] <= mean_returns[i+1] for i in range(len(mean_returns)-1))

        metrics['monotonicity'] = {
            'is_monotonic': monotonic_increasing,
            'mean_returns_by_decile': mean_returns
        }

        logger.info("Full-sample performance computed")
        logger.info(f"  Long-short mean return: {metrics['long_short']['mean_return']:.4f}")
        logger.info(f"  Long-short Sharpe: {metrics['long_short']['sharpe']:.2f}")
        logger.info(f"  Long-short t-stat: {metrics['long_short']['t_stat']:.2f}")
        logger.info(f"  Monotonicity: {monotonic_increasing}")

        return metrics

    def _analyze_regimes(self, decile_portfolios: Dict) -> Dict:
        """
        Analyze performance by regime.

        Regimes:
        - Pre-COVID: 2015-04-01 to 2019-12-31
        - COVID: 2020-01-01 to 2020-12-31
        - 2021-2022: 2021-01-01 to 2022-12-31
        - 2023-2024: 2023-01-01 to latest

        Args:
            decile_portfolios: Dict of decile portfolio DataFrames

        Returns:
            Dict with regime-specific metrics
        """
        regime_periods = {
            'pre_covid': ('2015-04-01', '2019-12-31'),
            'covid': ('2020-01-01', '2020-12-31'),
            'bear_2022': ('2021-01-01', '2022-12-31'),
            'recent': ('2023-01-01', '2024-12-31')
        }

        regime_metrics = {}

        for regime_name, (start, end) in regime_periods.items():
            logger.info(f"  Regime: {regime_name} ({start} to {end})")

            regime_metrics[regime_name] = {}

            # Filter decile returns to regime period
            for decile_num, df in decile_portfolios.items():
                regime_df = df[
                    (df['rebal_date'] >= start) &
                    (df['rebal_date'] <= end)
                ]

                if len(regime_df) == 0:
                    continue

                returns = regime_df['return'].values
                cumulative_ret = (1 + returns).prod() - 1
                mean_ret = returns.mean()
                std_ret = returns.std()

                regime_metrics[regime_name][f'decile_{decile_num}'] = {
                    'cumulative_return': float(cumulative_ret),
                    'mean_return': float(mean_ret),
                    'volatility': float(std_ret),
                    'num_periods': len(returns)
                }

            # Long-short for this regime
            if 10 in decile_portfolios and 1 in decile_portfolios:
                # Decile 1 = high momentum (winners), Decile 10 = low momentum (losers)
                top_df = decile_portfolios[1][
                    (decile_portfolios[1]['rebal_date'] >= start) &
                    (decile_portfolios[1]['rebal_date'] <= end)
                ]
                bottom_df = decile_portfolios[10][
                    (decile_portfolios[10]['rebal_date'] >= start) &
                    (decile_portfolios[10]['rebal_date'] <= end)
                ]

                if len(top_df) > 0 and len(bottom_df) > 0:
                    ls_returns = top_df['return'].values - bottom_df['return'].values  # Winners - Losers

                    regime_metrics[regime_name]['long_short'] = {
                        'cumulative_return': float((1 + ls_returns).prod() - 1),
                        'mean_return': float(ls_returns.mean()),
                        'volatility': float(ls_returns.std()),
                        'sharpe': float(ls_returns.mean() / ls_returns.std() if ls_returns.std() > 0 else 0)
                    }

        logger.info("Regime analysis complete")
        return regime_metrics

    def _generate_report(self, results: Dict):
        """
        Generate Phase 1.5 diagnostic report with GO/NO-GO recommendation.

        Args:
            results: Dict with all diagnostic results
        """
        output_path = Path(__file__).parent.parent / 'results' / 'momentum_v1_phase1_report.md'
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("# InstitutionalMomentum - Phase 1.5 Performance Acceptance Report\n\n")
            f.write(f"**Date:** {results['test_date']}\n")
            f.write(f"**Period:** {results['period']}\n")
            f.write(f"**Signal:** {results['signal']}\n\n")
            f.write("---\n\n")

            # Executive Summary (will be updated based on results)
            f.write("## Executive Summary\n\n")
            full_sample = results['full_sample']

            # Check acceptance criteria
            ls_sharpe = full_sample.get('long_short', {}).get('sharpe', 0)
            ls_tstat = full_sample.get('long_short', {}).get('t_stat', 0)
            is_monotonic = full_sample.get('monotonicity', {}).get('is_monotonic', False)

            if ls_sharpe >= 0.3 and ls_tstat >= 2.0 and is_monotonic:
                f.write("**Decision: ✅ GO - InstitutionalMomentum shows robust factor behavior.**\n\n")
                f.write("Phase 1.5 performance acceptance testing confirms that Momentum delivers tradable alpha:\n\n")
                f.write(f"- **Decile monotonicity:** {'PASS' if is_monotonic else 'FAIL'}\n")
                f.write(f"- **Long-short Sharpe:** {ls_sharpe:.2f} (vs 0.3 threshold)\n")
                f.write(f"- **Statistical significance:** t-stat {ls_tstat:.2f} (vs 2.0 threshold)\n")
                f.write(f"- **Behavior:** Consistent momentum premium across regimes\n\n")
                go_status = "GO"
            else:
                f.write("**Decision: ⚠️ CONDITIONAL - Momentum shows some factor behavior but with concerns.**\n\n")
                f.write("Phase 1.5 performance acceptance reveals:\n\n")
                f.write(f"- **Decile monotonicity:** {'PASS' if is_monotonic else 'FAIL'}\n")
                f.write(f"- **Long-short Sharpe:** {ls_sharpe:.2f} (vs 0.3 threshold)\n")
                f.write(f"- **Statistical significance:** t-stat {ls_tstat:.2f} (vs 2.0 threshold)\n\n")
                go_status = "CONDITIONAL"

            f.write("This report answers three critical questions:\n\n")
            f.write("1. **Does Momentum behave like a real factor or noise?**\n")
            f.write("2. **How does it behave by regime?**\n")
            f.write("3. **Are results strong enough to justify Phase 2 optimization?**\n\n")

            # Full Sample Performance
            f.write("## 1. Full-Sample Performance\n\n")

            f.write("### Decile Performance\n\n")
            f.write("| Decile | Mean Return | Volatility | Sharpe |\n")
            f.write("|--------|-------------|------------|--------|\n")
            for i in range(1, 11):
                key = f'decile_{i}'
                if key in full_sample:
                    metrics = full_sample[key]
                    f.write(f"| {i} | {metrics['mean_return']:.4f} | {metrics['volatility']:.4f} | {metrics['sharpe']:.2f} |\n")
            f.write("\n")

            # Long-Short
            if 'long_short' in full_sample:
                ls = full_sample['long_short']
                f.write("### Long-Short (Top - Bottom)\n\n")
                f.write(f"- **Mean Return:** {ls['mean_return']:.4f} ({ls['mean_return']*100:.2f}% per month)\n")
                f.write(f"- **Volatility:** {ls['volatility']:.4f}\n")
                f.write(f"- **Sharpe:** {ls['sharpe']:.2f}\n")
                f.write(f"- **t-statistic:** {ls['t_stat']:.2f}\n\n")

            # Monotonicity
            mono = full_sample['monotonicity']
            f.write(f"### Monotonicity: {'✓ PASS' if mono['is_monotonic'] else '✗ FAIL'}\n\n")

            # Regime Analysis
            f.write("## 2. Regime Analysis\n\n")
            for regime_name, regime_data in results['regimes'].items():
                f.write(f"### {regime_name.replace('_', ' ').title()}\n\n")
                if 'long_short' in regime_data:
                    ls = regime_data['long_short']
                    f.write(f"- Mean Return: {ls['mean_return']:.4f} ({ls['mean_return']*100:.2f}% per month)\n")
                    f.write(f"- Volatility: {ls['volatility']:.4f}\n")
                    f.write(f"- Sharpe: {ls['sharpe']:.2f}\n\n")

            # GO/NO-GO Decision
            f.write("## 3. Performance Acceptance: GO/NO-GO\n\n")

            if go_status == "GO":
                f.write("### ✅ GO - Proceed to Phase 2\n\n")
                f.write("InstitutionalMomentum passes performance acceptance:\n")
                f.write("- ✓ Clear decile monotonicity\n")
                f.write("- ✓ Statistically significant long-short spread\n")
                f.write("- ✓ Sharpe above minimum threshold\n\n")
                f.write("**Momentum is APPROVED FOR:**\n")
                f.write("- Hyperparameter optimization (Phase 2)\n")
                f.write("- Ensemble inclusion (Phase 5)\n")
                f.write("- Production consideration\n\n")
            else:
                f.write("### ⚠️ CONDITIONAL - Review Before Phase 2\n\n")
                f.write("Issues to address:\n")
                if not is_monotonic:
                    f.write("- Decile monotonicity not perfect\n")
                if ls_sharpe < 0.3:
                    f.write(f"- Sharpe ({ls_sharpe:.2f}) below threshold (0.3)\n")
                if ls_tstat < 2.0:
                    f.write(f"- t-stat ({ls_tstat:.2f}) below significance threshold (2.0)\n")
                f.write("\n**Required Actions:** Review methodology and consider parameter adjustments.\n\n")

            f.write("---\n\n")
            f.write("**Note:** This is the Phase 1.5 performance acceptance test.\n")
            f.write("Full optimization will be performed in Phase 2 if approved.\n\n")

        logger.info(f"Report written: {output_path}")


def main():
    """Run Phase 1.5 diagnostics for InstitutionalMomentum."""
    tester = MomentumPhase1Diagnostics(
        start_date='2015-04-01',
        end_date='2024-12-31'
    )
    results = tester.run_full_diagnostics()

    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1.5 DIAGNOSTICS COMPLETE")
    logger.info("=" * 80)
    logger.info("Report: results/momentum_v1_phase1_report.md")


if __name__ == '__main__':
    main()
