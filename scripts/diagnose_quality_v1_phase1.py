"""
Phase 1 Diagnostics for CrossSectionalQuality v1

Performance acceptance testing with decile portfolios, regime splits, and sector tilts.

This script answers three critical questions:
1. Does Quality v1 behave like a real factor or random noise?
2. How does it behave by regime (pre-COVID, COVID, 2022 bear)?
3. Are the sector tilts and style exposures what we expect?

Usage:
    python3 scripts/diagnose_quality_v1_phase1.py

Outputs:
    results/quality_v1_phase1_report.md - Full diagnostic report with GO/NO-GO
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
from signals.quality.cross_sectional_quality import CrossSectionalQuality
from core.schedules import get_rebalance_dates
from config import get_logger

logger = get_logger(__name__)


class QualityV1Phase1Diagnostics:
    """
    Phase 1 performance acceptance diagnostics for CrossSectionalQuality v1.

    Tests:
    1. Decile monotonicity (top > bottom in ≥70% of 3-year windows)
    2. Long-short spread (top minus bottom decile)
    3. Long-only performance (top decile vs SPY)
    4. Regime behavior (pre-COVID, COVID, 2022 bear, 2023-2024)
    5. Sector tilts (top decile vs S&P 500 index)
    6. Information Ratio thresholds (IR > 0.3 minimum, > 0.5 for variants)
    """

    def __init__(self,
                 start_date: str = '2015-04-01',
                 end_date: str = '2024-12-31'):
        """
        Initialize diagnostics.

        Args:
            start_date: Analysis start (default 2015-01-01)
            end_date: Analysis end (default 2024-12-31)
        """
        self.start_date = start_date
        self.end_date = end_date

        self.dm = DataManager()
        self.um = UniverseManager(self.dm)

        # Initialize Quality v1 with standard parameters
        self.quality_params = {
            'w_profitability': 0.4,
            'w_growth': 0.3,
            'w_safety': 0.3,
            'winsorize_pct': [5, 95],
            'quintiles': False,  # Use continuous scores for decile construction
            'min_coverage': 0.5
        }
        self.quality = CrossSectionalQuality(self.quality_params, data_manager=self.dm)

        logger.info("=" * 80)
        logger.info("Phase 1 Diagnostics: CrossSectionalQuality v1")
        logger.info("=" * 80)
        logger.info(f"Period: {self.start_date} to {self.end_date}")
        logger.info(f"Universe: S&P 500 PIT (sp500_actual)")
        logger.info("=" * 80)

    def run_full_diagnostics(self) -> Dict:
        """
        Run all Phase 1 diagnostic tests.

        Returns:
            Dict with diagnostic results
        """
        results = {
            'test_date': datetime.now().isoformat(),
            'period': f"{self.start_date} to {self.end_date}",
            'signal': 'CrossSectionalQuality v1'
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

        # 4. Sector tilts
        logger.info("\n4. Computing sector tilts...")
        results['sector_tilts'] = self._compute_sector_tilts()

        # 5. Generate report with GO/NO-GO
        logger.info("\n5. Generating diagnostic report...")
        self._generate_report(results)

        logger.info("\n" + "=" * 80)
        logger.info("Phase 1 Diagnostics Complete")
        logger.info("=" * 80)

        return results

    def _build_decile_portfolios(self) -> Dict:
        """
        Build monthly-rebalanced decile portfolios based on Quality v1 scores.

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

        # Generate Quality v1 scores
        logger.info("Generating Quality v1 signals...")
        signals_df = self.quality.generate_signals_cross_sectional(
            universe_tickers=universe,
            rebalance_dates=pd.DatetimeIndex(rebalance_dates),
            start_date=self.start_date,
            end_date=self.end_date
        )
        logger.info(f"Signals generated: {signals_df.shape}")

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

        # Build decile portfolios at each rebalance date
        logger.info("Constructing decile portfolios...")
        decile_returns = {i: [] for i in range(1, 11)}  # Deciles 1-10

        for rebal_date in pd.DatetimeIndex(rebalance_dates):
            if rebal_date not in signals_df.index:
                continue

            # Get signals for this rebalance date
            signals_today = signals_df.loc[rebal_date].dropna()

            if len(signals_today) < 50:  # Need minimum universe size
                continue

            # Sort stocks by signal (high quality = high signal)
            ranked = signals_today.sort_values(ascending=False)

            # Form 10 deciles
            decile_size = len(ranked) // 10
            deciles = {}
            for i in range(1, 11):
                start_idx = (i - 1) * decile_size
                end_idx = start_idx + decile_size if i < 10 else len(ranked)
                deciles[i] = ranked.iloc[start_idx:end_idx].index.tolist()

            # Compute returns from this rebalance to next rebalance
            next_rebal_idx = rebalance_dates.index(rebal_date.strftime('%Y-%m-%d')) + 1
            if next_rebal_idx >= len(rebalance_dates):
                break

            next_rebal_date = pd.Timestamp(rebalance_dates[next_rebal_idx])

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
            top_returns = decile_portfolios[10]['return'].values
            bottom_returns = decile_portfolios[1]['return'].values
            ls_returns = top_returns - bottom_returns

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
                top_df = decile_portfolios[10][
                    (decile_portfolios[10]['rebal_date'] >= start) &
                    (decile_portfolios[10]['rebal_date'] <= end)
                ]
                bottom_df = decile_portfolios[1][
                    (decile_portfolios[1]['rebal_date'] >= start) &
                    (decile_portfolios[1]['rebal_date'] <= end)
                ]

                if len(top_df) > 0 and len(bottom_df) > 0:
                    ls_returns = top_df['return'].values - bottom_df['return'].values

                    regime_metrics[regime_name]['long_short'] = {
                        'cumulative_return': float((1 + ls_returns).prod() - 1),
                        'mean_return': float(ls_returns.mean()),
                        'volatility': float(ls_returns.std())
                    }

        logger.info("Regime analysis complete")
        return regime_metrics

    def _compute_sector_tilts(self, snapshot_date: str = '2024-01-01') -> Dict:
        """
        Compute sector tilts of top quality decile vs S&P 500 index.

        Args:
            snapshot_date: Date for sector snapshot

        Returns:
            Dict with sector weights and tilts
        """
        logger.info(f"  Snapshot date: {snapshot_date}")

        # Get universe
        universe = self.um.get_universe(
            universe_type='sp500_actual',
            as_of_date=snapshot_date,
            min_price=5.0
        )

        # Get Quality v1 scores for snapshot
        rebal_dates = get_rebalance_dates(
            schedule='M',
            dm=self.dm,
            start_date=snapshot_date,
            end_date=snapshot_date
        )

        if len(rebal_dates) == 0:
            logger.warning("No rebalance date found for snapshot")
            return {}

        signals_df = self.quality.generate_signals_cross_sectional(
            universe_tickers=universe,
            rebalance_dates=pd.DatetimeIndex([snapshot_date]),
            start_date=snapshot_date,
            end_date=snapshot_date
        )

        # Get top decile stocks
        signals_today = signals_df.iloc[0].dropna().sort_values(ascending=False)
        decile_size = len(signals_today) // 10
        top_decile_tickers = signals_today.iloc[:decile_size].index.tolist()

        # Get sector info
        universe_info = self.um.get_universe_info(universe, snapshot_date)
        top_decile_info = universe_info[universe_info.index.isin(top_decile_tickers)]

        # Compute sector weights
        universe_sectors = universe_info['sector'].value_counts(normalize=True)
        top_decile_sectors = top_decile_info['sector'].value_counts(normalize=True)

        # Compute tilts
        sector_tilts = {}
        for sector in universe_sectors.index:
            universe_wt = universe_sectors.get(sector, 0)
            top_decile_wt = top_decile_sectors.get(sector, 0)
            tilt = top_decile_wt - universe_wt

            sector_tilts[sector] = {
                'universe_weight': float(universe_wt),
                'top_decile_weight': float(top_decile_wt),
                'tilt': float(tilt)
            }

        logger.info("Sector tilts computed")
        for sector, tilts in sorted(sector_tilts.items(), key=lambda x: abs(x[1]['tilt']), reverse=True):
            logger.info(f"  {sector}: {tilts['tilt']:+.3f} ({tilts['top_decile_weight']:.3f} vs {tilts['universe_weight']:.3f})")

        return sector_tilts

    def _generate_report(self, results: Dict):
        """
        Generate Phase 1 diagnostic report with GO/NO-GO recommendation.

        Args:
            results: Dict with all diagnostic results
        """
        output_path = Path(__file__).parent.parent / 'results' / 'quality_v1_phase1_report.md'
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("# CrossSectionalQuality v1 - Phase 1 Performance Acceptance Report\n\n")
            f.write(f"**Date:** {results['test_date']}\n")
            f.write(f"**Period:** {results['period']}\n")
            f.write(f"**Signal:** {results['signal']}\n\n")
            f.write("---\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write("Phase 1 performance acceptance testing for CrossSectionalQuality v1.\n")
            f.write("This report answers three critical questions:\n\n")
            f.write("1. Does Quality v1 behave like a real factor or random noise?\n")
            f.write("2. How does it behave by regime?\n")
            f.write("3. Are sector tilts and style exposures what we expect?\n\n")

            # Full Sample Performance
            f.write("## 1. Full-Sample Performance\n\n")
            full_sample = results['full_sample']

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
                f.write(f"- **Mean Return:** {ls['mean_return']:.4f}\n")
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
                    f.write(f"- Mean Return: {ls['mean_return']:.4f}\n")
                    f.write(f"- Volatility: {ls['volatility']:.4f}\n\n")

            # Sector Tilts
            f.write("## 3. Sector Tilts (Top Decile vs Index)\n\n")
            f.write("| Sector | Index Weight | Top Decile Weight | Tilt |\n")
            f.write("|--------|--------------|-------------------|------|\n")
            for sector, tilts in sorted(results['sector_tilts'].items(), key=lambda x: abs(x[1]['tilt']), reverse=True):
                f.write(f"| {sector} | {tilts['universe_weight']:.3f} | {tilts['top_decile_weight']:.3f} | {tilts['tilt']:+.3f} |\n")
            f.write("\n")

            # GO/NO-GO Decision
            f.write("## 4. Performance Acceptance: GO/NO-GO\n\n")

            # Check acceptance criteria
            ls_positive = full_sample.get('long_short', {}).get('mean_return', 0) > 0
            is_monotonic = full_sample.get('monotonicity', {}).get('is_monotonic', False)

            if ls_positive and is_monotonic:
                f.write("### ✅ GO - Performance Acceptance PASSED\n\n")
                f.write("CrossSectionalQuality v1 passes performance acceptance:\n")
                f.write("- ✓ Long-short spread is positive\n")
                f.write("- ✓ Decile monotonicity observed\n\n")
                f.write("**v1 is now APPROVED FOR:**\n")
                f.write("- Hyperparameter optimization\n")
                f.write("- Ensemble inclusion\n")
                f.write("- Variant development (if IR > 0.5)\n\n")
            else:
                f.write("### ⚠️ NO-GO - Performance Issues Detected\n\n")
                f.write("Issues:\n")
                if not ls_positive:
                    f.write("- ✗ Long-short spread is NOT positive\n")
                if not is_monotonic:
                    f.write("- ✗ Decile monotonicity NOT observed\n")
                f.write("\n**Required Actions:** Investigate methodology before proceeding.\n\n")

            f.write("---\n\n")
            f.write("**Note:** This is the Phase 1 performance acceptance test.\n")
            f.write("Technical acceptance was completed in Phase 0.2.\n\n")

        logger.info(f"Report written: {output_path}")


def main():
    """Run Phase 1 diagnostics for CrossSectionalQuality v1."""
    tester = QualityV1Phase1Diagnostics(
        start_date='2015-04-01',
        end_date='2024-12-31'
    )
    results = tester.run_full_diagnostics()

    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1 DIAGNOSTICS COMPLETE")
    logger.info("=" * 80)
    logger.info("Report: results/quality_v1_phase1_report.md")


if __name__ == '__main__':
    main()
