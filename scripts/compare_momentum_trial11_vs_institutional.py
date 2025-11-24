"""
Compare stock selection between Trial 11 manual logic and InstitutionalMomentum class.

Investigates why ensemble baseline significantly outperforms Trial 11 diagnostic.

For representative dates, compares:
- Universe size
- Long leg composition (Trial 11 vs InstitutionalMomentum)
- Overlap and correlation of signals

This helps identify if the performance gap is due to:
- Different stock selection logic
- Different signal scoring
- Edge case handling in winsorization or ranking
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from data.data_manager import DataManager
from core.universe_manager import UniverseManager
from signals.momentum.institutional_momentum import InstitutionalMomentum
from config import get_logger

logger = get_logger(__name__)


class MomentumComparer:
    """
    Compares Trial 11 manual momentum calculation with InstitutionalMomentum class.
    """

    def __init__(self):
        """Initialize comparer."""
        self.dm = DataManager()
        self.um = UniverseManager(self.dm)

        # Trial 11 canonical params
        self.formation_period = 308
        self.skip_period = 0
        self.winsorize_pct = 9.2  # Two-sided

        # Initialize InstitutionalMomentum
        self.momentum_signal = InstitutionalMomentum(params={
            'formation_period': 308,
            'skip_period': 0,
            'winsorize_pct': [0.4, 99.6],  # Equivalent to 9.2% two-sided
            'rebalance_frequency': 'monthly',
            'quintiles': True  # Returns [-1, -0.5, 0, 0.5, 1]
        })

    def compare_at_date(self, rebal_date: str) -> Dict:
        """
        Compare stock selection at a single rebalance date.

        Args:
            rebal_date: Rebalance date (YYYY-MM-DD)

        Returns:
            Dict with comparison metrics
        """
        logger.info("=" * 80)
        logger.info(f"Comparing at date: {rebal_date}")
        logger.info("=" * 80)

        results = {
            'date': rebal_date,
            'universe_size': 0,
            'trial11_long_size': 0,
            'institutional_long_size': 0,
            'overlap_size': 0,
            'jaccard': 0.0,
            'score_correlation': np.nan
        }

        # 1. Build S&P 500 PIT universe
        universe = self.um.get_universe(
            universe_type='sp500_actual',
            as_of_date=rebal_date,
            min_price=5.0
        )

        if isinstance(universe, pd.Series):
            universe = universe.tolist()
        elif isinstance(universe, pd.DataFrame):
            universe = universe.index.tolist()

        results['universe_size'] = len(universe)
        logger.info(f"Universe size: {len(universe)} stocks")

        # 2. Fetch price data
        # Need ~308 trading days = ~445 calendar days, add buffer for safety
        lookback_buffer = timedelta(days=500)
        rebal_dt = pd.Timestamp(rebal_date)
        price_start_date = (rebal_dt - lookback_buffer).strftime('%Y-%m-%d')

        prices_dict = {}
        for ticker in universe:
            try:
                prices = self.dm.get_prices(ticker, price_start_date, rebal_date)
                if len(prices) > 0 and 'close' in prices.columns:
                    prices_dict[ticker] = prices
            except:
                pass

        logger.info(f"Price data loaded: {len(prices_dict)} stocks")

        # 3. Calculate Trial 11 style momentum
        trial11_scores = self._calculate_trial11_momentum(
            prices_dict,
            rebal_dt
        )

        # 4. Calculate InstitutionalMomentum style scores
        institutional_scores = self._calculate_institutional_momentum(
            prices_dict,
            rebal_dt
        )

        logger.info(f"Trial 11 scores: {len(trial11_scores)} stocks")
        logger.info(f"InstitutionalMomentum scores: {len(institutional_scores)} stocks")

        # 5. Compare long legs
        # Trial 11: top quintile by raw momentum
        trial11_sorted = trial11_scores.sort_values(ascending=False)
        trial11_quintile_size = len(trial11_sorted) // 5
        trial11_long = set(trial11_sorted.iloc[:trial11_quintile_size].index)

        # InstitutionalMomentum: top quintile (signal == 1.0)
        institutional_long = set(institutional_scores[institutional_scores == 1.0].index)

        results['trial11_long_size'] = len(trial11_long)
        results['institutional_long_size'] = len(institutional_long)

        # 6. Calculate overlap
        overlap = trial11_long & institutional_long
        union = trial11_long | institutional_long

        results['overlap_size'] = len(overlap)
        results['jaccard'] = len(overlap) / len(union) if len(union) > 0 else 0.0

        logger.info(f"\nLong leg comparison:")
        logger.info(f"  Trial 11 long: {len(trial11_long)} stocks")
        logger.info(f"  InstitutionalMomentum long: {len(institutional_long)} stocks")
        logger.info(f"  Overlap: {len(overlap)} stocks")
        logger.info(f"  Jaccard similarity: {results['jaccard']:.2%}")

        # 7. Calculate score correlation
        common_tickers = set(trial11_scores.index) & set(institutional_scores.index)
        if len(common_tickers) > 0:
            trial11_common = trial11_scores.loc[list(common_tickers)]
            institutional_common = institutional_scores.loc[list(common_tickers)]
            results['score_correlation'] = trial11_common.corr(institutional_common)
            logger.info(f"  Score correlation: {results['score_correlation']:.3f}")

        # 8. Show sample differences
        logger.info(f"\nSample stock selection differences:")

        # In Trial 11 but not Institutional
        trial11_only = trial11_long - institutional_long
        if len(trial11_only) > 0:
            sample = list(trial11_only)[:5]
            logger.info(f"  Trial 11 only (first 5): {sample}")

        # In Institutional but not Trial 11
        institutional_only = institutional_long - trial11_long
        if len(institutional_only) > 0:
            sample = list(institutional_only)[:5]
            logger.info(f"  InstitutionalMomentum only (first 5): {sample}")

        return results

    def _calculate_trial11_momentum(
        self,
        prices_dict: Dict[str, pd.DataFrame],
        rebal_date: pd.Timestamp
    ) -> pd.Series:
        """
        Calculate momentum using Trial 11 manual logic.

        Args:
            prices_dict: Dict of ticker -> price DataFrame
            rebal_date: Rebalance date

        Returns:
            Series of momentum scores
        """
        logger.info(f"Calculating Trial 11 momentum for {len(prices_dict)} tickers")
        momentum_dict = {}

        for ticker, prices in prices_dict.items():
            if 'close' not in prices.columns:
                continue

            # Handle duplicate dates
            if prices.index.duplicated().any():
                prices = prices[~prices.index.duplicated(keep='last')]

            # Momentum calculation (same as Trial 11)
            # pct_change will produce NaN where insufficient data, filtered later
            mom = prices['close'].pct_change(
                periods=self.formation_period,
                fill_method=None
            ).shift(self.skip_period)

            # Drop duplicates from momentum series
            if mom.index.duplicated().any():
                mom = mom[~mom.index.duplicated(keep='last')]

            momentum_dict[ticker] = mom

        # Convert to DataFrame
        logger.info(f"Created momentum dict with {len(momentum_dict)} tickers")
        momentum_df = pd.DataFrame(momentum_dict)
        if len(momentum_df) == 0:
            logger.warning("Momentum DataFrame is empty")
            return pd.Series(dtype=float)

        logger.info(f"Momentum DataFrame shape: {momentum_df.shape}, index range: {momentum_df.index.min()} to {momentum_df.index.max()}")
        momentum_df = momentum_df.sort_index()

        # Handle duplicate dates
        if momentum_df.index.duplicated().any():
            logger.info(f"Removing {momentum_df.index.duplicated().sum()} duplicate dates")
            momentum_df = momentum_df[~momentum_df.index.duplicated(keep='last')]

        # Get momentum at rebalance date
        valid_idx = momentum_df.index[momentum_df.index <= rebal_date]
        logger.info(f"Valid index dates <= {rebal_date}: {len(valid_idx)} dates")
        if len(valid_idx) == 0:
            logger.warning(f"No valid index dates found <= {rebal_date}")
            return pd.Series(dtype=float)

        mom_date = valid_idx[-1]
        logger.info(f"Selected momentum date: {mom_date}")
        mom_today = momentum_df.loc[mom_date].dropna()
        logger.info(f"Momentum values at {mom_date}: {len(mom_today)} stocks (after dropna)")

        if len(mom_today) < 20:
            logger.warning(f"Only {len(mom_today)} stocks with momentum < 20 minimum")
            return pd.Series(dtype=float)

        # Winsorize (same as Trial 11)
        winsor_lower = self.winsorize_pct / 100
        winsor_upper = (100 - self.winsorize_pct) / 100
        lower_bound = mom_today.quantile(winsor_lower)
        upper_bound = mom_today.quantile(winsor_upper)
        mom_wins = mom_today.clip(lower=lower_bound, upper=upper_bound)

        return mom_wins

    def _calculate_institutional_momentum(
        self,
        prices_dict: Dict[str, pd.DataFrame],
        rebal_date: pd.Timestamp
    ) -> pd.Series:
        """
        Calculate momentum using InstitutionalMomentum class.

        Args:
            prices_dict: Dict of ticker -> price DataFrame
            rebal_date: Rebalance date

        Returns:
            Series of momentum scores
        """
        # Convert to dict of Series (close prices only)
        prices_series = {}
        for ticker, df in prices_dict.items():
            if 'close' in df.columns:
                prices_series[ticker] = df['close']

        # Generate signals for each ticker
        scores = {}
        for ticker, px_series in prices_series.items():
            # Build DataFrame for signal API
            data = pd.DataFrame({
                'close': px_series,
                'ticker': ticker
            })

            try:
                sig_series = self.momentum_signal.generate_signals(data)
                if len(sig_series) > 0:
                    signal_value = sig_series.iloc[-1]
                    if pd.notna(signal_value) and signal_value != 0:
                        scores[ticker] = signal_value
            except:
                pass

        return pd.Series(scores)

    def run_comparison(self, dates: List[str]):
        """
        Run comparison across multiple dates.

        Args:
            dates: List of rebalance dates (YYYY-MM-DD)
        """
        logger.info("\n" + "=" * 80)
        logger.info("MOMENTUM COMPARISON: Trial 11 vs InstitutionalMomentum")
        logger.info("=" * 80)
        logger.info(f"\nDates to compare: {dates}")
        logger.info("")

        all_results = []

        for date in dates:
            results = self.compare_at_date(date)
            all_results.append(results)
            logger.info("")

        # Summary
        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info("")
        logger.info(f"{'Date':<15} {'Universe':<10} {'T11 Long':<10} {'Inst Long':<10} "
                   f"{'Overlap':<10} {'Jaccard':<10} {'Correlation':<12}")
        logger.info("-" * 95)

        for r in all_results:
            logger.info(f"{r['date']:<15} {r['universe_size']:<10} "
                       f"{r['trial11_long_size']:<10} {r['institutional_long_size']:<10} "
                       f"{r['overlap_size']:<10} {r['jaccard']:<10.2%} "
                       f"{r['score_correlation']:<12.3f}")

        logger.info("")
        logger.info("Interpretation:")
        avg_jaccard = np.mean([r['jaccard'] for r in all_results])
        logger.info(f"  Average Jaccard: {avg_jaccard:.2%}")

        if avg_jaccard > 0.85:
            logger.info("  ✅ HIGH overlap - stock selection is very similar")
        elif avg_jaccard > 0.70:
            logger.info("  ⚠️  MODERATE overlap - some differences in stock selection")
        else:
            logger.info("  ❌ LOW overlap - significant differences in stock selection")

        avg_corr = np.nanmean([r['score_correlation'] for r in all_results])
        logger.info(f"  Average Score Correlation: {avg_corr:.3f}")

        if avg_corr > 0.90:
            logger.info("  ✅ HIGH correlation - score rankings are very similar")
        elif avg_corr > 0.70:
            logger.info("  ⚠️  MODERATE correlation - some differences in rankings")
        else:
            logger.info("  ❌ LOW correlation - significant differences in rankings")

        logger.info("=" * 80)


def main():
    """Run comparison."""
    comparer = MomentumComparer()

    # Representative dates spanning different market regimes
    test_dates = [
        '2016-06-30',  # Mid 2016
        '2019-12-31',  # Pre-COVID
        '2022-12-31',  # Post-COVID
    ]

    comparer.run_comparison(test_dates)


if __name__ == "__main__":
    main()
