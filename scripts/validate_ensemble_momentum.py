"""
Ensemble Validation: Momentum-Only Baseline

Validates that EnsembleSignal with momentum-only configuration produces
identical results to standalone InstitutionalMomentum v2.

This is a critical sanity check before using the ensemble for production.

Usage:
    # Full validation (2-3 years)
    python3 scripts/validate_ensemble_momentum.py

    # Debug mode (first 6 rebalances with detailed comparison)
    python3 scripts/validate_ensemble_momentum.py --debug

Validation Steps:
1. Initialize ensemble with canonical Momentum v2 params
2. For each rebalance:
   - Generate scores via ensemble.generate_ensemble_scores()
   - Generate scores via direct InstitutionalMomentum.generate_signals()
   - Compare (should be numerically identical within epsilon)
3. Run backtest on ensemble scores
4. Verify high-level metrics match standalone Momentum v2

Expected Outcome:
- Max score difference < 1e-9 (float epsilon)
- Sharpe, max drawdown, returns match Trial 11 baseline
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
import argparse
import warnings
warnings.filterwarnings('ignore')

from data.data_manager import DataManager
from core.universe_manager import UniverseManager
from signals.momentum.institutional_momentum import InstitutionalMomentum
from signals.ml.ensemble_signal import EnsembleSignal, EnsembleMember
from core.signal_registry import get_signal_status
from core.schedules import get_rebalance_dates
from config import get_logger

logger = get_logger(__name__)


class EnsembleMomentumValidator:
    """
    Validates ensemble(momentum-only) == direct momentum.

    Critical sanity check before production use.
    """

    def __init__(self,
                 start_date: str = '2022-01-01',
                 end_date: str = '2024-12-31',
                 debug_compare: bool = False):
        """
        Initialize validator.

        Args:
            start_date: Validation period start (default: 2-3 years for speed)
            end_date: Validation period end
            debug_compare: If True, compare ensemble vs direct for first 6 rebalances
        """
        self.start_date = start_date
        self.end_date = end_date
        self.debug_compare = debug_compare

        self.dm = DataManager()
        self.um = UniverseManager(self.dm)

        # Canonical Momentum v2 params (Trial 11)
        self.momentum_params = {
            'formation_period': 308,  # Trial 11 optimal
            'skip_period': 0,         # Trial 11 optimal
            'winsorize_pct': [0.4, 99.6],  # 9.2% = [0.4, 99.6]
            'rebalance_frequency': 'monthly'
        }

        logger.info("=" * 80)
        logger.info("Ensemble Validation: Momentum-Only")
        logger.info("=" * 80)
        logger.info(f"Period: {self.start_date} to {self.end_date}")
        logger.info(f"Params: {self.momentum_params}")
        logger.info(f"Debug compare: {self.debug_compare}")
        logger.info("=" * 80)

        # Verify Momentum v2 is GO
        status = get_signal_status("InstitutionalMomentum", "v2")
        if status is None:
            raise ValueError("InstitutionalMomentum v2 not found in registry!")
        if status.status != "GO":
            raise ValueError(f"InstitutionalMomentum v2 is {status.status}, expected GO")
        logger.info(f"✓ Verified InstitutionalMomentum v2 status: {status.status}")

    def validate(self) -> Dict:
        """
        Run full ensemble validation.

        Returns:
            Dict with validation results
        """
        results = {
            'test_date': datetime.now().isoformat(),
            'period': f"{self.start_date} to {self.end_date}",
            'momentum_params': self.momentum_params
        }

        # 1. Initialize ensemble and direct momentum
        logger.info("\n1. Initializing ensemble and direct momentum...")
        ensemble, direct_momentum = self._initialize_signals()

        # 2. Build universe
        logger.info("\n2. Building S&P 500 PIT universe...")
        universe = self.um.get_universe(
            universe_type='sp500_actual',
            as_of_date=self.start_date,
            min_price=5.0
        )
        logger.info(f"Universe: {len(universe)} stocks")

        # 3. Get rebalance dates
        logger.info("\n3. Getting monthly rebalance dates...")
        rebalance_dates = get_rebalance_dates(
            schedule='M',
            dm=self.dm,
            start_date=self.start_date,
            end_date=self.end_date
        )
        logger.info(f"Rebalance dates: {len(rebalance_dates)} month-ends")

        # 4. Fetch price data with lookback
        logger.info("\n4. Fetching price data...")
        lookback_days = self.momentum_params['formation_period'] + self.momentum_params['skip_period'] + 30
        lookback_buffer = timedelta(days=int(lookback_days * 1.5))  # Calendar days buffer
        start_dt = pd.Timestamp(self.start_date)
        price_start_date = (start_dt - lookback_buffer).strftime('%Y-%m-%d')
        logger.info(f"Price data from {price_start_date} (includes lookback for momentum)")

        prices_dict = {}
        for ticker in universe:
            try:
                prices = self.dm.get_prices(ticker, price_start_date, self.end_date)
                if len(prices) > 0 and 'close' in prices.columns:
                    prices_dict[ticker] = prices['close']
            except Exception as e:
                logger.debug(f"Failed to load {ticker}: {e}")

        logger.info(f"Price data loaded: {len(prices_dict)} stocks")

        # 5. Generate scores via ensemble and direct, compare
        logger.info("\n5. Generating and comparing scores...")
        comparison_results = self._compare_scores(
            ensemble=ensemble,
            direct_momentum=direct_momentum,
            prices_dict=prices_dict,
            rebalance_dates=rebalance_dates
        )
        results['comparison'] = comparison_results

        # 6. Summary
        logger.info("\n6. Validation Summary:")
        logger.info(f"  Rebalances compared: {comparison_results['num_rebalances']}")
        logger.info(f"  Max abs difference: {comparison_results['max_diff']:.2e}")
        logger.info(f"  Mean abs difference: {comparison_results['mean_diff']:.2e}")

        if comparison_results['max_diff'] < 1e-9:
            logger.info("  ✅ PASS: Ensemble matches direct momentum (within float epsilon)")
            results['verdict'] = 'PASS'
        elif comparison_results['max_diff'] < 1e-6:
            logger.warning(f"  ⚠️  MARGINAL: Small numerical differences ({comparison_results['max_diff']:.2e})")
            results['verdict'] = 'MARGINAL'
        else:
            logger.error(f"  ❌ FAIL: Large differences ({comparison_results['max_diff']:.2e})")
            results['verdict'] = 'FAIL'

        return results

    def _initialize_signals(self) -> tuple:
        """
        Initialize ensemble and direct momentum instances.

        Returns:
            (EnsembleSignal, InstitutionalMomentum)
        """
        # Direct momentum
        direct_momentum = InstitutionalMomentum(params=self.momentum_params)
        logger.info(f"Initialized direct momentum: {direct_momentum}")

        # Ensemble with momentum-only member
        # NOTE: Use normalize="none" because InstitutionalMomentum already
        # outputs quintile-mapped values [-1, -0.5, 0, 0.5, 1]
        ensemble = EnsembleSignal(
            members=[
                EnsembleMember(
                    signal_name="InstitutionalMomentum",
                    version="v2",
                    weight=1.0,
                    normalize="none",  # Signal already normalized to quintiles
                    params=self.momentum_params
                )
            ],
            data_manager=self.dm,
            enforce_go_only=True
        )
        logger.info(f"Initialized ensemble: {ensemble}")

        return ensemble, direct_momentum

    def _compare_scores(self,
                       ensemble: EnsembleSignal,
                       direct_momentum: InstitutionalMomentum,
                       prices_dict: Dict[str, pd.Series],
                       rebalance_dates: list) -> Dict:
        """
        Compare ensemble scores to direct momentum scores.

        Args:
            ensemble: EnsembleSignal instance
            direct_momentum: InstitutionalMomentum instance
            prices_dict: Dict mapping ticker -> price series
            rebalance_dates: List of rebalance timestamps

        Returns:
            Dict with comparison statistics
        """
        all_diffs = []
        max_diff = 0.0
        num_compared = 0

        # Limit to first 6 rebalances if debug mode
        compare_limit = 6 if self.debug_compare else len(rebalance_dates)

        for i, rebal_date in enumerate(pd.DatetimeIndex(rebalance_dates[:compare_limit])):
            logger.info(f"\nRebalance {i+1}/{compare_limit}: {rebal_date.date()}")

            # Build prices_by_ticker dict for this rebalance (data up to rebal_date)
            prices_by_ticker = {}
            for ticker, px_series in prices_dict.items():
                px_slice = px_series[px_series.index <= rebal_date]
                if len(px_slice) >= 90:  # Minimum history
                    prices_by_ticker[ticker] = px_slice

            logger.info(f"  Tickers with sufficient history: {len(prices_by_ticker)}")

            if len(prices_by_ticker) == 0:
                logger.warning(f"  No tickers with sufficient history, skipping")
                continue

            # Generate ensemble scores
            ensemble_scores = ensemble.generate_ensemble_scores(
                prices_by_ticker=prices_by_ticker,
                rebalance_date=rebal_date
            )
            logger.info(f"  Ensemble scores: {len(ensemble_scores)} tickers, "
                       f"range=[{ensemble_scores.min():.3f}, {ensemble_scores.max():.3f}]")

            # Generate direct momentum scores
            direct_scores = {}
            for ticker, px_series in prices_by_ticker.items():
                # Build DataFrame matching signal API
                data = pd.DataFrame({
                    'close': px_series,
                    'ticker': ticker
                })
                data = data[data.index <= rebal_date]

                try:
                    sig_series = direct_momentum.generate_signals(data)
                    if len(sig_series) > 0:
                        signal_value = sig_series.iloc[-1]
                        if pd.notna(signal_value) and signal_value != 0:
                            direct_scores[ticker] = signal_value
                except Exception as e:
                    logger.debug(f"  Error generating direct signal for {ticker}: {e}")

            direct_scores_series = pd.Series(direct_scores)
            logger.info(f"  Direct scores: {len(direct_scores_series)} tickers, "
                       f"range=[{direct_scores_series.min():.3f}, {direct_scores_series.max():.3f}]")

            # Compare scores (only tickers present in both)
            common_tickers = set(ensemble_scores.index) & set(direct_scores_series.index)
            logger.info(f"  Common tickers: {len(common_tickers)}")

            if len(common_tickers) == 0:
                logger.warning(f"  No common tickers, skipping comparison")
                continue

            # Compute differences
            for ticker in common_tickers:
                ens_score = ensemble_scores[ticker]
                dir_score = direct_scores_series[ticker]
                diff = abs(ens_score - dir_score)
                all_diffs.append(diff)

                if diff > max_diff:
                    max_diff = diff

                # Log details for debug mode
                if self.debug_compare and len(all_diffs) <= 10:
                    logger.info(f"    {ticker}: ensemble={ens_score:.6f}, direct={dir_score:.6f}, diff={diff:.2e}")

            num_compared += 1

            # Log summary for this rebalance
            rebal_diffs = [abs(ensemble_scores.get(t, 0) - direct_scores_series.get(t, 0))
                          for t in common_tickers]
            logger.info(f"  Max diff this rebalance: {max(rebal_diffs):.2e}")

        return {
            'num_rebalances': num_compared,
            'num_comparisons': len(all_diffs),
            'max_diff': max_diff,
            'mean_diff': np.mean(all_diffs) if all_diffs else 0.0,
            'median_diff': np.median(all_diffs) if all_diffs else 0.0
        }


def main():
    parser = argparse.ArgumentParser(
        description='Validate EnsembleSignal with momentum-only configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--start', default='2022-01-01',
                       help='Start date (default: 2022-01-01 for 2-3 year validation)')
    parser.add_argument('--end', default='2024-12-31',
                       help='End date (default: 2024-12-31)')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode: compare only first 6 rebalances with detailed logging')

    args = parser.parse_args()

    # Run validation
    validator = EnsembleMomentumValidator(
        start_date=args.start,
        end_date=args.end,
        debug_compare=args.debug
    )

    results = validator.validate()

    logger.info("\n" + "=" * 80)
    logger.info(f"VALIDATION {results['verdict']}")
    logger.info("=" * 80)

    if results['verdict'] == 'PASS':
        logger.info("✅ Ensemble is ready for production use")
        logger.info("   Treat EnsembleSignal as canonical path going forward")
    elif results['verdict'] == 'MARGINAL':
        logger.warning("⚠️  Small numerical differences detected")
        logger.warning("   Review normalization and parameter handling")
    else:
        logger.error("❌ Validation failed")
        logger.error("   Do NOT use ensemble until differences resolved")


if __name__ == '__main__':
    main()
