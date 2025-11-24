"""
CrossSectionalQuality v1 - Proper QMJ Implementation

Implementation of Asness-Frazzini-Pedersen (2018) Quality Minus Junk
with CORRECT cross-sectional ranking methodology.

Supersedes: InstitutionalQuality v0 (time-series scaled, deprecated)
Specification: See docs/QUALITY_SPEC.md for complete mathematical definition

Key Difference from v0:
- v0: Time-series ranking (each stock vs its own history)
- v1: Cross-sectional ranking (all stocks vs each other at each rebalance)

This is the ONLY Quality signal eligible for:
- S&P 500 baseline testing (Phase 1)
- Hyperparameter optimization (Phase 3)
- Ensemble construction (Phase 5)

Academic Reference:
- Asness, C., Frazzini, A., & Pedersen, L. H. (2018). "Quality Minus Junk."
  Review of Accounting Studies.

Implementation Date: 2025-11-21
Version: 1.0
"""

from typing import Dict, Any, Optional, List, Tuple, Sequence
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from core.institutional_base import InstitutionalSignal
from data.data_manager import DataManager
from config import get_logger

logger = get_logger(__name__)


class CrossSectionalQuality(InstitutionalSignal):
    """
    Quality Minus Junk signal with proper cross-sectional ranking.

    Methodology (per QUALITY_SPEC.md §2):
    1. At each month-end rebalance date t
    2. For all stocks i in U_SP500_PIT(t):
       a. Compute raw metrics: P_i(t), G_i(t), S_i(t)
       b. Cross-sectional z-score: Z_P_i(t), Z_G_i(t), Z_S_i(t)
       c. Composite: Q_i(t) = w_P * Z_P + w_G * Z_G + w_S * Z_S
       d. Percentile rank: R_i(t) in [0, 1]
       e. Signal: 2 * R_i(t) - 1 in [-1, 1]
    3. Forward-fill signals to daily

    Parameters:
        w_profitability: Weight on profitability component (default: 0.4)
        w_growth: Weight on growth component (default: 0.3)
        w_safety: Weight on safety component (default: 0.3)
        winsorize_pct: [lower, upper] percentiles for outlier handling (default: [5, 95])
        quintiles: Use quintile discretization vs continuous (default: True)
        min_coverage: Minimum % of universe with valid scores (default: 0.5)

    Constraints:
        - w_profitability + w_growth + w_safety == 1.0
        - Rebalancing: Monthly (month-end only)
        - Point-in-time: 33-day filing lag enforced

    See docs/QUALITY_SPEC.md for complete mathematical specification.
    """

    def __init__(self,
                 params: Dict[str, Any],
                 data_manager: Optional[DataManager] = None,
                 name: str = 'CrossSectionalQuality'):
        # Copy params to avoid mutation
        params = params.copy()

        # Set defaults (QUALITY_SPEC.md §2.4)
        params.setdefault('w_profitability', 0.4)
        params.setdefault('w_growth', 0.3)
        params.setdefault('w_safety', 0.3)
        params.setdefault('winsorize_pct', [5, 95])
        params.setdefault('quintiles', True)
        params.setdefault('min_coverage', 0.5)  # At least 50% of universe
        params.setdefault('rebalance_frequency', 'monthly')

        # Validate weight constraint
        w_sum = params['w_profitability'] + params['w_growth'] + params['w_safety']
        if not np.isclose(w_sum, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {w_sum}")

        super().__init__(params, name)

        self.data_manager = data_manager or DataManager()

        # Component weights (QUALITY_SPEC.md §2.4)
        self.w_profitability = params['w_profitability']
        self.w_growth = params['w_growth']
        self.w_safety = params['w_safety']

        # Configuration
        self.winsorize_pct = params['winsorize_pct']
        self.quintiles = params['quintiles']
        self.min_coverage = params['min_coverage']

        # Filing lag (QUALITY_SPEC.md §2.1 - Point-in-Time Constraint)
        self.filing_lag_days = 33  # Conservative minimum for quarterly filings

        logger.info(f"Initialized {name} v1 (cross-sectional)")
        logger.info(f"  Weights: P={self.w_profitability:.1f}, "
                   f"G={self.w_growth:.1f}, S={self.w_safety:.1f}")
        logger.info(f"  Quintiles: {self.quintiles}, MinCoverage: {self.min_coverage:.0%}")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate quality signals for a SINGLE stock (compatibility method).

        Note: This method exists for API compatibility with single-stock backtests.
        For proper cross-sectional Quality signals, use generate_signals_cross_sectional()
        with the full universe.

        This implementation computes quality metrics for the single stock but cannot
        perform proper cross-sectional ranking without other stocks. Signals will be
        neutral (0) to avoid misleading results.

        Args:
            data: DataFrame with 'close' prices and 'ticker'

        Returns:
            Series with neutral signals (0) - cross-sectional ranking requires full universe
        """
        logger.warning(f"CrossSectionalQuality.generate_signals() called for single stock.")
        logger.warning(f"Cross-sectional ranking requires full universe.")
        logger.warning(f"Returning neutral signals (0). Use generate_signals_cross_sectional() instead.")

        return pd.Series(0.0, index=data.index, dtype=float)

    def generate_signals_cross_sectional(self,
                                        universe_tickers: List[str],
                                        rebalance_dates: pd.DatetimeIndex,
                                        start_date: str,
                                        end_date: str) -> pd.DataFrame:
        """
        Generate cross-sectional quality signals for entire universe.

        This is the PRIMARY method for CrossSectionalQuality.
        Implements proper academic QMJ methodology per QUALITY_SPEC.md §2.

        Args:
            universe_tickers: List of tickers in universe (e.g., S&P 500 on start_date)
            rebalance_dates: Monthly rebalance dates (month-ends)
            start_date: Start date for signal generation
            end_date: End date for signal generation

        Returns:
            DataFrame with:
                - Index: All trading days from start_date to end_date
                - Columns: Tickers in universe
                - Values: Signals in [-1, 1] (or quintile values if quintiles=True)

        Process (QUALITY_SPEC.md §2):
            1. For each rebalance date t:
                a. Fetch fundamentals for all tickers (PIT with filing lag)
                b. Compute raw P/G/S scores for each ticker
                c. Cross-sectional z-score standardization
                d. Weighted composite quality score
                e. Percentile rank to signal [-1, 1]
            2. Forward-fill signals to daily
        """
        logger.info("="*80)
        logger.info(f"CrossSectionalQuality v1: Generating signals")
        logger.info(f"  Universe: {len(universe_tickers)} tickers")
        logger.info(f"  Rebalance dates: {len(rebalance_dates)} month-ends")
        logger.info(f"  Period: {start_date} to {end_date}")
        logger.info("="*80)

        # Get all trading days for forward-fill
        all_trading_days = pd.date_range(start=start_date, end=end_date, freq='B')

        # Initialize signal DataFrame
        signals_df = pd.DataFrame(0.0, index=all_trading_days, columns=universe_tickers)

        # Process each rebalance date
        for i, rebal_date in enumerate(rebalance_dates):
            logger.info(f"\nRebalance {i+1}/{len(rebalance_dates)}: {rebal_date.date()}")

            # Compute cross-sectional quality scores at this date
            quality_scores = self._compute_quality_scores_at_date(
                tickers=universe_tickers,
                as_of_date=rebal_date
            )

            if len(quality_scores) == 0:
                logger.warning(f"  No valid quality scores for {rebal_date.date()}, skipping")
                continue

            # Check coverage
            coverage_pct = len(quality_scores) / len(universe_tickers)
            logger.info(f"  Coverage: {len(quality_scores)}/{len(universe_tickers)} "
                       f"({coverage_pct:.1%})")

            if coverage_pct < self.min_coverage:
                logger.warning(f"  Coverage {coverage_pct:.1%} < minimum {self.min_coverage:.1%}, skipping")
                continue

            # Convert quality scores to signals (QUALITY_SPEC.md §2.5)
            signals = self._quality_scores_to_signals(quality_scores)

            # Assign signals (forward-fill until next rebalance)
            next_rebal_idx = i + 1
            if next_rebal_idx < len(rebalance_dates):
                next_rebal_date = rebalance_dates[next_rebal_idx]
                fill_mask = (signals_df.index >= rebal_date) & (signals_df.index < next_rebal_date)
            else:
                # Last rebalance: fill to end
                fill_mask = signals_df.index >= rebal_date

            for ticker, signal_value in signals.items():
                if ticker in signals_df.columns:
                    signals_df.loc[fill_mask, ticker] = signal_value

            logger.info(f"  Signals: mean={signals.mean():.3f}, "
                       f"std={signals.std():.3f}, "
                       f"range=[{signals.min():.2f}, {signals.max():.2f}]")

        logger.info("\n" + "="*80)
        logger.info("CrossSectionalQuality v1: Signal generation complete")
        logger.info("="*80)

        return signals_df

    def generate_cross_sectional_scores(
        self,
        rebal_date: pd.Timestamp,
        universe: Sequence[str],
        data_manager: "DataManager",
    ) -> pd.Series:
        """
        Generate cross-sectional quality scores for universe at a single rebalance date.

        **Backtest Integration Method** (Phase 3 Milestone 2)

        This implementation reuses the proven CrossSectionalQuality v1 methodology
        by calling into _compute_quality_scores_at_date() and _quality_scores_to_signals().
        Semantics are intentionally aligned with:
        - test_cross_sectional_quality_v1.py acceptance tests
        - docs/QUALITY_SPEC.md mathematical specification

        Args:
            rebal_date: Rebalance date (all data must be PIT as-of this date)
            universe: List of ticker symbols to score
            data_manager: DataManager instance for fetching fundamental data

        Returns:
            pd.Series indexed by ticker with quality signals in [-1, 1]
            (subset of universe - tickers with insufficient data excluded)

        PIT Correctness:
            - Enforces 33-day filing lag (self.filing_lag_days)
            - Uses DataManager.get_fundamentals(..., as_of_date=rebal_date)
            - No lookahead bias - all data known by rebal_date

        Data Requirements:
            - Fundamentals: profitability (ROE, ROA, GP/A)
            - Fundamentals: growth (revenue growth, earnings growth)
            - Fundamentals: safety (leverage, volatility)
            - Minimum 4 quarters of data per ticker

        Lookback:
            - 3-year fundamental history (hardcoded in _compute_quality_scores_at_date)
            - This matches the existing v1 implementation

        Notes:
            - Same semantics as generate_signals_cross_sectional() for a single date
            - Coverage check NOT applied here (applied at backtest level if needed)
            - Returns empty Series if no tickers have valid quality scores
        """
        logger.debug(f"CrossSectionalQuality: Generating scores for {len(universe)} tickers at {rebal_date.date()}")

        # Compute composite quality scores using existing proven method
        quality_scores = self._compute_quality_scores_at_date(
            tickers=list(universe),
            as_of_date=rebal_date
        )

        if len(quality_scores) == 0:
            logger.warning(f"CrossSectionalQuality: No valid quality scores at {rebal_date.date()}")
            return pd.Series(dtype=float)

        # Convert quality scores to trading signals using existing method
        signals = self._quality_scores_to_signals(quality_scores)

        logger.debug(f"CrossSectionalQuality: Generated {len(signals)} signals, "
                    f"coverage={len(signals)/len(universe):.1%}")

        return signals

    def _compute_quality_scores_at_date(self,
                                       tickers: List[str],
                                       as_of_date: pd.Timestamp) -> pd.Series:
        """
        Compute composite quality scores for all tickers at a single date.

        Implements QUALITY_SPEC.md §2.2 (Raw Metrics) and §2.3 (Cross-Sectional Standardization).

        Args:
            tickers: List of tickers
            as_of_date: Date for point-in-time quality computation

        Returns:
            Series with composite quality scores Q_i(t), index=tickers with valid data
        """
        # Enforce filing lag (QUALITY_SPEC.md §2.1)
        pit_date = as_of_date - timedelta(days=self.filing_lag_days)
        pit_date_str = pit_date.strftime('%Y-%m-%d')
        as_of_str = as_of_date.strftime('%Y-%m-%d')

        # Collect raw scores for all tickers
        profitability_scores = {}
        growth_scores = {}
        safety_scores = {}

        for ticker in tickers:
            try:
                # Get fundamentals with PIT constraint
                fundamentals = self.data_manager.get_fundamentals(
                    ticker,
                    start_date=(as_of_date - timedelta(days=365*3)).strftime('%Y-%m-%d'),  # 3Y history
                    end_date=pit_date_str,
                    dimension='ARQ',  # As-reported quarterly
                    as_of_date=as_of_str
                )

                if len(fundamentals) < 4:  # Need at least 4 quarters
                    continue

                # Compute raw scores (QUALITY_SPEC.md §2.2)
                P = self._compute_profitability(fundamentals)
                G = self._compute_growth(fundamentals)
                S = self._compute_safety(fundamentals)

                # Store if valid
                if pd.notna(P):
                    profitability_scores[ticker] = P
                if pd.notna(G):
                    growth_scores[ticker] = G
                if pd.notna(S):
                    safety_scores[ticker] = S

            except Exception as e:
                logger.debug(f"Error computing quality for {ticker}: {e}")
                continue

        # Convert to Series
        P_series = pd.Series(profitability_scores)
        G_series = pd.Series(growth_scores)
        S_series = pd.Series(safety_scores)

        logger.debug(f"  Raw scores: P={len(P_series)}, G={len(G_series)}, S={len(S_series)}")

        # Cross-sectional standardization (QUALITY_SPEC.md §2.3)
        Z_P = self._cross_sectional_zscore(P_series)
        Z_G = self._cross_sectional_zscore(G_series)
        Z_S = self._cross_sectional_zscore(S_series)

        # Combine into composite quality (QUALITY_SPEC.md §2.4)
        # Q_i(t) = w_P * Z_P + w_G * Z_G + w_S * Z_S
        all_tickers = set(Z_P.index) | set(Z_G.index) | set(Z_S.index)
        quality_scores = pd.Series(0.0, index=list(all_tickers))

        for ticker in all_tickers:
            z_p = Z_P.get(ticker, 0.0)
            z_g = Z_G.get(ticker, 0.0)
            z_s = Z_S.get(ticker, 0.0)
            quality_scores[ticker] = (self.w_profitability * z_p +
                                     self.w_growth * z_g +
                                     self.w_safety * z_s)

        return quality_scores

    def _compute_profitability(self, fundamentals: pd.DataFrame) -> float:
        """
        Compute profitability score (QUALITY_SPEC.md §2.2).

        Components:
        - ROE (Return on Equity)
        - ROA (Return on Assets)
        - GP/A (Gross Profit / Total Assets)

        Returns:
            Mean of available metrics (single scalar value for most recent quarter)
        """
        scores = []

        # Most recent quarter
        latest = fundamentals.iloc[-1]

        # ROE
        if 'roe' in fundamentals.columns and pd.notna(latest.get('roe')):
            roe = latest['roe']
            if np.isfinite(roe):
                scores.append(roe)
        elif 'netinc' in fundamentals.columns and 'equity' in fundamentals.columns:
            # Compute ROE from raw fields
            netinc = latest.get('netinc')
            equity = latest.get('equity')
            if pd.notna(netinc) and pd.notna(equity) and equity != 0:
                roe = netinc / equity
                if np.isfinite(roe):
                    scores.append(roe)

        # ROA
        if 'roa' in fundamentals.columns and pd.notna(latest.get('roa')):
            roa = latest['roa']
            if np.isfinite(roa):
                scores.append(roa)
        elif 'netinc' in fundamentals.columns and 'assets' in fundamentals.columns:
            # Compute ROA from raw fields
            netinc = latest.get('netinc')
            assets = latest.get('assets')
            if pd.notna(netinc) and pd.notna(assets) and assets != 0:
                roa = netinc / assets
                if np.isfinite(roa):
                    scores.append(roa)

        # Gross Profit / Assets
        if 'gp' in fundamentals.columns and 'assets' in fundamentals.columns:
            gp = latest.get('gp')
            assets = latest.get('assets')
            if pd.notna(gp) and pd.notna(assets) and assets != 0:
                gp_a = gp / assets
                if np.isfinite(gp_a):
                    scores.append(gp_a)

        return np.mean(scores) if len(scores) > 0 else np.nan

    def _compute_growth(self, fundamentals: pd.DataFrame) -> float:
        """
        Compute growth score (QUALITY_SPEC.md §2.2).

        Components:
        - Revenue YoY growth (4 quarters)
        - Net Income YoY growth (4 quarters)

        Returns:
            Mean of available growth metrics
        """
        scores = []

        if len(fundamentals) < 5:  # Need 5+ quarters for YoY
            return np.nan

        # Revenue growth
        if 'revenue' in fundamentals.columns:
            rev = fundamentals['revenue']
            if len(rev) >= 5:
                rev_growth = (rev.iloc[-1] - rev.iloc[-5]) / abs(rev.iloc[-5])
                if pd.notna(rev_growth) and np.isfinite(rev_growth):
                    # Cap extreme growth (QUALITY_SPEC.md §2.2)
                    rev_growth = np.clip(rev_growth, -2.0, 2.0)
                    scores.append(rev_growth)

        # Net income growth
        if 'netinc' in fundamentals.columns:
            ni = fundamentals['netinc']
            if len(ni) >= 5:
                ni_growth = (ni.iloc[-1] - ni.iloc[-5]) / (abs(ni.iloc[-5]) + 1e-9)
                if pd.notna(ni_growth) and np.isfinite(ni_growth):
                    # Cap extreme growth
                    ni_growth = np.clip(ni_growth, -2.0, 2.0)
                    scores.append(ni_growth)

        return np.mean(scores) if len(scores) > 0 else np.nan

    def _compute_safety(self, fundamentals: pd.DataFrame) -> float:
        """
        Compute safety score (QUALITY_SPEC.md §2.2).

        Components:
        - Low leverage (inverted D/E ratio)
        - Low earnings volatility (inverted ROE std)
        - Profitability consistency (positive ROE indicator)

        Returns:
            Mean of available safety metrics
        """
        scores = []

        latest = fundamentals.iloc[-1]

        # Low leverage (invert: low D/E = high safety)
        if 'de' in fundamentals.columns and pd.notna(latest.get('de')):
            de = latest['de']
            if np.isfinite(de):
                scores.append(-de)  # Inverted

        # ROE stability (invert volatility)
        if 'roe' in fundamentals.columns and len(fundamentals) >= 8:
            roe_vol = fundamentals['roe'].tail(8).std()
            if pd.notna(roe_vol) and np.isfinite(roe_vol) and roe_vol > 0:
                scores.append(-roe_vol)  # Inverted

        # Positive ROE (binary indicator)
        if 'roe' in fundamentals.columns and pd.notna(latest.get('roe')):
            roe = latest['roe']
            if np.isfinite(roe):
                scores.append(1.0 if roe > 0 else 0.0)

        return np.mean(scores) if len(scores) > 0 else np.nan

    def _cross_sectional_zscore(self, values: pd.Series) -> pd.Series:
        """
        Cross-sectional z-score standardization (QUALITY_SPEC.md §2.3).

        Process:
        1. Winsorize to handle outliers
        2. Compute mean and std across all stocks
        3. Standardize: Z_i = (x_i - mean) / std
        4. Handle edge cases (std=0, too few stocks)

        Args:
            values: Series of raw values, index=tickers

        Returns:
            Series of z-scores, index=tickers
        """
        if len(values) < 5:  # Too few stocks for meaningful standardization
            return pd.Series(0.0, index=values.index)

        # Winsorize (QUALITY_SPEC.md §2.3)
        lower_pct, upper_pct = self.winsorize_pct
        values_wins = values.copy()
        lower_bound = values.quantile(lower_pct / 100.0)
        upper_bound = values.quantile(upper_pct / 100.0)
        values_wins = values_wins.clip(lower=lower_bound, upper=upper_bound)

        # Z-score
        mean = values_wins.mean()
        std = values_wins.std()

        if std == 0 or not np.isfinite(std):
            # All values identical or invalid
            return pd.Series(0.0, index=values.index)

        z_scores = (values_wins - mean) / std

        # Handle infinities
        z_scores = z_scores.replace([np.inf, -np.inf], 0.0)

        return z_scores

    def _quality_scores_to_signals(self, quality_scores: pd.Series) -> pd.Series:
        """
        Convert quality scores to trading signals (QUALITY_SPEC.md §2.5).

        Process:
        1. Percentile rank: R_i(t) in [0, 1]
        2. Convert to signal: Signal = 2 * R - 1 in [-1, 1]
        3. Optional: Discretize to quintiles

        Args:
            quality_scores: Series of composite quality scores Q_i(t)

        Returns:
            Series of signals in [-1, 1]
        """
        # Percentile rank (QUALITY_SPEC.md §2.5)
        ranks = quality_scores.rank(pct=True)

        # Convert to [-1, 1]
        signals = 2.0 * ranks - 1.0

        # Optional: Discretize to quintiles (QUALITY_SPEC.md §2.5)
        if self.quintiles:
            signals = self._to_quintiles(signals)

        return signals

    def _to_quintiles(self, signals: pd.Series) -> pd.Series:
        """
        Discretize continuous signals to quintile values.

        Mapping (QUALITY_SPEC.md §2.5):
        - [0.0, 0.2): -1.0 (quintile 1, worst quality)
        - [0.2, 0.4): -0.5 (quintile 2)
        - [0.4, 0.6):  0.0 (quintile 3, neutral)
        - [0.6, 0.8): +0.5 (quintile 4)
        - [0.8, 1.0]: +1.0 (quintile 5, best quality)
        """
        # Convert [-1, 1] back to [0, 1] for quintile assignment
        ranks = (signals + 1.0) / 2.0

        quintile_signals = pd.Series(0.0, index=signals.index)
        quintile_signals[ranks < 0.2] = -1.0
        quintile_signals[(ranks >= 0.2) & (ranks < 0.4)] = -0.5
        quintile_signals[(ranks >= 0.4) & (ranks < 0.6)] = 0.0
        quintile_signals[(ranks >= 0.6) & (ranks < 0.8)] = 0.5
        quintile_signals[ranks >= 0.8] = 1.0

        return quintile_signals

    def get_parameter_space(self) -> Dict[str, tuple]:
        """
        Define parameter space for optimization (QUALITY_SPEC.md §4).

        Returns:
            Dict with Optuna-compatible parameter specifications
        """
        return {
            # Component weights (must sum to 1.0)
            'w_profitability': ('float', 0.2, 0.6),
            'w_growth': ('float', 0.1, 0.5),
            'w_safety': ('float', 0.1, 0.5),

            # Winsorization bounds
            'winsorize_pct': ('categorical', [[1, 99], [5, 95], [10, 90]]),

            # Signal discretization
            'quintiles': ('categorical', [True, False]),

            # Coverage threshold
            'min_coverage': ('float', 0.3, 0.7),
        }

    def __repr__(self) -> str:
        return (f"CrossSectionalQuality(v1, "
                f"weights=[{self.w_profitability:.1f}, {self.w_growth:.1f}, {self.w_safety:.1f}], "
                f"quintiles={self.quintiles})")


if __name__ == '__main__':
    print("CrossSectionalQuality v1 - Proper QMJ Implementation")
    print("="*60)
    print()
    print("Methodology: Cross-sectional ranking (academic QMJ)")
    print("Supersedes: InstitutionalQuality v0 (time-series, deprecated)")
    print()
    print("Components:")
    print("  - Profitability (40%): ROE, ROA, GP/Assets")
    print("  - Growth (30%): Revenue YoY, Net Income YoY")
    print("  - Safety (30%): Low leverage, low volatility, positive ROE")
    print()
    print("Key Features:")
    print("  - Cross-sectional z-score at each month-end")
    print("  - Point-in-time enforcement (33-day filing lag)")
    print("  - Quintile discretization for clean signals")
    print("  - Coverage checks (minimum 50% of universe)")
    print()
    print("Specification: docs/QUALITY_SPEC.md")
    print("Version: 1.0")
    print("Date: 2025-11-21")
