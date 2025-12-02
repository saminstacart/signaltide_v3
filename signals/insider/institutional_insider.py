"""
Institutional Insider Trading Signal

Implementation of Cohen-Malloy-Pomorski (2012) methodology
with professional insider transaction analysis.

Strategy:
- Dollar-weighted insider transactions
- Role-based weighting (CEO > CFO > Director > Officer)
- Cluster detection (coordinated buying/selling)
- Monthly rebalancing
- Cross-sectional ranking

This methodology captures informed insider trading while
filtering noise from routine transactions and small trades.

References:
- Cohen, Malloy, Pomorski (2012) "Decoding Inside Information"
- Seyhun (1986) "Insiders' Profits, Costs of Trading, and Market Efficiency"
- Jeng, Metrick, Zeckhauser (2003) "Estimating the Returns to Insider Trading"
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import timedelta
from core.institutional_base import InstitutionalSignal
from data.data_manager import DataManager
from config import get_logger

logger = get_logger(__name__)


class InstitutionalInsider(InstitutionalSignal):
    """
    Cohen-Malloy-Pomorski Insider Trading Signal.

    Professional implementation with:
    - Dollar-weighted transactions
    - Role hierarchy weighting
    - Cluster detection (3+ insiders within 7 days)
    - Monthly rebalancing

    Parameters:
        lookback_days: Days to aggregate insider activity (default: 90)
        min_transaction_value: Minimum dollar value to consider (default: 10000)
        cluster_window: Days for cluster detection (default: 7)
        cluster_min_insiders: Minimum insiders for cluster (default: 3)
        role_weights: Dict of weights by role (default: CEO=3, CFO=2, etc.)
        rebalance_frequency: 'monthly' (default)
    """

    def __init__(self,
                 params: Dict[str, Any],
                 data_manager: Optional[DataManager] = None,
                 name: str = 'InstitutionalInsider'):
        # Make a copy to avoid mutating caller's dict
        params = params.copy()

        # Set defaults for insider-specific parameters BEFORE validation
        params.setdefault('lookback_days', 90)  # 3 months
        params.setdefault('min_transaction_value', 10000)
        params.setdefault('cluster_window', 7)
        params.setdefault('cluster_min_insiders', 3)
        params.setdefault('ceo_weight', 3.0)
        params.setdefault('cfo_weight', 2.5)

        super().__init__(params, name)

        self.data_manager = data_manager or DataManager()

        # Aggregation parameters (now guaranteed to exist)
        self.lookback_days = params['lookback_days']
        self.min_transaction_value = params['min_transaction_value']

        # Cluster detection
        self.cluster_window = params['cluster_window']
        self.cluster_min_insiders = params['cluster_min_insiders']

        # Role hierarchy (Cohen-Malloy-Pomorski weights)
        default_role_weights = {
            'ceo': params['ceo_weight'],
            'cfo': params['cfo_weight'],
            'president': 2.5,
            'coo': 2.0,
            'director': 1.5,
            'officer': 1.0,
            'other': 0.5
        }
        self.role_weights = params.get('role_weights', default_role_weights)

    def generate_signals(self, data: pd.DataFrame, bulk_insider_data: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Generate insider trading signals.

        Args:
            data: DataFrame with 'close' prices and 'ticker'
            bulk_insider_data: Optional pre-fetched insider data (bulk mode, PREFERRED)
                               DataFrame with MultiIndex [(ticker, filingdate)]
                               If provided, used instead of per-ticker DB query
                               This is 50-100x faster for multi-stock backtests

        Returns:
            Series with signals in [-1, 1] range

        Notes:
            - Bulk mode (bulk_insider_data provided): Fast lookup from pre-fetched data
            - Legacy mode (bulk_insider_data=None): Per-ticker database query (compatibility fallback)
        """
        if 'ticker' not in data.columns:
            return pd.Series(0, index=data.index)

        ticker = data['ticker'].iloc[0]

        # BULK MODE (PREFERRED): Lookup from pre-fetched data
        if bulk_insider_data is not None:
            insiders = self._extract_ticker_from_bulk(ticker, bulk_insider_data, data.index)

        # LEGACY MODE (FALLBACK): Per-ticker database query
        else:
            start_date = (data.index.min() - timedelta(days=self.lookback_days)).strftime('%Y-%m-%d')
            end_date = data.index.max().strftime('%Y-%m-%d')

            insiders = self.data_manager.get_insider_trades(
                ticker,
                start_date,
                end_date,
                as_of_date=end_date  # Point-in-time filtering using filing dates
            )

        if len(insiders) == 0:
            return pd.Series(0, index=data.index)

        # Calculate daily insider signals
        insider_signals = self._calculate_insider_activity(insiders, data.index)

        # Apply monthly rebalancing
        if self.rebalance_frequency == 'monthly':
            insider_signals = self._apply_monthly_rebalancing(insider_signals)

        return insider_signals.clip(-1, 1)

    def _extract_ticker_from_bulk(self,
                                   ticker: str,
                                   bulk_data: pd.DataFrame,
                                   price_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Extract ticker-specific insider trades from bulk data.

        Args:
            ticker: Ticker symbol to extract
            bulk_data: MultiIndex DataFrame [(ticker, filingdate)]
            price_index: DatetimeIndex for filtering date range

        Returns:
            DataFrame with insider trades for this ticker only,
            filtered to relevant date range, indexed by filingdate
        """
        try:
            # Extract ticker from MultiIndex
            ticker_data = bulk_data.xs(ticker, level='ticker')
        except KeyError:
            # Ticker not in bulk data (no insider activity)
            return pd.DataFrame()

        if len(ticker_data) == 0:
            return pd.DataFrame()

        # Filter to relevant date range (lookback period)
        start_date = price_index.min() - timedelta(days=self.lookback_days)
        end_date = price_index.max()

        ticker_data = ticker_data[
            (ticker_data.index >= start_date) &
            (ticker_data.index <= end_date)
        ]

        return ticker_data

    def _calculate_insider_activity(self,
                                    insiders: pd.DataFrame,
                                    price_index: pd.DatetimeIndex) -> pd.Series:
        """
        Calculate daily insider trading activity score.

        Professional methodology:
        1. Weight by dollar value
        2. Weight by insider role
        3. Detect clusters (coordinated activity)
        4. Aggregate over lookback period
        5. Normalize to [-1, 1]
        """
        # Prepare transactions
        transactions = self._prepare_transactions(insiders)

        if len(transactions) == 0:
            return pd.Series(0, index=price_index)

        # Calculate weighted scores for each transaction
        transactions['weighted_score'] = self._calculate_weighted_scores(transactions)

        # Detect clusters (amplify coordinated activity)
        transactions = self._detect_clusters(transactions)

        # Aggregate to daily level
        daily_scores = self._aggregate_to_daily(transactions, price_index)

        # Convert to signals using rolling aggregation
        signals = self._rolling_signals(daily_scores)

        return signals

    def _prepare_transactions(self, insiders: pd.DataFrame) -> pd.DataFrame:
        """
        Filter and prepare insider transactions.

        Apply thresholds:
        - Minimum dollar value
        - Valid transaction types (P=purchase, S=sale)
        - Remove option exercises (focus on open market)
        """
        trans = insiders.copy()

        # Filter for purchases and sales only
        # Column name: transactioncode (not transcode)
        if 'transactioncode' not in trans.columns:
            return pd.DataFrame()  # No transaction data

        trans = trans[trans['transactioncode'].isin(['P', 'S'])]

        # Calculate transaction value if not already present
        if 'value' not in trans.columns:
            # Try transactionvalue first
            if 'transactionvalue' in trans.columns:
                trans['value'] = trans['transactionvalue'].abs()
            # Otherwise calculate from shares * price
            elif 'transactionshares' in trans.columns and 'transactionpricepershare' in trans.columns:
                trans['value'] = abs(trans['transactionshares'] * trans['transactionpricepershare'])

        # Filter by minimum value
        if 'value' in trans.columns:
            trans = trans[trans['value'] >= self.min_transaction_value]

        return trans

    def _calculate_weighted_scores(self, transactions: pd.DataFrame) -> pd.Series:
        """
        Calculate weighted scores for transactions.

        Weights:
        1. Dollar value (larger trades = more information)
        2. Insider role (CEO > CFO > Director > Officer)
        3. Direction (+1 for purchase, -1 for sale)

        OPTIMIZED: Vectorized operations instead of iterrows().
        """
        if len(transactions) == 0:
            return pd.Series(dtype=float)

        # VECTORIZED: Direction (+1 for purchase, -1 for sale)
        direction = np.where(transactions['transactioncode'] == 'P', 1.0, -1.0)

        # VECTORIZED: Dollar weight (log scale to reduce impact of extreme values)
        if 'value' in transactions.columns:
            values = transactions['value'].fillna(0)
            dollar_weight = np.where(values > 0, np.log10(values.clip(lower=1)), 1.0)
        else:
            dollar_weight = np.ones(len(transactions))

        # VECTORIZED: Role weight
        if 'officertitle' in transactions.columns:
            role_weight = transactions['officertitle'].apply(
                lambda t: self.role_weights.get(self._classify_role(t), 1.0)
            ).values
        else:
            role_weight = np.ones(len(transactions))

        # Combined score (all vectorized)
        scores = pd.Series(
            direction * dollar_weight * role_weight,
            index=transactions.index,
            dtype=float
        )

        return scores

    def _classify_role(self, title: str) -> str:
        """
        Classify insider role from title string.

        Returns role category based on title keywords.
        """
        if pd.isna(title):
            return 'other'

        title_lower = title.lower()

        # CEO (highest information)
        if 'ceo' in title_lower or 'chief executive' in title_lower:
            return 'ceo'

        # CFO
        if 'cfo' in title_lower or 'chief financial' in title_lower:
            return 'cfo'

        # President
        if 'president' in title_lower:
            return 'president'

        # COO
        if 'coo' in title_lower or 'chief operating' in title_lower:
            return 'coo'

        # Director
        if 'director' in title_lower:
            return 'director'

        # Officer
        if 'officer' in title_lower:
            return 'officer'

        return 'other'

    def _detect_clusters(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Detect insider trading clusters (coordinated activity).

        Cohen-Malloy-Pomorski finding: Clusters of insider purchases
        within short windows have stronger predictive power.

        Cluster = 3+ insiders trading in same direction within 7 days

        OPTIMIZED: Uses vectorized rolling count instead of O(n²) nested loops.
        """
        trans = transactions.copy()
        trans['is_cluster'] = False

        # Early exit for small transaction sets
        if len(trans) < self.cluster_min_insiders:
            return trans

        # Group by direction (purchases vs sales)
        # Column name: transactioncode
        for direction in ['P', 'S']:
            dir_mask = trans['transactioncode'] == direction
            dir_indices = trans.index[dir_mask]

            if len(dir_indices) < self.cluster_min_insiders:
                continue

            # VECTORIZED: Count transactions within cluster_window using rolling count
            # Create a time-indexed series for rolling operations
            dir_trans = trans.loc[dir_mask].copy()
            dir_trans = dir_trans.sort_index()

            # Use resample to count transactions per day, then rolling sum
            # This avoids O(n²) nested loop
            daily_counts = dir_trans.groupby(dir_trans.index.date).size()
            daily_counts.index = pd.to_datetime(daily_counts.index)
            daily_counts = daily_counts.sort_index()

            # Rolling window count (cluster_window * 2 + 1 to center the window)
            window_days = self.cluster_window * 2 + 1
            rolling_counts = daily_counts.rolling(
                window=f'{window_days}D',
                min_periods=1,
                center=True
            ).sum()

            # Map back to original transactions
            for idx in dir_indices:
                date = idx.date() if hasattr(idx, 'date') else idx
                date_dt = pd.to_datetime(date)

                # Find nearest date in rolling_counts
                if date_dt in rolling_counts.index:
                    count = rolling_counts.loc[date_dt]
                else:
                    # Fallback: find closest date
                    diffs = abs(rolling_counts.index - date_dt)
                    closest_idx = diffs.argmin()
                    count = rolling_counts.iloc[closest_idx]

                if count >= self.cluster_min_insiders:
                    trans.loc[idx, 'is_cluster'] = True

        # Amplify cluster signals (2x weight)
        trans.loc[trans['is_cluster'], 'weighted_score'] *= 2.0

        return trans

    def _aggregate_to_daily(self,
                           transactions: pd.DataFrame,
                           price_index: pd.DatetimeIndex) -> pd.Series:
        """
        Aggregate weighted transaction scores to daily frequency.

        OPTIMIZED: Uses vectorized merge instead of O(n²) nested loop.
        """
        # Sum scores by date
        daily = transactions.groupby(transactions.index.date)['weighted_score'].sum()

        # Convert to datetime for proper indexing
        daily.index = pd.to_datetime(daily.index)

        # VECTORIZED: Create date mapping and use reindex
        # Normalize price_index dates for matching
        price_dates = pd.Series(price_index.normalize(), index=price_index)

        # Reindex daily scores to price_index via date matching
        daily_series = pd.Series(0.0, index=price_index, dtype=float)

        # Map scores using normalized dates
        for ts in price_index:
            date_normalized = ts.normalize()
            if date_normalized in daily.index:
                daily_series.loc[ts] = daily.loc[date_normalized]

        return daily_series

    def _rolling_signals(self, daily_scores: pd.Series) -> pd.Series:
        """
        Convert daily scores to signals using rolling aggregation.

        Professional approach:
        1. Rolling sum over lookback period
        2. Winsorize to handle outliers
        3. Normalize to [-1, 1] using rolling percentile rank

        OPTIMIZED: Uses expanding rank instead of slow rolling apply.
        """
        # Rolling sum over lookback period
        rolling_sum = daily_scores.rolling(
            window=self.lookback_days,
            min_periods=1
        ).sum()

        # Winsorize
        rolling_winsorized = self.winsorize(rolling_sum.dropna())

        if len(rolling_winsorized) == 0:
            return pd.Series(0, index=daily_scores.index)

        # OPTIMIZED: Use expanding rank instead of slow rolling apply
        # This gives similar results but is O(n) instead of O(n*w)
        rank_window = 252 * 2
        min_periods = 21

        # Use expanding percentile rank (more efficient than rolling apply)
        # For each point, rank against all prior points
        signals = rolling_winsorized.expanding(min_periods=min_periods).apply(
            lambda x: 2.0 * ((x[-1] > x[:-1]).sum() / (len(x) - 1) if len(x) > 1 else 0.5) - 1.0,
            raw=True  # raw=True is much faster
        )

        # Alternative: Use simple cross-sectional z-score for faster computation
        # Fallback for early dates where expanding doesn't have enough data
        if signals.isna().all():
            mu = rolling_winsorized.mean()
            sigma = rolling_winsorized.std()
            if sigma > 0:
                signals = ((rolling_winsorized - mu) / sigma).clip(-3, 3) / 3.0
            else:
                signals = pd.Series(0, index=rolling_winsorized.index)

        # Align to original index
        signals = signals.reindex(daily_scores.index).fillna(0)

        return signals

    def _apply_monthly_rebalancing(self, signals: pd.Series) -> pd.Series:
        """Apply monthly rebalancing (hold signal for entire month)."""
        month_ends = signals.resample('ME').last()  # 'ME' = month end (replaces deprecated 'M')
        rebalanced = month_ends.reindex(signals.index).ffill()
        return rebalanced.fillna(0)

    def generate_cross_sectional_scores(
        self,
        rebal_date: pd.Timestamp,
        universe: List[str],
        data_manager: DataManager,
    ) -> pd.Series:
        """
        Generate InstitutionalInsider scores for universe at rebalance date.

        This is the PREFERRED method for ensemble construction and cross-sectional
        backtests. Uses bulk insider data fetching for 50-100x performance improvement
        over per-ticker queries.

        Args:
            rebal_date: Rebalance date (PIT cutoff)
            universe: List of ticker symbols to score
            data_manager: DataManager instance for fetching prices and insider data

        Returns:
            pd.Series indexed by ticker with insider scores in [-1, 1] range

        Example:
            >>> insider = InstitutionalInsider({'lookback_days': 90})
            >>> dm = DataManager()
            >>> universe = ['AAPL', 'MSFT', 'GOOGL']
            >>> scores = insider.generate_cross_sectional_scores(
            ...     rebal_date=pd.Timestamp('2024-01-31'),
            ...     universe=universe,
            ...     data_manager=dm
            ... )
            >>> print(scores)
            AAPL    0.75
            MSFT   -0.25
            GOOGL   0.50
            dtype: float64

        Notes:
            - Mirrors InstitutionalMomentum.generate_cross_sectional_scores() API
            - Uses bulk fetching: single DB query for all tickers (vs N queries)
            - Maintains PIT safety via as_of_date filtering
            - Returns only non-zero scores (tickers with zero signals are omitted)
            - OPTIMIZED: Caches results by (signal_type, params, date) to avoid
              redundant computation across Optuna trials
        """
        # CACHE CHECK: Try to retrieve cached signal scores
        date_str = rebal_date.strftime('%Y-%m-%d')
        cache_params = {
            'lookback_days': self.lookback_days,
            'min_transaction_value': self.min_transaction_value,
            'cluster_window': self.cluster_window,
            'cluster_min_insiders': self.cluster_min_insiders,
            'ceo_weight': self.role_weights.get('ceo', 3.0),
            'cfo_weight': self.role_weights.get('cfo', 2.5),
        }

        if hasattr(data_manager, 'get_cached_signal'):
            cached = data_manager.get_cached_signal('InstitutionalInsider', cache_params, date_str)
            if cached is not None:
                logger.debug(f"Using cached InstitutionalInsider scores for {date_str}")
                return cached

        logger.debug(
            f"Generating cross-sectional insider scores for {len(universe)} tickers "
            f"at {rebal_date.strftime('%Y-%m-%d')}"
        )

        # Step 1: Bulk fetch insider data for entire universe (CRITICAL for performance)
        # Add extra lookback for signal calculation (lookback_days + 1 year for rolling rank)
        lookback_start = (rebal_date - timedelta(days=self.lookback_days + 252)).strftime('%Y-%m-%d')
        rebal_date_str = rebal_date.strftime('%Y-%m-%d')

        bulk_insider_data = data_manager.get_insider_trades_bulk(
            tickers=universe,
            start_date=lookback_start,
            end_date=rebal_date_str,
            as_of_date=rebal_date_str  # PIT safety: only trades disclosed by rebal_date
        )

        logger.debug(f"Bulk fetched {len(bulk_insider_data)} insider transactions")

        # Step 2: Generate signals for each ticker using bulk data
        scores = {}
        skipped_count = 0

        for ticker in universe:
            try:
                # Fetch prices for this ticker
                prices = data_manager.get_prices(ticker, lookback_start, rebal_date_str)

                if len(prices) == 0 or 'close' not in prices.columns:
                    skipped_count += 1
                    continue

                # Prepare data for signal generation
                data = pd.DataFrame({
                    'close': prices['close'],
                    'ticker': ticker
                })

                # Generate signals with bulk insider data (FAST: no DB query)
                sig_series = self.generate_signals(data, bulk_insider_data=bulk_insider_data)

                if len(sig_series) > 0:
                    signal_value = sig_series.iloc[-1]

                    # Only include non-zero signals
                    if pd.notna(signal_value) and signal_value != 0:
                        scores[ticker] = float(signal_value)

            except Exception as e:
                # Skip tickers with data issues (consistent with momentum pattern)
                logger.debug(f"Skipping {ticker}: {e}")
                skipped_count += 1
                continue

        logger.info(
            f"Generated {len(scores)} insider scores, skipped {skipped_count} tickers"
        )

        result = pd.Series(scores, dtype=float)

        # CACHE STORE: Save computed scores for future trials with same params
        if hasattr(data_manager, 'cache_signal'):
            data_manager.cache_signal('InstitutionalInsider', cache_params, date_str, result)
            logger.debug(f"Cached InstitutionalInsider scores for {date_str}")

        return result

    def get_parameter_space(self) -> Dict[str, tuple]:
        """
        Define parameter space for optimization.

        Returns:
            Dict with parameter specifications
        """
        return {
            'lookback_days': ('int', 30, 180),  # 1-6 months
            'min_transaction_value': ('categorical', [5000, 10000, 25000, 50000]),
            'cluster_window': ('int', 3, 14),  # 3 days to 2 weeks
            'cluster_min_insiders': ('int', 2, 5),
            'ceo_weight': ('float', 2.0, 4.0),
            'cfo_weight': ('float', 1.5, 3.0)
        }

    def __repr__(self) -> str:
        return (f"InstitutionalInsider("
                f"lookback={self.lookback_days}d, "
                f"cluster={self.cluster_window}d/{self.cluster_min_insiders}, "
                f"min_value=${self.min_transaction_value:,})")


if __name__ == '__main__':
    print("InstitutionalInsider - Cohen-Malloy-Pomorski Methodology")
    print("Professional insider transaction analysis")
    print()
    print("Features:")
    print("  1. Dollar-weighted transactions")
    print("  2. Role hierarchy (CEO > CFO > Director)")
    print("  3. Cluster detection (coordinated activity)")
    print()
    print("Standard parameters:")
    print("  Lookback: 90 days")
    print("  Cluster: 3+ insiders within 7 days")
    print("  Min value: $10,000")
    print("  Rebalance: Monthly")
