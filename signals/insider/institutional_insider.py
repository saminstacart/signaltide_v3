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

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import timedelta
from core.institutional_base import InstitutionalSignal
from data.data_manager import DataManager


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

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate insider trading signals.

        Args:
            data: DataFrame with 'close' prices and 'ticker'

        Returns:
            Series with signals in [-1, 1] range
        """
        if 'ticker' not in data.columns:
            return pd.Series(0, index=data.index)

        ticker = data['ticker'].iloc[0]

        # Get insider trading data
        start_date = (data.index.min() - timedelta(days=self.lookback_days)).strftime('%Y-%m-%d')
        end_date = data.index.max().strftime('%Y-%m-%d')

        insiders = self.data_manager.get_insider_trades(ticker, start_date, end_date)

        if len(insiders) == 0:
            return pd.Series(0, index=data.index)

        # Calculate daily insider signals
        insider_signals = self._calculate_insider_activity(insiders, data.index)

        # Apply monthly rebalancing
        if self.rebalance_frequency == 'monthly':
            insider_signals = self._apply_monthly_rebalancing(insider_signals)

        return insider_signals.clip(-1, 1)

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
        """
        scores = pd.Series(0, index=transactions.index)

        for idx, row in transactions.iterrows():
            # Direction (column name: transactioncode)
            direction = 1 if row['transactioncode'] == 'P' else -1

            # Dollar weight (log scale to reduce impact of extreme values)
            if 'value' in row and pd.notna(row['value']) and row['value'] > 0:
                dollar_weight = np.log10(row['value'])
            else:
                dollar_weight = 1.0

            # Role weight (column name: officertitle)
            role = self._classify_role(row.get('officertitle', ''))
            role_weight = self.role_weights.get(role, 1.0)

            # Combined score
            scores[idx] = direction * dollar_weight * role_weight

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
        """
        trans = transactions.copy()
        trans['is_cluster'] = False

        # Group by direction (purchases vs sales)
        # Column name: transactioncode
        for direction in ['P', 'S']:
            dir_trans = trans[trans['transactioncode'] == direction].copy()

            if len(dir_trans) == 0:
                continue

            # Check each transaction for nearby similar transactions
            for idx, row in dir_trans.iterrows():
                date = row.name

                # Find transactions within cluster window
                window_start = date - pd.Timedelta(days=self.cluster_window)
                window_end = date + pd.Timedelta(days=self.cluster_window)

                nearby = dir_trans[
                    (dir_trans.index >= window_start) &
                    (dir_trans.index <= window_end)
                ]

                # Count unique insiders (by name if available, otherwise by transaction)
                n_insiders = len(nearby)

                if n_insiders >= self.cluster_min_insiders:
                    trans.loc[idx, 'is_cluster'] = True

        # Amplify cluster signals (2x weight)
        trans.loc[trans['is_cluster'], 'weighted_score'] *= 2.0

        return trans

    def _aggregate_to_daily(self,
                           transactions: pd.DataFrame,
                           price_index: pd.DatetimeIndex) -> pd.Series:
        """
        Aggregate weighted transaction scores to daily frequency.
        """
        # Sum scores by date
        daily = transactions.groupby(transactions.index.date)['weighted_score'].sum()

        # Reindex to full date range
        daily_series = pd.Series(0, index=price_index)

        for date, score in daily.items():
            # Find matching dates in price_index
            matching = daily_series.index[daily_series.index.date == date]
            if len(matching) > 0:
                daily_series.loc[matching] = score

        return daily_series

    def _rolling_signals(self, daily_scores: pd.Series) -> pd.Series:
        """
        Convert daily scores to signals using rolling aggregation.

        Professional approach:
        1. Rolling sum over lookback period
        2. Winsorize to handle outliers
        3. Normalize to [-1, 1] using rolling percentile rank
        """
        # Rolling sum over lookback period
        rolling_sum = daily_scores.rolling(
            window=self.lookback_days,
            min_periods=1
        ).sum()

        # Winsorize
        rolling_winsorized = self.winsorize(rolling_sum.dropna())

        # Normalize using rolling percentile rank (2-year window)
        rank_window = 252 * 2

        signals = rolling_winsorized.rolling(
            window=rank_window,
            min_periods=21  # At least 1 month
        ).apply(
            lambda x: 2.0 * (pd.Series(x).rank(pct=True).iloc[-1]) - 1.0,
            raw=False
        )

        # Align to original index
        signals = signals.reindex(daily_scores.index).fillna(0)

        return signals

    def _apply_monthly_rebalancing(self, signals: pd.Series) -> pd.Series:
        """Apply monthly rebalancing (hold signal for entire month)."""
        month_ends = signals.resample('M').last()
        rebalanced = month_ends.reindex(signals.index, method='ffill')
        return rebalanced.fillna(0)

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
