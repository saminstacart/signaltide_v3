"""
Insider Signal - Corporate insider trading activity.

Methodology:
- Insider buying/selling momentum (net buying over time)
- Cluster detection (multiple insiders trading)
- Value-weighted transactions (larger trades = stronger signal)
- Recency weighting (recent trades more important)
- Insider title weighting (CEO/CFO trades more informative)

Economic Rationale:
Insiders have superior information about company prospects:
1. Information asymmetry - insiders know true value
2. Skin in the game - insiders trade on conviction
3. Legal constraint - only trade when confident (avoiding insider trading laws)
4. Empirical evidence - insider buying predicts outperformance

This signal uses Sharadar insider trading data with proper point-in-time
access via DataManager to prevent lookahead bias.

CRITICAL: Uses filing_date not trade_date to ensure we only use
information that was publicly available at the time.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from core.base_signal import BaseSignal
from data.data_manager import DataManager


class InsiderSignal(BaseSignal):
    """
    Insider trading signal using Sharadar data.

    Analyzes patterns in corporate insider transactions:
    - Net insider buying/selling
    - Number of insiders trading
    - Transaction size and value
    - Insider title importance
    - Trading clusters

    Requires DataManager to access insider data with point-in-time constraints.
    """

    def __init__(self, params: Dict[str, Any],
                 data_manager: Optional[DataManager] = None,
                 name: str = 'InsiderSignal'):
        """
        Initialize insider signal.

        Args:
            params: Signal parameters (see get_parameter_space)
            data_manager: DataManager instance for insider data access
            name: Signal name
        """
        super().__init__(params, name)
        self.data_manager = data_manager or DataManager()

        # Title weights (more senior = more important)
        self.title_weights = {
            'CEO': 3.0,
            'CFO': 2.5,
            'COO': 2.0,
            'President': 2.5,
            'Chairman': 2.5,
            'Director': 1.5,
            'Officer': 1.0,
            'VP': 1.0,
            'Other': 0.5
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate insider trading signals.

        Args:
            data: DataFrame with price data and index as dates
                  Must have 'ticker' in columns or be single-ticker data

        Returns:
            Series with signals in [-1, 1], same index as data
        """
        # Get ticker from data
        if 'ticker' in data.columns:
            ticker = data['ticker'].iloc[0]
        else:
            # No ticker available
            return pd.Series(0, index=data.index)

        # Get insider trading data for the period
        insider_trades = self._get_insider_trades_for_dates(
            ticker=ticker,
            dates=data.index
        )

        if insider_trades is None or len(insider_trades) == 0:
            # No insider trading data
            return pd.Series(0, index=data.index)

        # Calculate insider trading scores
        insider_scores = self._calculate_insider_scores(
            insider_trades,
            data.index
        )

        # Align with price data
        signals = insider_scores.reindex(data.index, method='ffill').fillna(0)

        # Normalize to [-1, 1] range
        # Use rolling rank for adaptive normalization
        rank_window = self.params.get('rank_window', 252)
        signals = signals.rolling(window=rank_window, min_periods=20).apply(
            lambda x: 2.0 * (pd.Series(x).rank().iloc[-1] / len(x)) - 1.0,
            raw=False
        )

        # Apply threshold
        threshold = self.params.get('signal_threshold', 0.0)
        if threshold > 0:
            signals = signals.where(signals.abs() > threshold, 0)

        # Fill NaN with 0
        signals = signals.fillna(0)

        # Clip to [-1, 1]
        signals = signals.clip(-1, 1)

        return signals

    def _get_insider_trades_for_dates(self, ticker: str,
                                     dates: pd.DatetimeIndex) -> Optional[pd.DataFrame]:
        """
        Get insider trading data for ticker covering the date range.

        Uses point-in-time retrieval based on filing_date.

        Args:
            ticker: Stock ticker
            dates: Dates to cover

        Returns:
            DataFrame with insider trades or None
        """
        if len(dates) == 0:
            return None

        start_date = dates.min()
        end_date = dates.max()

        # Get insider trading data
        # Lookback period for insider activity
        lookback_days = self.params.get('lookback_days', 180)

        insider_trades = self.data_manager.get_insider_trades(
            ticker=ticker,
            start_date=start_date - pd.Timedelta(days=lookback_days),
            end_date=end_date,
            as_of=end_date  # Only trades filed by end date
        )

        return insider_trades if len(insider_trades) > 0 else None

    def _calculate_insider_scores(self, trades: pd.DataFrame,
                                  dates: pd.DatetimeIndex) -> pd.Series:
        """
        Calculate insider trading scores for each date.

        Args:
            trades: DataFrame with insider trades
            dates: Dates to generate scores for

        Returns:
            Series of insider scores indexed by date
        """
        # Create daily scores
        daily_scores = pd.Series(0.0, index=dates)

        if len(trades) == 0:
            return daily_scores

        # Ensure filing_date is datetime
        if 'filing_date' in trades.columns:
            trades = trades.copy()
            trades['filing_date'] = pd.to_datetime(trades['filing_date'])

            # Calculate score for each trade
            trades['trade_score'] = trades.apply(
                lambda row: self._score_trade(row),
                axis=1
            )

            # Aggregate trades by filing date
            lookback = self.params.get('aggregation_window', 30)

            for date in dates:
                # Get trades filed in lookback window before this date
                window_start = date - pd.Timedelta(days=lookback)
                window_trades = trades[
                    (trades['filing_date'] >= window_start) &
                    (trades['filing_date'] <= date)
                ]

                if len(window_trades) > 0:
                    # Calculate aggregate score
                    score = self._aggregate_trade_scores(window_trades, date)
                    daily_scores[date] = score

        return daily_scores

    def _score_trade(self, trade: pd.Series) -> float:
        """
        Score a single insider trade.

        Args:
            trade: Series with trade information

        Returns:
            Score (positive for buying, negative for selling)
        """
        # Base score from transaction type
        if trade['transaction_type'] == 'P':  # Purchase
            base_score = 1.0
        elif trade['transaction_type'] == 'S':  # Sale
            base_score = -1.0
        else:
            base_score = 0.0

        # Weight by insider title
        title_weight = self._get_title_weight(trade.get('insider_title', 'Other'))

        # Weight by transaction value
        value_weight = 1.0
        if 'shares' in trade and 'price_per_share' in trade:
            value = trade['shares'] * trade['price_per_share']
            # Log scale for value (larger trades more important)
            if value > 0:
                value_weight = np.log1p(value) / 10.0  # Normalize
                value_weight = min(value_weight, 5.0)  # Cap at 5x

        # Combined score
        score = base_score * title_weight * value_weight

        return score

    def _get_title_weight(self, title: str) -> float:
        """Get weight for insider title."""
        if pd.isna(title):
            return self.title_weights['Other']

        title_upper = str(title).upper()

        for key, weight in self.title_weights.items():
            if key.upper() in title_upper:
                return weight

        return self.title_weights['Other']

    def _aggregate_trade_scores(self, trades: pd.DataFrame, date: pd.Timestamp) -> float:
        """
        Aggregate multiple trades into a single score.

        Args:
            trades: DataFrame of trades in window
            date: Current date

        Returns:
            Aggregated score
        """
        if len(trades) == 0:
            return 0.0

        # Sum trade scores
        total_score = trades['trade_score'].sum()

        # Apply recency weighting (more recent = more important)
        if self.params.get('use_recency_weight', True):
            recency_halflife = self.params.get('recency_halflife', 30)
            # Calculate days since each trade
            days_since = (date - trades['filing_date']).dt.days

            # Exponential decay
            recency_weights = np.exp(-days_since / recency_halflife)

            # Reweight scores
            total_score = (trades['trade_score'] * recency_weights).sum()

        # Cluster bonus (multiple insiders trading = stronger signal)
        if self.params.get('use_cluster_bonus', True):
            n_insiders = trades['insider_name'].nunique()
            if n_insiders > 1:
                cluster_multiplier = self.params.get('cluster_multiplier', 1.5)
                total_score *= min(1.0 + (n_insiders - 1) * 0.1 * cluster_multiplier, 3.0)

        return total_score

    def get_parameter_space(self) -> Dict[str, tuple]:
        """
        Define Optuna search space for insider signal.

        Returns:
            Dict of parameter specifications
        """
        return {
            # Data lookback
            'lookback_days': ('int', 90, 365),

            # Aggregation
            'aggregation_window': ('int', 7, 90),

            # Recency weighting
            'use_recency_weight': ('categorical', [True, False]),
            'recency_halflife': ('int', 7, 90),

            # Cluster detection
            'use_cluster_bonus': ('categorical', [True, False]),
            'cluster_multiplier': ('float', 1.0, 3.0),

            # Signal processing
            'signal_threshold': ('float', 0.0, 0.3),
            'rank_window': ('int', 60, 252),
        }

    def validate_params(self) -> None:
        """Validate insider-specific parameters."""
        super().validate_params()

        # Aggregation window should be <= lookback days
        if self.params['aggregation_window'] > self.params['lookback_days']:
            raise ValueError(
                "aggregation_window must be <= lookback_days"
            )

    def __repr__(self) -> str:
        """String representation."""
        return (f"InsiderSignal(lookback={self.params['lookback_days']}d, "
                f"window={self.params['aggregation_window']}d)")
