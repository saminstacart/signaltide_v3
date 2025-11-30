"""
Opportunistic Insider Trading Signal based on Cohen, Malloy & Pomorski (2012).

Reference:
    Cohen, L., Malloy, C., & Pomorski, L. (2012).
    "Decoding Inside Information". Journal of Finance, 67(3), 1009-1043.

Key insight: Distinguish between "routine" and "opportunistic" insider trades.
Routine trades (same month/pattern each year) are uninformative.
Opportunistic trades (deviation from pattern) have 8.6% annual alpha.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import timedelta
from collections import defaultdict

import pandas as pd
import numpy as np

from .base import BaseSignal
from signaltide_v4.data.base import PITDataManager
from signaltide_v4.config.settings import get_settings

logger = logging.getLogger(__name__)


class OpportunisticInsiderSignal(BaseSignal):
    """
    Signal based on opportunistic (non-routine) insider trading.

    Process:
    1. Load insider transactions
    2. Classify as routine vs opportunistic
    3. Score based on opportunistic buying/selling
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        db_path: Optional[str] = None,
    ):
        """
        Initialize insider signal.

        Args:
            params: Parameter overrides
            db_path: Database path
        """
        super().__init__(name='opportunistic_insider', params=params)

        # Ensure params is a dict for .get() calls
        params = params or {}

        settings = get_settings()
        self.lookback_days = params.get('lookback_days', settings.insider_lookback_days)
        self.min_transaction_value = params.get(
            'min_transaction_value',
            settings.insider_min_transaction_value
        )

        # Routine trade threshold: trades in N+ consecutive years = routine
        self.routine_years_threshold = params.get('routine_years_threshold', 3)

        # Store db_path for insider queries
        self.db_path = db_path or settings.db_path

    def compute_raw_scores(
        self,
        tickers: List[str],
        as_of_date: str
    ) -> pd.Series:
        """
        Compute opportunistic insider trading score.

        Positive = net opportunistic buying
        Negative = net opportunistic selling
        """
        if not tickers:
            return pd.Series(dtype=float)

        # Get insider transactions
        transactions = self._get_insider_transactions(tickers, as_of_date)

        if len(transactions) == 0:
            logger.warning(f"No insider transactions found for {as_of_date}")
            return pd.Series(np.nan, index=tickers)

        # Classify routine vs opportunistic
        transactions = self._classify_routine(transactions, as_of_date)

        # Filter to opportunistic only
        opportunistic = transactions[~transactions['is_routine']]

        if len(opportunistic) == 0:
            logger.warning(f"No opportunistic trades found for {as_of_date}")
            return pd.Series(0.0, index=tickers)  # Neutral if no signal

        # Calculate net opportunistic activity per ticker
        scores = {}
        for ticker in tickers:
            ticker_txns = opportunistic[opportunistic['ticker'] == ticker]

            if len(ticker_txns) == 0:
                scores[ticker] = 0.0  # Neutral
                continue

            # Net buying value (positive = buying, negative = selling)
            net_value = ticker_txns['transaction_value'].sum()

            # Normalize by market cap or use raw value
            scores[ticker] = net_value

        result = pd.Series(scores)

        # Log summary
        n_opp = len(opportunistic)
        n_total = len(transactions)
        logger.debug(
            f"Insider: {n_opp}/{n_total} opportunistic trades "
            f"({n_opp/n_total:.1%} if n_total > 0 else 0)"
        )

        return result

    def _get_insider_transactions(
        self,
        tickers: List[str],
        as_of_date: str
    ) -> pd.DataFrame:
        """Get insider transactions from database."""
        import sqlite3

        # Calculate lookback period
        start_date = (pd.Timestamp(as_of_date) - timedelta(days=self.lookback_days))
        start_date = start_date.strftime('%Y-%m-%d')

        # Need historical data for routine classification
        history_start = (pd.Timestamp(as_of_date) - timedelta(days=365 * 4))
        history_start = history_start.strftime('%Y-%m-%d')

        try:
            conn = sqlite3.connect(self.db_path)

            placeholders = ','.join(['?' for _ in tickers])
            query = f"""
                SELECT ticker, filingdate, transactiondate, transactioncode,
                       transactionshares, transactionpricepershare,
                       ownername
                FROM sharadar_insiders
                WHERE ticker IN ({placeholders})
                AND filingdate BETWEEN ? AND ?
                AND filingdate <= ?
                AND transactioncode IN ('P', 'S')
                ORDER BY filingdate
            """

            params = tuple(tickers) + (history_start, as_of_date, as_of_date)
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()

            if len(df) == 0:
                return pd.DataFrame()

            # Calculate transaction value
            df['transactionshares'] = pd.to_numeric(df['transactionshares'], errors='coerce')
            df['transactionpricepershare'] = pd.to_numeric(df['transactionpricepershare'], errors='coerce')

            df['transaction_value'] = df['transactionshares'] * df['transactionpricepershare']

            # Make buys positive, sells negative
            df.loc[df['transactioncode'] == 'S', 'transaction_value'] *= -1

            # Filter by minimum value
            df = df[df['transaction_value'].abs() >= self.min_transaction_value]

            # Parse dates
            df['filingdate'] = pd.to_datetime(df['filingdate'])
            df['transactiondate'] = pd.to_datetime(df['transactiondate'])

            return df

        except Exception as e:
            logger.error(f"Error loading insider transactions: {e}")
            return pd.DataFrame()

    def _classify_routine(
        self,
        transactions: pd.DataFrame,
        as_of_date: str
    ) -> pd.DataFrame:
        """
        Classify trades as routine vs opportunistic.

        Routine = same insider trades in same month for N+ consecutive years.
        """
        if len(transactions) == 0:
            return transactions

        transactions = transactions.copy()
        transactions['is_routine'] = False

        # Group by insider
        for (ticker, owner), group in transactions.groupby(['ticker', 'ownername']):
            # Get months this insider has traded
            group = group.copy()
            group['trade_month'] = group['transactiondate'].dt.month
            group['trade_year'] = group['transactiondate'].dt.year

            # Check each trade
            for idx, row in group.iterrows():
                month = row['trade_month']

                # Count years this insider traded in this month
                same_month_trades = group[group['trade_month'] == month]
                years_traded = same_month_trades['trade_year'].nunique()

                # If traded same month for N+ years, it's routine
                if years_traded >= self.routine_years_threshold:
                    transactions.loc[idx, 'is_routine'] = True

        return transactions

    def get_diagnostics(
        self,
        raw_scores: pd.Series,
        normalized_scores: pd.Series
    ) -> Dict[str, Any]:
        """Add insider-specific diagnostics."""
        base_diag = super().get_diagnostics(raw_scores, normalized_scores)

        valid = raw_scores.dropna()
        if len(valid) == 0:
            return base_diag

        base_diag.update({
            'pct_net_buyers': float((valid > 0).mean()),
            'pct_net_sellers': float((valid < 0).mean()),
            'pct_neutral': float((valid == 0).mean()),
            'routine_years_threshold': self.routine_years_threshold,
            'min_transaction_value': self.min_transaction_value,
        })

        return base_diag
