"""
Simple Insider Signal

Strategy: Buy when insiders buy, sell when insiders sell.

Net buying = insider buys - insider sells
That's it. Under 100 lines.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from core.base_signal import BaseSignal
from data.data_manager import DataManager


class SimpleInsider(BaseSignal):
    """
    Dead simple insider trading signal.

    Count insider purchases vs sales in lookback window.
    More buying = positive signal, more selling = negative signal.

    Parameters:
        lookback_days: Days to look back for insider trades (default: 90)
    """

    def __init__(self,
                 params: Dict[str, Any],
                 data_manager: Optional[DataManager] = None,
                 name: str = 'SimpleInsider'):
        super().__init__(params, name)
        self.data_manager = data_manager or DataManager()

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate insider trading signals.

        Args:
            data: DataFrame with prices and 'ticker', datetime index

        Returns:
            Series with signals in [-1, 1]
        """
        # Get ticker
        if 'ticker' not in data.columns:
            return pd.Series(0, index=data.index)

        ticker = data['ticker'].iloc[0]

        # Get insider trading data
        lookback_days = self.params.get('lookback_days', 90)
        start_date = (data.index.min() - pd.Timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        end_date = data.index.max().strftime('%Y-%m-%d')

        insider_trades = self.data_manager.get_insider_trades(
            ticker,
            start_date,
            end_date
        )

        if len(insider_trades) == 0:
            return pd.Series(0, index=data.index)

        # Calculate net buying for each date
        signals = pd.Series(0.0, index=data.index)

        for date in data.index:
            # Get trades in lookback window before this date
            window_start = date - pd.Timedelta(days=lookback_days)

            window_trades = insider_trades[
                (insider_trades.index >= window_start) &
                (insider_trades.index <= date)
            ]

            if len(window_trades) > 0:
                # Count buys and sells
                n_buys = (window_trades['transactioncode'] == 'P').sum()
                n_sells = (window_trades['transactioncode'] == 'S').sum()

                # Net buying signal
                net = n_buys - n_sells
                signals[date] = net

        # Normalize to [-1, 1] using rolling rank
        rank_window = self.params.get('rank_window', 252)

        signals = signals.rolling(
            window=rank_window,
            min_periods=20
        ).apply(
            lambda x: 2.0 * (pd.Series(x).rank(pct=True).iloc[-1]) - 1.0 if x.max() != x.min() else 0.0,
            raw=False
        )

        # Fill NaN with 0
        signals = signals.fillna(0)

        # Clip to [-1, 1]
        signals = signals.clip(-1, 1)

        return signals

    def get_parameter_space(self) -> Dict[str, tuple]:
        """
        Define parameter space for Optuna.

        Returns:
            Dict with parameter specs
        """
        return {
            'lookback_days': ('int', 30, 180),     # 1-6 months
            'rank_window': ('int', 60, 252),       # Ranking window
        }

    def validate_params(self) -> None:
        """Validate parameters."""
        super().validate_params()

        if self.params['lookback_days'] < 1:
            raise ValueError("lookback_days must be >= 1")

    def __repr__(self) -> str:
        return f"SimpleInsider(lookback={self.params['lookback_days']}d)"


if __name__ == '__main__':
    print("SimpleInsider signal requires actual Sharadar insider data to test.")
    print("Use pytest tests for validation with mock data.")
