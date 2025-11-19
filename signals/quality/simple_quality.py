"""
Simple Quality Signal

Strategy: Buy high-quality companies using ROE (Return on Equity).

That's it. One simple metric. Under 100 lines.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from core.base_signal import BaseSignal
from data.data_manager import DataManager


class SimpleQuality(BaseSignal):
    """
    Dead simple quality signal using ROE.

    ROE = Net Income / Shareholder Equity
    Higher ROE = better quality = positive signal

    Parameters:
        None (ROE is the metric)
    """

    def __init__(self,
                 params: Dict[str, Any],
                 data_manager: Optional[DataManager] = None,
                 name: str = 'SimpleQuality'):
        super().__init__(params, name)
        self.data_manager = data_manager or DataManager()

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate quality signals based on ROE.

        Args:
            data: DataFrame with 'close' prices and 'ticker', datetime index

        Returns:
            Series with signals in [-1, 1]
        """
        # Get ticker
        if 'ticker' not in data.columns:
            return pd.Series(0, index=data.index)

        ticker = data['ticker'].iloc[0]

        # Get fundamental data
        start_date = data.index.min().strftime('%Y-%m-%d')
        end_date = data.index.max().strftime('%Y-%m-%d')

        fundamentals = self.data_manager.get_fundamentals(
            ticker,
            start_date,
            end_date,
            dimension='ARQ'  # As-reported quarterly
        )

        if len(fundamentals) == 0:
            return pd.Series(0, index=data.index)

        # Calculate quality score (just use ROE directly)
        roe = fundamentals['roe']

        # Reindex to daily (forward fill fundamentals)
        roe_daily = roe.reindex(data.index, method='ffill').fillna(0)

        # Normalize to [-1, 1] using cross-sectional rank
        # (For single stock, use time-series rank)
        rank_window = self.params.get('rank_window', 252 * 2)  # 2 years

        signals = roe_daily.rolling(
            window=rank_window,
            min_periods=4  # At least 4 quarters
        ).apply(
            lambda x: 2.0 * (pd.Series(x).rank(pct=True).iloc[-1]) - 1.0,
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
            'rank_window': ('int', 252, 252 * 3),  # 1-3 years of ranking
        }

    def __repr__(self) -> str:
        return "SimpleQuality(metric=ROE)"


if __name__ == '__main__':
    print("SimpleQuality signal requires actual Sharadar data to test.")
    print("Use pytest tests for validation with mock data.")
