"""
Simple Momentum Signal

Strategy: Buy winners, sell losers. Just measure price change over lookback period.

That's it. No volume confirmation, no volatility adjustment, no complexity.
Under 100 lines as promised.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
from core.base_signal import BaseSignal


class SimpleMomentum(BaseSignal):
    """
    Dead simple momentum signal.

    Just: return = (price_now - price_then) / price_then

    Parameters:
        lookback: Days to look back (default: 90)
    """

    def __init__(self, params: Dict[str, Any], name: str = 'SimpleMomentum'):
        super().__init__(params, name)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate momentum signals.

        Args:
            data: DataFrame with 'close' prices, datetime index

        Returns:
            Series with signals in [-1, 1]
        """
        # Get lookback parameter
        lookback = self.params.get('lookback', 90)

        # Calculate returns over lookback period
        returns = data['close'].pct_change(lookback)

        # Normalize to [-1, 1] using rolling rank
        rank_window = self.params.get('rank_window', 252)

        signals = returns.rolling(
            window=rank_window,
            min_periods=lookback
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
            'lookback': ('int', 5, 252),          # 1 week to 1 year
            'rank_window': ('int', 60, 252),      # Ranking window
        }

    def validate_params(self) -> None:
        """Validate parameters."""
        super().validate_params()

        if self.params['lookback'] < 1:
            raise ValueError("lookback must be >= 1")

        if self.params['lookback'] > self.params['rank_window']:
            raise ValueError("lookback should be <= rank_window")

    def __repr__(self) -> str:
        return f"SimpleMomentum(lookback={self.params['lookback']})"


if __name__ == '__main__':
    # Quick test
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]  # signals/momentum/ -> repo root
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from data.mock_generator import create_test_data
    from datetime import datetime, timedelta

    print("Testing SimpleMomentum...")

    # Generate mock data
    data = create_test_data(n_tickers=1, n_years=2, seed=42)
    prices = data['prices']

    # Create signal
    params = {'lookback': 90, 'rank_window': 252}
    signal = SimpleMomentum(params)

    # Generate signals
    signals = signal.generate_signals(prices)

    print(f"\nGenerated {len(signals)} signals")
    print(f"Min: {signals.min():.3f}, Max: {signals.max():.3f}")
    print(f"Mean: {signals.mean():.3f}, Std: {signals.std():.3f}")
    print(f"\nLast 10 signals:")
    print(signals.tail(10))

    print(f"\n{repr(signal)}")
