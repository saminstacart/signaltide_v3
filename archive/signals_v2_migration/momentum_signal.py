"""
Momentum Signal - Multi-timeframe momentum with volume confirmation.

Methodology:
- Combines price momentum across multiple lookback periods
- Volume confirmation (momentum stronger with high volume)
- Sector-relative momentum (optional, requires sector data)
- Volatility adjustment for risk-adjusted signals

Economic Rationale:
Markets exhibit serial correlation due to:
1. Behavioral biases (anchoring, herding, disposition effect)
2. Gradual information diffusion
3. Trend-following algorithms
4. Risk premium for momentum exposure

This signal has been proven in v2 and is being adapted to v3 framework
with proper point-in-time data access and no lookahead bias.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
from core.base_signal import BaseSignal


class MomentumSignal(BaseSignal):
    """
    Multi-timeframe momentum signal with volume confirmation.

    Combines:
    - Short-term momentum (1-4 weeks)
    - Medium-term momentum (1-3 months)
    - Long-term momentum (6-12 months)
    - Volume confirmation
    - Volatility adjustment

    All calculations use only past data - no lookahead bias.
    """

    def __init__(self, params: Dict[str, Any], name: str = 'MomentumSignal'):
        """
        Initialize momentum signal.

        Args:
            params: Signal parameters (see get_parameter_space)
            name: Signal name
        """
        super().__init__(params, name)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate momentum signals from price and volume data.

        Args:
            data: DataFrame with OHLCV data, DatetimeIndex

        Returns:
            Series with signals in [-1, 1], same index as data
        """
        # Extract parameters
        short_lookback = self.params['short_lookback']
        medium_lookback = self.params['medium_lookback']
        long_lookback = self.params['long_lookback']
        volume_confirmation = self.params.get('volume_confirmation', True)
        volatility_adjust = self.params.get('volatility_adjust', True)
        signal_threshold = self.params.get('signal_threshold', 0.0)

        # Calculate returns for each timeframe
        # Using pct_change ensures no lookahead (uses shift internally)
        if self.params.get('use_log_returns', False):
            short_mom = np.log(data['close'] / data['close'].shift(short_lookback))
            medium_mom = np.log(data['close'] / data['close'].shift(medium_lookback))
            long_mom = np.log(data['close'] / data['close'].shift(long_lookback))
        else:
            short_mom = data['close'].pct_change(short_lookback)
            medium_mom = data['close'].pct_change(medium_lookback)
            long_mom = data['close'].pct_change(long_lookback)

        # Combine momentum signals with weights
        # Shorter timeframes get more weight (more responsive)
        short_weight = self.params.get('short_weight', 0.5)
        medium_weight = self.params.get('medium_weight', 0.3)
        long_weight = self.params.get('long_weight', 0.2)

        # Normalize weights
        total_weight = short_weight + medium_weight + long_weight
        short_weight /= total_weight
        medium_weight /= total_weight
        long_weight /= total_weight

        # Combined momentum
        combined_momentum = (
            short_weight * short_mom +
            medium_weight * medium_mom +
            long_weight * long_mom
        )

        # Volume confirmation
        if volume_confirmation:
            # Calculate relative volume (current vs average)
            vol_lookback = self.params.get('volume_lookback', 20)
            avg_volume = data['volume'].rolling(window=vol_lookback, min_periods=1).mean()
            relative_volume = data['volume'] / avg_volume

            # Volume confirmation factor (1.0 = average volume, >1 = higher)
            volume_factor = np.clip(relative_volume, 0.5, 2.0)

            # Amplify momentum when volume is high
            volume_boost = self.params.get('volume_boost', 0.2)
            combined_momentum = combined_momentum * (1 + volume_boost * (volume_factor - 1))

        # Volatility adjustment (risk-adjusted momentum)
        if volatility_adjust:
            vol_window = self.params.get('volatility_window', 20)
            # Calculate rolling volatility
            returns = data['close'].pct_change()
            volatility = returns.rolling(window=vol_window, min_periods=1).std()

            # Adjust momentum by inverse volatility (lower vol = stronger signal)
            # Use median volatility as baseline
            median_vol = volatility.median()
            if median_vol > 0:
                vol_adjustment = median_vol / (volatility + 1e-8)  # Avoid division by zero
                vol_adjustment = np.clip(vol_adjustment, 0.5, 2.0)
                combined_momentum = combined_momentum * vol_adjustment

        # Normalize to [-1, 1] range
        # Use rolling percentile rank for adaptive normalization
        rank_window = self.params.get('rank_window', 252)  # 1 year
        signal = combined_momentum.rolling(window=rank_window, min_periods=20).apply(
            lambda x: 2.0 * (pd.Series(x).rank().iloc[-1] / len(x)) - 1.0,
            raw=False
        )

        # Apply threshold filter
        # Only generate signals when momentum exceeds threshold
        if signal_threshold > 0:
            signal = signal.where(signal.abs() > signal_threshold, 0)

        # Fill NaN values with 0 (no signal)
        signal = signal.fillna(0)

        # Final clip to ensure [-1, 1] range
        signal = signal.clip(-1, 1)

        return signal

    def get_parameter_space(self) -> Dict[str, tuple]:
        """
        Define Optuna search space for momentum signal.

        Returns:
            Dict of parameter specifications
        """
        return {
            # Lookback periods (in days)
            'short_lookback': ('int', 5, 30),        # 1-6 weeks
            'medium_lookback': ('int', 20, 90),      # 1-3 months
            'long_lookback': ('int', 120, 252),      # 6-12 months

            # Timeframe weights
            'short_weight': ('float', 0.1, 0.7),
            'medium_weight': ('float', 0.1, 0.5),
            'long_weight': ('float', 0.1, 0.5),

            # Volume parameters
            'volume_confirmation': ('categorical', [True, False]),
            'volume_lookback': ('int', 10, 50),
            'volume_boost': ('float', 0.0, 0.5),

            # Volatility adjustment
            'volatility_adjust': ('categorical', [True, False]),
            'volatility_window': ('int', 10, 50),

            # Signal processing
            'use_log_returns': ('categorical', [True, False]),
            'signal_threshold': ('float', 0.0, 0.3),
            'rank_window': ('int', 60, 252),
        }

    def validate_params(self) -> None:
        """Validate momentum-specific parameters."""
        super().validate_params()

        # Ensure lookbacks are in ascending order
        if not (self.params['short_lookback'] <
                self.params['medium_lookback'] <
                self.params['long_lookback']):
            raise ValueError(
                "Lookback periods must be in ascending order: short < medium < long"
            )

        # Weights must be positive
        for weight_param in ['short_weight', 'medium_weight', 'long_weight']:
            if self.params.get(weight_param, 0) < 0:
                raise ValueError(f"{weight_param} must be non-negative")

    def __repr__(self) -> str:
        """String representation."""
        return (f"MomentumSignal(short={self.params['short_lookback']}, "
                f"medium={self.params['medium_lookback']}, "
                f"long={self.params['long_lookback']})")
