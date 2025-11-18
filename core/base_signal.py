"""
Base signal class that all trading signals must implement.

This enforces a consistent interface and helps prevent lookahead bias.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class BaseSignal(ABC):
    """
    Abstract base class for all trading signals.

    All signals must implement:
    1. generate_signals(): Create trading signals from data
    2. get_parameter_space(): Define Optuna search space

    Signals must NOT use future data - only data up to each timestamp.
    """

    def __init__(self, params: Dict[str, Any], name: Optional[str] = None):
        """
        Initialize signal.

        Args:
            params: Signal parameters (from optimization or default)
            name: Optional signal name (defaults to class name)
        """
        self.params = params
        self.name = name or self.__class__.__name__
        self.validate_params()

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from market data.

        CRITICAL: This method must NOT use any future data.
        Signal at time t can only use data up to and including time t.

        Args:
            data: OHLCV DataFrame with DatetimeIndex
                  Required columns: open, high, low, close, volume

        Returns:
            Series with same index as data, values in [-1, 1]
            -1 = strong sell signal
             0 = neutral / no signal
             1 = strong buy signal

        Example:
            ```python
            def generate_signals(self, data):
                # Calculate momentum using only past data
                lookback = self.params['lookback']
                momentum = data['close'].pct_change(lookback)

                # Normalize to [-1, 1] and return
                return momentum.clip(-1, 1)
            ```
        """
        pass

    @abstractmethod
    def get_parameter_space(self) -> Dict[str, tuple]:
        """
        Define the Optuna search space for this signal's parameters.

        Returns:
            Dict mapping parameter names to (type, min, max) tuples

        Example:
            ```python
            def get_parameter_space(self):
                return {
                    'lookback': ('int', 5, 200),
                    'threshold': ('float', 0.0, 0.5),
                    'method': ('categorical', ['A', 'B', 'C']),
                }
            ```
        """
        pass

    def validate_params(self) -> None:
        """
        Validate that parameters are within acceptable ranges.

        Override this method to add custom validation.
        Raises ValueError if parameters are invalid.
        """
        # Base validation - check required parameters exist
        param_space = self.get_parameter_space()
        for param_name in param_space.keys():
            if param_name not in self.params:
                raise ValueError(f"Missing required parameter: {param_name}")

    def validate_no_lookahead(self, data: pd.DataFrame, signals: pd.Series) -> bool:
        """
        Verify that signals don't use future data.

        This is a sanity check - computes signal incrementally and checks
        that signal at time t doesn't change when future data is added.

        Args:
            data: OHLCV data
            signals: Generated signals

        Returns:
            True if no lookahead bias detected, False otherwise

        Note: This is computationally expensive (O(n²)) so use sparingly.
        """
        if len(data) < 10:
            # Skip for very small datasets
            return True

        # Sample random timestamps to check
        n_checks = min(10, len(data) // 10)
        check_indices = np.random.choice(
            range(len(data) // 2, len(data)),
            size=n_checks,
            replace=False
        )

        for idx in check_indices:
            # Generate signal using data up to this point
            partial_data = data.iloc[:idx + 1]
            partial_signal = self.generate_signals(partial_data).iloc[-1]

            # Original signal at this point
            original_signal = signals.iloc[idx]

            # Signals should match (within floating point tolerance)
            if not np.isclose(partial_signal, original_signal, atol=1e-6):
                print(f"Lookahead bias detected at index {idx}:")
                print(f"  Partial: {partial_signal}")
                print(f"  Original: {original_signal}")
                return False

        return True

    def __repr__(self) -> str:
        """String representation of signal."""
        return f"{self.name}(params={self.params})"

    def __str__(self) -> str:
        """Human-readable string."""
        return self.name


class CompositeSignal(BaseSignal):
    """
    Combines multiple signals with weights.

    This allows creating ensemble signals from individual signals.
    """

    def __init__(self, signals: Dict[str, BaseSignal], weights: Dict[str, float],
                 params: Optional[Dict[str, Any]] = None):
        """
        Initialize composite signal.

        Args:
            signals: Dict of {name: signal} pairs
            weights: Dict of {name: weight} pairs (must sum to 1.0)
            params: Optional parameters for the composite signal
        """
        self.signals = signals
        self.weights = weights

        # Validate weights
        if not np.isclose(sum(weights.values()), 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights.values())}")

        if set(signals.keys()) != set(weights.keys()):
            raise ValueError("Signal names and weight names must match")

        super().__init__(params or {}, name="CompositeSignal")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate combined signal as weighted average of component signals.

        Args:
            data: OHLCV data

        Returns:
            Weighted combination of signals
        """
        combined = pd.Series(0.0, index=data.index)

        for name, signal in self.signals.items():
            weight = self.weights[name]
            signal_values = signal.generate_signals(data)
            combined += weight * signal_values

        # Clip to [-1, 1] range
        return combined.clip(-1, 1)

    def get_parameter_space(self) -> Dict[str, tuple]:
        """
        Composite signals can optimize individual signal parameters
        as well as weights.
        """
        param_space = {}

        # Add parameter spaces from all component signals
        for name, signal in self.signals.items():
            signal_params = signal.get_parameter_space()
            # Prefix with signal name to avoid collisions
            for param_name, param_spec in signal_params.items():
                param_space[f"{name}_{param_name}"] = param_spec

        # Add weights as parameters
        for name in self.signals.keys():
            param_space[f"weight_{name}"] = ('float', 0.0, 1.0)

        return param_space


# Example signal implementation for reference
class ExampleMomentumSignal(BaseSignal):
    """
    Example momentum signal for demonstration.

    This is a simple price momentum signal that serves as a template
    for implementing new signals.
    """

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate momentum signals based on price change.

        Methodology:
        1. Calculate percentage change over lookback period
        2. Normalize to [-1, 1] range
        3. Apply threshold filter
        """
        lookback = self.params['lookback']
        threshold = self.params['threshold']

        # Calculate momentum (uses shift implicitly in pct_change - no lookahead)
        momentum = data['close'].pct_change(lookback)

        # Normalize to [-1, 1] by clipping
        signals = momentum.clip(-1, 1)

        # Apply threshold - only signal when momentum > threshold
        signals = signals.where(signals.abs() > threshold, 0)

        return signals

    def get_parameter_space(self) -> Dict[str, tuple]:
        """Define parameter search space."""
        return {
            'lookback': ('int', 5, 200),
            'threshold': ('float', 0.0, 0.5),
        }

    def validate_params(self) -> None:
        """Custom parameter validation."""
        super().validate_params()

        if self.params['lookback'] < 1:
            raise ValueError("lookback must be >= 1")

        if not 0 <= self.params['threshold'] <= 1:
            raise ValueError("threshold must be in [0, 1]")
