"""
Trading signals for SignalTide v3.

All signals inherit from BaseSignal and implement:
- generate_signals(): Signal generation logic
- get_parameter_space(): Optuna search space
"""

# Import example signal
from core.base_signal import ExampleMomentumSignal

__all__ = [
    'ExampleMomentumSignal',
]

# Add new signals here as they are implemented
