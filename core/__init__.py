"""
Core framework for SignalTide v3.

This module contains base classes and fundamental abstractions.

Note: DataManager is in the data layer, not core. Import it directly:
    from data.data_manager import DataManager
"""

from core.base_signal import BaseSignal
from core.portfolio import Portfolio
from core.types import SignalValue, Position, Trade

__all__ = [
    'BaseSignal',
    'Portfolio',
    'SignalValue',
    'Position',
    'Trade',
]
