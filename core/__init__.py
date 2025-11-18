"""
Core framework for SignalTide v3.

This module contains base classes and fundamental abstractions.
"""

from core.base_signal import BaseSignal
from core.portfolio import Portfolio
from core.types import SignalValue, Position, Trade
from data.data_manager import DataManager

__all__ = [
    'BaseSignal',
    'Portfolio',
    'SignalValue',
    'Position',
    'Trade',
    'DataManager',
]
