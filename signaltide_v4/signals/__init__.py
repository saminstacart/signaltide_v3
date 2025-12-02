"""Signal modules for SignalTide v4."""

from .base import BaseSignal, SignalResult
from .residual_momentum import ResidualMomentumSignal
from .quality import QualitySignal
from .insider import OpportunisticInsiderSignal
from .tone_change import ToneChangeSignal

__all__ = [
    'BaseSignal',
    'SignalResult',
    'ResidualMomentumSignal',
    'QualitySignal',
    'OpportunisticInsiderSignal',
    'ToneChangeSignal',
]
