"""
Validation framework for SignalTide v3.

Prevents overfitting through rigorous validation techniques.
"""

from validation.purged_kfold import PurgedKFold
from validation.monte_carlo import MonteCarloValidator
from validation.statistical_tests import StatisticalTests
from validation.deflated_sharpe import DeflatedSharpe

__all__ = [
    'PurgedKFold',
    'MonteCarloValidator',
    'StatisticalTests',
    'DeflatedSharpe',
]
