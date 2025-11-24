"""
Quality-based trading signals.

Signals based on fundamental quality metrics from financial statements.

CrossSectionalQuality v1: Production-ready QMJ implementation (use this)
InstitutionalQuality v0: DEPRECATED time-series version (archived)
SimpleQuality: Simple baseline for comparison
"""

from .cross_sectional_quality import CrossSectionalQuality  # v1 - Production
from .institutional_quality import InstitutionalQuality  # v0 - DEPRECATED
from .simple_quality import SimpleQuality

__all__ = [
    'CrossSectionalQuality',  # v1 - Use this for all production work
    'InstitutionalQuality',    # v0 - DEPRECATED, archived for reference
    'SimpleQuality'
]
