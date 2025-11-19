"""
SignalTide v3 - Signal Module

Institutional-grade signals implementing professional quantitative methodologies.

Available Signals:
- InstitutionalMomentum: Jegadeesh-Titman 12-1 momentum
- InstitutionalQuality: Asness-Frazzini-Pedersen Quality Minus Junk
- InstitutionalInsider: Cohen-Malloy-Pomorski insider trading analysis

All signals use:
- Cross-sectional methodology
- Monthly rebalancing
- Quintile construction
- Professional winsorization and ranking

Previous simple signals archived in: archive/simple_signals_v1/
"""

# Institutional-grade signals (production)
from signals.momentum.institutional_momentum import InstitutionalMomentum, CrossSectionalMomentum
from signals.quality.institutional_quality import InstitutionalQuality
from signals.insider.institutional_insider import InstitutionalInsider

# Export list
__all__ = [
    'InstitutionalMomentum',
    'CrossSectionalMomentum',
    'InstitutionalQuality',
    'InstitutionalInsider',
]

# Version info
__version__ = '3.0.0'
__status__ = 'institutional'
