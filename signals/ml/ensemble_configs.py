"""
Production Ensemble Configurations

Centralized definitions for production-ready ensembles.
Each config is validated against the signal registry and backed by diagnostic reports.

Usage:
    from signals.ml.ensemble_configs import get_momentum_v2_ensemble

    ensemble = get_momentum_v2_ensemble(dm)
    scores = ensemble.generate_ensemble_scores(prices_by_ticker, rebalance_date)

Available Configs:
- get_momentum_v2_ensemble: Single-signal momentum (Trial 11, GO status)
- [Future]: get_momentum_quality_ensemble, get_research_insider_ensemble, etc.
"""

from typing import Dict, Any
from dataclasses import dataclass

from signals.ml.ensemble_signal import EnsembleSignal, EnsembleMember
from data.data_manager import DataManager
from config import get_logger

logger = get_logger(__name__)


@dataclass
class EnsembleDefinition:
    """
    Metadata for an ensemble configuration.

    Used for documentation and validation purposes.
    """
    name: str
    description: str
    status: str  # "PRODUCTION", "RESEARCH", "ARCHIVED"
    validation_report: str


def get_momentum_v1_legacy_quintile_ensemble(dm: DataManager) -> EnsembleSignal:
    """
    Legacy ensemble: Momentum with hard 20% quintiles (Trial 11 spec).

    This configuration exactly matches the original Trial 11 manual backtest logic.
    Uses quintile_mode='hard_20pct' to ensure exactly 20% of stocks per quintile
    via rank-based assignment.

    Configuration:
    - Formation period: 308 days (~14 months)
    - Skip period: 0 days (no gap)
    - Winsorization: [0.4, 99.6] (9.2% trim)
    - Rebalance: Monthly
    - Quintile mode: hard_20pct (exactly 20% per bin)

    Performance (2015-2024, S&P 500 PIT, from Trial 11):
    - Total Return: 89.53%
    - CAGR: 3.33%
    - Sharpe: 0.245
    - Max Drawdown: -44.76%

    Reference:
    - Trial 11 diagnostic: results/MOMENTUM_PHASE2_TRIAL11_DIAGNOSTIC.md
    - Quintile mode: hard_20pct (rank-based, exactly 20%)

    Args:
        dm: DataManager instance

    Returns:
        EnsembleSignal configured with legacy Trial 11 spec
    """
    momentum_params: Dict[str, Any] = {
        "formation_period": 308,
        "skip_period": 0,
        "winsorize_pct": [0.4, 99.6],
        "rebalance_frequency": "monthly",
        "quintiles": True,
        "quintile_mode": "hard_20pct",  # Legacy Trial 11 behavior
    }

    members = [
        EnsembleMember(
            signal_name="InstitutionalMomentum",
            version="v2",
            weight=1.0,
            normalize="none",
            params=momentum_params,
        )
    ]

    logger.info("Initializing legacy ensemble: Momentum v1 (hard 20% quintiles)")
    logger.info(f"  Formation: {momentum_params['formation_period']}d, "
               f"Skip: {momentum_params['skip_period']}d, "
               f"Winsor: {momentum_params['winsorize_pct']}")
    logger.info(f"  Quintile mode: {momentum_params['quintile_mode']}")

    return EnsembleSignal(
        members=members,
        data_manager=dm,
        enforce_go_only=True,
    )


def get_momentum_v2_adaptive_quintile_ensemble(dm: DataManager) -> EnsembleSignal:
    """
    Production ensemble: Momentum with adaptive quintiles (production default).

    Uses quintile_mode='adaptive' which allows bins to merge when momentum values
    cluster. This can result in >20% of stocks in top/bottom bins during periods
    of high correlation, providing better diversification.

    Configuration:
    - Formation period: 308 days (~14 months)
    - Skip period: 0 days (no gap)
    - Winsorization: [0.4, 99.6] (9.2% trim)
    - Rebalance: Monthly
    - Quintile mode: adaptive (bins merge when values cluster)

    Performance (2015-2024, S&P 500 PIT, via unified harness):
    - Total Return: 129.89%
    - CAGR: 8.89%
    - Sharpe: 0.601
    - Max Drawdown: -30.20%

    This is the production default configuration as of Phase 2 Step 3.

    Args:
        dm: DataManager instance

    Returns:
        EnsembleSignal configured for production use with adaptive quintiles
    """
    momentum_params: Dict[str, Any] = {
        "formation_period": 308,
        "skip_period": 0,
        "winsorize_pct": [0.4, 99.6],
        "rebalance_frequency": "monthly",
        "quintiles": True,
        "quintile_mode": "adaptive",  # Production default
    }

    members = [
        EnsembleMember(
            signal_name="InstitutionalMomentum",
            version="v2",
            weight=1.0,
            normalize="none",
            params=momentum_params,
        )
    ]

    logger.info("Initializing production ensemble: Momentum v2 (adaptive quintiles)")
    logger.info(f"  Formation: {momentum_params['formation_period']}d, "
               f"Skip: {momentum_params['skip_period']}d, "
               f"Winsor: {momentum_params['winsorize_pct']}")
    logger.info(f"  Quintile mode: {momentum_params['quintile_mode']}")

    return EnsembleSignal(
        members=members,
        data_manager=dm,
        enforce_go_only=True,
    )


def get_momentum_v2_ensemble(dm: DataManager) -> EnsembleSignal:
    """
    Backward compatibility alias for momentum_v2_adaptive_quintile.

    This function maintains compatibility with existing code while
    clearly delegating to the canonical adaptive quintile configuration.

    For new code, prefer calling get_momentum_v2_adaptive_quintile_ensemble()
    directly to make the quintile mode explicit.

    Args:
        dm: DataManager instance

    Returns:
        EnsembleSignal configured with adaptive quintiles (production default)
    """
    return get_momentum_v2_adaptive_quintile_ensemble(dm)


# Metadata registry (for documentation/introspection)
ENSEMBLE_REGISTRY = {
    "momentum_v1_legacy_quintile": EnsembleDefinition(
        name="momentum_v1_legacy_quintile",
        description="Momentum with hard 20% quintiles (Trial 11 spec, legacy reference)",
        status="ARCHIVED",
        validation_report="results/MOMENTUM_PHASE2_TRIAL11_DIAGNOSTIC.md"
    ),
    "momentum_v2_adaptive_quintile": EnsembleDefinition(
        name="momentum_v2_adaptive_quintile",
        description="Momentum with adaptive quintiles (production default)",
        status="PRODUCTION",
        validation_report="results/ensemble_baselines/momentum_v2_diagnostic.md"
    ),
    "momentum_v2": EnsembleDefinition(
        name="momentum_v2",
        description="Alias for momentum_v2_adaptive_quintile (backward compatibility)",
        status="PRODUCTION",
        validation_report="results/ensemble_baselines/momentum_v2_diagnostic.md"
    ),
    # Future configs:
    # "momentum_quality_v2": EnsembleDefinition(...),
    # "research_insider": EnsembleDefinition(..., status="RESEARCH"),
}


def list_available_ensembles() -> Dict[str, EnsembleDefinition]:
    """
    List all available ensemble configurations.

    Returns:
        Dict mapping ensemble name to metadata
    """
    return ENSEMBLE_REGISTRY.copy()


if __name__ == '__main__':
    print("Production Ensemble Configurations")
    print("=" * 80)
    print()

    for name, config in ENSEMBLE_REGISTRY.items():
        print(f"{name}:")
        print(f"  Description: {config.description}")
        print(f"  Status: {config.status}")
        print(f"  Report: {config.validation_report}")
        print()

    print("\nUsage:")
    print("  from signals.ml.ensemble_configs import get_momentum_v2_ensemble")
    print("  ensemble = get_momentum_v2_ensemble(dm)")
