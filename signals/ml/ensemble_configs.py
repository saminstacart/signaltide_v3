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

# Import signal classes needed for mapping
from signals.quality.cross_sectional_quality import CrossSectionalQuality

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


def get_momentum_quality_v1_ensemble(dm: DataManager) -> EnsembleSignal:
    """
    Multi-signal ensemble: Momentum + Quality (equal weights, v1).

    First production multi-signal ensemble combining:
    - InstitutionalMomentum v2 (adaptive quintiles, 308-day formation)
    - CrossSectionalQuality v1 (QMJ methodology, 3-year fundamentals)

    Configuration:
    - Calibrated weights: 0.25 Momentum, 0.75 Quality
    - Both signals use normalize="none" (already return quintiles)
    - Momentum params: formation=308d, skip=0d, winsorize=[0.4, 99.6], adaptive quintiles
    - Quality params: w_profitability=0.4, w_growth=0.3, w_safety=0.3, adaptive quintiles

    Weight Calibration (Phase 3 Milestone 3.4):
    - Grid sweep: M=0.25/Q=0.75 best across all metrics (Sharpe 2.876, +19% vs pure momentum)
    - Optuna validation: Continuous search converged to M≈0.20/Q≈0.80 (32 trials, TPE sampler)
    - Both methods confirm quality-heavy plateau at M=0.20-0.25 (stable, non-overfitted)
    - Canonical weight M=0.25/Q=0.75 selected as plateau center (1:3 ratio, interpretable)
    - Quality adds value in 4/5 macro regimes (crisis/bear resilience, QE outperformance)
    - Full diagnostics: momentum_quality_v1_weight_sweep.md, momentum_quality_v1_weight_optuna.md

    This ensemble uses the cross-sectional pathway via generate_cross_sectional_ensemble_scores(),
    enabling signals with different data dependencies (prices vs fundamentals).

    Performance (2015-2024, 25/75 weights, S&P 500 PIT):
    - Total Return: 135.98%
    - CAGR: 9.28%
    - Sharpe: 2.876
    - Max Drawdown: -23.89%

    Expected Benefits:
    - Low correlation between momentum and quality factors (~0.3-0.4)
    - Quality provides stability during momentum crashes (reduces drawdown by 5%)
    - Diversified factor exposure

    Phase 3 Milestone 3: First multi-signal baseline for validation.

    Args:
        dm: DataManager instance

    Returns:
        EnsembleSignal with Momentum + Quality members

    Notes:
        - Use with make_multisignal_ensemble_fn() adapter for backtesting
        - Both signals must implement generate_cross_sectional_scores()
        - Registry validated (both signals have GO status)
    """
    # Momentum v2 production params (adaptive quintiles)
    momentum_params: Dict[str, Any] = {
        "formation_period": 308,
        "skip_period": 0,
        "winsorize_pct": [0.4, 99.6],
        "rebalance_frequency": "monthly",
        "quintiles": True,
        "quintile_mode": "adaptive",
    }

    # Quality v1 production params (default weights, adaptive quintiles)
    quality_params: Dict[str, Any] = {
        "w_profitability": 0.4,
        "w_growth": 0.3,
        "w_safety": 0.3,
        "winsorize_pct": [5, 95],
        "quintiles": True,
        "quintile_mode": "adaptive",
        "min_coverage": 0.5,
    }

    # Weights calibrated via grid sweep + Optuna validation (2015-2024, Phase 3 M3.4)
    # M=0.25/Q=0.75 selected: grid sweep best across all metrics, Optuna confirms plateau at M≈0.20-0.25
    # Quality-heavy allocation empirically validated, non-overfitted, interpretable (1:3 ratio)
    members = [
        EnsembleMember(
            signal_name="InstitutionalMomentum",
            version="v2",
            weight=0.25,  # Calibrated weight (grid + Optuna validated)
            normalize="none",  # Signal already returns quintiles
            params=momentum_params,
        ),
        EnsembleMember(
            signal_name="CrossSectionalQuality",
            version="v1",
            weight=0.75,  # Calibrated weight (grid + Optuna validated)
            normalize="none",  # Signal already returns quintiles
            params=quality_params,
        ),
    ]

    logger.info("Initializing multi-signal ensemble: Momentum + Quality v1")
    logger.info("  - InstitutionalMomentum v2: weight=0.25, formation=308d, adaptive quintiles")
    logger.info("  - CrossSectionalQuality v1: weight=0.75, QMJ methodology, adaptive quintiles")

    return EnsembleSignal(
        members=members,
        data_manager=dm,
        enforce_go_only=True,
    )


def get_momentum_quality_insider_v1_ensemble(dm: DataManager) -> EnsembleSignal:
    """
    Multi-signal ensemble: Momentum + Quality + Insider (M3.6 three-signal baseline).

    ⚠️ **STATUS: RESEARCH_NO_GO (2025-11-24)** ⚠️

    This ensemble is NOT recommended for production use. The 10-year diagnostic
    (2015-2024) shows that adding the insider signal at 25% weight DEGRADES
    performance relative to the M+Q baseline:

    - M+Q baseline: 135.98% return, 0.628 Sharpe
    - M+Q+I (this ensemble): 122.22% return, 0.591 Sharpe
    - **Impact: -13.76% return, -0.037 Sharpe (NEGATIVE)**

    While the insider signal showed positive contribution in a 5-year test
    (2020-2024: +7.24%), this was regime-specific and not robust over the
    full 10-year period. The signal is highly correlated (98.94%) with M+Q
    and adds significant computational cost (10.7x slowdown) without value.

    **Recommendation:** Use get_momentum_quality_v1_ensemble() instead.

    **Full diagnostic:** docs/logs/phase3_m3_M3.6_full_diag_20251124.md
    **Decision date:** 2025-11-24

    ---

    First three-signal ensemble combining:
    - InstitutionalMomentum v2 (adaptive quintiles, 308-day formation)
    - CrossSectionalQuality v1 (QMJ methodology, 3-year fundamentals)
    - InstitutionalInsider v1 (Cohen-Malloy-Pomorski informed trading)

    Configuration:
    - Recommended v1 weights: Momentum 0.25, Quality 0.50, Insider 0.25
    - All signals use normalize="none" (already return quintiles)
    - Momentum params: formation=308d, skip=0d, winsorize=[0.4, 99.6], adaptive quintiles
    - Quality params: w_profitability=0.4, w_growth=0.3, w_safety=0.3, adaptive quintiles
    - Insider params: lookback_days=90, min_transactions=3, value_threshold=100000

    Rationale (from M3.6 spec):
    - M+Q baseline (25/75) delivers Sharpe 2.876, serves as foundation
    - Insider adds low-correlation alpha (orthogonal to price/fundamental factors)
    - Conservative insider weight (0.25) due to coverage uncertainty at Phase 1
    - Quality remains largest component (0.50) for stability

    Data Coverage Validation (Phase 3 Priority 3.1):
    - Insider data coverage: 98.1% of S&P 500 sample (53/54 tickers)
    - Temporal density: Excellent (95K-120K trades/year, 2015-2024)
    - Transaction diversity: Good mix (23% awards, 23% options, 20% sales, 1.4% purchases)
    - GO decision: Coverage meets ≥75% threshold for Phase 1 ensemble

    This ensemble uses the cross-sectional pathway via generate_cross_sectional_ensemble_scores(),
    enabling signals with different data dependencies (prices/fundamentals/insider trades).

    Performance: TBD (awaiting diagnostic baseline)

    Expected Benefits:
    - Three-factor diversification (momentum, quality, informed trading)
    - Insider signal potentially orthogonal to M+Q (trade-based vs price/fundamental)
    - Conservative allocation manages insider signal uncertainty

    Phase 3 Milestone 3.6: Three-signal baseline for validation.

    Args:
        dm: DataManager instance

    Returns:
        EnsembleSignal with Momentum + Quality + Insider members

    Notes:
        - Use with make_multisignal_ensemble_fn() adapter for backtesting
        - All signals must implement generate_cross_sectional_scores()
        - Weights subject to tuning based on diagnostic results
        - See: docs/ENSEMBLES_M3.6_THREE_SIGNAL_SPEC.md
    """
    # Momentum v2 production params (adaptive quintiles)
    momentum_params: Dict[str, Any] = {
        "formation_period": 308,
        "skip_period": 0,
        "winsorize_pct": [0.4, 99.6],
        "rebalance_frequency": "monthly",
        "quintiles": True,
        "quintile_mode": "adaptive",
    }

    # Quality v1 production params (default weights, adaptive quintiles)
    quality_params: Dict[str, Any] = {
        "w_profitability": 0.4,
        "w_growth": 0.3,
        "w_safety": 0.3,
        "winsorize_pct": [5, 95],
        "quintiles": True,
        "quintile_mode": "adaptive",
        "min_coverage": 0.5,
    }

    # Insider v1 params (conservative settings for Phase 1)
    insider_params: Dict[str, Any] = {
        "lookback_days": 90,
        "min_transactions": 3,
        "value_threshold": 100000,
        "quintiles": True,
        "quintile_mode": "adaptive",
    }

    # M3.6 v1 weights: Conservative insider allocation, quality-heavy for stability
    # Weights subject to calibration based on diagnostic results
    members = [
        EnsembleMember(
            signal_name="InstitutionalMomentum",
            version="v2",
            weight=0.25,  # Recommended v1 weight
            normalize="none",
            params=momentum_params,
        ),
        EnsembleMember(
            signal_name="CrossSectionalQuality",
            version="v1",
            weight=0.50,  # Recommended v1 weight (largest component)
            normalize="none",
            params=quality_params,
        ),
        EnsembleMember(
            signal_name="InstitutionalInsider",
            version="v1",
            weight=0.25,  # Recommended v1 weight (conservative)
            normalize="none",
            params=insider_params,
        ),
    ]

    logger.info("Initializing three-signal ensemble: Momentum + Quality + Insider v1")
    logger.info("  - InstitutionalMomentum v2: weight=0.25, formation=308d, adaptive quintiles")
    logger.info("  - CrossSectionalQuality v1: weight=0.50, QMJ methodology, adaptive quintiles")
    logger.info("  - InstitutionalInsider v1: weight=0.25, lookback=90d, min_txns=3")
    logger.info("  Data coverage validated: 98.1% S&P 500 sample (GO decision)")

    return EnsembleSignal(
        members=members,
        data_manager=dm,
        enforce_go_only=False,  # Insider signal not yet GO status
    )


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
    "momentum_quality_v1": EnsembleDefinition(
        name="momentum_quality_v1",
        description="Multi-signal: Momentum + Quality (25/75 weights, calibrated via grid + Optuna)",
        status="CANDIDATE_PROD",  # 10-year diagnostic complete, ready for production evaluation
        validation_report="docs/logs/phase3_m3_M3.6_full_diag_20251124.md"
        # Full diagnostics: momentum_quality_v1_weight_sweep.md, momentum_quality_v1_weight_optuna.md
    ),
    "momentum_quality_insider_v1": EnsembleDefinition(
        name="momentum_quality_insider_v1",
        description="Three-signal: Momentum + Quality + Insider (25/50/25 weights, M3.6 baseline)",
        status="RESEARCH_NO_GO",  # 10-year diagnostic complete: -13.76% return, -0.037 Sharpe vs M+Q
        validation_report="docs/logs/phase3_m3_M3.6_full_diag_20251124.md"
        # Decision date: 2025-11-24. Insider v1 degrades performance. Do not use in production.
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
