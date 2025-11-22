"""
Cross-Sectional Ensemble Signal

Combines multiple validated signals into a single portfolio score using
configurable weights and normalization methods.

Design Principles:
- Registry-validated: Only GO signals allowed by default
- Cross-sectional: Normalizes signals within universe at each rebalance
- Monthly rebalancing: Aligns with institutional signal frequency
- Transparent: Simple weighted combination with clear normalization

Usage:
    from signals.ml.ensemble_signal import EnsembleSignal, EnsembleMember

    # Momentum-only ensemble
    ensemble = EnsembleSignal(
        members=[
            EnsembleMember(
                signal_name="InstitutionalMomentum",
                version="v2",
                weight=1.0,
                normalize="zscore"
            )
        ],
        data_manager=dm,
        enforce_go_only=True
    )

    # Generate scores for rebalance
    scores = ensemble.generate_ensemble_scores(
        prices_by_ticker=prices_dict,
        rebalance_date=rebalance_date
    )
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np

from core.signal_registry import get_signal_status
from data.data_manager import DataManager
from signals.momentum.institutional_momentum import InstitutionalMomentum
from signals.insider.institutional_insider import InstitutionalInsider
from config import get_logger

logger = get_logger(__name__)


@dataclass
class EnsembleMember:
    """
    Configuration for a single signal within an ensemble.

    Attributes:
        signal_name: Name from signal registry (e.g., 'InstitutionalMomentum')
        version: Version identifier (e.g., 'v2')
        weight: Contribution weight (will be normalized by total weight)
        allow_no_go: If True, allows NO_GO signals (for research/testing)
        normalize: Normalization method ('zscore', 'rank', 'none')
        params: Optional signal-specific parameters (overrides defaults)
    """
    signal_name: str
    version: str
    weight: float
    allow_no_go: bool = False
    normalize: str = "zscore"  # or "rank", "none"
    params: Optional[Dict[str, Any]] = None


# Signal class registry (maps (name, version) to implementation class)
_SIGNAL_CLASS_MAP = {
    ("InstitutionalMomentum", "v2"): InstitutionalMomentum,
    ("InstitutionalInsider", "v1"): InstitutionalInsider,
    # Add new signals here as they pass validation
}


class EnsembleSignal:
    """
    Cross-sectional ensemble of multiple signals.

    Generates combined portfolio scores by:
    1. Computing each signal's cross-sectional scores
    2. Normalizing within universe (z-score or rank)
    3. Combining using weighted average

    Enforces signal registry validation to prevent using NO_GO signals
    in production unless explicitly allowed.
    """

    def __init__(self,
                 members: List[EnsembleMember],
                 data_manager: Optional[DataManager] = None,
                 enforce_go_only: bool = True):
        """
        Initialize ensemble with signal members.

        Args:
            members: List of EnsembleMember configurations
            data_manager: Optional DataManager (creates new if None)
            enforce_go_only: If True, rejects NO_GO signals unless allow_no_go=True

        Raises:
            ValueError: If signal not found in registry or is NO_GO (when enforced)
        """
        self.members = members
        self.dm = data_manager or DataManager()
        self.enforce_go_only = enforce_go_only

        self._validate_members()
        self._instantiate_signals()

        logger.info(f"Initialized EnsembleSignal with {len(self.members)} members")
        for m in self.members:
            logger.info(f"  - {m.signal_name} {m.version}: weight={m.weight:.2f}, normalize={m.normalize}")

    def _validate_members(self) -> None:
        """
        Validate all ensemble members against signal registry.

        Checks:
        - Signal exists in registry
        - Signal status is GO (unless allow_no_go=True)

        Raises:
            ValueError: If validation fails
        """
        for m in self.members:
            status = get_signal_status(m.signal_name, m.version)

            if status is None:
                raise ValueError(
                    f"Signal {m.signal_name} v{m.version} not found in registry. "
                    f"Add to core/signal_registry.py first."
                )

            if self.enforce_go_only and status.status != "GO" and not m.allow_no_go:
                raise ValueError(
                    f"Signal {m.signal_name} v{m.version} has status {status.status}. "
                    f"Set allow_no_go=True to include NO_GO signals explicitly, "
                    f"or set enforce_go_only=False for testing."
                )

            logger.debug(f"Validated {m.signal_name} {m.version}: {status.status}")

    def _instantiate_signals(self) -> None:
        """
        Create signal object instances for all members.

        Uses _SIGNAL_CLASS_MAP to look up implementation classes.
        Passes member.params if provided, otherwise uses signal defaults.

        Raises:
            ValueError: If signal class not mapped
        """
        self._signal_objects = {}

        for m in self.members:
            key = (m.signal_name, m.version)
            cls = _SIGNAL_CLASS_MAP.get(key)

            if cls is None:
                raise ValueError(
                    f"No implementation class mapped for {key}. "
                    f"Add to _SIGNAL_CLASS_MAP in ensemble_signal.py"
                )

            # Use member params if provided, otherwise signal defaults
            params = m.params if m.params is not None else {}

            # Instantiate signal (handle different __init__ signatures)
            # InstitutionalInsider takes data_manager, InstitutionalMomentum doesn't
            try:
                # Try with data_manager first (InstitutionalInsider)
                self._signal_objects[key] = cls(params=params, data_manager=self.dm)
            except TypeError:
                # Fall back to params-only (InstitutionalMomentum)
                self._signal_objects[key] = cls(params=params)

            logger.debug(f"Instantiated {m.signal_name} {m.version}")

    def generate_ensemble_scores(
        self,
        prices_by_ticker: Dict[str, pd.Series],
        rebalance_date: pd.Timestamp,
        bulk_insider_data: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Generate combined ensemble scores for a single rebalance.

        Args:
            prices_by_ticker: Dict mapping ticker -> price series up to rebalance_date
            rebalance_date: Current rebalance timestamp
            bulk_insider_data: Optional bulk insider data for InstitutionalInsider

        Returns:
            pd.Series indexed by ticker with combined ensemble scores

        Process:
            1. For each signal, compute cross-sectional scores
            2. Normalize each signal's scores (z-score, rank, or none)
            3. Weight and combine into final ensemble score
        """
        if not prices_by_ticker:
            logger.warning("Empty prices_by_ticker, returning empty scores")
            return pd.Series(dtype=float)

        logger.debug(f"Generating ensemble scores for {len(prices_by_ticker)} tickers at {rebalance_date.date()}")

        per_signal_scores = {}  # (name, version) -> pd.Series[ticker -> score]

        # 1. Compute each signal's cross-sectional scores
        for m in self.members:
            key = (m.signal_name, m.version)
            sig_obj = self._signal_objects[key]

            scores = self._compute_single_signal_scores(
                member=m,
                sig_obj=sig_obj,
                prices_by_ticker=prices_by_ticker,
                rebalance_date=rebalance_date,
                bulk_insider_data=bulk_insider_data,
            )

            # 2. Normalize scores
            normalized = self._normalize(scores, mode=m.normalize)
            per_signal_scores[key] = normalized

            logger.debug(f"  {m.signal_name} {m.version}: {len(scores)} non-zero scores, "
                        f"normalized with {m.normalize}")

        # 3. Weighted combination
        tickers = sorted(prices_by_ticker.keys())
        combined = pd.Series(0.0, index=tickers)

        total_weight = sum(m.weight for m in self.members if m.weight != 0)
        if total_weight == 0:
            logger.warning("Total ensemble weight is zero, returning zeros")
            return combined

        for m in self.members:
            if m.weight == 0:
                continue

            key = (m.signal_name, m.version)
            w = m.weight / total_weight

            # Add weighted signal scores (fill missing with 0)
            combined = combined.add(per_signal_scores[key] * w, fill_value=0.0)

        logger.info(f"Generated ensemble scores: {len(combined.dropna())} tickers, "
                   f"range=[{combined.min():.3f}, {combined.max():.3f}]")

        return combined

    def _compute_single_signal_scores(
        self,
        member: EnsembleMember,
        sig_obj,
        prices_by_ticker: Dict[str, pd.Series],
        rebalance_date: pd.Timestamp,
        bulk_insider_data: Optional[pd.DataFrame],
    ) -> pd.Series:
        """
        Compute cross-sectional scores for a single signal.

        Calls signal's generate_signals() method for each ticker,
        using the same pattern as baseline scripts.

        Args:
            member: Ensemble member config
            sig_obj: Instantiated signal object
            prices_by_ticker: Price data by ticker
            rebalance_date: Current rebalance date
            bulk_insider_data: Bulk insider data (for InstitutionalInsider)

        Returns:
            pd.Series mapping ticker -> signal score
        """
        scores = {}

        for ticker, px in prices_by_ticker.items():
            # Build DataFrame matching baseline pattern
            data = pd.DataFrame({"close": px, "ticker": ticker})
            data = data[data.index <= rebalance_date]

            if data.empty or len(data) < 90:  # Need minimum history
                continue

            try:
                # Call signal's generate_signals()
                # InstitutionalInsider needs bulk_insider_data
                if isinstance(sig_obj, InstitutionalInsider):
                    sig_series = sig_obj.generate_signals(data, bulk_insider_data=bulk_insider_data)
                else:
                    sig_series = sig_obj.generate_signals(data)

                if len(sig_series) == 0:
                    continue

                # Take most recent signal value
                signal_value = sig_series.iloc[-1]

                if pd.notna(signal_value) and signal_value != 0:
                    scores[ticker] = signal_value

            except Exception as e:
                logger.debug(f"Error generating {member.signal_name} signal for {ticker}: {e}")
                continue

        if not scores:
            logger.debug(f"No valid scores for {member.signal_name} {member.version}")
            return pd.Series(dtype=float)

        return pd.Series(scores)

    def _normalize(self, s: pd.Series, mode: str) -> pd.Series:
        """
        Normalize signal scores cross-sectionally.

        Args:
            s: Raw signal scores
            mode: Normalization method ('zscore', 'rank', 'none')

        Returns:
            Normalized scores

        Methods:
            - zscore: (x - mean) / std
            - rank: Percentile rank scaled to [-0.5, 0.5]
            - none: Pass through unchanged
        """
        if s.empty or mode == "none":
            return s.fillna(0.0)

        if mode == "rank":
            # Cross-sectional percentile rank, centered at 0
            r = s.rank(method="average", na_option="keep", pct=True)
            # Scale to [-0.5, 0.5]
            r = r - 0.5
            return r.fillna(0.0)

        # Default: z-score normalization
        if mode == "zscore":
            mu = s.mean()
            sigma = s.std(ddof=0)

            if sigma <= 1e-12:
                logger.debug("Zero variance in signal scores, returning zeros")
                return s * 0.0

            return (s - mu) / sigma

        logger.warning(f"Unknown normalization mode '{mode}', using 'none'")
        return s.fillna(0.0)

    def __repr__(self) -> str:
        member_strs = [f"{m.signal_name} {m.version} (w={m.weight:.2f})"
                      for m in self.members]
        return f"EnsembleSignal({', '.join(member_strs)})"


if __name__ == '__main__':
    print("EnsembleSignal - Cross-Sectional Multi-Signal Combiner")
    print()
    print("Features:")
    print("  - Registry-validated signal members")
    print("  - Configurable normalization (zscore, rank, none)")
    print("  - Weighted combination")
    print("  - GO/NO_GO enforcement")
    print()
    print("Example:")
    print("  ensemble = EnsembleSignal(")
    print("      members=[")
    print("          EnsembleMember('InstitutionalMomentum', 'v2', weight=1.0)")
    print("      ]")
    print("  )")
