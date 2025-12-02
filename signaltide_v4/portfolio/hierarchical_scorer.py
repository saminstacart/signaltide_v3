"""
Hierarchical Signal Scorer for V5 Strategy.

Implements the Phase 6 recommended hierarchical scoring logic:
  Stage 1: Quality Filter (safety gate)
  Stage 2: Insider Signal (alpha source) - 60% weight
  Stage 3: Momentum Signal (timing) - 40% weight

Supports two gating modes:
  - HARD GATE: Stocks failing quality threshold get score = -999 (excluded)
  - SOFT GATE: Quality score acts as a multiplier (preserves best insider signals)

References:
    - Phase 6 Deep Research Final Verdict (2025-12-01)
    - Cohen, Malloy & Pomorski (2012) - Insider signals
    - Asness, Frazzini & Pedersen (2018) - Quality factors
    - Jegadeesh & Titman (1993) - Momentum timing
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

from signaltide_v4.signals.base import BaseSignal, SignalResult

logger = logging.getLogger(__name__)


class GateMode(Enum):
    """Quality gate modes."""
    HARD = "hard"  # Quality < threshold -> exclude (score = -999)
    SOFT = "soft"  # Quality score acts as multiplier


@dataclass
class HierarchicalScoreResult:
    """Container for hierarchical scoring results."""

    # Final composite scores
    scores: pd.Series  # Ticker -> final score

    # Metadata
    as_of_date: str
    gate_mode: GateMode

    # Component scores (for diagnostics)
    quality_scores: pd.Series
    insider_scores: pd.Series
    momentum_scores: pd.Series

    # Gate statistics
    quality_pass_mask: pd.Series  # True if passed quality gate
    n_quality_passed: int
    n_quality_failed: int

    # Configuration used
    insider_weight: float
    momentum_weight: float
    quality_threshold_percentile: float
    soft_gate_multiplier: float

    # Diagnostics
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'as_of_date': self.as_of_date,
            'gate_mode': self.gate_mode.value,
            'n_quality_passed': self.n_quality_passed,
            'n_quality_failed': self.n_quality_failed,
            'quality_pass_rate': self.n_quality_passed / (self.n_quality_passed + self.n_quality_failed)
                if (self.n_quality_passed + self.n_quality_failed) > 0 else 0,
            'insider_weight': self.insider_weight,
            'momentum_weight': self.momentum_weight,
            'quality_threshold_percentile': self.quality_threshold_percentile,
            'soft_gate_multiplier': self.soft_gate_multiplier,
            'score_stats': {
                'mean': float(self.scores.mean()) if len(self.scores.dropna()) > 0 else None,
                'std': float(self.scores.std()) if len(self.scores.dropna()) > 0 else None,
                'min': float(self.scores.min()) if len(self.scores.dropna()) > 0 else None,
                'max': float(self.scores.max()) if len(self.scores.dropna()) > 0 else None,
            },
            'diagnostics': self.diagnostics,
        }


class HierarchicalScorer:
    """
    Hierarchical signal scorer implementing the V5 strategy logic.

    Architecture:
        1. QUALITY FILTER (Gate): Filter/penalize low-quality stocks
        2. INSIDER ALPHA (60%): Opportunistic insider buying/selling
        3. MOMENTUM TIMING (40%): Residual momentum for entry timing

    Gate Modes:
        HARD: Quality below threshold -> excluded completely
        SOFT: Quality below threshold -> score multiplied by penalty factor

    The soft gate mode addresses the concern that hard filtering might
    exclude the best insider signals (which often occur in lower-quality
    stocks that are about to improve).
    """

    # Exclusion score for hard gate failures
    EXCLUSION_SCORE = -999.0

    def __init__(
        self,
        quality_signal: Optional[BaseSignal] = None,
        insider_signal: Optional[BaseSignal] = None,
        momentum_signal: Optional[BaseSignal] = None,
        gate_mode: GateMode = GateMode.HARD,
        insider_weight: float = 0.6,
        momentum_weight: float = 0.4,
        quality_threshold_percentile: float = 40.0,
        soft_gate_multiplier: float = 0.5,
    ):
        """
        Initialize hierarchical scorer.

        Args:
            quality_signal: Quality signal for filtering
            insider_signal: Insider signal (alpha source)
            momentum_signal: Momentum signal (timing)
            gate_mode: HARD (exclude) or SOFT (penalize)
            insider_weight: Weight for insider signal (default 0.6)
            momentum_weight: Weight for momentum signal (default 0.4)
            quality_threshold_percentile: Top X% pass quality gate (default 40)
            soft_gate_multiplier: Score multiplier for stocks failing soft gate (default 0.5)
        """
        self.quality_signal = quality_signal
        self.insider_signal = insider_signal
        self.momentum_signal = momentum_signal

        self.gate_mode = gate_mode
        self.insider_weight = insider_weight
        self.momentum_weight = momentum_weight
        self.quality_threshold_percentile = quality_threshold_percentile
        self.soft_gate_multiplier = soft_gate_multiplier

        # Validate weights sum to 1
        total_weight = insider_weight + momentum_weight
        if abs(total_weight - 1.0) > 0.001:
            logger.warning(f"Signal weights sum to {total_weight}, normalizing to 1.0")
            self.insider_weight = insider_weight / total_weight
            self.momentum_weight = momentum_weight / total_weight

        logger.info(
            f"HierarchicalScorer initialized: "
            f"mode={gate_mode.value}, "
            f"quality_threshold={quality_threshold_percentile}%, "
            f"insider_w={self.insider_weight:.0%}, "
            f"momentum_w={self.momentum_weight:.0%}"
        )

        if gate_mode == GateMode.SOFT:
            logger.info(f"  Soft gate multiplier: {soft_gate_multiplier:.2f}x")

    def score(
        self,
        tickers: List[str],
        as_of_date: str,
        precomputed_signals: Optional[Dict[str, pd.Series]] = None,
    ) -> HierarchicalScoreResult:
        """
        Compute hierarchical scores for all tickers.

        Args:
            tickers: Universe of tickers
            as_of_date: Point-in-time date
            precomputed_signals: Optional dict of precomputed signal scores
                                 Keys: 'quality', 'insider', 'momentum'

        Returns:
            HierarchicalScoreResult with composite scores and diagnostics
        """
        if not tickers:
            return self._empty_result(as_of_date)

        # Get signal scores (either precomputed or generate now)
        quality_scores = self._get_signal_scores(
            'quality', tickers, as_of_date, precomputed_signals
        )
        insider_scores = self._get_signal_scores(
            'insider', tickers, as_of_date, precomputed_signals
        )
        momentum_scores = self._get_signal_scores(
            'momentum', tickers, as_of_date, precomputed_signals
        )

        # Stage 1: Apply quality gate
        quality_pass_mask, gate_multiplier = self._apply_quality_gate(quality_scores)

        # Stage 2 & 3: Compute weighted alpha score
        alpha_scores = (
            self.insider_weight * insider_scores.fillna(0) +
            self.momentum_weight * momentum_scores.fillna(0)
        )

        # Apply gate effect
        if self.gate_mode == GateMode.HARD:
            # Hard gate: exclude failures completely
            final_scores = alpha_scores.copy()
            final_scores[~quality_pass_mask] = self.EXCLUSION_SCORE
        else:
            # Soft gate: multiply by penalty factor
            final_scores = alpha_scores * gate_multiplier

        # Calculate statistics
        n_passed = quality_pass_mask.sum()
        n_failed = (~quality_pass_mask).sum()

        # Build diagnostics
        diagnostics = self._build_diagnostics(
            quality_scores, insider_scores, momentum_scores,
            quality_pass_mask, final_scores
        )

        logger.info(
            f"Hierarchical scoring for {as_of_date}: "
            f"{n_passed} passed quality gate, {n_failed} failed "
            f"({n_passed/(n_passed+n_failed):.1%} pass rate)"
        )

        return HierarchicalScoreResult(
            scores=final_scores,
            as_of_date=as_of_date,
            gate_mode=self.gate_mode,
            quality_scores=quality_scores,
            insider_scores=insider_scores,
            momentum_scores=momentum_scores,
            quality_pass_mask=quality_pass_mask,
            n_quality_passed=int(n_passed),
            n_quality_failed=int(n_failed),
            insider_weight=self.insider_weight,
            momentum_weight=self.momentum_weight,
            quality_threshold_percentile=self.quality_threshold_percentile,
            soft_gate_multiplier=self.soft_gate_multiplier,
            diagnostics=diagnostics,
        )

    def _get_signal_scores(
        self,
        signal_name: str,
        tickers: List[str],
        as_of_date: str,
        precomputed: Optional[Dict[str, pd.Series]],
    ) -> pd.Series:
        """Get signal scores, either from precomputed or by generating."""

        # Check for precomputed scores
        if precomputed and signal_name in precomputed:
            scores = precomputed[signal_name]
            # Ensure alignment with tickers
            aligned = pd.Series(np.nan, index=tickers)
            common = set(tickers) & set(scores.index)
            if common:
                aligned.loc[list(common)] = scores.loc[list(common)]
            logger.debug(f"Using precomputed {signal_name} scores: {len(common)} tickers")
            return aligned

        # Get the signal object
        signal_map = {
            'quality': self.quality_signal,
            'insider': self.insider_signal,
            'momentum': self.momentum_signal,
        }
        signal = signal_map.get(signal_name)

        if signal is None:
            logger.warning(f"No {signal_name} signal provided, using zeros")
            return pd.Series(0.0, index=tickers)

        # Generate signals
        result = signal.generate_signals(tickers, as_of_date)
        return result.scores

    def _apply_quality_gate(
        self,
        quality_scores: pd.Series,
    ) -> tuple:
        """
        Apply quality gate based on mode.

        Returns:
            (pass_mask, multiplier)
            - pass_mask: Boolean series, True = passed gate
            - multiplier: Series of multipliers (1.0 for passed, penalty for failed)
        """
        valid_scores = quality_scores.dropna()

        if len(valid_scores) < 5:
            logger.warning("Insufficient quality scores for gating, passing all")
            pass_mask = pd.Series(True, index=quality_scores.index)
            multiplier = pd.Series(1.0, index=quality_scores.index)
            return pass_mask, multiplier

        # Calculate threshold (higher score = higher quality)
        # quality_threshold_percentile=40 means top 40% pass
        # So threshold is at 60th percentile of scores
        threshold_pct = 100 - self.quality_threshold_percentile
        threshold = np.percentile(valid_scores, threshold_pct)

        # Determine pass/fail
        pass_mask = quality_scores >= threshold

        # For tickers without quality data, apply neutral treatment
        # This prevents excluding stocks just because quality data is missing
        missing_quality = quality_scores.isna()

        if self.gate_mode == GateMode.HARD:
            # Hard gate: missing quality data = fail (exclude)
            pass_mask = pass_mask.fillna(False)
            multiplier = pd.Series(1.0, index=quality_scores.index)
        else:
            # Soft gate: missing quality data = neutral (multiplier = 1.0)
            pass_mask = pass_mask.fillna(True)  # Don't penalize missing

            # Create multiplier: passed = 1.0, failed = soft_gate_multiplier
            multiplier = pd.Series(1.0, index=quality_scores.index)
            multiplier[~pass_mask & ~missing_quality] = self.soft_gate_multiplier

        logger.debug(
            f"Quality gate: threshold={threshold:.3f}, "
            f"passed={pass_mask.sum()}, failed={(~pass_mask).sum()}"
        )

        return pass_mask, multiplier

    def _build_diagnostics(
        self,
        quality_scores: pd.Series,
        insider_scores: pd.Series,
        momentum_scores: pd.Series,
        quality_pass_mask: pd.Series,
        final_scores: pd.Series,
    ) -> Dict[str, Any]:
        """Build diagnostic information."""

        # Signal correlations
        correlations = {}
        for name1, s1 in [('quality', quality_scores), ('insider', insider_scores), ('momentum', momentum_scores)]:
            for name2, s2 in [('quality', quality_scores), ('insider', insider_scores), ('momentum', momentum_scores)]:
                if name1 < name2:
                    common = s1.dropna().index.intersection(s2.dropna().index)
                    if len(common) > 10:
                        corr = s1.loc[common].corr(s2.loc[common])
                        correlations[f"{name1}_vs_{name2}"] = float(corr)

        # Coverage stats
        coverage = {
            'quality': float(quality_scores.notna().mean()),
            'insider': float(insider_scores.notna().mean()),
            'momentum': float(momentum_scores.notna().mean()),
        }

        # Score distributions by gate status
        passed_scores = final_scores[quality_pass_mask & (final_scores > self.EXCLUSION_SCORE + 1)]
        failed_scores = final_scores[~quality_pass_mask | (final_scores <= self.EXCLUSION_SCORE + 1)]

        return {
            'signal_correlations': correlations,
            'signal_coverage': coverage,
            'passed_score_stats': {
                'mean': float(passed_scores.mean()) if len(passed_scores) > 0 else None,
                'std': float(passed_scores.std()) if len(passed_scores) > 0 else None,
            },
            'failed_score_stats': {
                'mean': float(failed_scores[failed_scores > self.EXCLUSION_SCORE + 1].mean())
                    if len(failed_scores[failed_scores > self.EXCLUSION_SCORE + 1]) > 0 else None,
            },
            'top_insider_in_failed_quality': self._check_top_insider_filtered(
                insider_scores, quality_pass_mask
            ),
        }

    def _check_top_insider_filtered(
        self,
        insider_scores: pd.Series,
        quality_pass_mask: pd.Series,
    ) -> Dict[str, Any]:
        """
        Check if top insider signals are being filtered by quality gate.

        This is a key diagnostic for the hard vs soft gate decision.
        """
        if len(insider_scores.dropna()) < 10:
            return {'warning': 'insufficient_data'}

        # Get top 10% insider scores
        top_10_pct = insider_scores.quantile(0.90)
        top_insider = insider_scores >= top_10_pct

        # How many of top insider signals failed quality gate?
        top_insider_failed = top_insider & ~quality_pass_mask
        n_top_insider = top_insider.sum()
        n_top_insider_failed = top_insider_failed.sum()

        pct_filtered = n_top_insider_failed / n_top_insider if n_top_insider > 0 else 0

        return {
            'n_top_10pct_insider': int(n_top_insider),
            'n_top_insider_failed_quality': int(n_top_insider_failed),
            'pct_top_insider_filtered': float(pct_filtered),
            'warning': 'high_filter_rate' if pct_filtered > 0.3 else None,
        }

    def _empty_result(self, as_of_date: str) -> HierarchicalScoreResult:
        """Return empty result for empty universe."""
        return HierarchicalScoreResult(
            scores=pd.Series(dtype=float),
            as_of_date=as_of_date,
            gate_mode=self.gate_mode,
            quality_scores=pd.Series(dtype=float),
            insider_scores=pd.Series(dtype=float),
            momentum_scores=pd.Series(dtype=float),
            quality_pass_mask=pd.Series(dtype=bool),
            n_quality_passed=0,
            n_quality_failed=0,
            insider_weight=self.insider_weight,
            momentum_weight=self.momentum_weight,
            quality_threshold_percentile=self.quality_threshold_percentile,
            soft_gate_multiplier=self.soft_gate_multiplier,
            diagnostics={'warning': 'empty_universe'},
        )


def create_v5_scorer_hard(
    quality_signal: Optional[BaseSignal] = None,
    insider_signal: Optional[BaseSignal] = None,
    momentum_signal: Optional[BaseSignal] = None,
) -> HierarchicalScorer:
    """
    Create V5 scorer with hard quality gate (Phase 6 default).

    Configuration:
        - Quality gate: Top 40% pass
        - Insider weight: 60%
        - Momentum weight: 40%
        - Failed quality: EXCLUDED (score = -999)
    """
    return HierarchicalScorer(
        quality_signal=quality_signal,
        insider_signal=insider_signal,
        momentum_signal=momentum_signal,
        gate_mode=GateMode.HARD,
        insider_weight=0.6,
        momentum_weight=0.4,
        quality_threshold_percentile=40.0,
    )


def create_v5_scorer_soft(
    quality_signal: Optional[BaseSignal] = None,
    insider_signal: Optional[BaseSignal] = None,
    momentum_signal: Optional[BaseSignal] = None,
    soft_multiplier: float = 0.5,
) -> HierarchicalScorer:
    """
    Create V5 scorer with soft quality gate.

    This variant preserves stocks with strong insider signals
    even if they fail the quality gate, applying a penalty instead
    of complete exclusion.

    Configuration:
        - Quality gate: Top 40% pass at full score
        - Failed quality: Multiplied by soft_multiplier (default 0.5)
        - Insider weight: 60%
        - Momentum weight: 40%
    """
    return HierarchicalScorer(
        quality_signal=quality_signal,
        insider_signal=insider_signal,
        momentum_signal=momentum_signal,
        gate_mode=GateMode.SOFT,
        insider_weight=0.6,
        momentum_weight=0.4,
        quality_threshold_percentile=40.0,
        soft_gate_multiplier=soft_multiplier,
    )


# Configuration presets for comparison testing
V5_SCORER_CONFIGS = {
    'V5-Hierarchical-Hard': {
        'gate_mode': GateMode.HARD,
        'insider_weight': 0.6,
        'momentum_weight': 0.4,
        'quality_threshold_percentile': 40.0,
        'description': 'Phase 6 default: hard quality gate',
    },
    'V5-Hierarchical-Soft': {
        'gate_mode': GateMode.SOFT,
        'insider_weight': 0.6,
        'momentum_weight': 0.4,
        'quality_threshold_percentile': 40.0,
        'soft_gate_multiplier': 0.5,
        'description': 'Soft gate: penalize but preserve strong insider signals',
    },
    'V5-Hierarchical-Soft-Light': {
        'gate_mode': GateMode.SOFT,
        'insider_weight': 0.6,
        'momentum_weight': 0.4,
        'quality_threshold_percentile': 40.0,
        'soft_gate_multiplier': 0.7,
        'description': 'Light soft gate: smaller penalty for failed quality',
    },
    'V5-InsiderHeavy-Hard': {
        'gate_mode': GateMode.HARD,
        'insider_weight': 0.7,
        'momentum_weight': 0.3,
        'quality_threshold_percentile': 40.0,
        'description': 'More insider weight (70/30)',
    },
    'V5-MomentumHeavy-Hard': {
        'gate_mode': GateMode.HARD,
        'insider_weight': 0.5,
        'momentum_weight': 0.5,
        'quality_threshold_percentile': 40.0,
        'description': 'Equal weights (50/50)',
    },
}
