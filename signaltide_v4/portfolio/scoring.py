"""
Signal aggregation and scoring for portfolio construction.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np

from signaltide_v4.signals.base import BaseSignal, SignalResult
from signaltide_v4.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class AggregatedScore:
    """Container for aggregated signal scores."""

    scores: pd.Series  # Ticker -> composite score
    as_of_date: str
    signal_weights: Dict[str, float]
    component_scores: Dict[str, pd.Series]
    regime_tilt: Optional[str] = None
    diagnostics: Dict[str, Any] = None


class SignalAggregator:
    """
    Aggregates multiple signals into a composite score.

    Features:
    - Configurable signal weights
    - Regime-aware tilting
    - Missing signal handling
    """

    def __init__(
        self,
        signals: Optional[List[BaseSignal]] = None,
        weights: Optional[Dict[str, float]] = None,
        min_signals_required: int = 2,
    ):
        """
        Initialize signal aggregator.

        Args:
            signals: List of signal objects (optional, can be empty)
            weights: Optional weight overrides (signal_name -> weight)
            min_signals_required: Minimum valid signals to produce score
        """
        signals = signals or []
        self.signals = {s.name: s for s in signals}
        self.signal_names = list(self.signals.keys())

        # Default equal weights
        if weights is None and len(signals) > 0:
            weights = {name: 1.0 / len(signals) for name in self.signal_names}
        elif weights is None:
            weights = {}

        # Normalize weights
        total_weight = sum(weights.get(name, 0) for name in self.signal_names)
        if total_weight > 0:
            self.weights = {
                name: weights.get(name, 0) / total_weight
                for name in self.signal_names
            }
        else:
            self.weights = {}

        self.min_signals_required = min_signals_required

        logger.info(f"SignalAggregator initialized with {len(signals)} signals")
        for name, weight in self.weights.items():
            logger.info(f"  {name}: {weight:.1%}")

    def aggregate(
        self,
        tickers: List[str],
        as_of_date: str,
        regime: Optional[str] = None,
    ) -> AggregatedScore:
        """
        Aggregate all signals for given universe.

        Args:
            tickers: Universe of tickers
            as_of_date: Point-in-time date
            regime: Optional market regime for tilting

        Returns:
            AggregatedScore with composite scores
        """
        component_scores = {}
        signal_results = {}

        # Generate each signal
        for name, signal in self.signals.items():
            try:
                result = signal.generate_signals(tickers, as_of_date)
                signal_results[name] = result
                component_scores[name] = result.scores
                logger.debug(f"Signal {name}: {len(result.scores.dropna())} valid scores")
            except Exception as e:
                logger.error(f"Error generating signal {name}: {e}")
                component_scores[name] = pd.Series(np.nan, index=tickers)

        # Apply regime tilting if specified
        adjusted_weights = self._apply_regime_tilt(regime)

        # Aggregate scores
        composite = pd.Series(0.0, index=tickers)
        valid_weight_sum = pd.Series(0.0, index=tickers)

        for name, scores in component_scores.items():
            weight = adjusted_weights.get(name, 0)
            if weight <= 0:
                continue

            # Add weighted score where valid
            valid_mask = scores.notna()
            composite[valid_mask] += weight * scores[valid_mask]
            valid_weight_sum[valid_mask] += weight

        # Normalize by total valid weight
        composite = composite / valid_weight_sum.replace(0, np.nan)

        # Count valid signals per ticker
        signal_counts = pd.DataFrame(component_scores).notna().sum(axis=1)
        composite[signal_counts < self.min_signals_required] = np.nan

        # Build diagnostics
        diagnostics = {
            'signal_coverage': {
                name: float(scores.notna().mean())
                for name, scores in component_scores.items()
            },
            'signal_correlations': self._compute_correlations(component_scores),
            'regime_applied': regime,
        }

        return AggregatedScore(
            scores=composite,
            as_of_date=as_of_date,
            signal_weights=adjusted_weights,
            component_scores=component_scores,
            regime_tilt=regime,
            diagnostics=diagnostics,
        )

    def _apply_regime_tilt(
        self,
        regime: Optional[str]
    ) -> Dict[str, float]:
        """
        Adjust signal weights based on market regime.

        Regimes:
        - 'bull': Tilt toward momentum
        - 'bear': Tilt toward quality
        - 'volatile': Reduce momentum, increase quality
        - None: Use base weights
        """
        adjusted = self.weights.copy()

        if regime is None:
            return adjusted

        # Regime-specific tilts
        tilts = {
            'bull': {
                'residual_momentum': 1.3,
                'quality': 0.8,
                'opportunistic_insider': 1.0,
                'tone_change': 0.9,
            },
            'bear': {
                'residual_momentum': 0.6,
                'quality': 1.4,
                'opportunistic_insider': 1.0,
                'tone_change': 1.2,
            },
            'volatile': {
                'residual_momentum': 0.5,
                'quality': 1.5,
                'opportunistic_insider': 0.8,
                'tone_change': 1.2,
            },
        }

        if regime in tilts:
            for name in adjusted:
                tilt = tilts[regime].get(name, 1.0)
                adjusted[name] *= tilt

            # Renormalize
            total = sum(adjusted.values())
            adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted

    def _compute_correlations(
        self,
        component_scores: Dict[str, pd.Series]
    ) -> Dict[str, float]:
        """Compute pairwise correlations between signals."""
        correlations = {}
        names = list(component_scores.keys())

        for i, name1 in enumerate(names):
            for name2 in names[i+1:]:
                s1 = component_scores[name1].dropna()
                s2 = component_scores[name2].dropna()

                common = s1.index.intersection(s2.index)
                if len(common) > 10:
                    corr = s1.loc[common].corr(s2.loc[common])
                    correlations[f"{name1}_vs_{name2}"] = float(corr)

        return correlations

    def get_top_n(
        self,
        aggregated: AggregatedScore,
        n: int = 25,
        sector_max_pct: float = 0.40,
        sectors: Optional[Dict[str, str]] = None,
    ) -> pd.Series:
        """
        Get top N tickers by score with optional sector constraints.

        Args:
            aggregated: AggregatedScore result
            n: Number of positions
            sector_max_pct: Maximum weight in any sector
            sectors: Ticker -> sector mapping

        Returns:
            Series with selected tickers and scores
        """
        scores = aggregated.scores.dropna().sort_values(ascending=False)

        if sectors is None or not sectors:
            # No sector constraint
            return scores.head(n)

        # Apply sector constraints
        selected = []
        sector_counts = {}

        for ticker, score in scores.items():
            sector = sectors.get(ticker, 'Unknown')
            current_count = sector_counts.get(sector, 0)

            # Check if sector is at max
            max_in_sector = int(n * sector_max_pct)
            if current_count >= max_in_sector:
                continue

            selected.append(ticker)
            sector_counts[sector] = current_count + 1

            if len(selected) >= n:
                break

        return scores.loc[selected]
