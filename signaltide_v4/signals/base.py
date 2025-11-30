"""
Base signal class with standardized interface and diagnostics.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

import pandas as pd
import numpy as np

from signaltide_v4.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class SignalResult:
    """Container for signal computation results."""

    scores: pd.Series  # Ticker -> score mapping
    as_of_date: str
    signal_name: str
    coverage: float  # Fraction of universe with valid signals
    computation_time_ms: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'signal_name': self.signal_name,
            'as_of_date': self.as_of_date,
            'coverage': self.coverage,
            'computation_time_ms': self.computation_time_ms,
            'n_valid': len(self.scores.dropna()),
            'n_total': len(self.scores),
            'score_stats': {
                'mean': float(self.scores.mean()) if len(self.scores) > 0 else None,
                'std': float(self.scores.std()) if len(self.scores) > 0 else None,
                'min': float(self.scores.min()) if len(self.scores) > 0 else None,
                'max': float(self.scores.max()) if len(self.scores) > 0 else None,
            },
            'diagnostics': self.diagnostics,
        }


class BaseSignal(ABC):
    """
    Abstract base class for all signals.

    Provides:
    - Standardized interface for signal generation
    - Automatic timing and diagnostics
    - Cross-sectional normalization to [-1, 1]
    """

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize signal.

        Args:
            name: Signal name for logging/tracking
            params: Optional parameter overrides
        """
        self.name = name
        self.params = params or {}
        self.settings = get_settings()
        logger.info(f"Initialized signal: {name}")

    @abstractmethod
    def compute_raw_scores(
        self,
        tickers: List[str],
        as_of_date: str
    ) -> pd.Series:
        """
        Compute raw (unnormalized) signal scores.

        Must be implemented by subclasses.

        Args:
            tickers: Universe of tickers to score
            as_of_date: Point-in-time date for computation

        Returns:
            Series with raw scores (can be any range)
        """
        pass

    def generate_signals(
        self,
        tickers: List[str],
        as_of_date: str,
        normalize: bool = True
    ) -> SignalResult:
        """
        Generate signals with timing and diagnostics.

        Args:
            tickers: Universe of tickers
            as_of_date: Point-in-time date
            normalize: Whether to normalize to [-1, 1]

        Returns:
            SignalResult with scores and metadata
        """
        start_time = datetime.now()

        # Compute raw scores
        raw_scores = self.compute_raw_scores(tickers, as_of_date)

        # Normalize if requested
        if normalize and len(raw_scores.dropna()) > 0:
            scores = self.normalize_cross_sectional(raw_scores)
        else:
            scores = raw_scores

        # Calculate timing
        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Calculate coverage
        n_valid = len(scores.dropna())
        n_total = len(tickers)
        coverage = n_valid / n_total if n_total > 0 else 0.0

        # Build diagnostics
        diagnostics = self.get_diagnostics(raw_scores, scores)

        result = SignalResult(
            scores=scores,
            as_of_date=as_of_date,
            signal_name=self.name,
            coverage=coverage,
            computation_time_ms=elapsed_ms,
            diagnostics=diagnostics,
        )

        logger.info(
            f"{self.name}: {n_valid}/{n_total} tickers scored "
            f"({coverage:.1%} coverage) in {elapsed_ms:.1f}ms"
        )

        return result

    def normalize_cross_sectional(
        self,
        scores: pd.Series,
        method: str = 'quintile'
    ) -> pd.Series:
        """
        Normalize scores cross-sectionally to [-1, 1].

        Args:
            scores: Raw scores
            method: 'quintile', 'rank', or 'zscore'

        Returns:
            Normalized scores in [-1, 1]
        """
        valid = scores.dropna()
        if len(valid) < 5:
            return pd.Series(np.nan, index=scores.index)

        if method == 'quintile':
            # Quintile-based: bottom 20% = -1, top 20% = 1
            try:
                quintiles = pd.qcut(valid, q=5, labels=False, duplicates='drop')
                # Map 0-4 to [-1, 1]: 0->-1, 1->-0.5, 2->0, 3->0.5, 4->1
                normalized = (quintiles - 2) / 2
            except ValueError:
                # Fall back to rank if too few unique values
                normalized = self.normalize_cross_sectional(scores, method='rank')
                return normalized

        elif method == 'rank':
            # Rank-based: linear mapping
            ranks = valid.rank(pct=True)
            normalized = (ranks - 0.5) * 2  # Map [0,1] to [-1,1]

        elif method == 'zscore':
            # Z-score with winsorization
            mean = valid.mean()
            std = valid.std()
            if std == 0:
                return pd.Series(0.0, index=scores.index)
            z = (valid - mean) / std
            # Winsorize to [-3, 3] then scale to [-1, 1]
            z = z.clip(-3, 3)
            normalized = z / 3

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        # Reindex to include all original tickers (NaN for missing)
        result = pd.Series(np.nan, index=scores.index)
        result.loc[normalized.index] = normalized.values

        return result

    def get_diagnostics(
        self,
        raw_scores: pd.Series,
        normalized_scores: pd.Series
    ) -> Dict[str, Any]:
        """
        Get diagnostic information for debugging.

        Override in subclasses for signal-specific diagnostics.
        """
        valid_raw = raw_scores.dropna()
        valid_norm = normalized_scores.dropna()

        return {
            'raw_score_range': [
                float(valid_raw.min()) if len(valid_raw) > 0 else None,
                float(valid_raw.max()) if len(valid_raw) > 0 else None,
            ],
            'normalized_score_range': [
                float(valid_norm.min()) if len(valid_norm) > 0 else None,
                float(valid_norm.max()) if len(valid_norm) > 0 else None,
            ],
            'score_correlation': float(
                valid_raw.corr(valid_norm)
            ) if len(valid_raw) > 2 else None,
        }
