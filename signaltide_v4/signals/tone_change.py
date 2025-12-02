"""
Semantic Tone Change Signal based on Tetlock et al. (2008).

Reference:
    Tetlock, P. C., Saar-Tsechansky, M., & Macskassy, S. (2008).
    "More Than Words: Quantifying Language to Measure Firms' Fundamentals".
    Journal of Finance, 63(3), 1437-1467.

Key insight: Negative tone in SEC filings predicts lower earnings and returns.
Change in tone (vs baseline) is more predictive than absolute levels.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import timedelta

import pandas as pd
import numpy as np

from .base import BaseSignal
from signaltide_v4.config.settings import get_settings

logger = logging.getLogger(__name__)


class ToneChangeSignal(BaseSignal):
    """
    Signal based on change in sentiment/tone of SEC filings.

    Note: This is a placeholder implementation.
    Full implementation would require:
    1. SEC filing text data
    2. NLP/LLM for tone analysis
    3. Historical tone baseline per company

    For now, returns neutral scores.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        db_path: Optional[str] = None,
    ):
        """
        Initialize tone change signal.

        Args:
            params: Parameter overrides
            db_path: Database path
        """
        super().__init__(name='tone_change', params=params)

        # Ensure params is a dict for .get() calls
        params = params or {}

        settings = get_settings()
        self.tone_window_days = params.get('tone_window_days', 30)

        self.db_path = db_path or settings.db_path

    def compute_raw_scores(
        self,
        tickers: List[str],
        as_of_date: str
    ) -> pd.Series:
        """
        Compute tone change scores.

        Note: Placeholder - returns neutral scores.
        Full implementation would:
        1. Get recent 10-K/10-Q filings
        2. Analyze tone with NLP/LLM
        3. Compare to historical baseline
        4. Score based on tone change
        """
        logger.warning(
            "ToneChangeSignal: Placeholder implementation. "
            "Full version requires SEC filing text analysis."
        )

        # Return neutral scores for now
        return pd.Series(0.0, index=tickers)

    def _analyze_filing_tone(self, filing_text: str) -> float:
        """
        Analyze tone of a filing.

        Placeholder - would use NLP/LLM in production.
        """
        # Simple negative word count approach (placeholder)
        negative_words = [
            'loss', 'decline', 'decrease', 'adverse', 'risk',
            'uncertain', 'difficult', 'challenge', 'concern',
            'impairment', 'restructuring', 'litigation'
        ]

        text_lower = filing_text.lower()
        word_count = len(text_lower.split())

        if word_count == 0:
            return 0.0

        negative_count = sum(text_lower.count(word) for word in negative_words)

        # Negative tone score (higher = more negative)
        tone = negative_count / word_count * 100

        return -tone  # Return inverted (positive = good tone)

    def get_diagnostics(
        self,
        raw_scores: pd.Series,
        normalized_scores: pd.Series
    ) -> Dict[str, Any]:
        """Add tone-specific diagnostics."""
        base_diag = super().get_diagnostics(raw_scores, normalized_scores)

        base_diag.update({
            'implementation': 'placeholder',
            'note': 'Full implementation requires SEC filing text analysis',
        })

        return base_diag
