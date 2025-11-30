"""
Quality Signal combining multiple quality factors.

References:
    - Ball et al. (2016): Cash-based Operating Profitability (CbOP)
    - Frazzini & Pedersen (2014): Betting Against Beta (BAB)
    - Grullon & Michaely (2004): Buyback Yield
    - Cooper, Gulen & Schill (2008): Asset Growth (negative screen)
    - Sloan (1996): Accrual Anomaly
    - Novy-Marx (2013): Gross Profitability
    - Eisfeldt & Papanikolaou (2013): Intangible Value (R&D capitalization)
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import timedelta

import pandas as pd
import numpy as np

from .base import BaseSignal
from signaltide_v4.data.fundamental_data import FundamentalDataProvider
from signaltide_v4.data.market_data import MarketDataProvider
from signaltide_v4.config.settings import get_settings

logger = logging.getLogger(__name__)


class QualitySignal(BaseSignal):
    """
    Multi-component quality signal.

    Components:
    1. CbOP (Cash-based Operating Profitability) - Ball et al. 2016
    2. BAB (Betting Against Beta) - Low beta preferred
    3. Buyback Yield - Net share repurchases
    4. Asset Growth Screen - Penalize high asset growth
    5. Accrual Anomaly - Sloan (1996) - Low accruals = high quality
    6. Gross Profitability - Novy-Marx (2013) - High GP/Assets = pricing power
    7. Share Reduction - Buyback proxy via share count change
    8. Intangible Value - Eisfeldt & Papanikolaou (2013) - R&D capitalization

    All components are normalized and combined with configurable weights.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        fundamental_data: Optional[FundamentalDataProvider] = None,
        market_data: Optional[MarketDataProvider] = None,
    ):
        """
        Initialize quality signal.

        Args:
            params: Parameter overrides including component weights
            fundamental_data: FundamentalDataProvider instance
            market_data: MarketDataProvider instance
        """
        super().__init__(name='quality', params=params)

        # Ensure params is a dict for .get() calls
        params = params or {}

        settings = get_settings()

        # Component weights (8 components totaling 1.0)
        # Original factors (reduced weights to make room for intangible)
        self.w_cbop = params.get('w_cbop', 0.15)          # Was 0.20, -0.05
        self.w_bab = params.get('w_bab', 0.10)            # Was 0.15, -0.05 (complemented by vol guardrail)
        self.w_buyback = params.get('w_buyback', 0.10)    # Unchanged
        self.w_asset_growth = params.get('w_asset_growth', 0.15)  # Unchanged

        # Quality factors from Sloan (1996) and Novy-Marx (2013)
        self.w_accrual = params.get('w_accrual', 0.15)         # Accrual Anomaly
        self.w_gross_profit = params.get('w_gross_profit', 0.15)  # Gross Profitability
        self.w_share_reduction = params.get('w_share_reduction', 0.10)  # Share buyback proxy

        # NEW: Intangible Value - Eisfeldt & Papanikolaou (2013)
        self.w_intangible = params.get('w_intangible', 0.10)  # R&D capitalization

        # Asset growth threshold (Cooper et al. 2008)
        self.max_asset_growth = params.get('max_asset_growth', 0.30)

        self.fundamental_data = fundamental_data or FundamentalDataProvider()
        self.market_data = market_data or MarketDataProvider()

    def compute_raw_scores(
        self,
        tickers: List[str],
        as_of_date: str
    ) -> pd.Series:
        """
        Compute composite quality score.

        Steps:
        1. Calculate each component
        2. Normalize each component cross-sectionally
        3. Apply asset growth screen
        4. Combine with weights
        """
        if not tickers:
            return pd.Series(dtype=float)

        # Initialize scores DataFrame
        scores_df = pd.DataFrame(index=tickers)

        # 1. CbOP (Cash-based Operating Profitability)
        cbop = self.fundamental_data.get_cbop(tickers, as_of_date)
        scores_df['cbop'] = self.normalize_cross_sectional(cbop, method='rank')
        logger.debug(f"CbOP: {len(cbop.dropna())}/{len(tickers)} scored")

        # 2. BAB (Betting Against Beta) - lower beta = higher score
        beta = self.market_data.get_beta(tickers, as_of_date)
        # Invert: low beta is good
        bab = -beta
        scores_df['bab'] = self.normalize_cross_sectional(bab, method='rank')
        logger.debug(f"BAB: {len(beta.dropna())}/{len(tickers)} scored")

        # 3. Buyback Yield
        buyback = self.fundamental_data.get_buyback_yield(tickers, as_of_date)
        scores_df['buyback'] = self.normalize_cross_sectional(buyback, method='rank')
        logger.debug(f"Buyback: {len(buyback.dropna())}/{len(tickers)} scored")

        # 4. Asset Growth (screen/penalty)
        asset_growth = self.fundamental_data.get_asset_growth(tickers, as_of_date)
        # High growth is bad (per Cooper et al. 2008)
        # Penalize if growth > threshold
        growth_penalty = pd.Series(0.0, index=tickers)
        for ticker in asset_growth.index:
            if ticker in growth_penalty.index:
                ag = asset_growth[ticker]
                if pd.notna(ag):
                    if ag > self.max_asset_growth:
                        growth_penalty[ticker] = -1.0  # Strong penalty
                    elif ag > 0.15:
                        growth_penalty[ticker] = -0.5  # Moderate penalty
                    else:
                        growth_penalty[ticker] = 0.5  # Neutral to slight positive

        scores_df['asset_growth'] = growth_penalty
        logger.debug(f"Asset Growth: {len(asset_growth.dropna())}/{len(tickers)} scored")

        # 5. Accrual Anomaly (Sloan 1996)
        # Uses MRT dimension for TTM to avoid seasonality bias
        accrual = self.fundamental_data.get_accrual_anomaly(tickers, as_of_date)
        scores_df['accrual'] = self.normalize_cross_sectional(accrual, method='rank')
        logger.debug(f"Accrual: {len(accrual.dropna())}/{len(tickers)} scored")

        # 6. Gross Profitability (Novy-Marx 2013)
        # Uses MRT dimension for TTM to avoid seasonality bias
        gp = self.fundamental_data.get_gross_profitability(tickers, as_of_date)
        scores_df['gross_profit'] = self.normalize_cross_sectional(gp, method='rank')
        logger.debug(f"Gross Profit: {len(gp.dropna())}/{len(tickers)} scored")

        # 7. Share Reduction (Buyback Proxy)
        # Uses ARQ dimension (correct for stock variables with explicit time-diffing)
        share_red = self.fundamental_data.get_share_reduction(tickers, as_of_date)
        scores_df['share_reduction'] = self.normalize_cross_sectional(share_red, method='rank')
        logger.debug(f"Share Reduction: {len(share_red.dropna())}/{len(tickers)} scored")

        # 8. Intangible Value (Eisfeldt & Papanikolaou 2013)
        # Uses MRT for R&D/SG&A (TTM flows) to fix "Old Economy" bias
        intangible = self.fundamental_data.get_intangible_yield(tickers, as_of_date)
        scores_df['intangible'] = self.normalize_cross_sectional(intangible, method='rank')
        logger.debug(f"Intangible: {len(intangible.dropna())}/{len(tickers)} scored")

        # Combine components (weights sum to 1.0)
        composite = (
            self.w_cbop * scores_df['cbop'].fillna(0) +
            self.w_bab * scores_df['bab'].fillna(0) +
            self.w_buyback * scores_df['buyback'].fillna(0) +
            self.w_asset_growth * scores_df['asset_growth'].fillna(0) +
            self.w_accrual * scores_df['accrual'].fillna(0) +
            self.w_gross_profit * scores_df['gross_profit'].fillna(0) +
            self.w_share_reduction * scores_df['share_reduction'].fillna(0) +
            self.w_intangible * scores_df['intangible'].fillna(0)
        )

        # Set score to NaN if we don't have at least 3 components (increased from 2)
        component_count = scores_df.notna().sum(axis=1)
        composite[component_count < 3] = np.nan

        return composite

    def get_diagnostics(
        self,
        raw_scores: pd.Series,
        normalized_scores: pd.Series
    ) -> Dict[str, Any]:
        """Add quality-specific diagnostics."""
        base_diag = super().get_diagnostics(raw_scores, normalized_scores)

        base_diag.update({
            'weights': {
                'cbop': self.w_cbop,
                'bab': self.w_bab,
                'buyback': self.w_buyback,
                'asset_growth': self.w_asset_growth,
                'accrual': self.w_accrual,
                'gross_profit': self.w_gross_profit,
                'share_reduction': self.w_share_reduction,
                'intangible': self.w_intangible,
            },
            'max_asset_growth_threshold': self.max_asset_growth,
        })

        return base_diag
