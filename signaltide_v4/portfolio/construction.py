"""
Portfolio construction with inverse volatility weighting and hysteresis.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from signaltide_v4.data.market_data import MarketDataProvider
from signaltide_v4.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class Portfolio:
    """Container for portfolio allocation."""

    positions: Dict[str, float]  # Ticker -> weight
    as_of_date: str
    method: str
    total_weight: float = 1.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        df = pd.DataFrame([
            {'ticker': ticker, 'weight': weight}
            for ticker, weight in self.positions.items()
        ])
        return df.sort_values('weight', ascending=False)


@dataclass
class PortfolioChange:
    """Container for portfolio rebalancing changes."""

    buys: Dict[str, float]  # Ticker -> target weight
    sells: Dict[str, float]  # Ticker -> current weight being sold
    holds: Dict[str, float]  # Ticker -> weight unchanged
    turnover: float  # Total portfolio turnover (0-1)


class PortfolioConstructor:
    """
    Constructs portfolio with:
    - Inverse volatility weighting
    - Hysteresis to reduce turnover
    - Position and sector limits
    """

    def __init__(
        self,
        market_data: Optional[MarketDataProvider] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize portfolio constructor.

        Args:
            market_data: MarketDataProvider for volatility calculations
            params: Optional parameter overrides
        """
        settings = get_settings()
        params = params or {}

        self.market_data = market_data or MarketDataProvider()

        # Position limits
        self.max_position_pct = params.get('max_position_pct', settings.max_position_pct)
        self.max_sector_pct = params.get('max_sector_pct', settings.max_sector_pct)

        # Weighting method
        self.use_inverse_vol = params.get(
            'use_inverse_vol',
            settings.use_inverse_vol_weighting
        )

        # Hysteresis for turnover reduction
        self.hysteresis_threshold = params.get(
            'hysteresis_threshold',
            settings.hysteresis_threshold
        )

        logger.info(
            f"PortfolioConstructor: inverse_vol={self.use_inverse_vol}, "
            f"hysteresis={self.hysteresis_threshold:.1%}"
        )

    def construct(
        self,
        selected_tickers: pd.Series,
        as_of_date: str,
        previous_portfolio: Optional[Portfolio] = None,
        sectors: Optional[Dict[str, str]] = None,
    ) -> Portfolio:
        """
        Construct portfolio from selected tickers.

        Args:
            selected_tickers: Series with ticker -> score
            as_of_date: Rebalance date
            previous_portfolio: Previous portfolio for hysteresis
            sectors: Ticker -> sector mapping

        Returns:
            Portfolio with final weights
        """
        if len(selected_tickers) == 0:
            return Portfolio(
                positions={},
                as_of_date=as_of_date,
                method='empty',
                diagnostics={'reason': 'No tickers selected'},
            )

        tickers = list(selected_tickers.index)

        # Apply hysteresis if we have previous portfolio
        if previous_portfolio is not None:
            tickers = self._apply_hysteresis(
                tickers,
                selected_tickers,
                previous_portfolio,
            )

        # Calculate weights
        if self.use_inverse_vol:
            weights = self._inverse_vol_weights(tickers, as_of_date)
        else:
            weights = self._equal_weights(tickers)

        # Apply position limits
        weights = self._apply_position_limits(weights)

        # Apply sector limits if provided
        if sectors:
            weights = self._apply_sector_limits(weights, sectors)

        # Normalize to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        # Build diagnostics
        diagnostics = {
            'method': 'inverse_vol' if self.use_inverse_vol else 'equal_weight',
            'n_positions': len(weights),
            'max_weight': max(weights.values()) if weights else 0,
            'min_weight': min(weights.values()) if weights else 0,
            'herfindahl': sum(w**2 for w in weights.values()) if weights else 0,
        }

        return Portfolio(
            positions=weights,
            as_of_date=as_of_date,
            method='inverse_vol' if self.use_inverse_vol else 'equal_weight',
            total_weight=sum(weights.values()),
            diagnostics=diagnostics,
        )

    def _equal_weights(self, tickers: List[str]) -> Dict[str, float]:
        """Calculate equal weights."""
        n = len(tickers)
        if n == 0:
            return {}
        weight = 1.0 / n
        return {ticker: weight for ticker in tickers}

    def _inverse_vol_weights(
        self,
        tickers: List[str],
        as_of_date: str,
        lookback_days: int = 60,
    ) -> Dict[str, float]:
        """
        Calculate inverse volatility weights.

        Lower volatility stocks get higher weight.
        """
        if not tickers:
            return {}

        # Get volatility for each ticker
        volatility = self.market_data.get_volatility(tickers, as_of_date, lookback_days)

        if len(volatility.dropna()) == 0:
            logger.warning("No volatility data, falling back to equal weights")
            return self._equal_weights(tickers)

        # Inverse volatility
        inv_vol = 1.0 / volatility.replace(0, np.nan)

        # Handle missing volatility with median
        median_inv_vol = inv_vol.median()
        for ticker in tickers:
            if ticker not in inv_vol or pd.isna(inv_vol.get(ticker)):
                inv_vol[ticker] = median_inv_vol

        # Normalize
        total = inv_vol.sum()
        if total == 0:
            return self._equal_weights(tickers)

        weights = {ticker: float(inv_vol.get(ticker, 0) / total) for ticker in tickers}

        return weights

    def _apply_position_limits(
        self,
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """Apply maximum position size limit."""
        if not weights:
            return weights

        # Cap any position above max
        adjusted = {}
        excess = 0.0

        for ticker, weight in weights.items():
            if weight > self.max_position_pct:
                excess += weight - self.max_position_pct
                adjusted[ticker] = self.max_position_pct
            else:
                adjusted[ticker] = weight

        # Redistribute excess proportionally to uncapped positions
        if excess > 0:
            uncapped = [t for t, w in adjusted.items() if w < self.max_position_pct]
            if uncapped:
                per_ticker = excess / len(uncapped)
                for ticker in uncapped:
                    adjusted[ticker] = min(
                        adjusted[ticker] + per_ticker,
                        self.max_position_pct
                    )

        return adjusted

    def _apply_sector_limits(
        self,
        weights: Dict[str, float],
        sectors: Dict[str, str],
    ) -> Dict[str, float]:
        """Apply sector concentration limits."""
        if not weights:
            return weights

        # Calculate sector weights
        sector_weights = {}
        for ticker, weight in weights.items():
            sector = sectors.get(ticker, 'Unknown')
            sector_weights[sector] = sector_weights.get(sector, 0) + weight

        # Check for violations
        adjusted = weights.copy()
        for sector, total_weight in sector_weights.items():
            if total_weight > self.max_sector_pct:
                # Reduce all positions in this sector proportionally
                scale = self.max_sector_pct / total_weight
                for ticker, weight in adjusted.items():
                    if sectors.get(ticker, 'Unknown') == sector:
                        adjusted[ticker] = weight * scale

        return adjusted

    def _apply_hysteresis(
        self,
        new_tickers: List[str],
        scores: pd.Series,
        previous: Portfolio,
    ) -> List[str]:
        """
        Apply hysteresis to reduce turnover.

        Keep current positions unless score rank drops significantly.
        """
        prev_tickers = set(previous.positions.keys())
        new_tickers_set = set(new_tickers)

        # Get ranks for new selection
        new_ranks = scores.rank(ascending=False)

        # Check each current position
        keep = []
        for ticker in prev_tickers:
            if ticker in new_tickers_set:
                # Already in new selection, keep it
                keep.append(ticker)
            elif ticker in new_ranks:
                # Check if rank drop is within hysteresis
                n = len(scores)
                rank = new_ranks.get(ticker, n)
                threshold_rank = len(new_tickers) * (1 + self.hysteresis_threshold)

                if rank <= threshold_rank:
                    # Within hysteresis, keep the position
                    keep.append(ticker)
                    logger.debug(f"Hysteresis: keeping {ticker} (rank {rank:.0f})")

        # Add new tickers that aren't in keep
        for ticker in new_tickers:
            if ticker not in keep:
                keep.append(ticker)

        # Limit to original selection size
        return keep[:len(new_tickers)]

    def calculate_rebalance(
        self,
        current: Portfolio,
        target: Portfolio,
    ) -> PortfolioChange:
        """
        Calculate changes needed to rebalance from current to target.

        Returns:
            PortfolioChange with buys, sells, holds, and turnover
        """
        current_positions = current.positions
        target_positions = target.positions

        buys = {}
        sells = {}
        holds = {}

        all_tickers = set(current_positions.keys()) | set(target_positions.keys())

        total_trade = 0.0

        for ticker in all_tickers:
            curr_weight = current_positions.get(ticker, 0)
            tgt_weight = target_positions.get(ticker, 0)

            diff = tgt_weight - curr_weight

            if diff > 0.001:  # Buy
                buys[ticker] = tgt_weight
                total_trade += diff
            elif diff < -0.001:  # Sell
                sells[ticker] = curr_weight
                total_trade += abs(diff)
            else:  # Hold
                if tgt_weight > 0:
                    holds[ticker] = tgt_weight

        # Turnover is half of total trade (buy + sell counted separately)
        turnover = total_trade / 2

        return PortfolioChange(
            buys=buys,
            sells=sells,
            holds=holds,
            turnover=turnover,
        )
