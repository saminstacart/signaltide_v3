"""
Stabilized Portfolio Construction with Phase 4 fixes.

Key improvements over base construction:
1. Percentile-based entry/exit hysteresis (not rank-based)
2. Minimum holding period enforcement
3. Hard sector cap with redistribution
4. Signal smoothing via rolling average
5. Signal coverage validation
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

import pandas as pd
import numpy as np

from signaltide_v4.data.market_data import MarketDataProvider
from signaltide_v4.config.settings import get_settings
from signaltide_v4.portfolio.construction import Portfolio, PortfolioChange

logger = logging.getLogger(__name__)


@dataclass
class HoldingState:
    """Track holding information for hysteresis."""
    months_held: int
    entry_score: float
    entry_date: str


class StabilizedPortfolioConstructor:
    """
    Portfolio construction with stabilization features:

    1. Percentile-based entry/exit:
       - Enter only if score >= entry_percentile (top 10% by default)
       - Exit only if score < exit_percentile (below top 50%)

    2. Minimum holding period:
       - Don't sell positions held < min_holding_months

    3. Hard sector cap:
       - Cap any sector at hard_sector_cap (35%)
       - Redistribute excess to other sectors pro-rata

    4. Signal smoothing:
       - Maintain rolling average of scores over smoothing_window
    """

    def __init__(
        self,
        market_data: Optional[MarketDataProvider] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        settings = get_settings()
        params = params or {}

        self.market_data = market_data or MarketDataProvider()

        # Basic position limits
        self.max_position_pct = params.get('max_position_pct', settings.max_position_pct)
        self.target_positions = params.get('top_n_positions', settings.top_n_positions)

        # Inverse vol weighting
        self.use_inverse_vol = params.get(
            'use_inverse_vol',
            settings.use_inverse_vol_weighting
        )

        # Stabilization parameters
        self.entry_percentile = params.get('entry_percentile', settings.entry_percentile)
        self.exit_percentile = params.get('exit_percentile', settings.exit_percentile)
        self.min_holding_months = params.get('min_holding_months', settings.min_holding_months)
        self.hard_sector_cap = params.get('hard_sector_cap', settings.hard_sector_cap)
        self.smoothing_window = params.get('smoothing_window', settings.signal_smoothing_window)

        # State tracking
        self._holdings: Dict[str, HoldingState] = {}  # ticker -> holding state
        self._score_history: Dict[str, List[float]] = defaultdict(list)  # ticker -> score history

        logger.info(
            f"StabilizedPortfolioConstructor initialized:\n"
            f"  entry_percentile={self.entry_percentile}%, exit_percentile={self.exit_percentile}%\n"
            f"  min_holding_months={self.min_holding_months}, hard_sector_cap={self.hard_sector_cap:.0%}\n"
            f"  smoothing_window={self.smoothing_window}, inverse_vol={self.use_inverse_vol}"
        )

    def construct(
        self,
        scores: pd.Series,
        as_of_date: str,
        sectors: Optional[Dict[str, str]] = None,
    ) -> Portfolio:
        """
        Construct portfolio using stabilized approach.

        Args:
            scores: Series with ticker -> composite score
            as_of_date: Rebalance date
            sectors: Ticker -> sector mapping

        Returns:
            Portfolio with stabilized weights
        """
        if len(scores) == 0 or scores.dropna().empty:
            return Portfolio(
                positions={},
                as_of_date=as_of_date,
                method='stabilized_empty',
                diagnostics={'reason': 'No valid scores'},
            )

        # Step 1: Apply signal smoothing
        smoothed_scores = self._smooth_scores(scores)

        # Step 2: Calculate percentile thresholds
        valid_scores = smoothed_scores.dropna()
        if len(valid_scores) == 0:
            return Portfolio(
                positions={},
                as_of_date=as_of_date,
                method='stabilized_empty',
                diagnostics={'reason': 'No valid smoothed scores'},
            )

        entry_cutoff = np.percentile(valid_scores, 100 - self.entry_percentile)
        exit_cutoff = np.percentile(valid_scores, 100 - self.exit_percentile)

        # Step 3: Determine which positions to keep/add/remove
        new_holdings = {}
        kept_count = 0
        forced_hold_count = 0
        exit_count = 0
        entry_count = 0

        # Evaluate existing holdings
        for ticker, state in list(self._holdings.items()):
            if ticker not in smoothed_scores.index:
                # Ticker no longer in universe (delisted?)
                exit_count += 1
                continue

            score = smoothed_scores.get(ticker, np.nan)
            if pd.isna(score):
                exit_count += 1
                continue

            # Check minimum holding period
            if state.months_held < self.min_holding_months:
                # MUST hold regardless of score
                new_holdings[ticker] = HoldingState(
                    months_held=state.months_held + 1,
                    entry_score=state.entry_score,
                    entry_date=state.entry_date,
                )
                forced_hold_count += 1
                continue

            # Check exit threshold
            if score >= exit_cutoff:
                # Score still acceptable, HOLD
                new_holdings[ticker] = HoldingState(
                    months_held=state.months_held + 1,
                    entry_score=state.entry_score,
                    entry_date=state.entry_date,
                )
                kept_count += 1
            else:
                # Score dropped too far, EXIT
                exit_count += 1

        # Add new entries
        remaining_slots = self.target_positions - len(new_holdings)
        if remaining_slots > 0:
            # Get candidates above entry threshold, not already held
            candidates = valid_scores[
                (valid_scores >= entry_cutoff) &
                (~valid_scores.index.isin(new_holdings.keys()))
            ].sort_values(ascending=False)

            for ticker in candidates.head(remaining_slots).index:
                new_holdings[ticker] = HoldingState(
                    months_held=1,
                    entry_score=float(smoothed_scores[ticker]),
                    entry_date=as_of_date,
                )
                entry_count += 1

        # Update holdings state
        self._holdings = new_holdings

        # Step 4: Calculate weights
        tickers = list(new_holdings.keys())
        if self.use_inverse_vol:
            weights = self._inverse_vol_weights(tickers, as_of_date)
        else:
            weights = self._equal_weights(tickers)

        # Step 5: Apply position limits
        weights = self._apply_position_limits(weights)

        # Step 6: Apply HARD sector cap with redistribution
        if sectors:
            weights = self._apply_hard_sector_cap(weights, sectors)

        # Normalize to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        # Calculate sector concentration for diagnostics
        sector_weights = {}
        for ticker, weight in weights.items():
            sector = sectors.get(ticker, 'Unknown') if sectors else 'Unknown'
            sector_weights[sector] = sector_weights.get(sector, 0) + weight

        max_sector = max(sector_weights.values()) if sector_weights else 0
        max_sector_name = max(sector_weights, key=sector_weights.get) if sector_weights else 'N/A'

        diagnostics = {
            'method': 'stabilized',
            'n_positions': len(weights),
            'kept': kept_count,
            'forced_hold': forced_hold_count,
            'exits': exit_count,
            'entries': entry_count,
            'entry_cutoff': float(entry_cutoff),
            'exit_cutoff': float(exit_cutoff),
            'max_sector_weight': max_sector,
            'max_sector_name': max_sector_name,
            'sector_weights': sector_weights,
        }

        logger.debug(
            f"Stabilized construction: {len(weights)} positions "
            f"(kept={kept_count}, forced={forced_hold_count}, exits={exit_count}, entries={entry_count})"
        )

        return Portfolio(
            positions=weights,
            as_of_date=as_of_date,
            method='stabilized',
            total_weight=sum(weights.values()),
            diagnostics=diagnostics,
        )

    def _smooth_scores(self, scores: pd.Series) -> pd.Series:
        """Apply temporal smoothing using rolling average."""
        smoothed = {}

        for ticker, score in scores.items():
            if pd.isna(score):
                smoothed[ticker] = np.nan
                continue

            # Add to history
            history = self._score_history[ticker]
            history.append(float(score))

            # Keep only last N scores
            if len(history) > self.smoothing_window:
                self._score_history[ticker] = history[-self.smoothing_window:]
                history = self._score_history[ticker]

            # Calculate smoothed score (simple average)
            smoothed[ticker] = np.mean(history)

        return pd.Series(smoothed)

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
        """Calculate inverse volatility weights."""
        if not tickers:
            return {}

        volatility = self.market_data.get_volatility(tickers, as_of_date, lookback_days)

        if len(volatility.dropna()) == 0:
            logger.warning("No volatility data, falling back to equal weights")
            return self._equal_weights(tickers)

        inv_vol = 1.0 / volatility.replace(0, np.nan)
        median_inv_vol = inv_vol.median()

        for ticker in tickers:
            if ticker not in inv_vol or pd.isna(inv_vol.get(ticker)):
                inv_vol[ticker] = median_inv_vol

        total = inv_vol.sum()
        if total == 0:
            return self._equal_weights(tickers)

        return {ticker: float(inv_vol.get(ticker, 0) / total) for ticker in tickers}

    def _apply_position_limits(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply maximum position size limit."""
        if not weights:
            return weights

        adjusted = {}
        excess = 0.0

        for ticker, weight in weights.items():
            if weight > self.max_position_pct:
                excess += weight - self.max_position_pct
                adjusted[ticker] = self.max_position_pct
            else:
                adjusted[ticker] = weight

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

    def _apply_hard_sector_cap(
        self,
        weights: Dict[str, float],
        sectors: Dict[str, str],
    ) -> Dict[str, float]:
        """
        Apply HARD sector cap with pro-rata redistribution.

        Unlike proportional scaling, this:
        1. Hard caps the sector at the limit
        2. Redistributes excess weight to other sectors
        """
        if not weights:
            return weights

        max_iterations = 10  # Prevent infinite loops

        for iteration in range(max_iterations):
            # Calculate sector totals
            sector_totals = {}
            sector_tickers = defaultdict(list)

            for ticker, weight in weights.items():
                sector = sectors.get(ticker, 'Unknown')
                sector_totals[sector] = sector_totals.get(sector, 0) + weight
                sector_tickers[sector].append(ticker)

            # Find violations
            violations = {
                s: total - self.hard_sector_cap
                for s, total in sector_totals.items()
                if total > self.hard_sector_cap + 0.001  # Small epsilon
            }

            if not violations:
                break  # No violations, done

            # Process each violation
            total_excess = 0
            for sector, excess in violations.items():
                total_excess += excess

                # Scale down all tickers in this sector proportionally
                scale = self.hard_sector_cap / sector_totals[sector]
                for ticker in sector_tickers[sector]:
                    weights[ticker] *= scale

                logger.debug(f"Sector {sector} capped: {sector_totals[sector]:.1%} -> {self.hard_sector_cap:.1%}")

            # Redistribute excess to uncapped sectors
            uncapped_tickers = [
                t for t, w in weights.items()
                if sectors.get(t, 'Unknown') not in violations
            ]

            if uncapped_tickers:
                # Distribute proportionally to current weights
                uncapped_total = sum(weights[t] for t in uncapped_tickers)
                if uncapped_total > 0:
                    for ticker in uncapped_tickers:
                        share = weights[ticker] / uncapped_total
                        weights[ticker] += total_excess * share

        return weights

    def calculate_turnover(
        self,
        previous_weights: Dict[str, float],
        current_weights: Dict[str, float],
    ) -> float:
        """Calculate one-way turnover between portfolios."""
        all_tickers = set(previous_weights.keys()) | set(current_weights.keys())

        total_change = 0.0
        for ticker in all_tickers:
            prev = previous_weights.get(ticker, 0)
            curr = current_weights.get(ticker, 0)
            total_change += abs(curr - prev)

        # One-way turnover is half of total change
        return total_change / 2

    def get_holdings_summary(self) -> Dict[str, Any]:
        """Get summary of current holdings state."""
        if not self._holdings:
            return {'count': 0, 'avg_holding_months': 0}

        months = [h.months_held for h in self._holdings.values()]
        return {
            'count': len(self._holdings),
            'avg_holding_months': np.mean(months),
            'min_holding_months': min(months),
            'max_holding_months': max(months),
        }

    def reset_state(self):
        """Reset all state tracking (for new backtests)."""
        self._holdings = {}
        self._score_history = defaultdict(list)
        logger.info("StabilizedPortfolioConstructor state reset")
