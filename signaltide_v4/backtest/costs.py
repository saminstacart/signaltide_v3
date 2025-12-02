"""
Enhanced transaction cost modeling with market microstructure.

References:
    - Almgren & Chriss (2000): "Optimal Execution of Portfolio Transactions"
    - Kissell & Glantz (2003): "Optimal Trading Strategies"

This module extends basic cost modeling with:
- ADV-based position sizing
- Market impact estimation
- Spread modeling by liquidity tier
"""

import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class LiquidityTier(Enum):
    """Liquidity classification for stocks."""
    MEGA_CAP = "mega_cap"      # >$200B, very liquid
    LARGE_CAP = "large_cap"    # $10B-$200B
    MID_CAP = "mid_cap"        # $2B-$10B
    SMALL_CAP = "small_cap"    # <$2B, less liquid


@dataclass
class CostBreakdown:
    """Detailed breakdown of transaction costs."""

    spread_cost: float      # Half-spread cost
    market_impact: float    # Price impact from trading
    commission: float       # Broker commission (typically 0 for Schwab)
    slippage: float        # Execution slippage
    total_cost: float      # Sum of all costs
    cost_bps: float        # Total in basis points

    def __repr__(self) -> str:
        return (
            f"CostBreakdown(spread={self.spread_cost:.2f}, "
            f"impact={self.market_impact:.2f}, "
            f"total={self.total_cost:.2f}, "
            f"{self.cost_bps:.1f}bps)"
        )


@dataclass
class ADVConstraints:
    """ADV-based position sizing constraints."""

    max_pct_adv: float = 0.01  # Max 1% of ADV per day
    max_days_to_liquidate: int = 5  # Max 5 days to exit
    min_adv_usd: float = 100_000  # Minimum ADV for trading


class EnhancedCostModel:
    """
    Comprehensive transaction cost model.

    Incorporates:
    1. Spread costs (liquidity-tier dependent)
    2. Market impact (Almgren-Chriss square-root model)
    3. ADV-based position sizing
    4. Slippage estimation

    Default assumptions for $50K Schwab account:
    - Zero commissions
    - Liquid large-cap focused
    - ~5 bps round-trip cost
    """

    # Spread estimates by tier (half-spread in bps)
    SPREAD_BY_TIER = {
        LiquidityTier.MEGA_CAP: 1.0,    # ~2 bps round-trip
        LiquidityTier.LARGE_CAP: 2.5,   # ~5 bps round-trip
        LiquidityTier.MID_CAP: 5.0,     # ~10 bps round-trip
        LiquidityTier.SMALL_CAP: 10.0,  # ~20 bps round-trip
    }

    # Market impact coefficients (empirical)
    IMPACT_COEFFICIENT = 0.1  # Almgren-Chriss eta parameter

    def __init__(
        self,
        base_cost_bps: float = 5.0,
        use_market_impact: bool = True,
        use_tiered_spread: bool = True,
        commission_per_trade: float = 0.0,  # Zero for Schwab
        adv_constraints: Optional[ADVConstraints] = None,
    ):
        """
        Initialize enhanced cost model.

        Args:
            base_cost_bps: Base cost assumption (fallback)
            use_market_impact: Whether to model price impact
            use_tiered_spread: Whether to use liquidity-based spreads
            commission_per_trade: Fixed commission per trade
            adv_constraints: ADV-based position constraints
        """
        self.base_cost_bps = base_cost_bps
        self.use_market_impact = use_market_impact
        self.use_tiered_spread = use_tiered_spread
        self.commission_per_trade = commission_per_trade
        self.adv_constraints = adv_constraints or ADVConstraints()

        logger.info(
            f"EnhancedCostModel: base={base_cost_bps}bps, "
            f"impact={use_market_impact}, tiered={use_tiered_spread}"
        )

    def get_liquidity_tier(self, market_cap: float) -> LiquidityTier:
        """Classify stock by market cap tier."""
        if market_cap >= 200_000_000_000:
            return LiquidityTier.MEGA_CAP
        elif market_cap >= 10_000_000_000:
            return LiquidityTier.LARGE_CAP
        elif market_cap >= 2_000_000_000:
            return LiquidityTier.MID_CAP
        else:
            return LiquidityTier.SMALL_CAP

    def estimate_spread_cost(
        self,
        trade_value: float,
        market_cap: Optional[float] = None,
    ) -> float:
        """
        Estimate half-spread cost.

        Args:
            trade_value: Dollar value of trade
            market_cap: Market cap for tier classification

        Returns:
            Spread cost in dollars
        """
        if not self.use_tiered_spread or market_cap is None:
            # Use base cost assumption
            return trade_value * (self.base_cost_bps / 2 / 10000)

        tier = self.get_liquidity_tier(market_cap)
        spread_bps = self.SPREAD_BY_TIER[tier]

        return trade_value * (spread_bps / 10000)

    def estimate_market_impact(
        self,
        trade_value: float,
        adv: Optional[float] = None,
        volatility: Optional[float] = None,
    ) -> float:
        """
        Estimate market impact using Almgren-Chriss model.

        Impact = eta * sigma * sqrt(trade_value / ADV)

        Args:
            trade_value: Dollar value of trade
            adv: Average daily volume in dollars
            volatility: Daily volatility (decimal)

        Returns:
            Market impact cost in dollars
        """
        if not self.use_market_impact or adv is None:
            return 0.0

        # Assume 2% daily vol if not provided
        sigma = volatility or 0.02

        # Participation rate
        participation = trade_value / adv if adv > 0 else 0.1

        # Square-root impact model
        impact_pct = self.IMPACT_COEFFICIENT * sigma * np.sqrt(participation)

        return trade_value * impact_pct

    def calculate_trade_cost(
        self,
        trade_value: float,
        ticker: Optional[str] = None,
        market_cap: Optional[float] = None,
        adv: Optional[float] = None,
        volatility: Optional[float] = None,
    ) -> CostBreakdown:
        """
        Calculate comprehensive trade cost.

        Args:
            trade_value: Absolute dollar value of trade
            ticker: Stock ticker (for logging)
            market_cap: Market capitalization
            adv: Average daily volume in dollars
            volatility: Daily volatility

        Returns:
            CostBreakdown with component costs
        """
        if trade_value <= 0:
            return CostBreakdown(
                spread_cost=0.0,
                market_impact=0.0,
                commission=0.0,
                slippage=0.0,
                total_cost=0.0,
                cost_bps=0.0,
            )

        # Component costs
        spread = self.estimate_spread_cost(trade_value, market_cap)
        impact = self.estimate_market_impact(trade_value, adv, volatility)
        commission = self.commission_per_trade

        # Slippage: small random component (~1 bp)
        slippage = trade_value * 0.0001

        total = spread + impact + commission + slippage
        cost_bps = (total / trade_value) * 10000 if trade_value > 0 else 0

        breakdown = CostBreakdown(
            spread_cost=spread,
            market_impact=impact,
            commission=commission,
            slippage=slippage,
            total_cost=total,
            cost_bps=cost_bps,
        )

        logger.debug(f"Trade cost for {ticker}: {breakdown}")

        return breakdown

    def check_adv_constraint(
        self,
        position_value: float,
        adv: float,
    ) -> Tuple[bool, float]:
        """
        Check if position passes ADV constraints.

        Args:
            position_value: Target position value
            adv: Average daily volume

        Returns:
            (passes_constraint, max_allowed_value)
        """
        if adv < self.adv_constraints.min_adv_usd:
            return False, 0.0

        # Max position based on daily participation
        max_daily = adv * self.adv_constraints.max_pct_adv
        max_total = max_daily * self.adv_constraints.max_days_to_liquidate

        passes = position_value <= max_total

        return passes, max_total

    def apply_to_rebalance(
        self,
        current_positions: Dict[str, float],
        target_positions: Dict[str, float],
        portfolio_value: float,
        market_data: Optional[Dict[str, Dict]] = None,
    ) -> Tuple[float, Dict[str, CostBreakdown]]:
        """
        Apply costs to full rebalance.

        Args:
            current_positions: Ticker -> weight (current)
            target_positions: Ticker -> weight (target)
            portfolio_value: Total portfolio value
            market_data: Dict with market_cap, adv, volatility per ticker

        Returns:
            (total_cost, per_ticker_breakdown)
        """
        market_data = market_data or {}

        all_tickers = set(current_positions.keys()) | set(target_positions.keys())

        total_cost = 0.0
        breakdowns = {}

        for ticker in all_tickers:
            curr_weight = current_positions.get(ticker, 0.0)
            tgt_weight = target_positions.get(ticker, 0.0)

            weight_change = abs(tgt_weight - curr_weight)
            trade_value = weight_change * portfolio_value

            if trade_value < 1.0:  # Skip trivial trades
                continue

            # Get market data for ticker
            ticker_data = market_data.get(ticker, {})

            breakdown = self.calculate_trade_cost(
                trade_value=trade_value,
                ticker=ticker,
                market_cap=ticker_data.get('market_cap'),
                adv=ticker_data.get('adv'),
                volatility=ticker_data.get('volatility'),
            )

            breakdowns[ticker] = breakdown
            total_cost += breakdown.total_cost

        logger.info(
            f"Rebalance total cost: ${total_cost:.2f} "
            f"({total_cost/portfolio_value*10000:.1f}bps), "
            f"{len(breakdowns)} trades"
        )

        return total_cost, breakdowns

    def estimate_annual_drag(
        self,
        annual_turnover: float,
        portfolio_value: float,
        avg_market_cap: float = 100_000_000_000,
    ) -> float:
        """
        Estimate annual cost drag from expected turnover.

        Args:
            annual_turnover: Expected annual turnover (e.g., 0.5 = 50%)
            portfolio_value: Portfolio value
            avg_market_cap: Average market cap of holdings

        Returns:
            Expected annual cost in dollars
        """
        # Total traded value (turnover = one-way)
        total_traded = annual_turnover * portfolio_value * 2  # Buy + sell

        # Estimate average cost
        tier = self.get_liquidity_tier(avg_market_cap)
        spread_bps = self.SPREAD_BY_TIER[tier]

        # Spread + small impact
        avg_cost_bps = spread_bps * 2 + 1  # Round-trip spread + slippage

        annual_cost = total_traded * (avg_cost_bps / 10000)

        logger.info(
            f"Annual cost drag: ${annual_cost:.2f} "
            f"({annual_cost/portfolio_value*100:.2f}%) "
            f"at {annual_turnover:.0%} turnover"
        )

        return annual_cost
