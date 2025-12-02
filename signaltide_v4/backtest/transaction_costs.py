"""
Transaction cost modeling for realistic backtesting.

Assumptions for $50K Schwab account:
- Zero commissions
- Tight spreads on liquid large caps (~2-5 bps)
- Default round-trip cost: ~5 bps
- Stress test at 10-20 bps for robustness
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TransactionCostResult:
    """Result of transaction cost calculation."""

    gross_value: float  # Value before costs
    net_value: float  # Value after costs
    total_cost: float  # Total cost in dollars
    cost_bps: float  # Cost in basis points
    turnover: float  # Portfolio turnover (0-1)
    trades: int  # Number of trades


class TransactionCostModel:
    """
    Models transaction costs for portfolio rebalancing.

    Supports:
    - Fixed percentage cost
    - Tiered costs by market cap
    - Slippage estimation
    """

    def __init__(
        self,
        cost_bps: float = 5.0,
        use_tiered: bool = False,
        tier_config: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize transaction cost model.

        Args:
            cost_bps: Base cost in basis points (default 5 bps for $50K Schwab)
            use_tiered: Whether to use tiered costs by market cap
            tier_config: Market cap tiers and their costs
        """
        self.base_cost_bps = cost_bps
        self.use_tiered = use_tiered

        # Default tier config (bps by market cap)
        self.tier_config = tier_config or {
            'mega': 2.0,   # >$200B - very liquid
            'large': 5.0,  # $10B-$200B
            'mid': 10.0,   # $2B-$10B
            'small': 20.0,  # <$2B - less liquid
        }

        logger.info(
            f"TransactionCostModel: {cost_bps:.1f} bps base, "
            f"tiered={use_tiered}"
        )

    def calculate_costs(
        self,
        trade_value: float,
        ticker: Optional[str] = None,
        market_cap: Optional[float] = None,
    ) -> float:
        """
        Calculate transaction cost for a trade.

        Args:
            trade_value: Absolute value of the trade
            ticker: Optional ticker for logging
            market_cap: Optional market cap for tiered costs

        Returns:
            Cost in dollars
        """
        if trade_value <= 0:
            return 0.0

        if self.use_tiered and market_cap is not None:
            cost_bps = self._get_tiered_cost(market_cap)
        else:
            cost_bps = self.base_cost_bps

        cost = trade_value * (cost_bps / 10000)

        logger.debug(
            f"Trade cost: ${trade_value:,.0f} @ {cost_bps:.1f} bps = ${cost:.2f}"
        )

        return cost

    def _get_tiered_cost(self, market_cap: float) -> float:
        """Get cost based on market cap tier."""
        if market_cap >= 200_000_000_000:  # $200B+
            return self.tier_config['mega']
        elif market_cap >= 10_000_000_000:  # $10B+
            return self.tier_config['large']
        elif market_cap >= 2_000_000_000:  # $2B+
            return self.tier_config['mid']
        else:
            return self.tier_config['small']

    def apply_to_rebalance(
        self,
        current_positions: Dict[str, float],
        target_positions: Dict[str, float],
        portfolio_value: float,
        prices: Optional[Dict[str, float]] = None,
        market_caps: Optional[Dict[str, float]] = None,
    ) -> TransactionCostResult:
        """
        Apply transaction costs to a portfolio rebalance.

        Args:
            current_positions: Ticker -> weight mapping (current)
            target_positions: Ticker -> weight mapping (target)
            portfolio_value: Total portfolio value
            prices: Optional ticker -> price mapping
            market_caps: Optional ticker -> market cap mapping

        Returns:
            TransactionCostResult with cost breakdown
        """
        total_cost = 0.0
        total_turnover = 0.0
        n_trades = 0

        all_tickers = set(current_positions.keys()) | set(target_positions.keys())

        for ticker in all_tickers:
            curr_weight = current_positions.get(ticker, 0.0)
            tgt_weight = target_positions.get(ticker, 0.0)

            # Weight change -> trade value
            weight_change = abs(tgt_weight - curr_weight)
            trade_value = weight_change * portfolio_value

            if trade_value > 0.01:  # Minimum trade threshold
                n_trades += 1
                total_turnover += weight_change

                # Get market cap for tiered cost
                mcap = market_caps.get(ticker) if market_caps else None
                cost = self.calculate_costs(trade_value, ticker, mcap)
                total_cost += cost

        # Turnover is counted as one-way (total of buys OR sells)
        turnover = total_turnover / 2

        gross_value = portfolio_value
        net_value = portfolio_value - total_cost
        cost_bps = (total_cost / portfolio_value) * 10000 if portfolio_value > 0 else 0

        result = TransactionCostResult(
            gross_value=gross_value,
            net_value=net_value,
            total_cost=total_cost,
            cost_bps=cost_bps,
            turnover=turnover,
            trades=n_trades,
        )

        logger.info(
            f"Rebalance cost: ${total_cost:,.2f} ({cost_bps:.1f} bps), "
            f"turnover={turnover:.1%}, trades={n_trades}"
        )

        return result

    def estimate_annual_drag(
        self,
        turnover_annual: float,
        portfolio_value: float,
    ) -> float:
        """
        Estimate annual cost drag from turnover.

        Args:
            turnover_annual: Expected annual turnover (e.g., 0.5 = 50%)
            portfolio_value: Portfolio value

        Returns:
            Expected annual cost in dollars
        """
        # Each turnover involves a round-trip cost
        total_traded = turnover_annual * portfolio_value * 2  # Buys + sells
        cost = total_traded * (self.base_cost_bps / 10000)

        return cost
