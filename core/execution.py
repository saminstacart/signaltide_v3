"""
Execution Models for Transaction Costs

Implements realistic transaction cost modeling including:
- Commissions (fixed + percentage)
- Slippage (market impact)
- Spread costs
- Market impact for large orders

References:
- Almgren & Chriss (2000) "Optimal Execution of Portfolio Transactions"
- Grinold & Kahn (2000) "Active Portfolio Management"
"""

from typing import Optional
import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DEFAULT_TRANSACTION_COSTS, get_logger

logger = get_logger(__name__)


class TransactionCostModel:
    """
    Transaction cost model for realistic backtesting.

    Components:
    1. Commission: Fixed + percentage of trade value
    2. Slippage: Market impact based on trade size
    3. Spread: Bid-ask spread cost
    """

    def __init__(self,
                 commission_pct: Optional[float] = None,
                 slippage_pct: Optional[float] = None,
                 spread_pct: Optional[float] = None,
                 min_commission: float = 1.0):
        """
        Initialize transaction cost model.

        Args:
            commission_pct: Commission as percentage (default: from config)
            slippage_pct: Slippage as percentage (default: from config)
            spread_pct: Spread as percentage (default: from config)
            min_commission: Minimum commission per trade in dollars
        """
        # Use config defaults if not specified
        if commission_pct is None:
            commission_pct = DEFAULT_TRANSACTION_COSTS['commission_pct']
        if slippage_pct is None:
            slippage_pct = DEFAULT_TRANSACTION_COSTS['slippage_pct']
        if spread_pct is None:
            spread_pct = DEFAULT_TRANSACTION_COSTS['spread_pct']

        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.spread_pct = spread_pct
        self.min_commission = min_commission

        logger.info(
            f"TransactionCostModel initialized: "
            f"commission={commission_pct*100:.2f}bps, "
            f"slippage={slippage_pct*100:.2f}bps, "
            f"spread={spread_pct*100:.2f}bps"
        )

    def calculate_costs(self,
                       trade_value: float,
                       is_buy: bool = True) -> dict:
        """
        Calculate all transaction costs for a trade.

        Args:
            trade_value: Dollar value of trade (absolute value)
            is_buy: True if buying, False if selling

        Returns:
            Dict with cost breakdown:
                - commission: Commission cost
                - slippage: Slippage cost
                - spread: Spread cost
                - total: Total cost
                - total_pct: Total cost as percentage
        """
        trade_value = abs(trade_value)

        # Commission
        commission = max(
            trade_value * self.commission_pct,
            self.min_commission
        )

        # Slippage (market impact)
        slippage = trade_value * self.slippage_pct

        # Spread cost
        spread = trade_value * self.spread_pct

        # Total
        total = commission + slippage + spread
        total_pct = total / trade_value if trade_value > 0 else 0

        return {
            'commission': commission,
            'slippage': slippage,
            'spread': spread,
            'total': total,
            'total_pct': total_pct,
            'trade_value': trade_value
        }

    def apply_costs_to_returns(self,
                               positions: pd.Series,
                               prices: pd.Series,
                               returns: pd.Series) -> pd.Series:
        """
        Apply transaction costs to strategy returns.

        Args:
            positions: Position sizes over time
            prices: Prices over time
            returns: Raw returns before costs

        Returns:
            Returns after transaction costs
        """
        # Calculate position changes (trades)
        position_changes = positions.diff().fillna(positions)

        # Calculate trade values
        trade_values = abs(position_changes * prices)

        # Calculate costs as fraction of portfolio
        costs = pd.Series(0.0, index=positions.index)
        for date, trade_value in trade_values.items():
            if trade_value > 0:
                cost_info = self.calculate_costs(trade_value)
                # Cost as negative return
                costs[date] = -cost_info['total'] / trade_value

        # Apply costs to returns
        returns_after_costs = returns + costs

        logger.debug(
            f"Applied transaction costs: "
            f"{(costs != 0).sum()} trades, "
            f"avg cost={(costs[costs != 0].mean()*10000):.1f}bps"
        )

        return returns_after_costs

    def estimate_turnover_cost(self,
                              monthly_turnover: float,
                              portfolio_value: float) -> float:
        """
        Estimate annual cost from turnover.

        Args:
            monthly_turnover: Average monthly turnover (0.5 = 50%)
            portfolio_value: Portfolio value in dollars

        Returns:
            Estimated annual cost in dollars
        """
        # Annual turnover
        annual_turnover = monthly_turnover * 12

        # Trade value per year
        annual_trade_value = annual_turnover * portfolio_value

        # Cost per trade
        avg_cost_pct = self.commission_pct + self.slippage_pct + self.spread_pct

        # Annual cost
        annual_cost = annual_trade_value * avg_cost_pct

        logger.info(
            f"Turnover cost estimate: "
            f"{monthly_turnover*100:.1f}% monthly turnover = "
            f"${annual_cost:,.0f}/year ({annual_cost/portfolio_value*100:.2f}% of portfolio)"
        )

        return annual_cost

    def __repr__(self) -> str:
        """String representation."""
        total_bps = (self.commission_pct + self.slippage_pct + self.spread_pct) * 10000
        return (
            f"TransactionCostModel("
            f"total={total_bps:.1f}bps, "
            f"commission={self.commission_pct*10000:.1f}bps, "
            f"slippage={self.slippage_pct*10000:.1f}bps, "
            f"spread={self.spread_pct*10000:.1f}bps)"
        )


class MarketImpactModel:
    """
    Advanced market impact model for large orders.

    Based on square-root market impact model:
    Impact ∝ sqrt(trade_size / average_volume)

    Reference: Almgren & Chriss (2000)
    """

    def __init__(self,
                 temporary_impact_coef: float = 0.01,
                 permanent_impact_coef: float = 0.005):
        """
        Initialize market impact model.

        Args:
            temporary_impact_coef: Coefficient for temporary impact
            permanent_impact_coef: Coefficient for permanent impact
        """
        self.temporary_coef = temporary_impact_coef
        self.permanent_coef = permanent_impact_coef

        logger.info(
            f"MarketImpactModel initialized: "
            f"temporary={temporary_impact_coef}, "
            f"permanent={permanent_impact_coef}"
        )

    def calculate_impact(self,
                        trade_shares: float,
                        avg_daily_volume: float,
                        price: float) -> dict:
        """
        Calculate market impact for a trade.

        Args:
            trade_shares: Number of shares to trade
            avg_daily_volume: Average daily volume
            price: Current price

        Returns:
            Dict with:
                - temporary_impact: Temporary price impact (%)
                - permanent_impact: Permanent price impact (%)
                - total_cost: Total impact cost in dollars
        """
        # Participation rate
        participation = abs(trade_shares) / avg_daily_volume if avg_daily_volume > 0 else 0

        # Square-root impact
        temporary_impact = self.temporary_coef * np.sqrt(participation)
        permanent_impact = self.permanent_coef * np.sqrt(participation)

        # Dollar cost
        trade_value = abs(trade_shares * price)
        total_cost = trade_value * (temporary_impact + permanent_impact)

        return {
            'temporary_impact_pct': temporary_impact,
            'permanent_impact_pct': permanent_impact,
            'total_impact_pct': temporary_impact + permanent_impact,
            'participation_rate': participation,
            'total_cost': total_cost,
            'trade_value': trade_value
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MarketImpactModel("
            f"temporary={self.temporary_coef}, "
            f"permanent={self.permanent_coef})"
        )


# Default global instance
default_cost_model = TransactionCostModel()


if __name__ == '__main__':
    # Test transaction cost model
    print("=" * 80)
    print("Transaction Cost Model Tests")
    print("=" * 80)

    model = TransactionCostModel()
    print(f"\n{model}\n")

    # Test trade costs
    test_trades = [1000, 10000, 100000, 1000000]
    print("Trade Cost Analysis:")
    print(f"{'Trade Value':>12} | {'Commission':>10} | {'Slippage':>10} | {'Spread':>10} | {'Total':>10} | {'Total %':>8}")
    print("-" * 80)

    for trade_value in test_trades:
        costs = model.calculate_costs(trade_value)
        print(
            f"${trade_value:>11,.0f} | "
            f"${costs['commission']:>9,.2f} | "
            f"${costs['slippage']:>9,.2f} | "
            f"${costs['spread']:>9,.2f} | "
            f"${costs['total']:>9,.2f} | "
            f"{costs['total_pct']*100:>7.3f}%"
        )

    # Test turnover cost
    print("\nTurnover Cost Estimate:")
    portfolio_value = 50000
    monthly_turnover = 0.05  # 5% per month
    annual_cost = model.estimate_turnover_cost(monthly_turnover, portfolio_value)

    # Test market impact
    print("\n" + "=" * 80)
    print("Market Impact Model Tests")
    print("=" * 80)

    impact_model = MarketImpactModel()
    print(f"\n{impact_model}\n")

    # Test impact for different trade sizes
    avg_volume = 1000000  # 1M shares average daily volume
    price = 100
    test_sizes = [1000, 10000, 50000, 100000]

    print("Market Impact Analysis (ADV = 1M shares, Price = $100):")
    print(f"{'Shares':>10} | {'Participation':>13} | {'Temp Impact':>12} | {'Perm Impact':>12} | {'Total Cost':>12}")
    print("-" * 80)

    for shares in test_sizes:
        impact = impact_model.calculate_impact(shares, avg_volume, price)
        print(
            f"{shares:>10,} | "
            f"{impact['participation_rate']*100:>12.3f}% | "
            f"{impact['temporary_impact_pct']*100:>11.3f}% | "
            f"{impact['permanent_impact_pct']*100:>11.3f}% | "
            f"${impact['total_cost']:>11,.2f}"
        )

    print("\n✅ Transaction cost models working correctly")
