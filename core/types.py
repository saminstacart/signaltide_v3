"""
Common type definitions for SignalTide v3.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class SignalValue(float):
    """
    Signal value in range [-1, 1].

    -1 = strong sell
     0 = neutral
     1 = strong buy
    """

    def __new__(cls, value: float):
        if not -1 <= value <= 1:
            raise ValueError(f"Signal value must be in [-1, 1], got {value}")
        return super().__new__(cls, value)


class PositionSide(Enum):
    """Position side."""
    LONG = "long"
    SHORT = "short"


class OrderType(Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Position:
    """
    Represents a trading position.
    """
    symbol: str
    side: PositionSide
    size: float  # Number of units
    entry_price: float
    entry_time: datetime
    current_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    @property
    def value(self) -> float:
        """Current market value of position."""
        return self.size * self.current_price

    @property
    def cost_basis(self) -> float:
        """Original cost of position."""
        return self.size * self.entry_price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        if self.side == PositionSide.LONG:
            return (self.current_price - self.entry_price) * self.size
        else:  # SHORT
            return (self.entry_price - self.current_price) * self.size

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized profit/loss as percentage."""
        return self.unrealized_pnl / self.cost_basis

    def should_stop_loss(self) -> bool:
        """Check if stop loss should trigger."""
        if self.stop_loss is None:
            return False

        if self.side == PositionSide.LONG:
            return self.current_price <= self.stop_loss
        else:  # SHORT
            return self.current_price >= self.stop_loss

    def should_take_profit(self) -> bool:
        """Check if take profit should trigger."""
        if self.take_profit is None:
            return False

        if self.side == PositionSide.LONG:
            return self.current_price >= self.take_profit
        else:  # SHORT
            return self.current_price <= self.take_profit


@dataclass
class Trade:
    """
    Represents a completed trade.
    """
    symbol: str
    side: OrderSide
    size: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    slippage: float = 0.0

    @property
    def gross_value(self) -> float:
        """Gross value of trade."""
        return self.size * self.price

    @property
    def total_cost(self) -> float:
        """Total cost including commission and slippage."""
        return self.commission + self.slippage

    @property
    def net_value(self) -> float:
        """Net value after costs."""
        if self.side == OrderSide.BUY:
            return -(self.gross_value + self.total_cost)
        else:  # SELL
            return self.gross_value - self.total_cost


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""
    total_return: float
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    n_trades: int
    n_winning_trades: int
    n_losing_trades: int


@dataclass
class SignalResult:
    """
    Result from signal generation.
    """
    signal_values: 'pd.Series'  # Series of signal values
    confidence: Optional['pd.Series'] = None  # Optional confidence scores
    metadata: Optional[dict] = None  # Optional metadata
