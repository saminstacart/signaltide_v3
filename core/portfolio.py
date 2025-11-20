"""
Portfolio management for SignalTide v3.

Handles position sizing, risk management, and execution.
"""

from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.types import Position, Trade, PositionSide, OrderSide, PortfolioMetrics
from core.execution import TransactionCostModel
from config import get_logger, DEFAULT_TRANSACTION_COSTS, DEFAULT_RISK_PARAMS

logger = get_logger(__name__)


class Portfolio:
    """
    Manages portfolio state, positions, and trade execution.

    Responsibilities:
    - Position sizing based on signals and risk parameters
    - Order execution with realistic costs
    - Risk management (stops, position limits, drawdown monitoring)
    - Performance tracking
    """

    def __init__(self, initial_capital: float, params: Optional[Dict] = None):
        """
        Initialize portfolio.

        Args:
            initial_capital: Starting capital in USD
            params: Portfolio parameters (position sizing, risk management, etc.)
                   If None, uses defaults from config
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital

        # Use defaults from config if not provided
        if params is None:
            params = {}

        # Merge with defaults
        self.params = {**DEFAULT_RISK_PARAMS, **DEFAULT_TRANSACTION_COSTS, **params}

        # Transaction cost model
        self.cost_model = TransactionCostModel(
            commission_pct=self.params.get('commission_pct'),
            slippage_pct=self.params.get('slippage_pct'),
            spread_pct=self.params.get('spread_pct')
        )

        # Positions and trades
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []

        # Performance tracking
        self.equity_curve: List[Dict] = []
        self.peak_equity = initial_capital
        self.current_drawdown = 0.0

        logger.info(f"Portfolio initialized: ${initial_capital:,.2f} capital")
        logger.debug(f"Portfolio params: {self.params}")

    def get_equity(self) -> float:
        """Calculate current total equity (cash + positions)."""
        position_value = sum(pos.value for pos in self.positions.values())
        return self.capital + position_value

    def get_available_capital(self) -> float:
        """Get capital available for new positions."""
        # Reserved for existing positions
        reserved = sum(pos.cost_basis for pos in self.positions.values())
        return self.initial_capital - reserved

    def calculate_position_size(self, symbol: str, signal: float,
                                 price: float) -> float:
        """
        Calculate position size based on signal strength and risk parameters.

        Args:
            symbol: Trading symbol
            signal: Signal value in [-1, 1]
            price: Current price

        Returns:
            Position size in units (0 if no position should be taken)
        """
        if abs(signal) < 1e-6:  # Effectively zero
            return 0.0

        method = self.params.get('position_sizing_method', 'equal_weight')
        max_position_size = self.params.get('max_position_size', 0.20)
        max_positions = self.params.get('max_positions', 5)

        # Calculate base position size
        available = self.get_available_capital()
        equity = self.get_equity()

        if method == 'equal_weight':
            # Equal weight across max positions
            base_size = (equity / max_positions) * max_position_size

        elif method == 'volatility_scaled':
            # Scale by inverse volatility (requires vol data - simplified here)
            base_size = (equity / max_positions) * max_position_size
            # TODO: Implement actual volatility scaling

        elif method == 'kelly':
            # Kelly criterion (simplified - needs win rate and payoff ratio)
            # For now, use fractional Kelly
            base_size = (equity / max_positions) * max_position_size * 0.25

        elif method == 'risk_parity':
            # Equal risk contribution
            base_size = (equity / max_positions) * max_position_size
            # TODO: Implement actual risk parity

        else:
            raise ValueError(f"Unknown position sizing method: {method}")

        # Scale by signal strength
        target_value = base_size * abs(signal)

        # Convert to units
        size = target_value / price

        # Apply drawdown scaling if in drawdown
        if self.current_drawdown > self.params.get('max_portfolio_drawdown', 0.25):
            scale_factor = self.params.get('drawdown_scale_factor', 0.5)
            size *= (1 - scale_factor)

        return size

    def execute_trade(self, symbol: str, side: OrderSide, size: float,
                      price: float, timestamp: datetime) -> Trade:
        """
        Execute a trade with realistic transaction costs.

        Args:
            symbol: Trading symbol
            side: BUY or SELL
            size: Number of units
            price: Execution price
            timestamp: Trade timestamp

        Returns:
            Executed trade
        """
        # Calculate transaction costs using cost model
        gross_value = size * price
        costs = self.cost_model.calculate_costs(gross_value, is_buy=(side == OrderSide.BUY))

        # Extract individual cost components
        commission = costs['commission']
        slippage = costs['slippage']
        spread = costs['spread']
        total_cost = costs['total']

        # Adjust execution price to reflect slippage and spread
        total_cost_pct = total_cost / gross_value if gross_value > 0 else 0
        if side == OrderSide.BUY:
            execution_price = price * (1 + total_cost_pct)
        else:  # SELL
            execution_price = price * (1 - total_cost_pct)

        # Create trade
        trade = Trade(
            symbol=symbol,
            side=side,
            size=size,
            price=execution_price,
            timestamp=timestamp,
            commission=commission,
            slippage=slippage
        )

        # Update capital
        self.capital += trade.net_value

        # Record trade
        self.trades.append(trade)

        return trade

    def update(self, timestamp: datetime, signals: Dict[str, float],
               prices: Dict[str, float]) -> List[Trade]:
        """
        Update portfolio based on signals and current prices.

        This is the main entry point called at each timestep.

        Args:
            timestamp: Current timestamp
            signals: Dict of {symbol: signal_value}
            prices: Dict of {symbol: current_price}

        Returns:
            List of executed trades
        """
        executed_trades = []

        # Update existing positions with current prices
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.current_price = prices[symbol]

        # Check stop losses and take profits
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]

            if position.should_stop_loss():
                # Close position at stop loss
                trade = self.execute_trade(
                    symbol=symbol,
                    side=OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY,
                    size=position.size,
                    price=position.stop_loss,
                    timestamp=timestamp
                )
                executed_trades.append(trade)
                del self.positions[symbol]

            elif position.should_take_profit():
                # Close position at take profit
                trade = self.execute_trade(
                    symbol=symbol,
                    side=OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY,
                    size=position.size,
                    price=position.take_profit,
                    timestamp=timestamp
                )
                executed_trades.append(trade)
                del self.positions[symbol]

        # Process new signals
        for symbol, signal in signals.items():
            if symbol not in prices:
                continue

            price = prices[symbol]
            current_position = self.positions.get(symbol)

            # Determine desired position
            if signal > 0:  # Buy signal
                target_side = PositionSide.LONG
                target_size = self.calculate_position_size(symbol, signal, price)

            elif signal < 0:  # Sell signal
                target_side = PositionSide.SHORT
                target_size = self.calculate_position_size(symbol, abs(signal), price)

            else:  # Neutral signal
                target_size = 0
                target_side = None

            # Execute trades to reach target position
            if current_position is None and target_size > 0:
                # Open new position
                trade = self.execute_trade(
                    symbol=symbol,
                    side=OrderSide.BUY if target_side == PositionSide.LONG else OrderSide.SELL,
                    size=target_size,
                    price=price,
                    timestamp=timestamp
                )
                executed_trades.append(trade)

                # Create position
                stop_loss_pct = self.params.get('stop_loss_pct', 0.05)
                take_profit_pct = self.params.get('take_profit_pct', 0.15)

                if target_side == PositionSide.LONG:
                    stop_loss = price * (1 - stop_loss_pct)
                    take_profit = price * (1 + take_profit_pct)
                else:  # SHORT
                    stop_loss = price * (1 + stop_loss_pct)
                    take_profit = price * (1 - take_profit_pct)

                self.positions[symbol] = Position(
                    symbol=symbol,
                    side=target_side,
                    size=target_size,
                    entry_price=price,
                    entry_time=timestamp,
                    current_price=price,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )

            elif current_position is not None and target_size == 0:
                # Close existing position
                trade = self.execute_trade(
                    symbol=symbol,
                    side=OrderSide.SELL if current_position.side == PositionSide.LONG else OrderSide.BUY,
                    size=current_position.size,
                    price=price,
                    timestamp=timestamp
                )
                executed_trades.append(trade)
                del self.positions[symbol]

        # Update drawdown
        equity = self.get_equity()
        if equity > self.peak_equity:
            self.peak_equity = equity

        self.current_drawdown = (self.peak_equity - equity) / self.peak_equity

        # Record equity
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': equity,
            'capital': self.capital,
            'positions_value': sum(p.value for p in self.positions.values()),
            'n_positions': len(self.positions),
            'drawdown': self.current_drawdown
        })

        return executed_trades

    def get_metrics(self) -> PortfolioMetrics:
        """
        Calculate comprehensive performance metrics.

        Returns:
            PortfolioMetrics with all performance stats
        """
        if len(self.equity_curve) < 2:
            # Not enough data for metrics
            return PortfolioMetrics(
                total_return=0.0,
                total_return_pct=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                max_drawdown_pct=0.0,
                calmar_ratio=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                n_trades=0,
                n_winning_trades=0,
                n_losing_trades=0
            )

        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)

        # Calculate returns
        returns = equity_df['equity'].pct_change().dropna()

        # Total return
        final_equity = equity_df['equity'].iloc[-1]
        total_return = final_equity - self.initial_capital
        total_return_pct = total_return / self.initial_capital

        # Sharpe ratio (annualized, assuming daily data)
        if returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe = 0.0

        # Sortino ratio (only downside volatility)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino = returns.mean() / downside_returns.std() * np.sqrt(252)
        else:
            sortino = 0.0

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = abs(drawdown.min())
        max_dd_value = equity_df['equity'].max() - equity_df['equity'].min()

        # Calmar ratio
        annual_return = total_return_pct * (252 / len(returns))
        calmar = annual_return / max_dd if max_dd > 0 else 0.0

        # Trade-level metrics
        if len(self.trades) >= 2:
            # Calculate P&L for each round-trip trade
            # Simplified: Pair buy/sell trades
            winning_trades = []
            losing_trades = []

            # Group trades by symbol and calculate P&L
            # (Simplified implementation - assumes alternating buy/sell)
            for i in range(0, len(self.trades) - 1, 2):
                if i + 1 < len(self.trades):
                    entry = self.trades[i]
                    exit_trade = self.trades[i + 1]

                    if entry.side == OrderSide.BUY and exit_trade.side == OrderSide.SELL:
                        pnl = (exit_trade.price - entry.price) * entry.size
                        pnl -= (entry.total_cost + exit_trade.total_cost)

                        if pnl > 0:
                            winning_trades.append(pnl)
                        else:
                            losing_trades.append(abs(pnl))

            n_winning = len(winning_trades)
            n_losing = len(losing_trades)
            n_total = n_winning + n_losing

            win_rate = n_winning / n_total if n_total > 0 else 0.0
            avg_win = np.mean(winning_trades) if n_winning > 0 else 0.0
            avg_loss = np.mean(losing_trades) if n_losing > 0 else 0.0

            gross_profit = sum(winning_trades) if n_winning > 0 else 0.0
            gross_loss = sum(losing_trades) if n_losing > 0 else 0.0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        else:
            win_rate = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            profit_factor = 0.0
            n_winning = 0
            n_losing = 0

        return PortfolioMetrics(
            total_return=total_return,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd_value,
            max_drawdown_pct=max_dd,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            n_trades=len(self.trades),
            n_winning_trades=n_winning,
            n_losing_trades=n_losing
        )
