"""
Backtest Engine for SignalTide v4.

Executes strategies over historical data with:
- Monthly rebalancing
- Transaction cost modeling
- Position tracking
- Performance metrics calculation
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import numpy as np

from signaltide_v4.config.settings import get_settings
from signaltide_v4.data.market_data import MarketDataProvider
from signaltide_v4.portfolio.scoring import SignalAggregator, AggregatedScore
from signaltide_v4.portfolio.construction import PortfolioConstructor, Portfolio
from signaltide_v4.backtest.transaction_costs import TransactionCostModel
from signaltide_v4.signals.residual_momentum import ResidualMomentumSignal
from signaltide_v4.signals.quality import QualitySignal

logger = logging.getLogger(__name__)


@dataclass
class RebalanceRecord:
    """Record of a single rebalance event."""

    date: str
    portfolio: Portfolio
    portfolio_value: float
    transaction_cost: float
    turnover: float
    n_positions: int


@dataclass
class BacktestResult:
    """Container for backtest results."""

    # Performance metrics
    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float

    # Risk metrics
    volatility: float
    downside_deviation: float
    var_95: float
    cvar_95: float

    # Trading metrics
    total_trades: int
    total_turnover: float
    total_costs: float
    avg_positions: float

    # Time series
    returns: pd.Series
    cumulative_returns: pd.Series
    drawdown_series: pd.Series
    portfolio_values: pd.Series

    # Rebalance history
    rebalances: List[RebalanceRecord]

    # Metadata
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    n_periods: int

    # Diagnostics
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'total_return': self.total_return,
            'cagr': self.cagr,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'calmar_ratio': self.calmar_ratio,
            'volatility': self.volatility,
            'total_trades': self.total_trades,
            'avg_positions': self.avg_positions,
            'total_costs': self.total_costs,
        }


class BacktestEngine:
    """
    Executes backtests with monthly rebalancing.

    Process:
    1. Generate signals at each rebalance date
    2. Construct portfolio from signals
    3. Apply transaction costs
    4. Track returns between rebalances
    5. Calculate performance metrics
    """

    def __init__(
        self,
        market_data: Optional[MarketDataProvider] = None,
        signal_aggregator: Optional[SignalAggregator] = None,
        portfolio_constructor: Optional[PortfolioConstructor] = None,
        transaction_costs: Optional[TransactionCostModel] = None,
        initial_capital: Optional[float] = None,
    ):
        """
        Initialize backtest engine.

        Args:
            market_data: MarketDataProvider for price data
            signal_aggregator: SignalAggregator for signal generation
            portfolio_constructor: PortfolioConstructor for portfolio building
            transaction_costs: TransactionCostModel for cost calculation
            initial_capital: Starting capital (default from settings)
        """
        settings = get_settings()

        self.market_data = market_data or MarketDataProvider()
        self.signal_aggregator = signal_aggregator or SignalAggregator()
        self.portfolio_constructor = portfolio_constructor or PortfolioConstructor()
        self.transaction_costs = transaction_costs or TransactionCostModel()
        self.initial_capital = initial_capital or settings.initial_capital

        logger.info(f"BacktestEngine: capital=${self.initial_capital:,.0f}")

    def run(
        self,
        universe: List[str],
        start_date: str,
        end_date: str,
        rebalance_freq: str = 'ME',
    ) -> BacktestResult:
        """
        Run backtest over date range.

        Args:
            universe: List of tickers to consider
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            rebalance_freq: Rebalance frequency ('M' for monthly)

        Returns:
            BacktestResult with full performance analysis
        """
        logger.info(
            f"Starting backtest: {start_date} to {end_date}, "
            f"{len(universe)} tickers, freq={rebalance_freq}"
        )

        # Generate rebalance dates
        rebalance_dates = self._generate_rebalance_dates(
            start_date, end_date, rebalance_freq
        )

        if len(rebalance_dates) < 2:
            logger.error(f"Insufficient rebalance dates: {len(rebalance_dates)}")
            return self._empty_result(start_date, end_date)

        # Initialize state
        portfolio_value = self.initial_capital
        current_portfolio = None
        rebalances = []
        daily_values = {}

        # Get all price data upfront
        price_data_raw = self.market_data.get_prices(
            universe, start_date, end_date
        )

        if price_data_raw.empty:
            logger.error("No price data available")
            return self._empty_result(start_date, end_date)

        # Convert MultiIndex to date x ticker format for return calculation
        # The get_prices returns MultiIndex (date, ticker), we need date as index, ticker as columns
        if isinstance(price_data_raw.index, pd.MultiIndex):
            # Use closeadj column and pivot to wide format
            if 'closeadj' in price_data_raw.columns:
                price_series = price_data_raw['closeadj']
            elif 'close' in price_data_raw.columns:
                price_series = price_data_raw['close']
            else:
                price_series = price_data_raw.iloc[:, 0]

            # Handle duplicates before unstacking
            if price_series.index.duplicated().any():
                price_series = price_series[~price_series.index.duplicated(keep='last')]

            price_data = price_series.unstack(level='ticker')
        else:
            price_data = price_data_raw

        logger.info(f"Processing {len(rebalance_dates)} rebalance dates")

        for i, rebal_date in enumerate(rebalance_dates):
            try:
                # Generate signals
                scores = self.signal_aggregator.aggregate(
                    universe, rebal_date
                )

                # Select top stocks
                selected = self._select_stocks(scores)

                if len(selected) == 0:
                    logger.warning(f"No stocks selected on {rebal_date}")
                    continue

                # Construct new portfolio
                new_portfolio = self.portfolio_constructor.construct(
                    selected,
                    rebal_date,
                    previous_portfolio=current_portfolio,
                )

                # Calculate transaction costs
                if current_portfolio is not None:
                    cost_result = self.transaction_costs.apply_to_rebalance(
                        current_portfolio.positions,
                        new_portfolio.positions,
                        portfolio_value,
                    )
                    portfolio_value -= cost_result.total_cost
                    turnover = cost_result.turnover
                else:
                    # Initial investment
                    cost_result = self.transaction_costs.apply_to_rebalance(
                        {},
                        new_portfolio.positions,
                        portfolio_value,
                    )
                    portfolio_value -= cost_result.total_cost
                    turnover = 1.0

                # Record rebalance
                rebalances.append(RebalanceRecord(
                    date=rebal_date,
                    portfolio=new_portfolio,
                    portfolio_value=portfolio_value,
                    transaction_cost=cost_result.total_cost,
                    turnover=turnover,
                    n_positions=len(new_portfolio.positions),
                ))

                current_portfolio = new_portfolio

                # Calculate returns until next rebalance
                if i < len(rebalance_dates) - 1:
                    next_date = rebalance_dates[i + 1]
                    period_return = self._calculate_period_return(
                        current_portfolio,
                        price_data,
                        rebal_date,
                        next_date,
                    )
                    portfolio_value *= (1 + period_return)

                    # Store daily values for this period
                    period_values = self._get_period_values(
                        current_portfolio,
                        price_data,
                        rebal_date,
                        next_date,
                        portfolio_value / (1 + period_return),
                    )
                    daily_values.update(period_values)

                logger.debug(
                    f"Rebalance {i+1}/{len(rebalance_dates)}: "
                    f"{rebal_date}, {len(new_portfolio.positions)} positions, "
                    f"value=${portfolio_value:,.0f}"
                )

            except Exception as e:
                logger.error(f"Error on rebalance {rebal_date}: {e}")
                continue

        # Build result
        return self._build_result(
            daily_values,
            rebalances,
            start_date,
            end_date,
        )

    def _generate_rebalance_dates(
        self,
        start_date: str,
        end_date: str,
        freq: str,
    ) -> List[str]:
        """Generate rebalance dates at month-end."""
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)

        # Use month-end business days
        rebal_dates = []
        for d in dates:
            # Get last business day of month
            month_end = d + pd.offsets.MonthEnd(0)
            bus_day = month_end - pd.offsets.BDay(0)
            if bus_day <= pd.Timestamp(end_date):
                rebal_dates.append(bus_day.strftime('%Y-%m-%d'))

        return sorted(set(rebal_dates))

    def _select_stocks(
        self,
        scores: AggregatedScore,
        top_n: Optional[int] = None,
    ) -> pd.Series:
        """Select top-scoring stocks."""
        settings = get_settings()
        top_n = top_n or settings.top_n_positions

        # Sort by score descending
        sorted_scores = scores.scores.sort_values(ascending=False)

        # Take top N
        selected = sorted_scores.head(top_n)

        return selected

    def _calculate_period_return(
        self,
        portfolio: Portfolio,
        price_data: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> float:
        """Calculate portfolio return over a period."""
        if portfolio is None or len(portfolio.positions) == 0:
            return 0.0

        # Get prices for period
        mask = (price_data.index >= start_date) & (price_data.index <= end_date)
        period_prices = price_data[mask]

        if len(period_prices) < 2:
            return 0.0

        # Calculate weighted return
        total_return = 0.0
        total_weight = 0.0

        for ticker, weight in portfolio.positions.items():
            if ticker in period_prices.columns:
                ticker_prices = period_prices[ticker].dropna()
                if len(ticker_prices) >= 2:
                    ticker_return = ticker_prices.iloc[-1] / ticker_prices.iloc[0] - 1
                    total_return += weight * ticker_return
                    total_weight += weight

        # Adjust if not all weights had data
        if total_weight > 0 and total_weight < 1:
            # Scale return to full portfolio
            total_return = total_return / total_weight

        return total_return

    def _get_period_values(
        self,
        portfolio: Portfolio,
        price_data: pd.DataFrame,
        start_date: str,
        end_date: str,
        start_value: float,
    ) -> Dict[str, float]:
        """Get daily portfolio values for a period."""
        if portfolio is None or len(portfolio.positions) == 0:
            return {}

        mask = (price_data.index >= start_date) & (price_data.index <= end_date)
        period_prices = price_data[mask]

        if len(period_prices) == 0:
            return {}

        daily_values = {}
        first_prices = period_prices.iloc[0]

        for idx in range(len(period_prices)):
            date_str = period_prices.index[idx].strftime('%Y-%m-%d')
            current_prices = period_prices.iloc[idx]

            # Calculate portfolio value
            value = 0.0
            for ticker, weight in portfolio.positions.items():
                if ticker in current_prices.index and ticker in first_prices.index:
                    first_price = first_prices[ticker]
                    current_price = current_prices[ticker]
                    if pd.notna(first_price) and pd.notna(current_price) and first_price > 0:
                        ticker_return = current_price / first_price
                        value += weight * start_value * ticker_return

            if value > 0:
                daily_values[date_str] = value

        return daily_values

    def _build_result(
        self,
        daily_values: Dict[str, float],
        rebalances: List[RebalanceRecord],
        start_date: str,
        end_date: str,
    ) -> BacktestResult:
        """Build BacktestResult from backtest data."""
        if not daily_values:
            return self._empty_result(start_date, end_date)

        # Convert to series
        values_series = pd.Series(daily_values).sort_index()
        values_series.index = pd.to_datetime(values_series.index)

        # Calculate returns
        returns = values_series.pct_change().dropna()

        # Calculate metrics
        total_return = values_series.iloc[-1] / values_series.iloc[0] - 1

        # CAGR
        days = (values_series.index[-1] - values_series.index[0]).days
        years = days / 365.25
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252)

        # Sharpe (assuming 0 risk-free rate)
        sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

        # Sortino (downside deviation)
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else volatility
        sortino = (returns.mean() * 252) / downside_std if downside_std > 0 else 0

        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar
        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

        # VaR and CVaR
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95

        # Trading metrics
        total_costs = sum(r.transaction_cost for r in rebalances)
        total_turnover = sum(r.turnover for r in rebalances)
        total_trades = sum(r.n_positions for r in rebalances)
        avg_positions = np.mean([r.n_positions for r in rebalances]) if rebalances else 0

        return BacktestResult(
            total_return=float(total_return),
            cagr=float(cagr),
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            max_drawdown=float(max_drawdown),
            calmar_ratio=float(calmar),
            volatility=float(volatility),
            downside_deviation=float(downside_std),
            var_95=float(var_95),
            cvar_95=float(cvar_95),
            total_trades=total_trades,
            total_turnover=float(total_turnover),
            total_costs=float(total_costs),
            avg_positions=float(avg_positions),
            returns=returns,
            cumulative_returns=cumulative,
            drawdown_series=drawdown,
            portfolio_values=values_series,
            rebalances=rebalances,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_value=float(values_series.iloc[-1]) if len(values_series) > 0 else self.initial_capital,
            n_periods=len(rebalances),
            diagnostics={
                'trading_days': len(returns),
                'years': years,
            },
        )

    def _empty_result(
        self,
        start_date: str,
        end_date: str,
    ) -> BacktestResult:
        """Return empty result for failed backtest."""
        empty_series = pd.Series(dtype=float)

        return BacktestResult(
            total_return=0.0,
            cagr=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            calmar_ratio=0.0,
            volatility=0.0,
            downside_deviation=0.0,
            var_95=0.0,
            cvar_95=0.0,
            total_trades=0,
            total_turnover=0.0,
            total_costs=0.0,
            avg_positions=0.0,
            returns=empty_series,
            cumulative_returns=empty_series,
            drawdown_series=empty_series,
            portfolio_values=empty_series,
            rebalances=[],
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_value=self.initial_capital,
            n_periods=0,
            diagnostics={'error': 'insufficient_data'},
        )


def run_backtest(
    universe: List[str],
    start_date: str,
    end_date: str,
    initial_capital: float = 50_000.0,
    transaction_cost_bps: float = 5.0,
) -> BacktestResult:
    """
    Convenience function to run a backtest.

    Args:
        universe: List of tickers
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        initial_capital: Starting capital
        transaction_cost_bps: Transaction cost in basis points

    Returns:
        BacktestResult
    """
    # Create signals
    signals = [
        ResidualMomentumSignal(),
        QualitySignal(),
    ]

    # Create signal aggregator
    signal_aggregator = SignalAggregator(
        signals=signals,
        min_signals_required=1,  # Accept stocks with at least 1 signal
    )

    engine = BacktestEngine(
        signal_aggregator=signal_aggregator,
        initial_capital=initial_capital,
        transaction_costs=TransactionCostModel(cost_bps=transaction_cost_bps),
    )

    return engine.run(universe, start_date, end_date)
