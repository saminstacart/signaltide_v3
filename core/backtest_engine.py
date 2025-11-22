"""
Unified backtest harness for signal validation.

Provides shared backtesting infrastructure to ensure fair comparisons between
signal implementations. Eliminates plumbing differences (equity tracking,
rebalancing logic, metric calculation) so performance gaps reflect only signal behavior.

Usage:
    config = BacktestConfig(
        start_date='2015-04-01',
        end_date='2024-12-31',
        initial_capital=100000.0,
        rebalance_schedule='M'
    )

    def my_universe_fn(rebal_date):
        return um.get_universe('sp500_actual', as_of_date=rebal_date, min_price=5.0)

    def my_signal_fn(rebal_date, tickers):
        # Generate scores for tickers at rebal_date
        return pd.Series({ticker: score for ticker, score in ...})

    result = run_backtest(my_universe_fn, my_signal_fn, config)
"""

from dataclasses import dataclass, field
from typing import Callable, List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import timedelta

from data.data_manager import DataManager
from core.schedules import get_rebalance_dates
from config import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtest execution."""

    start_date: str
    end_date: str
    initial_capital: float
    rebalance_schedule: str = 'M'  # 'M' for monthly, 'W' for weekly, etc.

    # Position sizing
    long_only: bool = True
    equal_weight: bool = True
    max_position_size: Optional[float] = None  # Fraction of portfolio

    # Execution
    lookback_days: int = 500  # For signal generation
    min_universe_size: int = 1  # Minimum stocks required (lowered for testing)

    # Transaction costs (future: plug in TransactionCostModel)
    transaction_costs: float = 0.0  # Basis points per round-trip

    # Tracking
    track_daily_equity: bool = False  # True for daily, False for rebalance-point only

    # Data manager (optional, created if not provided)
    data_manager: Optional[DataManager] = None

    def __post_init__(self):
        """Initialize data manager if not provided."""
        if self.data_manager is None:
            self.data_manager = DataManager()


@dataclass
class BacktestResult:
    """Results from backtest execution."""

    # Equity curve
    equity_curve: pd.Series

    # Performance metrics
    initial_capital: float
    final_equity: float
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    max_drawdown: float

    # Metadata
    num_rebalances: int
    rebalance_dates: List[str] = field(default_factory=list)

    # Diagnostics (optional)
    holdings_history: Optional[List[Dict]] = None
    signal_history: Optional[List[Dict]] = None


def run_backtest(
    universe_fn: Callable[[str], List[str]],
    signal_fn: Callable[[str, List[str]], pd.Series],
    config: BacktestConfig
) -> BacktestResult:
    """
    Run backtest with shared execution logic.

    Args:
        universe_fn: Function that returns list of tickers for a rebalance date
            Signature: universe_fn(rebal_date: str) -> List[str]

        signal_fn: Function that generates signals for given tickers at rebalance date
            Signature: signal_fn(rebal_date: str, tickers: List[str]) -> pd.Series
            Returns: Series with ticker index, signal scores as values

        config: Backtest configuration

    Returns:
        BacktestResult with equity curve and performance metrics
    """
    logger.info("=" * 80)
    logger.info("UNIFIED BACKTEST ENGINE")
    logger.info("=" * 80)
    logger.info(f"Period: {config.start_date} to {config.end_date}")
    logger.info(f"Initial capital: ${config.initial_capital:,.0f}")
    logger.info(f"Rebalance schedule: {config.rebalance_schedule}")
    logger.info(f"Daily equity tracking: {config.track_daily_equity}")
    logger.info("")

    # Get rebalance schedule
    rebalance_dates = get_rebalance_dates(
        schedule=config.rebalance_schedule,
        dm=config.data_manager,
        start_date=config.start_date,
        end_date=config.end_date
    )
    logger.info(f"Scheduled rebalances: {len(rebalance_dates)}")

    # Run backtest loop
    equity_curve = _backtest_loop(
        universe_fn,
        signal_fn,
        rebalance_dates,
        config
    )

    # Calculate metrics
    metrics = _calculate_metrics(equity_curve, config.initial_capital)

    # Package results (handle empty equity curve case)
    result = BacktestResult(
        equity_curve=equity_curve,
        initial_capital=config.initial_capital,
        final_equity=metrics.get('final_equity', config.initial_capital),
        total_return=metrics.get('total_return', 0.0),
        cagr=metrics.get('cagr', 0.0),
        volatility=metrics.get('volatility', 0.0),
        sharpe=metrics.get('sharpe', 0.0),
        max_drawdown=metrics.get('max_drawdown', 0.0),
        num_rebalances=len(rebalance_dates),
        rebalance_dates=rebalance_dates
    )

    # Print summary
    _print_summary(result)

    return result


def _backtest_loop(
    universe_fn: Callable[[str], List[str]],
    signal_fn: Callable[[str, List[str]], pd.Series],
    rebalance_dates: List[str],
    config: BacktestConfig
) -> pd.Series:
    """
    Core backtest execution loop.

    Returns:
        Equity curve (daily if config.track_daily_equity, else rebalance-point)
    """
    equity_data = []
    current_holdings = {}
    cash = config.initial_capital

    for i, rebal_date in enumerate(rebalance_dates):
        logger.info(f"Rebalance {i+1}/{len(rebalance_dates)}: {rebal_date}")

        # 1. Get universe
        universe = universe_fn(rebal_date)
        if isinstance(universe, pd.Series):
            universe = universe.tolist()
        elif isinstance(universe, pd.DataFrame):
            universe = universe.index.tolist()

        logger.info(f"  Universe: {len(universe)} stocks")

        if len(universe) < config.min_universe_size:
            logger.warning(f"  Universe too small ({len(universe)} < {config.min_universe_size})")
            continue

        # 2. Generate signals
        try:
            signals = signal_fn(rebal_date, universe)
        except Exception as e:
            logger.error(f"  Signal generation failed: {e}")
            continue

        logger.info(f"  Signals generated: {len(signals)} stocks")

        if len(signals) == 0:
            logger.warning("  No signals generated")
            continue

        # 3. Select positions (long only for now)
        if config.long_only:
            # Select tickers with positive signals (typically signal == 1.0 for top quintile)
            long_tickers = signals[signals > 0].index.tolist()
        else:
            # TODO: Long/short logic
            long_tickers = signals[signals > 0].index.tolist()

        logger.info(f"  Long positions: {len(long_tickers)} stocks")

        if len(long_tickers) == 0:
            logger.warning("  No long positions - holding cash")
            current_holdings = {}
            equity_data.append({
                'date': pd.Timestamp(rebal_date),
                'equity': cash
            })
            continue

        # 4. Get prices at rebalance date
        rebal_prices = {}
        for ticker in long_tickers:
            try:
                prices = config.data_manager.get_prices(ticker, rebal_date, rebal_date)
                if len(prices) > 0 and 'close' in prices.columns:
                    price = prices['close'].iloc[-1]
                    if isinstance(price, pd.Series):
                        price = price.iloc[0]
                    rebal_prices[ticker] = price
            except:
                pass

        if len(rebal_prices) == 0:
            logger.warning("  No prices available")
            continue

        # 5. Position sizing (equal-weight for now)
        if config.equal_weight:
            portfolio_value = cash
            position_size = portfolio_value / len(rebal_prices)

            new_holdings = {}
            for ticker, price in rebal_prices.items():
                shares = position_size / price
                new_holdings[ticker] = {
                    'shares': shares,
                    'entry_price': price
                }

            current_holdings = new_holdings
            cash = 0.0  # Fully invested

            logger.info(f"  Portfolio allocated: ${portfolio_value:,.0f} across {len(current_holdings)} stocks")

        # 6. Track equity
        equity_data.append({
            'date': pd.Timestamp(rebal_date),
            'equity': portfolio_value
        })

        # 7. Update cash for next rebalance (mark-to-market)
        if i + 1 < len(rebalance_dates):
            next_rebal = rebalance_dates[i + 1]
            portfolio_value = 0.0

            for ticker, holding in current_holdings.items():
                try:
                    price_df = config.data_manager.get_prices(ticker, next_rebal, next_rebal)
                    if len(price_df) > 0:
                        price = price_df['close'].iloc[-1]
                        if isinstance(price, pd.Series):
                            price = price.iloc[0]
                        portfolio_value += holding['shares'] * price
                except:
                    # Use entry price if can't get latest
                    portfolio_value += holding['shares'] * holding['entry_price']

            cash = portfolio_value

    # Convert to equity curve
    if len(equity_data) == 0:
        logger.error("No equity data generated!")
        return pd.Series(dtype=float)

    equity_df = pd.DataFrame(equity_data)
    equity_df = equity_df.drop_duplicates(subset=['date'], keep='last')
    equity_curve = equity_df.set_index('date')['equity'].sort_index()
    equity_curve = equity_curve.dropna()
    equity_curve = equity_curve[equity_curve > 0]

    logger.info(f"\nBacktest complete: {len(equity_curve)} equity points")

    return equity_curve


def _calculate_metrics(equity_curve: pd.Series, initial_capital: float) -> Dict:
    """Calculate performance metrics from equity curve."""
    if len(equity_curve) == 0:
        return {}

    final_equity = equity_curve.iloc[-1]
    if isinstance(final_equity, pd.Series):
        final_equity = final_equity.iloc[0]

    total_return = (final_equity / initial_capital) - 1

    # Calculate returns (could be daily or rebalance-point)
    returns = equity_curve.pct_change().dropna()

    # Annualized metrics
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    cagr = (final_equity / initial_capital) ** (1 / years) - 1 if years > 0 else 0

    # Annualization factor (assume monthly if < 100 points, else daily)
    periods_per_year = 252 if len(equity_curve) > 100 else 12

    # Volatility
    volatility = returns.std() * np.sqrt(periods_per_year) if len(returns) > 0 else 0

    # Sharpe
    sharpe = 0.0
    if len(returns) > 0 and returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(periods_per_year)

    # Max drawdown
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max.replace(0, 1)
    max_drawdown = drawdown.min()

    return {
        'final_equity': final_equity,
        'total_return': total_return,
        'cagr': cagr,
        'volatility': volatility,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown
    }


def _print_summary(result: BacktestResult):
    """Print backtest summary."""
    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Initial Capital:    ${result.initial_capital:,.0f}")
    logger.info(f"Final Equity:       ${result.final_equity:,.0f}")
    logger.info(f"Total Return:       {result.total_return:.2%}")
    logger.info(f"CAGR:               {result.cagr:.2%}")
    logger.info(f"Volatility:         {result.volatility:.2%}")
    logger.info(f"Sharpe Ratio:       {result.sharpe:.3f}")
    logger.info(f"Max Drawdown:       {result.max_drawdown:.2%}")
    logger.info(f"Rebalances:         {result.num_rebalances}")
    logger.info("")
    logger.info("=" * 80)
