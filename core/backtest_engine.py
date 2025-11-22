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
    # Validate config for unimplemented features
    if config.track_daily_equity:
        raise NotImplementedError(
            "track_daily_equity=True not yet implemented. "
            "Use track_daily_equity=False for rebalance-point equity tracking."
        )

    if not config.equal_weight:
        raise NotImplementedError(
            "equal_weight=False not yet implemented. "
            "Only equal-weight positioning is currently supported."
        )

    if not config.long_only:
        raise NotImplementedError(
            "long_only=False (long/short) not yet implemented. "
            "Only long-only backtests are currently supported."
        )

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

        # Validate signals (no NaN values allowed)
        if signals.isna().any():
            nan_count = signals.isna().sum()
            logger.warning(f"  Dropping {nan_count} signals with NaN values")
            signals = signals.dropna()

        if len(signals) == 0:
            logger.warning("  No valid signals after NaN filtering")
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
        price_fetch_errors = 0
        for ticker in long_tickers:
            try:
                prices = config.data_manager.get_prices(ticker, rebal_date, rebal_date)
                if len(prices) > 0 and 'close' in prices.columns:
                    price = prices['close'].iloc[-1]
                    if isinstance(price, pd.Series):
                        price = price.iloc[0]
                    rebal_prices[ticker] = price
                else:
                    logger.debug(f"    No price data for {ticker} at {rebal_date}")
                    price_fetch_errors += 1
            except Exception as e:
                logger.debug(f"    Error fetching {ticker} price at {rebal_date}: {e}")
                price_fetch_errors += 1

        if price_fetch_errors > 0:
            pct_failed = price_fetch_errors / len(long_tickers) * 100
            logger.warning(f"  Could not fetch prices for {price_fetch_errors}/{len(long_tickers)} tickers ({pct_failed:.1f}%)")

        if len(rebal_prices) == 0:
            logger.warning("  No prices available - skipping rebalance")
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
            missing_prices_count = 0

            for ticker, holding in current_holdings.items():
                try:
                    price_df = config.data_manager.get_prices(ticker, next_rebal, next_rebal)
                    if len(price_df) > 0:
                        price = price_df['close'].iloc[-1]
                        if isinstance(price, pd.Series):
                            price = price.iloc[0]
                        portfolio_value += holding['shares'] * price
                    else:
                        # No price data available, fall back to entry price
                        logger.warning(f"  No price for {ticker} at {next_rebal}, using entry price")
                        portfolio_value += holding['shares'] * holding['entry_price']
                        missing_prices_count += 1
                except Exception as e:
                    # Error fetching price, fall back to entry price
                    logger.warning(f"  Error fetching {ticker} price at {next_rebal}: {e}")
                    portfolio_value += holding['shares'] * holding['entry_price']
                    missing_prices_count += 1

            if missing_prices_count > 0:
                pct_missing = missing_prices_count / len(current_holdings) * 100
                logger.warning(f"  {missing_prices_count}/{len(current_holdings)} positions ({pct_missing:.1f}%) using fallback prices")

            cash = portfolio_value

    # 8. FINAL MARK-TO-MARKET at config.end_date
    # CRITICAL: Last rebalance only has entry prices, need to mark-to-market at backtest end
    if len(current_holdings) > 0:
        logger.info(f"\nFinal mark-to-market at {config.end_date}")
        final_portfolio_value = 0.0
        missing_prices_count = 0

        for ticker, holding in current_holdings.items():
            try:
                price_df = config.data_manager.get_prices(ticker, config.end_date, config.end_date)
                if len(price_df) > 0:
                    price = price_df['close'].iloc[-1]
                    if isinstance(price, pd.Series):
                        price = price.iloc[0]
                    final_portfolio_value += holding['shares'] * price
                else:
                    logger.warning(f"  No price for {ticker} at {config.end_date}, using entry price")
                    final_portfolio_value += holding['shares'] * holding['entry_price']
                    missing_prices_count += 1
            except Exception as e:
                logger.warning(f"  Error fetching {ticker} price at {config.end_date}: {e}")
                final_portfolio_value += holding['shares'] * holding['entry_price']
                missing_prices_count += 1

        if missing_prices_count > 0:
            pct_missing = missing_prices_count / len(current_holdings) * 100
            logger.warning(f"  Final MTM: {missing_prices_count}/{len(current_holdings)} positions ({pct_missing:.1f}%) using fallback prices")

        # Add final equity point
        equity_data.append({
            'date': pd.Timestamp(config.end_date),
            'equity': final_portfolio_value
        })
        logger.info(f"  Final equity: ${final_portfolio_value:,.0f}")

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
