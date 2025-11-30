#!/usr/bin/env python3
"""
Backtest with Dynamic S&P 500 Universe.

This script runs a backtest using point-in-time S&P 500 membership,
reconstituting the universe at each rebalance date.

Key difference from fixed universe:
- Uses dim_universe_membership table for point-in-time S&P 500
- Universe changes at each rebalance to match actual index composition
- Properly handles stocks that were added/removed from the index
"""

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from signaltide_v4.config.settings import get_settings
from signaltide_v4.data.market_data import MarketDataProvider
from signaltide_v4.portfolio.scoring import SignalAggregator
from signaltide_v4.portfolio.construction import PortfolioConstructor, Portfolio
from signaltide_v4.backtest.transaction_costs import TransactionCostModel
from signaltide_v4.backtest.engine import BacktestResult, RebalanceRecord
from signaltide_v4.signals.residual_momentum import ResidualMomentumSignal
from signaltide_v4.signals.quality import QualitySignal
from signaltide_v4.signals.insider import OpportunisticInsiderSignal
from signaltide_v4.validation.deflated_sharpe import DeflatedSharpeCalculator
from signaltide_v4.validation.walk_forward import WalkForwardValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/backtest_v4_dynamic.log'),
    ]
)
logger = logging.getLogger(__name__)


class DynamicSP500Provider:
    """Provides point-in-time S&P 500 membership."""

    def __init__(self, db_path: str = None):
        """Initialize with database path."""
        self.db_path = db_path or '/Users/samuelksherman/signaltide/data/signaltide.db'
        self._cache = {}  # Cache for universe lookups

        # Get all tickers that were ever in S&P 500 during our period
        self._all_tickers = self._get_all_sp500_tickers()
        logger.info(f"DynamicSP500Provider: {len(self._all_tickers)} total tickers in period")

    def _get_all_sp500_tickers(self) -> List[str]:
        """Get all tickers that were ever in S&P 500 during 2015-2025."""
        conn = sqlite3.connect(self.db_path)
        query = "SELECT DISTINCT ticker FROM dim_universe_membership WHERE universe_name = 'sp500_actual'"
        df = pd.read_sql(query, conn)
        conn.close()
        return sorted(df['ticker'].tolist())

    def get_universe(self, as_of_date: str) -> List[str]:
        """Get S&P 500 members as of a specific date."""
        if as_of_date in self._cache:
            return self._cache[as_of_date]

        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT ticker
            FROM dim_universe_membership
            WHERE universe_name = 'sp500_actual'
              AND membership_start_date <= ?
              AND (membership_end_date IS NULL OR membership_end_date > ?)
            ORDER BY ticker
        """
        df = pd.read_sql(query, conn, params=[as_of_date, as_of_date])
        conn.close()

        tickers = df['ticker'].tolist()
        self._cache[as_of_date] = tickers
        return tickers

    @property
    def all_tickers(self) -> List[str]:
        """Get all tickers for price data loading."""
        return self._all_tickers


def run_dynamic_backtest(
    start_date: str,
    end_date: str,
    initial_capital: float = 50_000.0,
    transaction_cost_bps: float = 5.0,
) -> BacktestResult:
    """
    Run backtest with dynamic S&P 500 universe.

    At each rebalance date, we:
    1. Get current S&P 500 members
    2. Generate signals only for those members
    3. Construct portfolio from top-ranked stocks
    """
    settings = get_settings()

    # Initialize components
    sp500_provider = DynamicSP500Provider()
    market_data = MarketDataProvider()

    # Create signals (Momentum + Quality + Insider)
    # Based on academic research:
    # - Momentum: Jegadeesh & Titman (1993), Blitz (2011) residual
    # - Quality: Ball et al. (2016), Asness (2018)
    # - Insider: Cohen, Malloy & Pomorski (2012)
    signals = [
        ResidualMomentumSignal(),
        QualitySignal(),
        OpportunisticInsiderSignal(),
    ]
    signal_aggregator = SignalAggregator(signals=signals, min_signals_required=1)
    portfolio_constructor = PortfolioConstructor()
    transaction_costs = TransactionCostModel(cost_bps=transaction_cost_bps)

    # Generate rebalance dates
    dates = pd.date_range(start=start_date, end=end_date, freq='ME')
    rebalance_dates = []
    for d in dates:
        month_end = d + pd.offsets.MonthEnd(0)
        bus_day = month_end - pd.offsets.BDay(0)
        if bus_day <= pd.Timestamp(end_date):
            rebalance_dates.append(bus_day.strftime('%Y-%m-%d'))
    rebalance_dates = sorted(set(rebalance_dates))

    logger.info(f"Backtest: {start_date} to {end_date}, {len(rebalance_dates)} rebalance dates")

    # Load price data for ALL possible tickers
    all_tickers = sp500_provider.all_tickers
    logger.info(f"Loading price data for {len(all_tickers)} potential tickers...")

    price_data_raw = market_data.get_prices(all_tickers, start_date, end_date)

    if price_data_raw.empty:
        logger.error("No price data available")
        raise ValueError("No price data")

    # Convert to wide format
    if isinstance(price_data_raw.index, pd.MultiIndex):
        if 'closeadj' in price_data_raw.columns:
            price_series = price_data_raw['closeadj']
        elif 'close' in price_data_raw.columns:
            price_series = price_data_raw['close']
        else:
            price_series = price_data_raw.iloc[:, 0]

        if price_series.index.duplicated().any():
            price_series = price_series[~price_series.index.duplicated(keep='last')]

        price_data = price_series.unstack(level='ticker')
    else:
        price_data = price_data_raw

    logger.info(f"Price data: {len(price_data)} dates, {len(price_data.columns)} tickers with data")

    # Run backtest
    portfolio_value = initial_capital
    current_portfolio = None
    rebalances = []
    daily_values = {}

    for i, rebal_date in enumerate(rebalance_dates):
        try:
            # GET CURRENT S&P 500 MEMBERS (DYNAMIC!)
            universe = sp500_provider.get_universe(rebal_date)
            logger.info(f"Rebalance {i+1}/{len(rebalance_dates)}: {rebal_date}, S&P 500 has {len(universe)} members")

            # Generate signals for current S&P 500 members only
            scores = signal_aggregator.aggregate(universe, rebal_date)

            # Select top stocks
            sorted_scores = scores.scores.sort_values(ascending=False)
            selected = sorted_scores.head(settings.top_n_positions)

            if len(selected) == 0:
                logger.warning(f"No stocks selected on {rebal_date}")
                continue

            # Construct portfolio
            new_portfolio = portfolio_constructor.construct(
                selected,
                rebal_date,
                previous_portfolio=current_portfolio,
            )

            # Calculate transaction costs
            if current_portfolio is not None:
                cost_result = transaction_costs.apply_to_rebalance(
                    current_portfolio.positions,
                    new_portfolio.positions,
                    portfolio_value,
                )
            else:
                cost_result = transaction_costs.apply_to_rebalance(
                    {},
                    new_portfolio.positions,
                    portfolio_value,
                )

            portfolio_value -= cost_result.total_cost
            turnover = cost_result.turnover if current_portfolio else 1.0

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
                period_return = _calculate_period_return(
                    current_portfolio, price_data, rebal_date, next_date
                )
                portfolio_value *= (1 + period_return)

                # Store daily values
                period_values = _get_period_values(
                    current_portfolio, price_data, rebal_date, next_date,
                    portfolio_value / (1 + period_return)
                )
                daily_values.update(period_values)

        except Exception as e:
            logger.error(f"Error on rebalance {rebal_date}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Build result
    return _build_result(daily_values, rebalances, start_date, end_date, initial_capital)


def _calculate_period_return(portfolio: Portfolio, price_data: pd.DataFrame,
                             start_date: str, end_date: str) -> float:
    """Calculate portfolio return over a period."""
    if portfolio is None or len(portfolio.positions) == 0:
        return 0.0

    mask = (price_data.index >= start_date) & (price_data.index <= end_date)
    period_prices = price_data[mask]

    if len(period_prices) < 2:
        return 0.0

    total_return = 0.0
    total_weight = 0.0

    for ticker, weight in portfolio.positions.items():
        if ticker in period_prices.columns:
            ticker_prices = period_prices[ticker].dropna()
            if len(ticker_prices) >= 2:
                ticker_return = ticker_prices.iloc[-1] / ticker_prices.iloc[0] - 1
                total_return += weight * ticker_return
                total_weight += weight

    if total_weight > 0 and total_weight < 1:
        total_return = total_return / total_weight

    return total_return


def _get_period_values(portfolio: Portfolio, price_data: pd.DataFrame,
                       start_date: str, end_date: str, start_value: float) -> dict:
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


def _build_result(daily_values: dict, rebalances: list,
                  start_date: str, end_date: str, initial_capital: float) -> BacktestResult:
    """Build BacktestResult from backtest data."""
    if not daily_values:
        raise ValueError("No daily values calculated")

    values_series = pd.Series(daily_values).sort_index()
    values_series.index = pd.to_datetime(values_series.index)

    returns = values_series.pct_change().dropna()
    total_return = values_series.iloc[-1] / values_series.iloc[0] - 1

    days = (values_series.index[-1] - values_series.index[0]).days
    years = days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    volatility = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

    negative_returns = returns[returns < 0]
    downside_std = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else volatility
    sortino = (returns.mean() * 252) / downside_std if downside_std > 0 else 0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95

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
        initial_capital=initial_capital,
        final_value=float(values_series.iloc[-1]),
        n_periods=len(rebalances),
        diagnostics={'trading_days': len(returns), 'years': years, 'universe_type': 'dynamic_sp500'},
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='SignalTide v4 Dynamic S&P 500 Backtest')
    parser.add_argument('--start', type=str, default='2015-07-01')
    parser.add_argument('--end', type=str, default='2024-12-31')
    parser.add_argument('--capital', type=float, default=50_000.0)
    parser.add_argument('--cost-bps', type=float, default=5.0)
    parser.add_argument('--output', type=str, default='results/backtest_v4_dynamic_sp500.json')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path('logs').mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("SignalTide v4 Dynamic S&P 500 Backtest")
    logger.info("=" * 60)
    logger.info(f"Period: {args.start} to {args.end}")
    logger.info(f"Capital: ${args.capital:,.0f}")
    logger.info(f"Transaction cost: {args.cost_bps} bps")

    start_time = datetime.now()

    result = run_dynamic_backtest(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        transaction_cost_bps=args.cost_bps,
    )

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Backtest completed in {elapsed:.1f} seconds")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Return:     {result.total_return:>10.2%}")
    logger.info(f"CAGR:             {result.cagr:>10.2%}")
    logger.info(f"Sharpe Ratio:     {result.sharpe_ratio:>10.3f}")
    logger.info(f"Sortino Ratio:    {result.sortino_ratio:>10.3f}")
    logger.info(f"Max Drawdown:     {result.max_drawdown:>10.2%}")
    logger.info(f"Calmar Ratio:     {result.calmar_ratio:>10.3f}")
    logger.info(f"Volatility:       {result.volatility:>10.2%}")
    logger.info(f"Total Costs:      ${result.total_costs:>9,.0f}")
    logger.info(f"Avg Positions:    {result.avg_positions:>10.1f}")
    logger.info(f"Final Value:      ${result.final_value:>9,.0f}")

    # Build output
    output = {
        'metadata': {
            'start_date': args.start,
            'end_date': args.end,
            'initial_capital': args.capital,
            'transaction_cost_bps': args.cost_bps,
            'universe_type': 'dynamic_sp500',
            'run_timestamp': datetime.now().isoformat(),
        },
        'performance': result.summary(),
        'final_value': result.final_value,
    }

    # Run validation if requested
    if args.validate:
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION")
        logger.info("=" * 60)

        # DSR
        dsr_calc = DeflatedSharpeCalculator()
        dsr_result = dsr_calc.calculate(result.returns)

        output['validation'] = {
            'dsr': {
                'observed_sharpe': dsr_result.observed_sharpe,
                'deflated_sharpe': dsr_result.deflated_sharpe,
                'p_value': dsr_result.p_value,
                'confidence': dsr_result.confidence_level,
                'is_significant': str(dsr_result.is_significant),
            },
        }

        # Walk-forward
        wf_validator = WalkForwardValidator()
        wf_result = wf_validator.validate_returns(result.returns)

        output['validation']['walk_forward'] = {
            'n_folds': wf_result.n_folds,
            'pct_positive': wf_result.pct_positive,
            'mean_test_sharpe': wf_result.mean_test_sharpe,
            'std_test_sharpe': wf_result.std_test_sharpe,
            'is_valid': wf_result.is_valid,
        }

        logger.info(f"DSR: Observed={dsr_result.observed_sharpe:.3f}, p={dsr_result.p_value:.3f}")
        logger.info(f"Walk-Forward: {wf_result.n_folds} folds, {wf_result.pct_positive:.0%} positive")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
