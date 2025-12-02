#!/usr/bin/env python3
"""
Phase 4 Stabilized Backtest Script for SignalTide V4.

Key improvements:
1. Warmup period - skip first N months while signals ramp up
2. Signal coverage validation - require 2+ signals with 30%+ coverage
3. Stabilized portfolio construction (percentile hysteresis, min holding, sector caps)
4. Signal contribution tracking
"""

import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict

import pandas as pd
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from signaltide_v4.config.settings import get_settings
from signaltide_v4.data.market_data import MarketDataProvider
from signaltide_v4.data.fundamental_data import FundamentalDataProvider
from signaltide_v4.data.factor_data import FactorDataProvider
from signaltide_v4.signals.residual_momentum import ResidualMomentumSignal
from signaltide_v4.signals.quality import QualitySignal
from signaltide_v4.signals.insider import OpportunisticInsiderSignal
from signaltide_v4.portfolio.scoring import SignalAggregator
from signaltide_v4.portfolio.stabilized_construction import StabilizedPortfolioConstructor
from signaltide_v4.backtest.transaction_costs import TransactionCostModel
from signaltide_v4.backtest.engine import BacktestResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class DynamicSP500Provider:
    """Provides point-in-time S&P 500 membership."""

    def __init__(self, db_path: Optional[str] = None):
        settings = get_settings()
        self.db_path = db_path or settings.db_path
        self._cache = {}  # Cache for universe lookups

        # Get all tickers that were ever in S&P 500 during our period
        self._all_tickers = self._get_all_sp500_tickers()
        logger.info(f"DynamicSP500Provider: {len(self._all_tickers)} total tickers in period")

    def _get_all_sp500_tickers(self) -> List[str]:
        """Get all tickers that were ever in S&P 500 during 2015-2025."""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        query = "SELECT DISTINCT ticker FROM dim_universe_membership WHERE universe_name = 'sp500_actual'"
        df = pd.read_sql(query, conn)
        conn.close()
        return sorted(df['ticker'].tolist())

    def get_members(self, as_of_date: str) -> List[str]:
        """Get S&P 500 members as of a specific date."""
        if as_of_date in self._cache:
            return self._cache[as_of_date]

        import sqlite3
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


class SignalCoverageValidator:
    """Validate signal coverage before trading."""

    def __init__(
        self,
        min_signals_required: int = 2,
        min_coverage_per_signal: float = 0.30,
    ):
        self.min_signals_required = min_signals_required
        self.min_coverage_per_signal = min_coverage_per_signal

    def validate(
        self,
        signal_coverages: Dict[str, float],
        as_of_date: str,
    ) -> tuple[bool, str]:
        """
        Validate if signal coverage is sufficient for trading.

        Returns:
            (is_valid, reason)
        """
        active_signals = [
            name for name, coverage in signal_coverages.items()
            if coverage >= self.min_coverage_per_signal
        ]

        if len(active_signals) < self.min_signals_required:
            reason = (
                f"Insufficient coverage: {len(active_signals)} signals >= "
                f"{self.min_coverage_per_signal:.0%} (need {self.min_signals_required}). "
                f"Coverages: {signal_coverages}"
            )
            return False, reason

        return True, "OK"


class SignalContributionTracker:
    """Track each signal's contribution to returns."""

    def __init__(self, signal_names: List[str]):
        self.signal_names = signal_names
        self.attribution_history: List[Dict[str, float]] = []

    def calculate_attribution(
        self,
        signal_scores: Dict[str, pd.Series],
        portfolio_weights: Dict[str, float],
        forward_returns: pd.Series,
    ) -> Dict[str, float]:
        """
        Calculate signal contribution using rank IC.

        For each signal, compute correlation between signal rank and return rank
        for the stocks in the portfolio.
        """
        attributions = {}

        for signal_name, scores in signal_scores.items():
            # Get portfolio tickers
            portfolio_tickers = list(portfolio_weights.keys())

            # Filter to portfolio
            signal_in_portfolio = scores.reindex(portfolio_tickers).dropna()
            returns_in_portfolio = forward_returns.reindex(signal_in_portfolio.index).dropna()

            common = signal_in_portfolio.index.intersection(returns_in_portfolio.index)

            if len(common) < 10:
                attributions[signal_name] = np.nan
                continue

            # Rank IC
            ic = signal_in_portfolio.loc[common].rank().corr(
                returns_in_portfolio.loc[common].rank()
            )
            attributions[signal_name] = ic

        self.attribution_history.append(attributions)
        return attributions

    def get_summary(self) -> Dict[str, float]:
        """Get average attribution per signal."""
        if not self.attribution_history:
            return {}

        df = pd.DataFrame(self.attribution_history)
        return df.mean().to_dict()


def run_stabilized_backtest(
    start_date: str = '2015-01-01',
    end_date: str = '2024-12-31',
    initial_capital: float = 50_000,
) -> BacktestResult:
    """
    Run stabilized backtest with all Phase 4 fixes.
    """
    settings = get_settings()

    # Configuration
    warmup_months = settings.warmup_months
    min_signals_required = settings.min_signals_required
    min_coverage_per_signal = settings.min_coverage_per_signal
    min_universe_size = settings.min_universe_size

    logger.info("=" * 70)
    logger.info("SIGNALTIDE V4 - PHASE 4 STABILIZED BACKTEST")
    logger.info("=" * 70)
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Warmup: {warmup_months} months")
    logger.info(f"Min signals: {min_signals_required} with {min_coverage_per_signal:.0%} coverage")
    logger.info(f"Min universe: {min_universe_size} tickers")
    logger.info("=" * 70)

    # Initialize components
    universe_provider = DynamicSP500Provider()
    market_data = MarketDataProvider()
    fundamental_data = FundamentalDataProvider()
    factor_data = FactorDataProvider()

    # Initialize signals
    momentum_signal = ResidualMomentumSignal(
        market_data=market_data,
        factor_data=factor_data,
    )
    quality_signal = QualitySignal(
        fundamental_data=fundamental_data,
        market_data=market_data,
    )
    insider_signal = OpportunisticInsiderSignal()

    signals = [momentum_signal, quality_signal, insider_signal]

    # Signal aggregator with equal weights
    aggregator = SignalAggregator(
        signals=signals,
        weights={'residual_momentum': 1/3, 'quality': 1/3, 'opportunistic_insider': 1/3},
        min_signals_required=min_signals_required,
    )

    # Stabilized portfolio constructor
    portfolio_constructor = StabilizedPortfolioConstructor(
        market_data=market_data,
        params={
            'entry_percentile': settings.entry_percentile,
            'exit_percentile': settings.exit_percentile,
            'min_holding_months': settings.min_holding_months,
            'hard_sector_cap': settings.hard_sector_cap,
            'smoothing_window': settings.signal_smoothing_window,
        },
    )

    # Validators and trackers
    coverage_validator = SignalCoverageValidator(
        min_signals_required=min_signals_required,
        min_coverage_per_signal=min_coverage_per_signal,
    )
    contribution_tracker = SignalContributionTracker(
        signal_names=['residual_momentum', 'quality', 'opportunistic_insider']
    )

    # Transaction cost model
    cost_model = TransactionCostModel(cost_bps=settings.transaction_cost_bps)

    # Generate rebalance dates
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    rebalance_dates = pd.date_range(start=start_dt, end=end_dt, freq='ME')  # Month-End
    rebalance_dates = [d.strftime('%Y-%m-%d') for d in rebalance_dates]

    logger.info(f"Backtest: {start_date} to {end_date}, {len(rebalance_dates)} rebalance dates")

    # Load price data
    all_tickers = universe_provider.all_tickers
    logger.info(f"Loading price data for {len(all_tickers)} potential tickers...")

    prices_df = market_data.get_prices(all_tickers, start_date, end_date)
    # prices_df has MultiIndex (date, ticker) - count unique tickers
    n_tickers = prices_df.index.get_level_values('ticker').nunique() if len(prices_df) > 0 else 0
    n_dates = prices_df.index.get_level_values('date').nunique() if len(prices_df) > 0 else 0
    logger.info(f"Price data: {n_dates} unique dates, {n_tickers} unique tickers")

    # Get sector mapping
    import sqlite3
    conn = sqlite3.connect(settings.db_path)
    sector_query = "SELECT ticker, sector FROM sharadar_tickers"
    sector_df = pd.read_sql_query(sector_query, conn)
    conn.close()
    sectors = dict(zip(sector_df['ticker'], sector_df['sector']))

    # Backtest state
    capital = initial_capital
    portfolio_value_series = []
    current_weights: Dict[str, float] = {}
    weights_history: Dict[str, Dict[str, float]] = {}  # rebalance_date -> weights
    rebalance_count = 0
    skipped_warmup = 0
    skipped_coverage = 0
    skipped_universe = 0
    total_turnover = 0.0
    rebalance_turnovers = []

    # Track signal coverages
    signal_coverage_history = []

    for i, rebalance_date in enumerate(rebalance_dates):
        rebalance_count += 1

        # Get universe
        universe = universe_provider.get_members(rebalance_date)

        # Check warmup period
        if i < warmup_months:
            logger.info(f"WARMUP {i+1}/{warmup_months}: Skipping {rebalance_date}")
            skipped_warmup += 1
            continue

        # Check universe size
        if len(universe) < min_universe_size:
            logger.warning(f"Universe too small: {len(universe)} < {min_universe_size}")
            skipped_universe += 1
            continue

        logger.info(f"Rebalance {i+1}/{len(rebalance_dates)}: {rebalance_date}, {len(universe)} members")

        # Generate signals
        aggregated = aggregator.aggregate(universe, rebalance_date)

        # Calculate coverage per signal
        signal_coverages = aggregated.diagnostics.get('signal_coverage', {})
        signal_coverage_history.append({
            'date': rebalance_date,
            **signal_coverages,
        })

        # Log coverages
        for name, coverage in signal_coverages.items():
            logger.debug(f"  {name}: {coverage:.1%} coverage")

        # Validate coverage
        is_valid, reason = coverage_validator.validate(signal_coverages, rebalance_date)
        if not is_valid:
            logger.warning(f"COVERAGE GATE: {reason}")
            skipped_coverage += 1
            continue

        # Get valid scores
        scores = aggregated.scores.dropna()
        if len(scores) < 20:
            logger.warning(f"Too few valid scores: {len(scores)}")
            continue

        # Construct portfolio using stabilized approach
        portfolio = portfolio_constructor.construct(
            scores=scores,
            as_of_date=rebalance_date,
            sectors=sectors,
        )

        new_weights = portfolio.positions

        # Calculate turnover
        turnover = portfolio_constructor.calculate_turnover(current_weights, new_weights)
        total_turnover += turnover
        rebalance_turnovers.append(turnover)

        # Apply transaction costs
        trade_value = capital * turnover * 2  # Both buy and sell
        cost = cost_model.calculate_costs(trade_value)

        logger.info(
            f"  Rebalance cost: ${cost:.2f} ({cost/capital*10000:.1f} bps), "
            f"turnover={turnover:.1%}, trades={len(new_weights)}"
        )

        # Log sector concentration
        max_sector = portfolio.diagnostics.get('max_sector_weight', 0)
        max_sector_name = portfolio.diagnostics.get('max_sector_name', 'N/A')
        logger.debug(f"  Max sector: {max_sector_name} at {max_sector:.1%}")

        # Update weights and store history
        current_weights = new_weights
        weights_history[rebalance_date] = new_weights.copy()  # Store for equity curve

    # Calculate performance using actual price data
    if len(portfolio_value_series) == 0:
        # Run equity curve using historical portfolio weights
        logger.info("\nCalculating equity curve using historical weights...")

        # Pivot prices_df from MultiIndex (date, ticker) to wide format (date x ticker)
        # prices_df has MultiIndex of (date, ticker) and 'closeadj' column
        price_series = prices_df['closeadj']
        if price_series.index.duplicated().any():
            price_series = price_series[~price_series.index.duplicated(keep='last')]
        prices_wide = price_series.unstack(level='ticker')

        # Ensure index is DatetimeIndex for proper comparison
        if not isinstance(prices_wide.index, pd.DatetimeIndex):
            prices_wide.index = pd.to_datetime(prices_wide.index)

        logger.info(f"  Prices wide shape: {prices_wide.shape}, date range: {prices_wide.index.min()} to {prices_wide.index.max()}")

        # Get sorted rebalance dates with weights
        active_rebalance_dates = sorted(weights_history.keys())
        logger.info(f"  Active rebalance dates with weights: {len(active_rebalance_dates)}")

        equity_curve = [initial_capital]
        portfolio_returns = []

        # Calculate MONTHLY returns between rebalance dates
        for i, rebal_date_str in enumerate(active_rebalance_dates[:-1]):
            rebal_date = pd.Timestamp(rebal_date_str)
            next_rebal_date_str = active_rebalance_dates[i + 1]
            next_rebal_date = pd.Timestamp(next_rebal_date_str)

            # Get the weights that were ACTIVE during this period
            period_weights = weights_history[rebal_date_str]

            if not period_weights:
                continue

            # Find closest price date to current rebalance date (look back)
            valid_start_dates = prices_wide.index[prices_wide.index <= rebal_date]
            if len(valid_start_dates) == 0:
                logger.debug(f"  No price data before {rebal_date_str}")
                continue
            start_price_date = valid_start_dates[-1]

            # Find closest price date to next rebalance date (look back)
            valid_end_dates = prices_wide.index[prices_wide.index <= next_rebal_date]
            if len(valid_end_dates) == 0:
                logger.debug(f"  No price data before {next_rebal_date_str}")
                continue
            end_price_date = valid_end_dates[-1]

            # Skip if dates are the same (no trading days in between)
            if start_price_date == end_price_date:
                continue

            # Calculate period return as weighted sum of individual stock returns
            period_return = 0.0
            weight_sum = 0.0
            for ticker, weight in period_weights.items():
                if ticker in prices_wide.columns:
                    start_price = prices_wide.loc[start_price_date, ticker]
                    end_price = prices_wide.loc[end_price_date, ticker]
                    if not pd.isna(start_price) and not pd.isna(end_price) and start_price > 0:
                        stock_return = (end_price / start_price) - 1
                        period_return += weight * stock_return
                        weight_sum += weight

            # Only count if we have significant weight coverage
            if weight_sum > 0.5:  # At least 50% of weights had valid prices
                # Scale return by actual weight coverage
                period_return = period_return / weight_sum * 1.0  # Assume missing weights had 0 return
                portfolio_returns.append(period_return)
                equity_curve.append(equity_curve[-1] * (1 + period_return))

        portfolio_returns = pd.Series(portfolio_returns)
        logger.info(f"  Equity curve: {len(equity_curve)} points, {len(portfolio_returns)} monthly returns")

    # Compute final metrics
    actual_rebalances = rebalance_count - skipped_warmup - skipped_coverage - skipped_universe

    # Annualized turnover
    years = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 365.25
    annualized_turnover = (total_turnover / actual_rebalances) * 12 if actual_rebalances > 0 else 0

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("STABILIZED BACKTEST SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total rebalances: {rebalance_count}")
    logger.info(f"  Skipped (warmup): {skipped_warmup}")
    logger.info(f"  Skipped (coverage): {skipped_coverage}")
    logger.info(f"  Skipped (universe): {skipped_universe}")
    logger.info(f"  Actual trades: {actual_rebalances}")
    logger.info(f"Total one-way turnover: {total_turnover:.1%}")
    logger.info(f"Annualized turnover: {annualized_turnover:.0%}")
    logger.info(f"Avg turnover per rebalance: {np.mean(rebalance_turnovers)*100:.1f}%" if rebalance_turnovers else "N/A")
    logger.info(f"Final holdings: {len(current_weights)} positions")
    logger.info(f"Holdings summary: {portfolio_constructor.get_holdings_summary()}")

    # Signal coverage summary
    coverage_df = pd.DataFrame(signal_coverage_history)
    logger.info("\nSignal Coverage Summary:")
    for col in coverage_df.columns:
        if col != 'date':
            logger.info(f"  {col}: mean={coverage_df[col].mean():.1%}, min={coverage_df[col].min():.1%}")

    # Signal contribution
    contribution_summary = contribution_tracker.get_summary()
    if contribution_summary:
        logger.info("\nSignal Contribution (Rank IC):")
        for signal, ic in contribution_summary.items():
            logger.info(f"  {signal}: {ic:.3f}")

    logger.info("=" * 70)

    # Build portfolio values series with proper datetime index
    # IMPORTANT: Use monthly frequency ('MS') since we have monthly data
    if len(equity_curve) > 0:
        # Create datetime index for equity curve (monthly data)
        values_series = pd.Series(
            equity_curve,
            index=pd.date_range(start=start_date, periods=len(equity_curve), freq='MS')
        )
    else:
        values_series = pd.Series([initial_capital])

    # Build returns series
    # IMPORTANT: Use monthly frequency since returns are monthly
    if len(portfolio_returns) > 0:
        returns_series = pd.Series(
            portfolio_returns.values if hasattr(portfolio_returns, 'values') else portfolio_returns,
            index=pd.date_range(start=start_date, periods=len(portfolio_returns), freq='MS')
        )
    else:
        returns_series = pd.Series(dtype=float)

    # Initialize defaults
    total_return = 0.0
    cagr = 0.0
    volatility = 0.0
    sharpe = 0.0
    sortino = 0.0
    downside_std = 0.0
    max_drawdown = 0.0
    calmar = 0.0
    var_95 = 0.0
    cvar_95 = 0.0
    cumulative = pd.Series(dtype=float)
    drawdown = pd.Series(dtype=float)

    # Calculate performance metrics
    if len(values_series) > 1 and len(returns_series) > 0:
        total_return = values_series.iloc[-1] / values_series.iloc[0] - 1

        # CAGR
        days = (values_series.index[-1] - values_series.index[0]).days
        years_calc = days / 365.25 if days > 0 else 1
        cagr = (1 + total_return) ** (1 / years_calc) - 1 if years_calc > 0 else 0

        # Volatility (annualized) - use sqrt(12) for monthly returns
        volatility = returns_series.std() * np.sqrt(12) if len(returns_series) > 0 else 0

        # Sharpe (assuming 0 risk-free rate) - use 12 for monthly annualization
        sharpe = (returns_series.mean() * 12) / (returns_series.std() * np.sqrt(12)) if returns_series.std() > 0 else 0

        # Sortino (downside deviation) - use sqrt(12) for monthly returns
        negative_returns = returns_series[returns_series < 0]
        downside_std = negative_returns.std() * np.sqrt(12) if len(negative_returns) > 0 else volatility
        sortino = (returns_series.mean() * 12) / downside_std if downside_std > 0 else 0

        # Drawdown
        cumulative = (1 + returns_series).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0

        # Calmar
        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

        # VaR and CVaR
        var_95 = returns_series.quantile(0.05) if len(returns_series) > 0 else 0
        cvar_95 = returns_series[returns_series <= var_95].mean() if len(returns_series[returns_series <= var_95]) > 0 else var_95

    # Print additional performance metrics
    logger.info(f"\nPerformance Metrics:")
    logger.info(f"  Total Return: {total_return:.2%}")
    logger.info(f"  CAGR: {cagr:.2%}")
    logger.info(f"  Sharpe Ratio: {sharpe:.2f}")
    logger.info(f"  Sortino Ratio: {sortino:.2f}")
    logger.info(f"  Max Drawdown: {max_drawdown:.2%}")
    logger.info(f"  Volatility: {volatility:.2%}")

    return BacktestResult(
        total_return=float(total_return),
        cagr=float(cagr),
        sharpe_ratio=float(sharpe),
        sortino_ratio=float(sortino),
        max_drawdown=float(max_drawdown),
        calmar_ratio=float(calmar),
        volatility=float(volatility),
        downside_deviation=float(downside_std) if downside_std > 0 else float(volatility),
        var_95=float(var_95),
        cvar_95=float(cvar_95),
        total_trades=actual_rebalances * 25,  # Approx trades
        total_turnover=float(total_turnover),
        total_costs=0.0,  # Costs tracked separately
        avg_positions=25.0,  # Target positions
        returns=returns_series,
        cumulative_returns=cumulative,
        drawdown_series=drawdown,
        portfolio_values=values_series,
        rebalances=[],  # Not tracking RebalanceRecord objects here
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        final_value=float(values_series.iloc[-1]) if len(values_series) > 0 else initial_capital,
        n_periods=actual_rebalances,
        diagnostics={
            'total_rebalances': rebalance_count,
            'actual_rebalances': actual_rebalances,
            'skipped_warmup': skipped_warmup,
            'skipped_coverage': skipped_coverage,
            'skipped_universe': skipped_universe,
            'total_turnover': total_turnover,
            'annualized_turnover': annualized_turnover,
            'signal_coverage': contribution_summary if contribution_summary else {},
        },
    )


if __name__ == '__main__':
    result = run_stabilized_backtest()
