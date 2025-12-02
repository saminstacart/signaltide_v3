#!/usr/bin/env python3
"""
V5 Comparison Matrix Runner.

Runs multiple V5 configuration variants and compares their performance against V4 baseline.

Configurations tested:
1. V4-Baseline: Original V4 parameters (equal weight signals)
2. V5-Hierarchical-Hard: Hierarchical scoring with hard quality gate
3. V5-Hierarchical-Soft: Hierarchical scoring with soft quality gate
4. V5-Full-Hard: All V5 changes + hard gate
5. V5-Full-Soft: All V5 changes + soft gate

Key V5 changes from V4:
- 40 positions (vs 25)
- Top 20% entry (vs 10%)
- Hierarchical signal logic (Quality -> Insider -> Momentum)
- No minimum holding period
"""

import sys
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from signaltide_v4.config.settings import get_settings
from signaltide_v4.config.v5_config import get_config, COMPARISON_CONFIGS
from signaltide_v4.data.market_data import MarketDataProvider
from signaltide_v4.data.fundamental_data import FundamentalDataProvider
from signaltide_v4.data.factor_data import FactorDataProvider
from signaltide_v4.signals.residual_momentum import ResidualMomentumSignal
from signaltide_v4.signals.quality import QualitySignal
from signaltide_v4.signals.insider import OpportunisticInsiderSignal
from signaltide_v4.portfolio.scoring import SignalAggregator
from signaltide_v4.portfolio.hierarchical_scorer import (
    HierarchicalScorer, GateMode, create_v5_scorer_hard, create_v5_scorer_soft
)
from signaltide_v4.portfolio.stabilized_construction import StabilizedPortfolioConstructor
from signaltide_v4.backtest.transaction_costs import TransactionCostModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


@dataclass
class BacktestMetrics:
    """Container for backtest performance metrics."""
    config_name: str
    cagr: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    avg_turnover: float
    total_return: float
    n_rebalances: int
    avg_positions: float

    # V5-specific metrics
    quality_pass_rate: Optional[float] = None  # For hierarchical configs
    top_insider_filter_rate: Optional[float] = None  # Key diagnostic

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DynamicSP500Provider:
    """Provides point-in-time S&P 500 membership."""

    def __init__(self, db_path: Optional[str] = None):
        settings = get_settings()
        self.db_path = db_path or settings.db_path
        self._cache = {}
        self._all_tickers = self._get_all_sp500_tickers()
        logger.info(f"DynamicSP500Provider: {len(self._all_tickers)} total tickers")

    def _get_all_sp500_tickers(self) -> List[str]:
        """Get all tickers that were ever in S&P 500."""
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
        return self._all_tickers


def run_single_config_backtest(
    config_name: str,
    start_date: str = '2015-07-01',
    end_date: str = '2024-12-31',
    initial_capital: float = 50_000,
    universe_provider: Optional[DynamicSP500Provider] = None,
    market_data: Optional[MarketDataProvider] = None,
    fundamental_data: Optional[FundamentalDataProvider] = None,
    factor_data: Optional[FactorDataProvider] = None,
    prices_df: Optional[pd.DataFrame] = None,
    sectors: Optional[Dict[str, str]] = None,
) -> BacktestMetrics:
    """
    Run backtest with a specific V5 configuration.

    Args:
        config_name: Name from COMPARISON_CONFIGS
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital
        universe_provider: Shared universe provider (for efficiency)
        market_data: Shared market data provider
        fundamental_data: Shared fundamental data provider
        factor_data: Shared factor data provider
        prices_df: Preloaded price data (for efficiency)
        sectors: Preloaded sector mapping

    Returns:
        BacktestMetrics with performance results
    """
    settings = get_settings()
    config = get_config(config_name)

    logger.info(f"\n{'='*70}")
    logger.info(f"RUNNING: {config_name}")
    logger.info(f"{'='*70}")

    # Get config parameters
    portfolio_cfg = config.get('portfolio', {})
    signals_cfg = config.get('signals', {})
    scorer_cfg = config.get('scorer', {})

    target_positions = portfolio_cfg.get('target_positions', 25)
    entry_percentile = portfolio_cfg.get('entry_percentile', 10)
    exit_percentile = portfolio_cfg.get('exit_percentile', 50)
    min_holding_months = portfolio_cfg.get('min_holding_months', 2)

    signal_logic = signals_cfg.get('logic', 'equal_weight')
    scorer_mode = scorer_cfg.get('mode', 'equal_weight')

    logger.info(f"  Positions: {target_positions}")
    logger.info(f"  Entry: Top {entry_percentile}%")
    logger.info(f"  Min hold: {min_holding_months} months")
    logger.info(f"  Signal logic: {signal_logic}")
    logger.info(f"  Scorer mode: {scorer_mode}")

    # Initialize providers if not passed
    if universe_provider is None:
        universe_provider = DynamicSP500Provider()
    if market_data is None:
        market_data = MarketDataProvider()
    if fundamental_data is None:
        fundamental_data = FundamentalDataProvider()
    if factor_data is None:
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

    # Choose scorer based on config
    hierarchical_scorer = None
    signal_aggregator = None

    if scorer_mode in ['hard', 'soft']:
        # Use hierarchical scorer
        gate_mode = GateMode.HARD if scorer_mode == 'hard' else GateMode.SOFT
        quality_threshold = scorer_cfg.get('quality_threshold_percentile', 40)
        insider_weight = scorer_cfg.get('insider_weight', 0.6)
        momentum_weight = scorer_cfg.get('momentum_weight', 0.4)
        soft_multiplier = scorer_cfg.get('soft_gate_multiplier', 0.5)

        hierarchical_scorer = HierarchicalScorer(
            quality_signal=quality_signal,
            insider_signal=insider_signal,
            momentum_signal=momentum_signal,
            gate_mode=gate_mode,
            insider_weight=insider_weight,
            momentum_weight=momentum_weight,
            quality_threshold_percentile=quality_threshold,
            soft_gate_multiplier=soft_multiplier,
        )
        logger.info(f"  Using HierarchicalScorer: {gate_mode.value} gate")
    else:
        # Use equal-weight aggregator
        signals = [momentum_signal, quality_signal, insider_signal]
        signal_aggregator = SignalAggregator(
            signals=signals,
            weights={'residual_momentum': 1/3, 'quality': 1/3, 'opportunistic_insider': 1/3},
            min_signals_required=2,
        )
        logger.info(f"  Using SignalAggregator: equal weights")

    # Portfolio constructor with config-specific parameters
    portfolio_constructor = StabilizedPortfolioConstructor(
        market_data=market_data,
        params={
            'entry_percentile': entry_percentile,
            'exit_percentile': exit_percentile,
            'min_holding_months': min_holding_months,
            'hard_sector_cap': 0.35,
            'smoothing_window': 3,
        },
    )

    # Transaction costs
    cost_model = TransactionCostModel(cost_bps=settings.transaction_cost_bps)

    # Generate rebalance dates
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    rebalance_dates = pd.date_range(start=start_dt, end=end_dt, freq='ME')
    rebalance_dates = [d.strftime('%Y-%m-%d') for d in rebalance_dates]

    # Load prices if not provided
    if prices_df is None:
        all_tickers = universe_provider.all_tickers
        prices_df = market_data.get_prices(all_tickers, start_date, end_date)

    # Load sectors if not provided
    if sectors is None:
        import sqlite3
        conn = sqlite3.connect(settings.db_path)
        sector_df = pd.read_sql_query("SELECT ticker, sector FROM sharadar_tickers", conn)
        conn.close()
        sectors = dict(zip(sector_df['ticker'], sector_df['sector']))

    # Backtest state
    warmup_months = 6
    capital = initial_capital
    portfolio_returns = []
    current_weights: Dict[str, float] = {}
    rebalance_count = 0
    total_turnover = 0.0
    positions_history = []

    # V5-specific tracking
    quality_pass_rates = []
    top_insider_filter_rates = []

    for i, rebalance_date in enumerate(rebalance_dates):
        # Get universe
        universe = universe_provider.get_members(rebalance_date)

        # Skip warmup
        if i < warmup_months:
            continue

        if len(universe) < 50:
            continue

        # Generate scores
        if hierarchical_scorer is not None:
            # Hierarchical scoring
            result = hierarchical_scorer.score(universe, rebalance_date)
            scores = result.scores

            # Track V5 diagnostics
            if result.n_quality_passed + result.n_quality_failed > 0:
                pass_rate = result.n_quality_passed / (result.n_quality_passed + result.n_quality_failed)
                quality_pass_rates.append(pass_rate)

            # Check top insider filter rate
            diag = result.diagnostics.get('top_insider_in_failed_quality', {})
            filter_rate = diag.get('pct_top_insider_filtered', 0)
            if filter_rate is not None:
                top_insider_filter_rates.append(filter_rate)

        else:
            # Equal weight aggregation
            aggregated = signal_aggregator.aggregate(universe, rebalance_date)
            scores = aggregated.scores

        # Filter valid scores (exclude -999 exclusion scores)
        valid_scores = scores[scores > -900].dropna()

        if len(valid_scores) < 20:
            continue

        # Construct portfolio
        portfolio = portfolio_constructor.construct(
            scores=valid_scores,
            as_of_date=rebalance_date,
            sectors=sectors,
        )

        new_weights = portfolio.positions
        positions_history.append(len(new_weights))

        # Calculate turnover
        turnover = sum(
            abs(new_weights.get(t, 0) - current_weights.get(t, 0))
            for t in set(new_weights.keys()) | set(current_weights.keys())
        ) / 2
        total_turnover += turnover

        # Get returns for next period
        next_idx = i + 1
        if next_idx >= len(rebalance_dates):
            break

        next_date = rebalance_dates[next_idx]

        # Calculate period return
        period_return = 0.0
        for ticker, weight in new_weights.items():
            try:
                # Get prices
                start_price = prices_df.xs(ticker, level='ticker').loc[rebalance_date:]['close'].iloc[0]
                end_price = prices_df.xs(ticker, level='ticker').loc[:next_date]['close'].iloc[-1]
                ticker_return = (end_price / start_price) - 1
                period_return += weight * ticker_return
            except (KeyError, IndexError):
                continue

        # Apply transaction costs (turnover * cost_bps / 10000)
        period_return -= turnover * (cost_model.base_cost_bps / 10000)
        portfolio_returns.append(period_return)

        current_weights = new_weights
        rebalance_count += 1

    # Calculate metrics
    if len(portfolio_returns) < 12:
        logger.warning(f"{config_name}: Too few returns ({len(portfolio_returns)})")
        return BacktestMetrics(
            config_name=config_name,
            cagr=0, volatility=0, sharpe_ratio=0, sortino_ratio=0,
            max_drawdown=0, calmar_ratio=0, win_rate=0, avg_turnover=0,
            total_return=0, n_rebalances=rebalance_count, avg_positions=0,
        )

    returns_series = pd.Series(portfolio_returns)

    # CAGR
    total_return = (1 + returns_series).prod() - 1
    n_years = len(returns_series) / 12
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Volatility (annualized)
    volatility = returns_series.std() * np.sqrt(12)

    # Sharpe (assuming 4% risk-free)
    rf_monthly = 0.04 / 12
    excess_returns = returns_series - rf_monthly
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(12) if excess_returns.std() > 0 else 0

    # Sortino
    downside_returns = returns_series[returns_series < 0]
    downside_std = downside_returns.std() * np.sqrt(12) if len(downside_returns) > 0 else 0
    sortino = (returns_series.mean() * 12) / downside_std if downside_std > 0 else 0

    # Max drawdown
    cum_returns = (1 + returns_series).cumprod()
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Calmar
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

    # Win rate
    win_rate = (returns_series > 0).mean()

    # Avg turnover
    avg_turnover = total_turnover / rebalance_count if rebalance_count > 0 else 0

    # Avg positions
    avg_positions = np.mean(positions_history) if positions_history else 0

    # V5 metrics
    avg_quality_pass_rate = np.mean(quality_pass_rates) if quality_pass_rates else None
    avg_insider_filter_rate = np.mean(top_insider_filter_rates) if top_insider_filter_rates else None

    logger.info(f"\n{config_name} Results:")
    logger.info(f"  CAGR: {cagr:.2%}")
    logger.info(f"  Sharpe: {sharpe:.2f}")
    logger.info(f"  Max DD: {max_drawdown:.2%}")
    logger.info(f"  Avg Positions: {avg_positions:.0f}")
    if avg_quality_pass_rate is not None:
        logger.info(f"  Quality Pass Rate: {avg_quality_pass_rate:.1%}")
    if avg_insider_filter_rate is not None:
        logger.info(f"  Top Insider Filter Rate: {avg_insider_filter_rate:.1%}")

    return BacktestMetrics(
        config_name=config_name,
        cagr=cagr,
        volatility=volatility,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_drawdown,
        calmar_ratio=calmar,
        win_rate=win_rate,
        avg_turnover=avg_turnover,
        total_return=total_return,
        n_rebalances=rebalance_count,
        avg_positions=avg_positions,
        quality_pass_rate=avg_quality_pass_rate,
        top_insider_filter_rate=avg_insider_filter_rate,
    )


def run_comparison_matrix(
    configs: Optional[List[str]] = None,
    start_date: str = '2015-07-01',
    end_date: str = '2024-12-31',
    initial_capital: float = 50_000,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run multiple configurations and compare results.

    Args:
        configs: List of config names to test. Default: key comparison configs
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital
        output_path: Path to save results JSON

    Returns:
        DataFrame with comparison metrics
    """
    # Default configs to compare
    if configs is None:
        configs = [
            'V4-Baseline',           # Original V4 (baseline)
            'V5-Hierarchical-Hard',  # Hierarchical with hard gate
            'V5-Hierarchical-Soft',  # Hierarchical with soft gate
            'V5-Full-Hard',          # All V5 changes + hard gate
            'V5-Full-Soft',          # All V5 changes + soft gate
        ]

    logger.info("=" * 70)
    logger.info("V5 COMPARISON MATRIX")
    logger.info("=" * 70)
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Configs: {configs}")
    logger.info("=" * 70)

    # Initialize shared providers for efficiency
    universe_provider = DynamicSP500Provider()
    market_data = MarketDataProvider()
    fundamental_data = FundamentalDataProvider()
    factor_data = FactorDataProvider()

    # Preload price data
    logger.info("Preloading price data...")
    all_tickers = universe_provider.all_tickers
    prices_df = market_data.get_prices(all_tickers, start_date, end_date)
    logger.info(f"Loaded prices for {len(all_tickers)} tickers")

    # Load sectors
    settings = get_settings()
    import sqlite3
    conn = sqlite3.connect(settings.db_path)
    sector_df = pd.read_sql_query("SELECT ticker, sector FROM sharadar_tickers", conn)
    conn.close()
    sectors = dict(zip(sector_df['ticker'], sector_df['sector']))

    # Run each config
    results = []
    for config_name in configs:
        try:
            metrics = run_single_config_backtest(
                config_name=config_name,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                universe_provider=universe_provider,
                market_data=market_data,
                fundamental_data=fundamental_data,
                factor_data=factor_data,
                prices_df=prices_df,
                sectors=sectors,
            )
            results.append(metrics)
        except Exception as e:
            logger.error(f"Error running {config_name}: {e}")
            import traceback
            traceback.print_exc()

    # Create comparison DataFrame
    df = pd.DataFrame([r.to_dict() for r in results])
    df = df.set_index('config_name')

    # Sort by Sharpe
    df = df.sort_values('sharpe_ratio', ascending=False)

    # Print summary
    print("\n" + "=" * 80)
    print("V5 COMPARISON MATRIX - RESULTS")
    print("=" * 80)
    print(df[['cagr', 'sharpe_ratio', 'max_drawdown', 'avg_positions',
              'quality_pass_rate', 'top_insider_filter_rate']].to_string())
    print("=" * 80)

    # Save results
    if output_path:
        results_dict = {
            'run_time': datetime.now().isoformat(),
            'period': {'start': start_date, 'end': end_date},
            'initial_capital': initial_capital,
            'results': {r.config_name: r.to_dict() for r in results},
        }
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")

    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run V5 Comparison Matrix')
    parser.add_argument('--start', default='2015-07-01', help='Start date')
    parser.add_argument('--end', default='2024-12-31', help='End date')
    parser.add_argument('--capital', type=float, default=50_000, help='Initial capital')
    parser.add_argument('--output', default='results/v5_comparison_matrix.json', help='Output path')
    parser.add_argument('--configs', nargs='+', help='Specific configs to run')
    parser.add_argument('--quick', action='store_true', help='Quick run (fewer configs)')

    args = parser.parse_args()

    # Quick mode runs fewer configs
    if args.quick:
        configs = ['V4-Baseline', 'V5-Full-Hard', 'V5-Full-Soft']
    else:
        configs = args.configs

    df = run_comparison_matrix(
        configs=configs,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        output_path=args.output,
    )

    print("\nDone!")
