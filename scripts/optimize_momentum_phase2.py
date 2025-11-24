"""
Phase 2 Momentum Optimization - Grid Search

Systematic hyperparameter search for InstitutionalMomentum signal.

Tests small discrete grid:
- Formation period: {126, 189, 252} days
- Skip period: {5, 10, 21} days
- Winsorization: {1%, 5%, 10%}
Total: 27 configurations

Outputs:
- results/momentum_phase2_trials.csv - All trial results
- results/momentum_phase2_trials.md - Human-readable table
- results/MOMENTUM_PHASE2_SUMMARY.md - Executive summary with passing configs

Usage:
    python3 scripts/optimize_momentum_phase2.py

Success criteria:
- At least 1 config passes all 4 acceptance gates
- OOS Sharpe ≥ 0.2 for passing configs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from data.data_manager import DataManager
from core.universe_manager import UniverseManager
from signals.momentum.institutional_momentum import InstitutionalMomentum
from core.schedules import get_rebalance_dates
from validation.simple_validation import (
    compute_basic_metrics,
    compute_monthly_returns,
    compute_regime_metrics,
    simple_deflated_sharpe,
    check_acceptance_gates
)
from config import get_logger

logger = get_logger(__name__)


# Fixed random seed for reproducibility
np.random.seed(42)


# ==============================================================================
# HYPERPARAMETER GRID
# ==============================================================================

FORMATION_PERIODS = [126, 189, 252]  # 6, 9, 12 months
SKIP_PERIODS = [5, 10, 21]           # 1 week, 2 weeks, 1 month
WINSOR_PCTS = [1, 5, 10]             # Two-sided winsorization percentiles

# Fixed parameters
START_DATE = '2015-04-01'  # Match Phase 1.5 - first date with S&P 500 PIT data
END_DATE = '2024-12-31'
IS_CUTOFF = '2022-12-31'  # In-sample ends here, OOS starts 2023-01-01
OOS_START = '2023-01-01'

INITIAL_CAPITAL = 50000


class MomentumPhase2Optimizer:
    """
    Phase 2 grid search optimizer for InstitutionalMomentum.

    Systematically tests all combinations of hyperparameters and
    evaluates each on in-sample, out-of-sample, and regime metrics.
    """

    def __init__(self):
        """Initialize optimizer."""
        self.dm = DataManager()
        self.um = UniverseManager(self.dm)

        self.start_date = START_DATE
        self.end_date = END_DATE
        self.is_cutoff = IS_CUTOFF
        self.oos_start = OOS_START

        # Calculate total number of trials (for DSR)
        self.n_trials = len(FORMATION_PERIODS) * len(SKIP_PERIODS) * len(WINSOR_PCTS)

        logger.info("=" * 80)
        logger.info("Phase 2 Momentum Optimization - Grid Search")
        logger.info("=" * 80)
        logger.info(f"Full period: {self.start_date} to {self.end_date}")
        logger.info(f"In-sample: {self.start_date} to {self.is_cutoff}")
        logger.info(f"Out-of-sample: {self.oos_start} to {self.end_date}")
        logger.info(f"Total configurations to test: {self.n_trials}")
        logger.info("=" * 80)

    def run_optimization(self) -> pd.DataFrame:
        """
        Run grid search over all hyperparameter combinations.

        Returns:
            DataFrame with one row per configuration and all metrics
        """
        results = []

        for i, formation in enumerate(FORMATION_PERIODS):
            for skip in SKIP_PERIODS:
                for winsor in WINSOR_PCTS:
                    config_num = len(results) + 1
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Config {config_num}/{self.n_trials}: formation={formation}, skip={skip}, winsor={winsor}%")
                    logger.info(f"{'='*60}")

                    try:
                        metrics = self._evaluate_config(formation, skip, winsor)
                        metrics['formation_period'] = formation
                        metrics['skip_period'] = skip
                        metrics['winsorize_pct'] = winsor
                        results.append(metrics)

                        # Log key metrics
                        logger.info(f"  Full Sharpe: {metrics['full_sharpe']:.3f}")
                        logger.info(f"  IS Sharpe: {metrics['is_sharpe']:.3f}")
                        logger.info(f"  OOS Sharpe: {metrics['oos_sharpe']:.3f}")
                        logger.info(f"  DSR: {metrics['dsr']:.3f}")
                        logger.info(f"  Passes all gates: {metrics['passes_all_gates']}")

                    except Exception as e:
                        logger.error(f"  ERROR: {e}")
                        # Still add the config with NaN metrics
                        results.append({
                            'formation_period': formation,
                            'skip_period': skip,
                            'winsorize_pct': winsor,
                            'error': str(e)
                        })

        return pd.DataFrame(results)

    def _evaluate_config(self, formation: int, skip: int, winsor: int) -> Dict:
        """
        Evaluate a single hyperparameter configuration.

        Args:
            formation: Formation period (days)
            skip: Skip period (days)
            winsor: Winsorization percentile (one-sided)

        Returns:
            Dict with all metrics for this configuration
        """
        # 1. Build momentum signal with these parameters
        params = {
            'formation_period': formation,
            'skip_period': skip,
            'winsorize_pct': [winsor, 100 - winsor],
            'quintiles': True,
            'rebalance_frequency': 'monthly'
        }

        # 2. Run backtest to get equity curve
        equity_curve = self._run_backtest(params)

        if equity_curve is None or len(equity_curve) < 10:
            raise ValueError("Backtest failed to produce valid equity curve")

        # 3. Split into IS and OOS (ensure we're comparing pandas Timestamps)
        is_cutoff_ts = pd.Timestamp(self.is_cutoff)
        oos_start_ts = pd.Timestamp(self.oos_start)

        is_equity = equity_curve[equity_curve.index <= is_cutoff_ts]
        oos_equity = equity_curve[equity_curve.index >= oos_start_ts]

        # 4. Compute basic metrics for each sample
        full_metrics = compute_basic_metrics(equity_curve)
        is_metrics = compute_basic_metrics(is_equity)
        oos_metrics = compute_basic_metrics(oos_equity)

        # 5. Compute regime metrics (on full sample)
        regime_metrics = compute_regime_metrics(equity_curve)

        # 6. Compute DSR
        dsr = simple_deflated_sharpe(
            sharpe=full_metrics['sharpe'],
            n_months=full_metrics['num_months'],
            n_trials=self.n_trials
        )

        # 7. Check acceptance gates
        gates = check_acceptance_gates(
            full_sharpe=full_metrics['sharpe'],
            oos_sharpe=oos_metrics['sharpe'],
            regime_metrics=regime_metrics,
            dsr=dsr
        )

        # 8. Package results
        result = {
            # Full sample metrics
            'full_sharpe': full_metrics['sharpe'],
            'full_annual_return': full_metrics['annual_return'],
            'full_volatility': full_metrics['volatility'],
            'full_max_drawdown': full_metrics['max_drawdown'],

            # In-sample metrics
            'is_sharpe': is_metrics['sharpe'],
            'is_annual_return': is_metrics['annual_return'],

            # Out-of-sample metrics
            'oos_sharpe': oos_metrics['sharpe'],
            'oos_annual_return': oos_metrics['annual_return'],

            # Deflated Sharpe
            'dsr': dsr,

            # Regime metrics
            'regime_covid_return': regime_metrics['covid']['mean_return'],
            'regime_covid_sharpe': regime_metrics['covid']['sharpe'],
            'regime_2022_return': regime_metrics['bear_2022']['mean_return'],
            'regime_2022_sharpe': regime_metrics['bear_2022']['sharpe'],
            'regime_recent_return': regime_metrics['recent']['mean_return'],
            'regime_recent_sharpe': regime_metrics['recent']['sharpe'],

            # Gates
            **gates
        }

        return result

    def _run_backtest(self, params: Dict) -> pd.Series:
        """
        Run backtest for given momentum parameters.

        Simplified version - builds decile portfolios and tracks equity.
        Reuses pattern from diagnose_momentum_phase1.py.

        Args:
            params: Momentum signal parameters

        Returns:
            Series of daily equity values (DatetimeIndex)
        """
        # Build universe (S&P 500 PIT)
        universe = self.um.get_universe(
            universe_type='sp500_actual',
            as_of_date=self.start_date,
            min_price=5.0
        )

        logger.debug(f"  Universe: {len(universe)} stocks")

        # Get monthly rebalance dates
        rebalance_dates = get_rebalance_dates(
            schedule='M',
            dm=self.dm,
            start_date=self.start_date,
            end_date=self.end_date
        )

        logger.debug(f"  Rebalance dates: {len(rebalance_dates)} months")

        # Fetch price data (need extra history for momentum calculation)
        lookback_buffer = timedelta(days=400)
        start_dt = pd.Timestamp(self.start_date)
        price_start_date = (start_dt - lookback_buffer).strftime('%Y-%m-%d')

        prices_dict = {}
        for ticker in universe:
            try:
                prices = self.dm.get_prices(ticker, price_start_date, self.end_date)
                if len(prices) > 0:
                    prices_dict[ticker] = prices
            except:
                pass

        logger.debug(f"  Price data loaded: {len(prices_dict)} stocks")

        # Calculate momentum for each stock
        momentum_dict = {}
        for ticker, prices in prices_dict.items():
            if 'close' not in prices.columns:
                continue
            if len(prices) < params['formation_period'] + params['skip_period']:
                continue

            # 12-1 style momentum
            mom = prices['close'].pct_change(
                periods=params['formation_period'],
                fill_method=None
            ).shift(params['skip_period'])

            momentum_dict[ticker] = mom

        # Convert to DataFrame
        momentum_df = pd.DataFrame(momentum_dict)
        momentum_df = momentum_df.sort_index()

        # Handle duplicate dates (from corporate actions)
        if momentum_df.index.duplicated().any():
            momentum_df = momentum_df[~momentum_df.index.duplicated(keep='last')]

        logger.debug(f"  Momentum calculated: {momentum_df.shape}")

        # Build price DataFrame for returns calculation
        prices_close = {}
        for ticker, prices in prices_dict.items():
            prices_close[ticker] = prices['close']
        prices_df = pd.DataFrame(prices_close)

        # Build simple equal-weight portfolio rebalanced monthly
        # Track portfolio value over time
        equity_series = []
        portfolio_value = INITIAL_CAPITAL
        current_holdings = {}  # ticker -> shares

        for i, rebal_date in enumerate(pd.DatetimeIndex(rebalance_dates)):
            # Get momentum scores
            valid_idx = momentum_df.index[momentum_df.index <= rebal_date]
            if len(valid_idx) == 0:
                continue

            mom_date = valid_idx[-1]
            mom_today = momentum_df.loc[mom_date].dropna()

            if len(mom_today) < 20:  # Need minimum universe
                continue

            # Winsorize
            lower_bound = mom_today.quantile(params['winsorize_pct'][0] / 100)
            upper_bound = mom_today.quantile(params['winsorize_pct'][1] / 100)
            mom_wins = mom_today.clip(lower=lower_bound, upper=upper_bound)

            # Take top quintile (highest momentum)
            sorted_mom = mom_wins.sort_values(ascending=False)
            quintile_size = len(sorted_mom) // 5
            top_quintile = sorted_mom.iloc[:quintile_size].index.tolist()

            # Equal-weight allocation
            if len(top_quintile) > 0:
                weight_per_stock = 1.0 / len(top_quintile)

                # Get prices on rebalance date
                if rebal_date in prices_df.index:
                    rebal_prices = prices_df.loc[rebal_date]
                else:
                    # Use nearest prior date
                    valid_price_idx = prices_df.index[prices_df.index <= rebal_date]
                    if len(valid_price_idx) > 0:
                        rebal_prices = prices_df.loc[valid_price_idx[-1]]
                    else:
                        continue

                # Rebalance portfolio
                current_holdings = {}
                for ticker in top_quintile:
                    # rebal_prices is a Series with tickers as index
                    if ticker in rebal_prices:
                        price = rebal_prices[ticker]
                        # Handle case where price might be a Series (duplicate indices)
                        if isinstance(price, pd.Series):
                            price = price.iloc[-1]
                        if pd.notna(price) and price > 0:
                            target_value = portfolio_value * weight_per_stock
                            shares = target_value / price
                            current_holdings[ticker] = shares

            # Track equity daily until next rebalance
            if i + 1 < len(rebalance_dates):
                next_rebal = pd.Timestamp(rebalance_dates[i + 1])
            else:
                next_rebal = pd.Timestamp(self.end_date)

            # Get daily prices between rebalances
            price_window = prices_df[
                (prices_df.index >= rebal_date) &
                (prices_df.index <= next_rebal)
            ]

            for date in price_window.index:
                # Calculate portfolio value
                total_value = 0
                for ticker, shares in current_holdings.items():
                    if ticker in price_window.columns:
                        price = price_window.loc[date, ticker]
                        # Handle duplicate date indices
                        if isinstance(price, pd.Series):
                            price = price.iloc[-1]
                        if pd.notna(price):
                            total_value += shares * price

                equity_series.append({
                    'date': date,
                    'equity': total_value if total_value > 0 else portfolio_value
                })

            # Update portfolio value for next iteration
            if equity_series:
                portfolio_value = equity_series[-1]['equity']

        # Convert to Series
        if not equity_series:
            logger.warning("  No equity data generated!")
            return None

        equity_df = pd.DataFrame(equity_series)
        equity_curve = equity_df.set_index('date')['equity']

        logger.debug(f"  Equity curve: {len(equity_curve)} days")

        return equity_curve


def main():
    """Run Phase 2 optimization."""
    logger.info("\nStarting Phase 2 Momentum Optimization...")

    # Run grid search
    optimizer = MomentumPhase2Optimizer()
    results_df = optimizer.run_optimization()

    # Save results to CSV
    csv_path = Path('results/momentum_phase2_trials.csv')
    results_df.to_csv(csv_path, index=False)
    logger.info(f"\nResults saved to: {csv_path}")

    # Generate markdown reports
    _generate_trials_report(results_df)
    _generate_summary_report(results_df)

    logger.info("\nPhase 2 optimization complete!")


def _generate_trials_report(results_df: pd.DataFrame):
    """Generate detailed trials report in markdown."""
    output_path = Path('results/momentum_phase2_trials.md')

    with open(output_path, 'w') as f:
        f.write("# Momentum Phase 2 - All Trial Results\n\n")
        f.write(f"**Date:** {datetime.now().isoformat()}\n")
        f.write(f"**Total Configurations:** {len(results_df)}\n\n")
        f.write("---\n\n")

        # Sort by OOS Sharpe
        sorted_df = results_df.sort_values('oos_sharpe', ascending=False, na_position='last')

        f.write("## All Configurations (Sorted by OOS Sharpe)\n\n")
        f.write("| Rank | Formation | Skip | Winsor | Full Sharpe | IS Sharpe | OOS Sharpe | DSR | Gates |\n")
        f.write("|------|-----------|------|--------|-------------|-----------|------------|-----|-------|\n")

        for rank, (_, row) in enumerate(sorted_df.iterrows(), 1):
            gates_str = "✅" if row.get('passes_all_gates', False) else "❌"
            f.write(
                f"| {rank} | {row['formation_period']} | {row['skip_period']} | "
                f"{row['winsorize_pct']}% | {row.get('full_sharpe', 0):.3f} | "
                f"{row.get('is_sharpe', 0):.3f} | {row.get('oos_sharpe', 0):.3f} | "
                f"{row.get('dsr', 0):.3f} | {gates_str} |\n"
            )

    logger.info(f"Trials report saved to: {output_path}")


def _generate_summary_report(results_df: pd.DataFrame):
    """Generate executive summary report."""
    output_path = Path('results/MOMENTUM_PHASE2_SUMMARY.md')

    # Filter passing configs
    passing = results_df[results_df.get('passes_all_gates', False) == True]
    conditional = results_df[
        (results_df.get('passes_all_gates', False) == False) &
        (
            results_df.get('passes_full_gate', False) +
            results_df.get('passes_oos_gate', False) +
            results_df.get('passes_regime_gate', False) +
            results_df.get('passes_dsr_gate', False) >= 3
        )
    ]

    with open(output_path, 'w') as f:
        f.write("# Momentum Phase 2 Optimization - Summary\n\n")
        f.write(f"**Date:** {datetime.now().isoformat()}\n")
        f.write(f"**Period:** {START_DATE} to {END_DATE}\n")
        f.write(f"**In-Sample:** {START_DATE} to {IS_CUTOFF}\n")
        f.write(f"**Out-of-Sample:** {OOS_START} to {END_DATE}\n\n")
        f.write("---\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total configurations tested:** {len(results_df)}\n")
        f.write(f"- **Configs passing all gates:** {len(passing)}\n")
        f.write(f"- **Configs passing 3/4 gates (conditional):** {len(conditional)}\n\n")

        if len(passing) == 0:
            f.write("**⚠️ NO CONFIGURATIONS PASSED ALL 4 GATES**\n\n")
            f.write("This suggests either:\n")
            f.write("1. Acceptance thresholds are too strict\n")
            f.write("2. Hyperparameter search space needs expansion\n")
            f.write("3. Momentum premium may be weaker than Phase 1.5 suggested\n\n")
        else:
            f.write(f"**✅ SUCCESS: {len(passing)} configuration(s) passed all gates**\n\n")

        # Passing configurations table
        if len(passing) > 0:
            f.write("## Configurations Passing All Gates\n\n")
            f.write("**Sorted by OOS Sharpe (descending):**\n\n")
            f.write("| Rank | Formation | Skip | Winsor | Full Sharpe | OOS Sharpe | DSR | Recent Regime Sharpe |\n")
            f.write("|------|-----------|------|--------|-------------|------------|-----|---------------------|\n")

            passing_sorted = passing.sort_values('oos_sharpe', ascending=False)
            for rank, (_, row) in enumerate(passing_sorted.iterrows(), 1):
                f.write(
                    f"| {rank} | {row['formation_period']} | {row['skip_period']} | "
                    f"{row['winsorize_pct']}% | {row['full_sharpe']:.3f} | "
                    f"{row['oos_sharpe']:.3f} | {row['dsr']:.3f} | "
                    f"{row['regime_recent_sharpe']:.3f} |\n"
                )
            f.write("\n")

        # Conditional configs
        if len(conditional) > 0:
            f.write("## Configurations Passing 3/4 Gates (Conditional)\n\n")
            f.write("**For manual review:**\n\n")
            f.write("| Formation | Skip | Winsor | Full Sharpe | OOS Sharpe | DSR | Gates Passed |\n")
            f.write("|-----------|------|--------|-------------|------------|-----|-------------|\n")

            for _, row in conditional.iterrows():
                gates_passed = sum([
                    row.get('passes_full_gate', False),
                    row.get('passes_oos_gate', False),
                    row.get('passes_regime_gate', False),
                    row.get('passes_dsr_gate', False)
                ])
                f.write(
                    f"| {row['formation_period']} | {row['skip_period']} | "
                    f"{row['winsorize_pct']}% | {row['full_sharpe']:.3f} | "
                    f"{row['oos_sharpe']:.3f} | {row['dsr']:.3f} | {gates_passed}/4 |\n"
                )
            f.write("\n")

        # Best overall config (by OOS Sharpe)
        best_config = results_df.loc[results_df['oos_sharpe'].idxmax()]
        f.write("## Best Configuration by OOS Sharpe\n\n")
        f.write(f"- **Formation period:** {best_config['formation_period']} days\n")
        f.write(f"- **Skip period:** {best_config['skip_period']} days\n")
        f.write(f"- **Winsorization:** {best_config['winsorize_pct']}%\n")
        f.write(f"- **Full Sharpe:** {best_config['full_sharpe']:.3f}\n")
        f.write(f"- **OOS Sharpe:** {best_config['oos_sharpe']:.3f}\n")
        f.write(f"- **DSR:** {best_config['dsr']:.3f}\n")
        f.write(f"- **Passes all gates:** {'Yes' if best_config['passes_all_gates'] else 'No'}\n\n")

        f.write("---\n\n")
        f.write("**Next Steps:**\n\n")
        if len(passing) > 0:
            f.write("1. Manual review of passing configuration(s)\n")
            f.write("2. Run Phase 1.5 diagnostics on selected config\n")
            f.write("3. Proceed to Phase 3 (Ensemble Design) if validated\n")
        else:
            f.write("1. Review why no configs passed\n")
            f.write("2. Consider relaxing acceptance thresholds\n")
            f.write("3. Or expand hyperparameter search space\n")

    logger.info(f"Summary report saved to: {output_path}")


if __name__ == '__main__':
    main()
