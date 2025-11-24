"""
Phase 2.1 Momentum Optimization - Optuna Bayesian Search

Advanced hyperparameter optimization using Optuna's Bayesian search.
This is the "real" optimizer after the Phase 2.0 grid smoke test.

Search space (Momentum-only, no portfolio/risk params):
- formation_days: [126, 378] step 7 (6-18 months)
- skip_days: [0, 42] step 3 (0-2 months)
- winsorize_pct: [1.0, 10.0] continuous

Objective: Maximize OOS Sharpe (2023-2024) subject to acceptance gates

Gates (same philosophy as Phase 2.0):
1. OOS Sharpe > 0.2
2. IS/OOS consistency (OOS not >50% below IS, both positive)
3. Max drawdown < 30% (OOS period)
4. Recent regime (2023-2024) viable (Sharpe > 0.2, returns > 0)

Outputs:
- results/momentum_phase2_optuna_trials.csv - All trial results
- results/momentum_phase2_optuna_summary.md - Top performers
- results/MOMENTUM_PHASE2_SHORTLIST.json - 1-3 shortlisted configs

Usage:
    python3 scripts/optimize_momentum_phase2_optuna.py [--n-trials 100]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
import json
import argparse
warnings.filterwarnings('ignore')

# Optuna
try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("ERROR: Optuna not installed. Run: pip install optuna")
    sys.exit(1)

from data.data_manager import DataManager
from core.universe_manager import UniverseManager
from signals.momentum.institutional_momentum import InstitutionalMomentum
from core.schedules import get_rebalance_dates
from validation.simple_validation import (
    compute_basic_metrics,
    compute_regime_metrics
)
from config import get_logger

logger = get_logger(__name__)


# Fixed random seed for reproducibility
np.random.seed(42)


# ==============================================================================
# FIXED CONFIGURATION
# ==============================================================================

START_DATE = '2015-04-01'  # Match Phase 1.5
END_DATE = '2024-12-31'
IS_CUTOFF = '2022-12-31'  # In-sample ends here, OOS starts 2023-01-01
OOS_START = '2023-01-01'

INITIAL_CAPITAL = 50000


# ==============================================================================
# ACCEPTANCE GATES
# ==============================================================================

def check_gates(
    is_sharpe: float,
    oos_sharpe: float,
    oos_max_dd: float,
    regime_recent_sharpe: float,
    regime_recent_return: float
) -> Dict[str, bool]:
    """
    Check if trial passes all acceptance gates.

    Gates:
    1. OOS Sharpe > 0.2
    2. IS and OOS both positive, OOS not more than 50% below IS
    3. Max drawdown (OOS) < 30%
    4. Recent regime (2023-2024): Sharpe > 0.2 and mean return > 0

    Args:
        is_sharpe: In-sample Sharpe ratio
        oos_sharpe: Out-of-sample Sharpe ratio
        oos_max_dd: Out-of-sample max drawdown (negative)
        regime_recent_sharpe: Recent regime Sharpe
        regime_recent_return: Recent regime mean monthly return

    Returns:
        Dict with gate pass/fail flags
    """
    # Gate 1: OOS Sharpe
    gate1 = oos_sharpe > 0.2

    # Gate 2: IS/OOS consistency
    gate2 = (
        is_sharpe > 0 and
        oos_sharpe > 0 and
        oos_sharpe >= (is_sharpe * 0.5)  # OOS not more than 50% below IS
    )

    # Gate 3: Max drawdown
    gate3 = oos_max_dd > -0.30  # Drawdown is negative, so -0.30 = 30% DD

    # Gate 4: Recent regime
    gate4 = (
        regime_recent_sharpe > 0.2 and
        regime_recent_return > 0
    )

    passes_all = gate1 and gate2 and gate3 and gate4

    return {
        'passes_oos_sharpe_gate': gate1,
        'passes_consistency_gate': gate2,
        'passes_drawdown_gate': gate3,
        'passes_recent_regime_gate': gate4,
        'passes_all_gates': passes_all
    }


# ==============================================================================
# BACKTEST ENGINE (Reused from Phase 2.0)
# ==============================================================================

class MomentumBacktester:
    """
    Simplified backtest engine for Momentum optimization.

    Reuses the same logic as Phase 2.0 grid search for consistency.
    """

    def __init__(self, debug: bool = False):
        """Initialize backtester.

        Args:
            debug: If True, enable detailed logging for debugging
        """
        self.dm = DataManager()
        self.um = UniverseManager(self.dm)

        self.start_date = START_DATE
        self.end_date = END_DATE
        self.debug = debug

    def run_backtest(self, params: Dict) -> Optional[pd.Series]:
        """
        Run backtest for given momentum parameters.

        Args:
            params: Momentum signal parameters

        Returns:
            Series of daily equity values (DatetimeIndex), or None if failed
        """
        try:
            # Build universe (S&P 500 PIT)
            universe = self.um.get_universe(
                universe_type='sp500_actual',
                as_of_date=self.start_date,
                min_price=5.0
            )

            # Ensure universe is a list
            if isinstance(universe, pd.Series):
                universe = universe.tolist()
            elif isinstance(universe, pd.DataFrame):
                universe = universe.index.tolist() if hasattr(universe, 'index') else []

            if not universe or len(universe) == 0:
                logger.warning("  Universe is empty!")
                return None

            # Get monthly rebalance dates
            rebalance_dates = get_rebalance_dates(
                schedule='M',
                dm=self.dm,
                start_date=self.start_date,
                end_date=self.end_date
            )

            if self.debug:
                logger.info(f"DEBUG: Total rebalance dates: {len(rebalance_dates)}")
                logger.info(f"DEBUG: First 5 rebalance dates: {rebalance_dates[:5]}")
                logger.info(f"DEBUG: Last 5 rebalance dates: {rebalance_dates[-5:]}")

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

            if len(prices_dict) == 0:
                logger.warning("  No price data loaded!")
                return None

            # Calculate momentum for each stock
            momentum_dict = {}
            for ticker, prices in prices_dict.items():
                if 'close' not in prices.columns:
                    continue
                if len(prices) < params['formation_period'] + params['skip_period']:
                    continue

                # Momentum calculation
                mom = prices['close'].pct_change(
                    periods=params['formation_period'],
                    fill_method=None
                ).shift(params['skip_period'])

                momentum_dict[ticker] = mom

            # Convert to DataFrame
            momentum_df = pd.DataFrame(momentum_dict)
            if len(momentum_df) == 0:
                return None

            momentum_df = momentum_df.sort_index()

            # Handle duplicate dates (from corporate actions)
            if momentum_df.index.duplicated().any():
                momentum_df = momentum_df[~momentum_df.index.duplicated(keep='last')]

            # Build price DataFrame for returns calculation
            prices_close = {}
            for ticker, prices in prices_dict.items():
                prices_close[ticker] = prices['close']
            prices_df = pd.DataFrame(prices_close)

            # Build simple equal-weight portfolio rebalanced monthly
            equity_series = []
            portfolio_value = INITIAL_CAPITAL
            current_holdings = {}

            if self.debug:
                logger.info(f"\nDEBUG: Starting portfolio loop with initial capital: ${portfolio_value:,.2f}")

            for i, rebal_date in enumerate(pd.DatetimeIndex(rebalance_dates)):
                # Debug logging for first 5 rebalances
                if self.debug and i < 5:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"DEBUG: Rebalance {i+1} on {rebal_date.date()}")
                    logger.info(f"DEBUG: Portfolio value before rebal: ${portfolio_value:,.2f}")

                # Get momentum scores
                valid_idx = momentum_df.index[momentum_df.index <= rebal_date]
                if len(valid_idx) == 0:
                    if self.debug and i < 5:
                        logger.info(f"DEBUG: No valid momentum data yet, skipping")
                    continue

                mom_date = valid_idx[-1]
                mom_today = momentum_df.loc[mom_date].dropna()

                if self.debug and i < 5:
                    logger.info(f"DEBUG: Momentum universe size: {len(mom_today)} stocks")

                if len(mom_today) < 20:
                    if self.debug and i < 5:
                        logger.info(f"DEBUG: Too few stocks ({len(mom_today)} < 20), skipping")
                    continue

                # Winsorize
                lower_bound = mom_today.quantile(params['winsorize_pct'][0] / 100)
                upper_bound = mom_today.quantile(params['winsorize_pct'][1] / 100)
                mom_wins = mom_today.clip(lower=lower_bound, upper=upper_bound)

                # Take top quintile (highest momentum)
                sorted_mom = mom_wins.sort_values(ascending=False)
                quintile_size = len(sorted_mom) // 5
                top_quintile = sorted_mom.iloc[:quintile_size].index.tolist()

                if self.debug and i < 5:
                    logger.info(f"DEBUG: Top quintile size: {len(top_quintile)} stocks")
                    if len(top_quintile) > 0:
                        logger.info(f"DEBUG: Sample top quintile tickers: {top_quintile[:5]}")

                # Equal-weight allocation
                if len(top_quintile) > 0:
                    weight_per_stock = 1.0 / len(top_quintile)

                    # Get prices on rebalance date
                    if rebal_date in prices_df.index:
                        rebal_prices = prices_df.loc[rebal_date]
                    else:
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

                    if self.debug and i < 5:
                        logger.info(f"DEBUG: Holdings after rebalance: {len(current_holdings)} positions")
                        if len(current_holdings) > 0:
                            # Show first 3 holdings as examples
                            sample_tickers = list(current_holdings.keys())[:3]
                            for ticker in sample_tickers:
                                shares = current_holdings[ticker]
                                price = rebal_prices[ticker]
                                if isinstance(price, pd.Series):
                                    price = price.iloc[-1]
                                notional = shares * price
                                logger.info(f"DEBUG:   {ticker}: {shares:.2f} shares @ ${price:.2f} = ${notional:,.2f}")

                # Track equity daily until next rebalance
                if i + 1 < len(rebalance_dates):
                    next_rebal = pd.Timestamp(rebalance_dates[i + 1])
                else:
                    next_rebal = pd.Timestamp(self.end_date)

                price_window = prices_df[
                    (prices_df.index >= rebal_date) &
                    (prices_df.index <= next_rebal)
                ]

                for date in price_window.index:
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

                if equity_series:
                    old_value = portfolio_value
                    portfolio_value = equity_series[-1]['equity']

                    if self.debug and i < 5:
                        logger.info(f"DEBUG: Portfolio value after tracking period: ${portfolio_value:,.2f}")
                        if old_value > 0:
                            ret = (portfolio_value - old_value) / old_value
                            logger.info(f"DEBUG: Return since last rebalance: {ret:.2%}")

            # Convert to Series
            if not equity_series:
                return None

            equity_df = pd.DataFrame(equity_series)
            equity_curve = equity_df.set_index('date')['equity']

            return equity_curve

        except Exception as e:
            import traceback
            logger.error(f"  Backtest error: {e}")
            logger.error(f"  Full traceback:\n{traceback.format_exc()}")
            return None


# ==============================================================================
# OPTUNA OBJECTIVE FUNCTION
# ==============================================================================

class OptunaObjective:
    """
    Optuna objective function for Momentum optimization.

    Maximizes OOS Sharpe while enforcing acceptance gates.
    """

    def __init__(self):
        """Initialize objective."""
        self.backtester = MomentumBacktester()
        self.trial_results = []

        self.is_cutoff = IS_CUTOFF
        self.oos_start = OOS_START

        logger.info("=" * 80)
        logger.info("Phase 2.1 Momentum Optimization - Optuna Bayesian Search")
        logger.info("=" * 80)
        logger.info(f"Full period: {START_DATE} to {END_DATE}")
        logger.info(f"In-sample: {START_DATE} to {self.is_cutoff}")
        logger.info(f"Out-of-sample: {self.oos_start} to {END_DATE}")
        logger.info("=" * 80)

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Objective function for a single Optuna trial.

        Args:
            trial: Optuna trial object

        Returns:
            OOS Sharpe ratio (or penalized value if gates fail)
        """
        # Suggest hyperparameters
        formation_days = trial.suggest_int('formation_days', 126, 378, step=7)
        skip_days = trial.suggest_int('skip_days', 0, 42, step=3)
        winsorize_pct = trial.suggest_float('winsorize_pct', 1.0, 10.0)

        logger.info(f"\nTrial {trial.number}: formation={formation_days}, skip={skip_days}, winsor={winsorize_pct:.1f}%")

        # Build momentum params
        params = {
            'formation_period': formation_days,
            'skip_period': skip_days,
            'winsorize_pct': [winsorize_pct, 100 - winsorize_pct],
            'quintiles': True,
            'rebalance_frequency': 'monthly'
        }

        # Run backtest
        equity_curve = self.backtester.run_backtest(params)

        if equity_curve is None or len(equity_curve) < 10:
            logger.warning("  Backtest failed!")
            self.trial_results.append({
                'trial_number': trial.number,
                'formation_days': formation_days,
                'skip_days': skip_days,
                'winsorize_pct': winsorize_pct,
                'error': 'Backtest failed'
            })
            return 0.01  # Penalize failed trials

        # Split into IS and OOS
        is_cutoff_ts = pd.Timestamp(self.is_cutoff)
        oos_start_ts = pd.Timestamp(self.oos_start)

        is_equity = equity_curve[equity_curve.index <= is_cutoff_ts]
        oos_equity = equity_curve[equity_curve.index >= oos_start_ts]

        # Compute metrics
        full_metrics = compute_basic_metrics(equity_curve)
        is_metrics = compute_basic_metrics(is_equity)
        oos_metrics = compute_basic_metrics(oos_equity)

        # Regime metrics
        regime_metrics = compute_regime_metrics(equity_curve)

        # Check gates
        gates = check_gates(
            is_sharpe=is_metrics['sharpe'],
            oos_sharpe=oos_metrics['sharpe'],
            oos_max_dd=oos_metrics['max_drawdown'],
            regime_recent_sharpe=regime_metrics['recent']['sharpe'],
            regime_recent_return=regime_metrics['recent']['mean_return']
        )

        # Store results
        result = {
            'trial_number': trial.number,
            'formation_days': formation_days,
            'skip_days': skip_days,
            'winsorize_pct': winsorize_pct,

            # Full sample
            'full_sharpe': full_metrics['sharpe'],
            'full_annual_return': full_metrics['annual_return'],
            'full_volatility': full_metrics['volatility'],
            'full_max_drawdown': full_metrics['max_drawdown'],

            # In-sample
            'is_sharpe': is_metrics['sharpe'],
            'is_annual_return': is_metrics['annual_return'],

            # Out-of-sample
            'oos_sharpe': oos_metrics['sharpe'],
            'oos_annual_return': oos_metrics['annual_return'],
            'oos_max_drawdown': oos_metrics['max_drawdown'],

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

        self.trial_results.append(result)

        # Log key metrics
        logger.info(f"  Full Sharpe: {full_metrics['sharpe']:.3f}")
        logger.info(f"  IS Sharpe: {is_metrics['sharpe']:.3f}")
        logger.info(f"  OOS Sharpe: {oos_metrics['sharpe']:.3f}")
        logger.info(f"  Passes all gates: {gates['passes_all_gates']}")

        # Return objective
        if gates['passes_all_gates']:
            # Return OOS Sharpe for gate passers
            return oos_metrics['sharpe']
        else:
            # Penalize gate failures but still return a value so Optuna explores
            return oos_metrics['sharpe'] * 0.1  # 90% penalty


def run_optimization(n_trials: int = 100) -> pd.DataFrame:
    """
    Run Optuna optimization.

    Args:
        n_trials: Number of trials to run

    Returns:
        DataFrame with all trial results
    """
    # Create objective
    objective = OptunaObjective()

    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )

    # Run optimization
    logger.info(f"\nStarting Optuna optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Get results
    results_df = pd.DataFrame(objective.trial_results)

    logger.info(f"\nOptimization complete!")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best OOS Sharpe: {study.best_value:.3f}")
    logger.info(f"Best params: {study.best_params}")

    return results_df


def generate_outputs(results_df: pd.DataFrame):
    """
    Generate output files from trial results.

    Args:
        results_df: DataFrame with all trial results
    """
    # Save CSV
    csv_path = Path('results/momentum_phase2_optuna_trials.csv')
    results_df.to_csv(csv_path, index=False)
    logger.info(f"\nResults saved to: {csv_path}")

    # Generate summary MD
    _generate_summary(results_df)

    # Generate shortlist JSON
    _generate_shortlist(results_df)


def _generate_summary(results_df: pd.DataFrame):
    """Generate summary markdown report."""
    output_path = Path('results/momentum_phase2_optuna_summary.md')

    # Filter passing trials
    if 'passes_all_gates' in results_df.columns:
        passing = results_df[results_df['passes_all_gates'] == True]
    else:
        passing = pd.DataFrame()

    with open(output_path, 'w') as f:
        f.write("# Momentum Phase 2.1 Optuna Optimization - Summary\n\n")
        f.write(f"**Date:** {datetime.now().isoformat()}\n")
        f.write(f"**Period:** {START_DATE} to {END_DATE}\n")
        f.write(f"**In-Sample:** {START_DATE} to {IS_CUTOFF}\n")
        f.write(f"**Out-of-Sample:** {OOS_START} to {END_DATE}\n\n")
        f.write("---\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total trials:** {len(results_df)}\n")
        f.write(f"- **Trials passing all gates:** {len(passing)}\n\n")

        if len(passing) == 0:
            f.write("**⚠️ NO TRIALS PASSED ALL 4 GATES**\n\n")
        else:
            f.write(f"**✅ SUCCESS: {len(passing)} trial(s) passed all gates**\n\n")

        # Top 5 passing trials
        if len(passing) > 0:
            f.write("## Top 5 Trials Passing All Gates\n\n")
            f.write("**Sorted by OOS Sharpe (descending):**\n\n")
            f.write("| Rank | Formation | Skip | Winsor | Full Sharpe | IS Sharpe | OOS Sharpe | OOS Max DD | Recent Sharpe |\n")
            f.write("|------|-----------|------|--------|-------------|-----------|------------|------------|---------------|\n")

            passing_sorted = passing.sort_values('oos_sharpe', ascending=False).head(5)
            for rank, (_, row) in enumerate(passing_sorted.iterrows(), 1):
                f.write(
                    f"| {rank} | {row['formation_days']} | {row['skip_days']} | "
                    f"{row['winsorize_pct']:.1f}% | {row['full_sharpe']:.3f} | "
                    f"{row['is_sharpe']:.3f} | {row['oos_sharpe']:.3f} | "
                    f"{row['oos_max_drawdown']:.2%} | {row['regime_recent_sharpe']:.3f} |\n"
                )
            f.write("\n")

        # Best overall trial (if any valid trials exist)
        if 'oos_sharpe' in results_df.columns and results_df['oos_sharpe'].notna().any():
            best_idx = results_df['oos_sharpe'].idxmax()
            best = results_df.loc[best_idx]
            f.write("## Best Trial by OOS Sharpe\n\n")
            f.write(f"- **Trial number:** {best['trial_number']}\n")
            f.write(f"- **Formation days:** {best['formation_days']}\n")
            f.write(f"- **Skip days:** {best['skip_days']}\n")
            f.write(f"- **Winsorization:** {best['winsorize_pct']:.1f}%\n")
            f.write(f"- **Full Sharpe:** {best['full_sharpe']:.3f}\n")
            f.write(f"- **OOS Sharpe:** {best['oos_sharpe']:.3f}\n")
            f.write(f"- **Passes all gates:** {'Yes' if best.get('passes_all_gates', False) else 'No'}\n\n")
        else:
            f.write("## Best Trial by OOS Sharpe\n\n")
            f.write("**No valid trials completed successfully.**\n\n")

    logger.info(f"Summary report saved to: {output_path}")


def _generate_shortlist(results_df: pd.DataFrame):
    """Generate shortlist JSON for top 1-3 configs."""
    output_path = Path('results/MOMENTUM_PHASE2_SHORTLIST.json')

    # Get passing trials
    if 'passes_all_gates' in results_df.columns:
        passing = results_df[results_df['passes_all_gates'] == True]
    else:
        passing = pd.DataFrame()

    if len(passing) == 0:
        logger.warning("No trials passed all gates - shortlist will be empty")
        shortlist = []
    else:
        # Sort by OOS Sharpe and take top 3
        passing_sorted = passing.sort_values('oos_sharpe', ascending=False)
        top_configs = passing_sorted.head(3)

        shortlist = []
        for _, row in top_configs.iterrows():
            shortlist.append({
                'trial_number': int(row['trial_number']),
                'formation_days': int(row['formation_days']),
                'skip_days': int(row['skip_days']),
                'winsorize_pct': float(row['winsorize_pct']),
                'full_sharpe': float(row['full_sharpe']),
                'is_sharpe': float(row['is_sharpe']),
                'oos_sharpe': float(row['oos_sharpe']),
                'oos_max_drawdown': float(row['oos_max_drawdown']),
                'regime_recent_sharpe': float(row['regime_recent_sharpe']),
                'regime_recent_return': float(row['regime_recent_return'])
            })

    with open(output_path, 'w') as f:
        json.dump(shortlist, f, indent=2)

    logger.info(f"Shortlist saved to: {output_path} ({len(shortlist)} configs)")


def debug_single_run():
    """
    Debug helper: Run a single known-good momentum config and inspect results.

    Uses standard 12-1 momentum: 252-day formation, 21-day skip, 5% winsorization.
    """
    logger.info("\n" + "="*80)
    logger.info("DEBUG SINGLE RUN - Known-Good Momentum Config")
    logger.info("="*80)

    # Known-good config (similar to Phase 1.5)
    params = {
        'formation_period': 252,    # 12 months
        'skip_period': 21,           # 1 month skip
        'winsorize_pct': [5.0, 95.0] # 5% two-sided winsorization
    }

    logger.info(f"Config: formation={params['formation_period']}, skip={params['skip_period']}, winsor={params['winsorize_pct'][0]}%")
    logger.info(f"Period: {START_DATE} to {END_DATE}")
    logger.info(f"Universe: S&P 500 PIT (sp500_actual)")
    logger.info("")

    # Run backtest with debug logging enabled
    backtester = MomentumBacktester(debug=True)
    equity_curve = backtester.run_backtest(params)

    if equity_curve is None:
        logger.error("DEBUG: Backtest returned None - something failed!")
        return

    logger.info("\n" + "="*80)
    logger.info("EQUITY CURVE INSPECTION")
    logger.info("="*80)

    # Show head and tail
    logger.info("\nFirst 10 days:")
    logger.info(equity_curve.head(10).to_string())

    logger.info("\nLast 10 days:")
    logger.info(equity_curve.tail(10).to_string())

    # Stats
    logger.info(f"\nEquity curve stats:")
    logger.info(f"  Min:   ${equity_curve.min():,.2f}")
    logger.info(f"  Max:   ${equity_curve.max():,.2f}")
    logger.info(f"  Final: ${equity_curve.iloc[-1]:,.2f}")
    logger.info(f"  Range: ${equity_curve.max() - equity_curve.min():,.2f}")

    # Check if flat
    is_flat = (equity_curve.max() - equity_curve.min()) < 1000  # Less than $1k movement
    if is_flat:
        logger.warning("\n⚠️  EQUITY CURVE IS ESSENTIALLY FLAT!")
        logger.warning("    Portfolio never meaningfully invested.")
    else:
        logger.info("\n✅ Equity curve is NOT flat - portfolio is trading")

    # Compute metrics
    logger.info("\n" + "="*80)
    logger.info("PERFORMANCE METRICS")
    logger.info("="*80)

    from validation.simple_validation import (
        compute_basic_metrics,
        compute_regime_metrics
    )

    # Full sample metrics
    full_metrics = compute_basic_metrics(equity_curve)
    logger.info(f"\nFull Sample (2015-04-01 to 2024-12-31):")
    logger.info(f"  Total Return:    {full_metrics['total_return']:.2%}")
    logger.info(f"  Annual Return:   {full_metrics['annual_return']:.2%}")
    logger.info(f"  Volatility:      {full_metrics['volatility']:.2%}")
    logger.info(f"  Sharpe Ratio:    {full_metrics['sharpe']:.3f}")
    logger.info(f"  Max Drawdown:    {full_metrics['max_drawdown']:.2%}")
    logger.info(f"  Num Days:        {full_metrics['num_days']}")

    # IS/OOS split
    is_cutoff = pd.Timestamp(IS_CUTOFF)
    oos_start = pd.Timestamp(OOS_START)

    is_equity = equity_curve[equity_curve.index <= is_cutoff]
    oos_equity = equity_curve[equity_curve.index >= oos_start]

    if len(is_equity) > 0:
        is_metrics = compute_basic_metrics(is_equity)
        logger.info(f"\nIn-Sample (2015-04-01 to 2022-12-31):")
        logger.info(f"  Sharpe Ratio:    {is_metrics['sharpe']:.3f}")
        logger.info(f"  Annual Return:   {is_metrics['annual_return']:.2%}")

    if len(oos_equity) > 0:
        oos_metrics = compute_basic_metrics(oos_equity)
        logger.info(f"\nOut-of-Sample (2023-01-01 to 2024-12-31):")
        logger.info(f"  Sharpe Ratio:    {oos_metrics['sharpe']:.3f}")
        logger.info(f"  Annual Return:   {oos_metrics['annual_return']:.2%}")
        logger.info(f"  Max Drawdown:    {oos_metrics['max_drawdown']:.2%}")

    # Regime metrics
    regime_metrics = compute_regime_metrics(equity_curve)
    logger.info(f"\nRegime Performance:")
    for regime_name, metrics in regime_metrics.items():
        logger.info(f"  {regime_name:12s}: Sharpe={metrics['sharpe']:6.3f}, Return={metrics['mean_return']:7.2%}")

    logger.info("\n" + "="*80)
    logger.info("DEBUG SINGLE RUN COMPLETE")
    logger.info("="*80)


def main():
    """Run Phase 2.1 Optuna optimization."""
    parser = argparse.ArgumentParser(description='Momentum Phase 2.1 Optuna Optimization')
    parser.add_argument('--n-trials', type=int, default=100,
                       help='Number of Optuna trials to run (default: 100)')
    parser.add_argument('--debug-single-run', action='store_true',
                       help='Run single debug config instead of optimization')
    args = parser.parse_args()

    # Handle debug mode
    if args.debug_single_run:
        debug_single_run()
        return

    logger.info("\nStarting Phase 2.1 Momentum Optuna Optimization...")

    # Run optimization
    results_df = run_optimization(n_trials=args.n_trials)

    # Generate outputs
    generate_outputs(results_df)

    logger.info("\nPhase 2.1 optimization complete!")
    logger.info("Check results in:")
    logger.info("  - results/momentum_phase2_optuna_trials.csv")
    logger.info("  - results/momentum_phase2_optuna_summary.md")
    logger.info("  - results/MOMENTUM_PHASE2_SHORTLIST.json")


if __name__ == '__main__':
    main()
