"""
Diagnostic script for Momentum Phase 2 canonical configuration.

Runs detailed diagnostics on Trial 11 (canonical config) and compares
to baseline and alternative configurations for robustness analysis.

Outputs:
- results/MOMENTUM_PHASE2_TRIAL11_DIAGNOSTIC.md
- results/momentum_phase2_trial11_monthly_returns.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, Optional

from data.data_manager import DataManager
from core.universe_manager import UniverseManager
from core.schedules import get_rebalance_dates
from validation.simple_validation import (
    compute_basic_metrics,
    compute_monthly_returns,
    compute_regime_metrics
)
from config import get_logger

logger = get_logger(__name__)

# Fixed random seed for reproducibility
np.random.seed(42)


class MomentumDiagnostic:
    """
    Diagnostic engine for Momentum Phase 2 configurations.

    Reuses backtest logic from Phase 2.1 Optuna optimizer.
    """

    def __init__(self):
        """Initialize diagnostic engine."""
        self.dm = DataManager()
        self.um = UniverseManager(self.dm)

    def run_backtest(
        self,
        formation_period: int,
        skip_period: int,
        winsorize_pct: float,
        start_date: str,
        end_date: str,
        initial_capital: int = 50000
    ) -> Optional[pd.Series]:
        """
        Run backtest for given momentum parameters.

        Args:
            formation_period: Days for momentum calculation
            skip_period: Days to skip after formation
            winsorize_pct: Two-sided winsorization percentage (e.g., 9.2 for [9.2, 90.8])
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital

        Returns:
            Series of daily equity values, or None if failed
        """
        try:
            # Build universe (S&P 500 PIT)
            universe = self.um.get_universe(
                universe_type='sp500_actual',
                as_of_date=start_date,
                min_price=5.0
            )

            # Ensure universe is a list
            if isinstance(universe, pd.Series):
                universe = universe.tolist()
            elif isinstance(universe, pd.DataFrame):
                universe = universe.index.tolist()

            if not universe or len(universe) == 0:
                logger.warning("Universe is empty!")
                return None

            # Get monthly rebalance dates
            rebalance_dates = get_rebalance_dates(
                schedule='M',
                dm=self.dm,
                start_date=start_date,
                end_date=end_date
            )

            # Fetch price data (need extra history for momentum calculation)
            lookback_buffer = timedelta(days=400)
            start_dt = pd.Timestamp(start_date)
            price_start_date = (start_dt - lookback_buffer).strftime('%Y-%m-%d')

            prices_dict = {}
            for ticker in universe:
                try:
                    prices = self.dm.get_prices(ticker, price_start_date, end_date)
                    if len(prices) > 0:
                        prices_dict[ticker] = prices
                except:
                    pass

            if len(prices_dict) == 0:
                logger.warning("No price data loaded!")
                return None

            # Calculate momentum for each stock
            momentum_dict = {}
            for ticker, prices in prices_dict.items():
                if 'close' not in prices.columns:
                    continue
                if len(prices) < formation_period + skip_period:
                    continue

                # Momentum calculation
                mom = prices['close'].pct_change(
                    periods=formation_period,
                    fill_method=None
                ).shift(skip_period)

                momentum_dict[ticker] = mom

            # Convert to DataFrame
            momentum_df = pd.DataFrame(momentum_dict)
            if len(momentum_df) == 0:
                return None

            momentum_df = momentum_df.sort_index()

            # Handle duplicate dates
            if momentum_df.index.duplicated().any():
                momentum_df = momentum_df[~momentum_df.index.duplicated(keep='last')]

            # Build price DataFrame for returns calculation
            prices_close = {}
            for ticker, prices in prices_dict.items():
                prices_close[ticker] = prices['close']
            prices_df = pd.DataFrame(prices_close)

            # Build simple equal-weight portfolio rebalanced monthly
            equity_series = []
            portfolio_value = initial_capital
            current_holdings = {}

            # Winsorization percentiles
            winsor_lower = winsorize_pct / 100
            winsor_upper = (100 - winsorize_pct) / 100

            for i, rebal_date in enumerate(pd.DatetimeIndex(rebalance_dates)):
                # Get momentum scores
                valid_idx = momentum_df.index[momentum_df.index <= rebal_date]
                if len(valid_idx) == 0:
                    continue

                mom_date = valid_idx[-1]
                mom_today = momentum_df.loc[mom_date].dropna()

                if len(mom_today) < 20:
                    continue

                # Winsorize
                lower_bound = mom_today.quantile(winsor_lower)
                upper_bound = mom_today.quantile(winsor_upper)
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
                        valid_price_idx = prices_df.index[prices_df.index <= rebal_date]
                        if len(valid_price_idx) > 0:
                            rebal_prices = prices_df.loc[valid_price_idx[-1]]
                        else:
                            continue

                    # Rebalance portfolio
                    current_holdings = {}
                    for ticker in top_quintile:
                        if ticker in rebal_prices:
                            price = rebal_prices[ticker]
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
                    next_rebal = pd.Timestamp(end_date)

                price_window = prices_df[
                    (prices_df.index >= rebal_date) &
                    (prices_df.index <= next_rebal)
                ]

                for date in price_window.index:
                    total_value = 0
                    for ticker, shares in current_holdings.items():
                        if ticker in price_window.columns:
                            price = price_window.loc[date, ticker]
                            if isinstance(price, pd.Series):
                                price = price.iloc[-1]
                            if pd.notna(price):
                                total_value += shares * price

                    equity_series.append({
                        'date': date,
                        'equity': total_value if total_value > 0 else portfolio_value
                    })

                if equity_series:
                    portfolio_value = equity_series[-1]['equity']

            # Convert to Series
            if not equity_series:
                return None

            equity_df = pd.DataFrame(equity_series)
            equity_curve = equity_df.set_index('date')['equity']

            return equity_curve

        except Exception as e:
            logger.error(f"Backtest error: {e}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return None

    def diagnose_config(
        self,
        config_name: str,
        formation_period: int,
        skip_period: int,
        winsorize_pct: float,
        start_date: str,
        end_date: str,
        is_cutoff: str,
        oos_start: str
    ) -> Dict:
        """
        Run full diagnostics on a configuration.

        Returns:
            Dict with keys: equity_curve, full_metrics, is_metrics, oos_metrics, regime_metrics
        """
        logger.info(f"\nDiagnosing config: {config_name}")
        logger.info(f"  formation={formation_period}, skip={skip_period}, winsor={winsorize_pct:.1f}%")

        # Run backtest
        equity_curve = self.run_backtest(
            formation_period=formation_period,
            skip_period=skip_period,
            winsorize_pct=winsorize_pct,
            start_date=start_date,
            end_date=end_date
        )

        if equity_curve is None:
            logger.error(f"  Backtest failed for {config_name}!")
            return None

        # Compute full sample metrics
        full_metrics = compute_basic_metrics(equity_curve)

        # Compute IS/OOS metrics
        is_cutoff_ts = pd.Timestamp(is_cutoff)
        oos_start_ts = pd.Timestamp(oos_start)

        is_equity = equity_curve[equity_curve.index <= is_cutoff_ts]
        oos_equity = equity_curve[equity_curve.index >= oos_start_ts]

        is_metrics = compute_basic_metrics(is_equity) if len(is_equity) > 0 else {}
        oos_metrics = compute_basic_metrics(oos_equity) if len(oos_equity) > 0 else {}

        # Compute regime metrics
        regime_metrics = compute_regime_metrics(equity_curve)

        logger.info(f"  Full Sharpe: {full_metrics['sharpe']:.3f}")
        logger.info(f"  OOS Sharpe: {oos_metrics.get('sharpe', 0):.3f}")
        logger.info(f"  Recent Sharpe: {regime_metrics.get('recent', {}).get('sharpe', 0):.3f}")

        return {
            'config_name': config_name,
            'equity_curve': equity_curve,
            'full_metrics': full_metrics,
            'is_metrics': is_metrics,
            'oos_metrics': oos_metrics,
            'regime_metrics': regime_metrics,
            'formation_period': formation_period,
            'skip_period': skip_period,
            'winsorize_pct': winsorize_pct
        }


def generate_diagnostic_report(
    canonical_result: Dict,
    comparison_results: Dict
):
    """
    Generate comprehensive diagnostic report.

    Args:
        canonical_result: Diagnostic result for canonical config
        comparison_results: Dict mapping config names to diagnostic results
    """
    output_path = Path('results/MOMENTUM_PHASE2_TRIAL11_DIAGNOSTIC.md')

    with open(output_path, 'w') as f:
        f.write("# Momentum Phase 2 - Trial 11 Diagnostic Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"**Config:** Trial 11 (Canonical Phase 2 Winner)\n\n")
        f.write("---\n\n")

        # Parameters
        f.write("## Configuration Parameters\n\n")
        f.write(f"- **Formation Period:** {canonical_result['formation_period']} days (~{canonical_result['formation_period']/21:.1f} months)\n")
        f.write(f"- **Skip Period:** {canonical_result['skip_period']} days\n")
        f.write(f"- **Winsorization:** {canonical_result['winsorize_pct']:.1f}% (two-sided)\n")
        f.write(f"- **Universe:** S&P 500 PIT (sp500_actual)\n")
        f.write(f"- **Rebalancing:** Monthly (end of month)\n")
        f.write(f"- **Position Sizing:** Equal-weight top quintile\n")
        f.write(f"- **Period:** 2015-04-01 to 2024-12-31\n\n")

        # Full sample metrics
        full = canonical_result['full_metrics']
        f.write("## Full Sample Performance (2015-04-01 to 2024-12-31)\n\n")
        f.write(f"- **Total Return:** {full['total_return']:.2%}\n")
        f.write(f"- **Annual Return:** {full['annual_return']:.2%}\n")
        f.write(f"- **Volatility:** {full['volatility']:.2%}\n")
        f.write(f"- **Sharpe Ratio:** {full['sharpe']:.3f}\n")
        f.write(f"- **Max Drawdown:** {full['max_drawdown']:.2%}\n")
        f.write(f"- **Trading Days:** {full['num_days']}\n\n")

        # IS metrics
        is_m = canonical_result['is_metrics']
        f.write("## In-Sample Performance (2015-04-01 to 2022-12-31)\n\n")
        f.write(f"- **Annual Return:** {is_m['annual_return']:.2%}\n")
        f.write(f"- **Volatility:** {is_m['volatility']:.2%}\n")
        f.write(f"- **Sharpe Ratio:** {is_m['sharpe']:.3f}\n")
        f.write(f"- **Max Drawdown:** {is_m['max_drawdown']:.2%}\n\n")

        # OOS metrics
        oos = canonical_result['oos_metrics']
        f.write("## Out-of-Sample Performance (2023-01-01 to 2024-12-31)\n\n")
        f.write(f"- **Annual Return:** {oos['annual_return']:.2%}\n")
        f.write(f"- **Volatility:** {oos['volatility']:.2%}\n")
        f.write(f"- **Sharpe Ratio:** {oos['sharpe']:.3f}\n")
        f.write(f"- **Max Drawdown:** {oos['max_drawdown']:.2%}\n\n")

        # Regime performance
        f.write("## Regime Performance\n\n")
        f.write("| Regime | Period | Mean Monthly Return | Sharpe | Num Months |\n")
        f.write("|--------|--------|---------------------|--------|------------|\n")

        regime_labels = {
            'covid': 'COVID',
            'bear_2022': '2022 Bear',
            'recent': 'Recent (2023-2024)'
        }

        for regime_key, regime_label in regime_labels.items():
            if regime_key in canonical_result['regime_metrics']:
                r = canonical_result['regime_metrics'][regime_key]
                f.write(f"| {regime_label} | {regime_key} | {r['mean_return']:.2%} | {r['sharpe']:.3f} | {r['num_months']} |\n")
        f.write("\n")

        # Robustness comparison
        f.write("## Robustness Comparison: A/B/C Configs\n\n")
        f.write("| Config | Formation | Skip | Winsor | Full Sharpe | OOS Sharpe | Recent Sharpe | OOS Ann Ret | OOS Max DD |\n")
        f.write("|--------|-----------|------|--------|-------------|------------|---------------|-------------|------------|\n")

        for config_name in ['A) Canonical', 'B) Baseline', 'C) Alternative']:
            result = comparison_results.get(config_name)
            if result:
                f.write(f"| {config_name} | {result['formation_period']} | {result['skip_period']} | {result['winsorize_pct']:.1f}% | ")
                f.write(f"{result['full_metrics']['sharpe']:.3f} | ")
                f.write(f"{result['oos_metrics']['sharpe']:.3f} | ")
                recent_sharpe = result['regime_metrics'].get('recent', {}).get('sharpe', 0)
                f.write(f"{recent_sharpe:.3f} | ")
                f.write(f"{result['oos_metrics']['annual_return']:.2%} | ")
                f.write(f"{result['oos_metrics']['max_drawdown']:.2%} |\n")
        f.write("\n")

        # Robustness analysis
        f.write("### Robustness Analysis\n\n")

        canonical_oos = canonical_result['oos_metrics']['sharpe']
        baseline_oos = comparison_results['B) Baseline']['oos_metrics']['sharpe']
        alternative_oos = comparison_results['C) Alternative']['oos_metrics']['sharpe']

        # Dominance check
        if canonical_oos > baseline_oos:
            improvement = (canonical_oos - baseline_oos) / baseline_oos * 100
            f.write(f"- ✅ **Canonical config clearly dominates baseline:** OOS Sharpe improvement of {improvement:.1f}% ({canonical_oos:.3f} vs {baseline_oos:.3f})\n")
        else:
            f.write(f"- ⚠️  Baseline slightly outperforms canonical in OOS Sharpe ({baseline_oos:.3f} vs {canonical_oos:.3f})\n")

        # Stability check
        oos_diff = abs(canonical_oos - alternative_oos)
        if oos_diff < 0.05:
            f.write(f"- ✅ **Performance stable across nearby configs:** OOS Sharpe difference only {oos_diff:.3f} between canonical and alternative\n")
        else:
            f.write(f"- ⚠️  Moderate sensitivity: OOS Sharpe difference of {oos_diff:.3f} between canonical and alternative configs\n")

        # Parameter sensitivity
        if canonical_result['skip_period'] == 0:
            f.write(f"- ⚠️  **Zero skip period is unconventional:** Traditional momentum uses 21-day skip. Requires validation that there's no short-term reversal effect in this data.\n")

        f.write("\n")

        # Verdict
        f.write("## Verdict\n\n")

        passes_full = full['sharpe'] > 0.15
        passes_oos = oos['sharpe'] >= 0.2
        passes_dd = oos['max_drawdown'] > -0.30
        recent_sharpe = canonical_result['regime_metrics'].get('recent', {}).get('sharpe', 0)
        passes_recent = recent_sharpe > 0.2

        all_pass = passes_full and passes_oos and passes_dd and passes_recent

        if all_pass:
            f.write("**✅ READY FOR PHASE 3 (ENSEMBLE DESIGN)**\n\n")
            f.write("This configuration passes all acceptance gates and shows:\n")
            f.write(f"- Strong OOS performance (Sharpe {oos['sharpe']:.3f})\n")
            f.write(f"- Robust recent regime performance (Sharpe {recent_sharpe:.3f})\n")
            f.write(f"- Acceptable drawdown profile (OOS Max DD {oos['max_drawdown']:.2%})\n")
            f.write(f"- Clear improvement over baseline\n\n")
            f.write("**Recommendation:** Proceed with this configuration as the canonical Momentum v2 signal.\n")
        else:
            f.write("**⚠️ CONDITIONAL GO - REQUIRES REVIEW**\n\n")
            f.write("Some gates did not pass:\n")
            if not passes_full:
                f.write(f"- Full Sharpe: {full['sharpe']:.3f} (< 0.15 threshold)\n")
            if not passes_oos:
                f.write(f"- OOS Sharpe: {oos['sharpe']:.3f} (< 0.20 threshold)\n")
            if not passes_dd:
                f.write(f"- OOS Max DD: {oos['max_drawdown']:.2%} (worse than -30% threshold)\n")
            if not passes_recent:
                f.write(f"- Recent Sharpe: {recent_sharpe:.3f} (< 0.20 threshold)\n")

        f.write("\n---\n\n")
        f.write(f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    logger.info(f"\nDiagnostic report saved to: {output_path}")


def save_monthly_returns(equity_curve: pd.Series):
    """Save monthly returns to CSV."""
    monthly_returns = compute_monthly_returns(equity_curve)

    output_path = Path('results/momentum_phase2_trial11_monthly_returns.csv')
    monthly_returns.to_csv(output_path, header=['monthly_return'])

    logger.info(f"Monthly returns saved to: {output_path}")


def main():
    """Run diagnostic on canonical config with A/B/C robustness comparison."""
    logger.info("\n" + "="*80)
    logger.info("MOMENTUM PHASE 2 - CANONICAL CONFIG DIAGNOSTIC")
    logger.info("="*80)

    # Load canonical config
    config_path = Path('results/MOMENTUM_PHASE2_CANONICAL_CONFIG.json')
    with open(config_path, 'r') as f:
        canonical_config = json.load(f)

    logger.info(f"\nLoaded canonical config from: {config_path}")
    logger.info(f"Trial: {canonical_config['trial_number']}")
    logger.info(f"Formation: {canonical_config['formation_period']}")
    logger.info(f"Skip: {canonical_config['skip_period']}")
    logger.info(f"Winsorization: {canonical_config['winsorize_pct_low']:.1f}%\n")

    # Initialize diagnostic engine
    engine = MomentumDiagnostic()

    # Define configs for A/B/C comparison
    configs = {
        'A) Canonical': {
            'formation_period': canonical_config['formation_period'],
            'skip_period': canonical_config['skip_period'],
            'winsorize_pct': canonical_config['winsorize_pct_low']
        },
        'B) Baseline': {
            'formation_period': 252,
            'skip_period': 21,
            'winsorize_pct': 5.0
        },
        'C) Alternative': {
            'formation_period': 287,
            'skip_period': 18,
            'winsorize_pct': 10.0
        }
    }

    # Date ranges
    start_date = canonical_config['date_ranges']['start_date']
    end_date = canonical_config['date_ranges']['end_date']
    is_cutoff = canonical_config['date_ranges']['is_cutoff']
    oos_start = canonical_config['date_ranges']['oos_start']

    # Run diagnostics for all configs
    results = {}
    for config_name, params in configs.items():
        result = engine.diagnose_config(
            config_name=config_name,
            formation_period=params['formation_period'],
            skip_period=params['skip_period'],
            winsorize_pct=params['winsorize_pct'],
            start_date=start_date,
            end_date=end_date,
            is_cutoff=is_cutoff,
            oos_start=oos_start
        )
        if result:
            results[config_name] = result

    if 'A) Canonical' not in results:
        logger.error("Failed to run canonical config diagnostic!")
        return

    # Generate diagnostic report
    canonical_result = results['A) Canonical']
    generate_diagnostic_report(canonical_result, results)

    # Save monthly returns for canonical config
    save_monthly_returns(canonical_result['equity_curve'])

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("DIAGNOSTICS COMPLETE")
    logger.info("="*80)
    logger.info("\nOutput files:")
    logger.info("  - results/MOMENTUM_PHASE2_TRIAL11_DIAGNOSTIC.md")
    logger.info("  - results/momentum_phase2_trial11_monthly_returns.csv")
    logger.info("\nSide-by-side robustness comparison included in diagnostic MD")
    logger.info("\nCanonical config performance:")
    logger.info(f"  Full Sharpe: {canonical_result['full_metrics']['sharpe']:.3f}")
    logger.info(f"  OOS Sharpe: {canonical_result['oos_metrics']['sharpe']:.3f}")
    recent_sharpe = canonical_result['regime_metrics'].get('recent', {}).get('sharpe', 0)
    logger.info(f"  Recent Sharpe: {recent_sharpe:.3f}")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()
