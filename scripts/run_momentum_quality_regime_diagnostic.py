"""
Momentum + Quality Ensemble Regime Diagnostic

Compares momentum-only vs momentum+quality performance across distinct market regimes:
1. Pre-COVID Expansion (2015-04-01 to 2019-12-31)
2. COVID Crash (2020-02-01 to 2020-04-30)
3. COVID/QE Recovery (2020-05-01 to 2021-12-31)
4. 2022 Bear Market (2022-01-01 to 2022-12-31)
5. Recent Period (2023-01-01 to 2024-12-31)

Outputs:
    results/ensemble_baselines/momentum_quality_v1_regime_comparison.csv
    results/ensemble_baselines/momentum_quality_v1_regime_diagnostic.md

Usage:
    python3 scripts/run_momentum_quality_regime_diagnostic.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from data.data_manager import DataManager
from core.universe_manager import UniverseManager
from core.backtest_engine import BacktestConfig, run_backtest, BacktestResult
from core.signal_adapters import make_ensemble_signal_fn, make_multisignal_ensemble_fn
from signals.ml.ensemble_configs import (
    get_momentum_v2_adaptive_quintile_ensemble,
    get_momentum_quality_v1_ensemble
)
from config import get_logger

logger = get_logger(__name__)


# ============================================================================
# REGIME DEFINITIONS
# ============================================================================

REGIMES = [
    {
        "name": "pre_covid_expansion",
        "label": "Pre-COVID Expansion",
        "start": "2015-04-01",
        "end": "2019-12-31",
        "description": "Bull market with steady growth, low volatility",
    },
    {
        "name": "covid_crash",
        "label": "COVID Crash",
        "start": "2020-02-01",
        "end": "2020-04-30",
        "description": "Pandemic-driven market crash (Feb-Apr 2020)",
    },
    {
        "name": "covid_recovery",
        "label": "COVID/QE Recovery",
        "start": "2020-05-01",
        "end": "2021-12-31",
        "description": "QE-driven recovery and growth stocks rally",
    },
    {
        "name": "bear_2022",
        "label": "2022 Bear Market",
        "start": "2022-01-01",
        "end": "2022-12-31",
        "description": "Inflation spike, rate hikes, value rotation",
    },
    {
        "name": "recent",
        "label": "Recent Period",
        "start": "2023-01-01",
        "end": "2024-12-31",
        "description": "AI boom, higher-for-longer rates",
    },
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_regime_metrics(
    equity_curve: pd.Series,
    regime_start: str,
    regime_end: str,
) -> Dict[str, float]:
    """
    Compute performance metrics for a specific regime period.

    Args:
        equity_curve: Full equity curve (indexed by date)
        regime_start: Regime start date (YYYY-MM-DD)
        regime_end: Regime end date (YYYY-MM-DD)

    Returns:
        Dict with metrics: total_return, cagr, volatility, sharpe, max_drawdown, num_periods
    """
    # Convert dates to pd.Timestamp for comparison
    start_ts = pd.Timestamp(regime_start)
    end_ts = pd.Timestamp(regime_end)

    # Slice equity curve to regime window
    regime_equity = equity_curve[(equity_curve.index >= start_ts) & (equity_curve.index <= end_ts)]

    if len(regime_equity) == 0:
        logger.warning(f"No data in regime {regime_start} to {regime_end}")
        return {
            'total_return': np.nan,
            'cagr': np.nan,
            'volatility': np.nan,
            'sharpe': np.nan,
            'max_drawdown': np.nan,
            'num_periods': 0,
        }

    # Compute returns
    regime_returns = regime_equity.pct_change().dropna()
    num_periods = len(regime_returns)

    if num_periods < 2:
        logger.warning(f"Too few periods ({num_periods}) in regime {regime_start} to {regime_end}")
        return {
            'total_return': np.nan,
            'cagr': np.nan,
            'volatility': np.nan,
            'sharpe': np.nan,
            'max_drawdown': np.nan,
            'num_periods': num_periods,
        }

    # Total return (compound return over regime)
    total_return = (regime_equity.iloc[-1] / regime_equity.iloc[0]) - 1.0

    # CAGR (annualized return)
    # Compute number of years
    days_in_regime = (end_ts - start_ts).days
    years = days_in_regime / 365.25
    if years > 0:
        cagr = (1 + total_return) ** (1 / years) - 1.0
    else:
        cagr = np.nan

    # Volatility (annualized std of returns)
    # Assume monthly returns (12 periods per year)
    periods_per_year = 12
    volatility = regime_returns.std() * np.sqrt(periods_per_year)

    # Sharpe ratio (assume 0 risk-free rate)
    if volatility > 0:
        sharpe = (regime_returns.mean() * periods_per_year) / volatility
    else:
        sharpe = np.nan

    # Max drawdown
    # Compute running max and drawdown from peak
    running_max = regime_equity.expanding().max()
    drawdown = (regime_equity - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        'total_return': total_return,
        'cagr': cagr,
        'volatility': volatility,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'num_periods': num_periods,
    }


def generate_narrative_observation(
    regime_label: str,
    delta_total_return: float,
    delta_sharpe: float,
    delta_max_drawdown: float,
) -> str:
    """
    Generate auto-narrative for a regime based on metric deltas.

    Args:
        regime_label: Human-readable regime name
        delta_total_return: M+Q total return - momentum total return
        delta_sharpe: M+Q Sharpe - momentum Sharpe
        delta_max_drawdown: M+Q max DD - momentum max DD (positive = improvement)

    Returns:
        Short narrative string
    """
    observations = []

    # Total return impact
    if abs(delta_total_return) < 0.02:
        observations.append("similar returns")
    elif delta_total_return > 0.05:
        observations.append(f"quality boosts returns (+{delta_total_return:.1%})")
    elif delta_total_return < -0.05:
        observations.append(f"quality hurts returns ({delta_total_return:.1%})")

    # Sharpe impact
    if delta_sharpe > 0.2:
        observations.append(f"strong Sharpe improvement (+{delta_sharpe:.2f})")
    elif delta_sharpe > 0.05:
        observations.append(f"modest Sharpe gain (+{delta_sharpe:.2f})")
    elif delta_sharpe < -0.2:
        observations.append(f"Sharpe deteriorates ({delta_sharpe:.2f})")

    # Drawdown impact (remember: less negative = better)
    if delta_max_drawdown > 0.03:
        observations.append(f"reduced drawdown (-{abs(delta_max_drawdown):.1%})")
    elif delta_max_drawdown < -0.03:
        observations.append(f"worse drawdown (+{abs(delta_max_drawdown):.1%})")

    if not observations:
        return f"**{regime_label}**: Minimal impact from quality factor"
    else:
        return f"**{regime_label}**: {', '.join(observations)}"


# ============================================================================
# MAIN DIAGNOSTIC RUNNER
# ============================================================================

class RegimeDiagnosticRunner:
    """
    Runs regime-specific diagnostic comparison of momentum-only vs momentum+quality.
    """

    def __init__(
        self,
        start_date: str = '2015-04-01',
        end_date: str = '2024-12-31',
        initial_capital: float = 100000.0,
    ):
        """
        Initialize regime diagnostic runner.

        Args:
            start_date: Full backtest start date (must cover all regimes)
            end_date: Full backtest end date (must cover all regimes)
            initial_capital: Starting capital
        """
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital

        self.dm = DataManager()
        self.um = UniverseManager(self.dm)

        logger.info("=" * 80)
        logger.info("MOMENTUM + QUALITY REGIME DIAGNOSTIC")
        logger.info("=" * 80)
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Capital: ${initial_capital:,.0f}")
        logger.info(f"Universe: sp500_actual (min_price=5.0)")
        logger.info(f"Regimes: {len(REGIMES)}")
        for regime in REGIMES:
            logger.info(f"  - {regime['label']}: {regime['start']} to {regime['end']}")
        logger.info("=" * 80)
        logger.info("")

    def run_diagnostic(self) -> Dict:
        """
        Run full regime diagnostic.

        Returns:
            Dict with regime metrics and comparison data
        """
        results = {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_capital': self.initial_capital,
            'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'regimes': REGIMES,
        }

        # Define shared universe function
        def universe_fn(rebal_date: str) -> List[str]:
            """Get S&P 500 PIT universe at rebalance date."""
            universe = self.um.get_universe(
                universe_type='sp500_actual',
                as_of_date=rebal_date,
                min_price=5.0
            )

            if isinstance(universe, pd.Series):
                return universe.tolist()
            elif isinstance(universe, pd.DataFrame):
                return universe.index.tolist()
            else:
                return list(universe)

        # Shared backtest config
        config = BacktestConfig(
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.initial_capital,
            rebalance_schedule='M',
            long_only=True,
            equal_weight=True,
            track_daily_equity=False,
            data_manager=self.dm
        )

        # 1. Run momentum-only ensemble (price-based path)
        logger.info("Running momentum-only baseline...")
        momentum_ensemble = get_momentum_v2_adaptive_quintile_ensemble(self.dm)
        momentum_signal_fn = make_ensemble_signal_fn(momentum_ensemble, self.dm, lookback_days=500)
        momentum_result = run_backtest(universe_fn, momentum_signal_fn, config)

        logger.info(f"  Momentum-only complete: {len(momentum_result.equity_curve)} equity points")
        logger.info("")

        # 2. Run momentum + quality ensemble (cross-sectional path)
        logger.info("Running momentum + quality ensemble...")
        multi_ensemble = get_momentum_quality_v1_ensemble(self.dm)
        multi_signal_fn = make_multisignal_ensemble_fn(multi_ensemble, self.dm)
        multi_result = run_backtest(universe_fn, multi_signal_fn, config)

        logger.info(f"  Momentum+Quality complete: {len(multi_result.equity_curve)} equity points")
        logger.info("")

        # 3. Compute per-regime metrics
        logger.info("Computing per-regime metrics...")
        regime_metrics = []

        for regime in REGIMES:
            logger.info(f"  Processing regime: {regime['label']}")

            # Momentum-only metrics
            momentum_metrics = compute_regime_metrics(
                momentum_result.equity_curve,
                regime['start'],
                regime['end'],
            )

            # Momentum+Quality metrics
            mq_metrics = compute_regime_metrics(
                multi_result.equity_curve,
                regime['start'],
                regime['end'],
            )

            # Store per-strategy rows
            regime_metrics.append({
                'regime_name': regime['name'],
                'regime_label': regime['label'],
                'start_date': regime['start'],
                'end_date': regime['end'],
                'strategy': 'momentum_v2',
                **momentum_metrics,
            })

            regime_metrics.append({
                'regime_name': regime['name'],
                'regime_label': regime['label'],
                'start_date': regime['start'],
                'end_date': regime['end'],
                'strategy': 'momentum_quality_v1',
                **mq_metrics,
            })

        logger.info("")

        # 4. Compute deltas (M+Q - momentum)
        logger.info("Computing regime deltas...")
        regime_deltas = []

        for regime in REGIMES:
            # Extract momentum and M+Q metrics for this regime
            momentum_row = next(m for m in regime_metrics if m['regime_name'] == regime['name'] and m['strategy'] == 'momentum_v2')
            mq_row = next(m for m in regime_metrics if m['regime_name'] == regime['name'] and m['strategy'] == 'momentum_quality_v1')

            delta = {
                'regime_name': regime['name'],
                'regime_label': regime['label'],
                'delta_total_return': mq_row['total_return'] - momentum_row['total_return'],
                'delta_cagr': mq_row['cagr'] - momentum_row['cagr'],
                'delta_volatility': mq_row['volatility'] - momentum_row['volatility'],
                'delta_sharpe': mq_row['sharpe'] - momentum_row['sharpe'],
                'delta_max_drawdown': mq_row['max_drawdown'] - momentum_row['max_drawdown'],  # Positive = improvement
            }

            regime_deltas.append(delta)

            logger.info(f"  {regime['label']}: ΔSharpe={delta['delta_sharpe']:+.3f}, "
                       f"ΔReturn={delta['delta_total_return']:+.2%}, "
                       f"ΔMaxDD={delta['delta_max_drawdown']:+.2%}")

        logger.info("")

        results['regime_metrics'] = regime_metrics
        results['regime_deltas'] = regime_deltas

        # 5. Save outputs
        self._save_regime_csv(results)
        self._save_regime_markdown(results)

        # 6. Print summary
        self._print_summary(results)

        return results

    def _save_regime_csv(self, results: Dict):
        """Save regime comparison CSV."""
        output_dir = Path('results/ensemble_baselines')
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / 'momentum_quality_v1_regime_comparison.csv'

        # Combine metrics and deltas into a single DataFrame
        metrics_df = pd.DataFrame(results['regime_metrics'])
        deltas_df = pd.DataFrame(results['regime_deltas'])

        # Merge on regime
        combined = pd.merge(
            metrics_df,
            deltas_df,
            on=['regime_name', 'regime_label'],
            how='left'
        )

        # Reorder columns for clarity
        col_order = [
            'regime_name', 'regime_label', 'start_date', 'end_date', 'strategy',
            'total_return', 'cagr', 'volatility', 'sharpe', 'max_drawdown', 'num_periods',
            'delta_total_return', 'delta_cagr', 'delta_volatility', 'delta_sharpe', 'delta_max_drawdown',
        ]
        combined = combined[[c for c in col_order if c in combined.columns]]

        combined.to_csv(csv_path, index=False)
        logger.info(f"Saved regime comparison CSV: {csv_path}")

    def _save_regime_markdown(self, results: Dict):
        """Save regime diagnostic Markdown report."""
        output_dir = Path('results/ensemble_baselines')
        md_path = output_dir / 'momentum_quality_v1_regime_diagnostic.md'

        content = f"""# Momentum + Quality v1 Regime Diagnostic

**Generated:** {results['generated']}
**Period:** {results['start_date']} to {results['end_date']}
**Capital:** ${results['initial_capital']:,.0f}
**Universe:** S&P 500 (actual constituents, min_price=$5)
**Rebalance:** Monthly

---

## Regime Definitions

This diagnostic breaks down performance across {len(results['regimes'])} distinct market regimes:

"""
        for regime in results['regimes']:
            content += f"**{regime['label']}** ({regime['start']} to {regime['end']})\n"
            content += f"- {regime['description']}\n\n"

        content += """---

## Per-Regime Performance Comparison

### Momentum-Only Baseline

| Regime | Total Return | CAGR | Volatility | Sharpe | Max Drawdown | Periods |
|--------|--------------|------|------------|--------|--------------|---------|
"""
        # Extract momentum-only rows
        momentum_rows = [m for m in results['regime_metrics'] if m['strategy'] == 'momentum_v2']
        for row in momentum_rows:
            content += f"| {row['regime_label']} | {row['total_return']:.2%} | {row['cagr']:.2%} | {row['volatility']:.2%} | {row['sharpe']:.3f} | {row['max_drawdown']:.2%} | {row['num_periods']} |\n"

        content += """
### Momentum + Quality v1

| Regime | Total Return | CAGR | Volatility | Sharpe | Max Drawdown | Periods |
|--------|--------------|------|------------|--------|--------------|---------|
"""
        # Extract M+Q rows
        mq_rows = [m for m in results['regime_metrics'] if m['strategy'] == 'momentum_quality_v1']
        for row in mq_rows:
            content += f"| {row['regime_label']} | {row['total_return']:.2%} | {row['cagr']:.2%} | {row['volatility']:.2%} | {row['sharpe']:.3f} | {row['max_drawdown']:.2%} | {row['num_periods']} |\n"

        content += """
---

## Regime Delta Analysis (M+Q minus Momentum)

**Note:** Positive ΔMax Drawdown = less severe drawdown (improvement)

| Regime | ΔTotal Return | ΔCAGR | ΔVolatility | ΔSharpe | ΔMax Drawdown |
|--------|---------------|-------|-------------|---------|---------------|
"""
        for delta in results['regime_deltas']:
            content += f"| {delta['regime_label']} | {delta['delta_total_return']:+.2%} | {delta['delta_cagr']:+.2%} | {delta['delta_volatility']:+.2%} | {delta['delta_sharpe']:+.3f} | {delta['delta_max_drawdown']:+.2%} |\n"

        content += """
---

## Observations by Regime

"""
        # Auto-generate narrative per regime
        for delta in results['regime_deltas']:
            narrative = generate_narrative_observation(
                delta['regime_label'],
                delta['delta_total_return'],
                delta['delta_sharpe'],
                delta['delta_max_drawdown'],
            )
            content += f"{narrative}\n\n"

        content += """---

## Summary: Where Does Quality Earn Its Keep?

"""
        # Find best and worst regimes for quality
        deltas_sorted_by_sharpe = sorted(results['regime_deltas'], key=lambda x: x['delta_sharpe'], reverse=True)
        best_regime = deltas_sorted_by_sharpe[0]
        worst_regime = deltas_sorted_by_sharpe[-1]

        content += f"""### Quality Shines Most In:
- **{best_regime['regime_label']}**: Sharpe improvement of {best_regime['delta_sharpe']:+.3f}, max DD improvement of {best_regime['delta_max_drawdown']:+.2%}

### Quality Adds Least (or Hurts) In:
- **{worst_regime['regime_label']}**: Sharpe change of {worst_regime['delta_sharpe']:+.3f}, max DD change of {worst_regime['delta_max_drawdown']:+.2%}

### Recommendations

"""
        # Check if quality is consistently helpful
        positive_sharpe_count = sum(1 for d in results['regime_deltas'] if d['delta_sharpe'] > 0)
        total_regimes = len(results['regime_deltas'])

        if positive_sharpe_count >= total_regimes * 0.8:
            content += f"- Quality improves Sharpe in {positive_sharpe_count}/{total_regimes} regimes → **Consider increasing quality weight**\n"
        elif positive_sharpe_count >= total_regimes * 0.5:
            content += f"- Quality helps in {positive_sharpe_count}/{total_regimes} regimes → **Maintain current 0.5/0.5 weights**\n"
        else:
            content += f"- Quality helps in only {positive_sharpe_count}/{total_regimes} regimes → **Consider reducing quality weight or regime-conditional allocation**\n"

        # Check drawdown protection
        positive_dd_count = sum(1 for d in results['regime_deltas'] if d['delta_max_drawdown'] > 0)
        if positive_dd_count >= total_regimes * 0.7:
            content += f"- Quality reduces drawdown in {positive_dd_count}/{total_regimes} regimes → **Strong defensive value**\n"

        content += """
---

**Status:** Phase 3 Milestone 3.3 regime diagnostic
**Note:** This uses the new cross-sectional ensemble pathway for momentum+quality
"""

        md_path.write_text(content)
        logger.info(f"Saved regime diagnostic: {md_path}")

    def _print_summary(self, results: Dict):
        """Print summary to stdout."""
        logger.info("=" * 80)
        logger.info("REGIME DIAGNOSTIC SUMMARY")
        logger.info("=" * 80)
        logger.info("")

        for delta in results['regime_deltas']:
            logger.info(f"{delta['regime_label']:25} | ΔSharpe: {delta['delta_sharpe']:+7.3f} | "
                       f"ΔReturn: {delta['delta_total_return']:+7.2%} | "
                       f"ΔMaxDD: {delta['delta_max_drawdown']:+7.2%}")

        logger.info("")
        logger.info("=" * 80)


def main():
    """Run regime diagnostic."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Run momentum + quality regime diagnostic',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--start', default='2015-04-01',
                       help='Start date (default: 2015-04-01)')
    parser.add_argument('--end', default='2024-12-31',
                       help='End date (default: 2024-12-31)')
    parser.add_argument('--capital', type=float, default=100000,
                       help='Initial capital (default: $100,000)')

    args = parser.parse_args()

    # Run diagnostic
    runner = RegimeDiagnosticRunner(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital
    )

    results = runner.run_diagnostic()

    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ REGIME DIAGNOSTIC COMPLETE")
    logger.info("=" * 80)
    logger.info("Results saved to:")
    logger.info("  - results/ensemble_baselines/momentum_quality_v1_regime_comparison.csv")
    logger.info("  - results/ensemble_baselines/momentum_quality_v1_regime_diagnostic.md")
    logger.info("")


if __name__ == '__main__':
    main()
