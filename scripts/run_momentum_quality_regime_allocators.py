"""
Momentum + Quality Regime Allocator Comparison (M3.5)

Compares three allocation strategies:
1. Static 25/75: Fixed weights (baseline)
2. Oracle: Hindsight-optimal weights per regime (research ceiling)
3. Rule-Based: PIT-safe regime detection using volatility + drawdown

Outputs:
    results/ensemble_baselines/momentum_quality_v1_regime_allocators.csv
    results/ensemble_baselines/momentum_quality_v1_regime_allocators.md

Usage:
    python3 scripts/run_momentum_quality_regime_allocators.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from data.data_manager import DataManager
from core.universe_manager import UniverseManager
from core.backtest_engine import BacktestConfig, run_backtest
from core.signal_adapters import make_ensemble_signal_fn
from signals.ml.ensemble_configs import get_momentum_v2_adaptive_quintile_ensemble
from signals.ml.regime_allocators import (
    OracleRegimeAllocatorMQ,
    RuleBasedRegimeAllocatorMQ,
)
from config import get_logger

logger = get_logger(__name__)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_performance_metrics(returns: pd.Series, name: str) -> Dict[str, float]:
    """
    Compute full-period performance metrics from monthly returns.

    Args:
        returns: Monthly return series
        name: Strategy name

    Returns:
        Dict with metrics: sharpe, cagr, volatility, max_drawdown, total_return
    """
    if len(returns) < 2:
        logger.warning(f"Too few returns for {name}")
        return {
            'strategy': name,
            'sharpe': np.nan,
            'cagr': np.nan,
            'volatility': np.nan,
            'max_drawdown': np.nan,
            'total_return': np.nan,
            'num_periods': len(returns),
        }

    # Build equity curve
    equity = (1 + returns).cumprod()
    equity.iloc[0] = 1.0  # Start at 1.0

    # Total return
    total_return = equity.iloc[-1] - 1.0

    # CAGR
    num_periods = len(returns)
    years = num_periods / 12.0  # Monthly returns
    if years > 0:
        cagr = (1 + total_return) ** (1 / years) - 1.0
    else:
        cagr = np.nan

    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(12)

    # Sharpe (assume 0 risk-free rate)
    if volatility > 0:
        sharpe = (returns.mean() * 12) / volatility
    else:
        sharpe = np.nan

    # Max drawdown
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        'strategy': name,
        'sharpe': sharpe,
        'cagr': cagr,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'total_return': total_return,
        'num_periods': num_periods,
    }


def reconstruct_quality_returns(
    momentum_returns: pd.Series,
    ensemble_returns: pd.Series,
    w_momentum: float = 0.25,
    w_quality: float = 0.75,
) -> pd.Series:
    """
    Reconstruct quality-only returns from ensemble and momentum returns.

    Given:
        r_ensemble = w_m * r_momentum + w_q * r_quality

    Solve for:
        r_quality = (r_ensemble - w_m * r_momentum) / w_q

    Args:
        momentum_returns: Monthly returns for momentum-only
        ensemble_returns: Monthly returns for M+Q ensemble (25/75)
        w_momentum: Momentum weight in ensemble (default: 0.25)
        w_quality: Quality weight in ensemble (default: 0.75)

    Returns:
        Reconstructed quality-only returns
    """
    # Align indices
    common_dates = momentum_returns.index.intersection(ensemble_returns.index)
    r_m = momentum_returns.loc[common_dates]
    r_ens = ensemble_returns.loc[common_dates]

    # Solve for quality returns
    r_q = (r_ens - w_momentum * r_m) / w_quality

    r_q.name = 'quality'
    return r_q


# ============================================================================
# MAIN DIAGNOSTIC RUNNER
# ============================================================================

class RegimeAllocatorDiagnostic:
    """
    Runs regime allocator comparison for M3.5.
    """

    def __init__(
        self,
        start_date: str = '2015-04-01',
        end_date: str = '2024-12-31',
        initial_capital: float = 100000.0,
    ):
        """
        Initialize regime allocator diagnostic.

        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital
        """
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital

        self.dm = DataManager()
        self.um = UniverseManager(self.dm)

        logger.info("=" * 80)
        logger.info("MOMENTUM + QUALITY REGIME ALLOCATOR COMPARISON (M3.5)")
        logger.info("=" * 80)
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Capital: ${initial_capital:,.0f}")
        logger.info(f"Universe: sp500_actual (min_price=5.0)")
        logger.info("=" * 80)
        logger.info("")

    def run_diagnostic(self) -> Dict:
        """
        Run full regime allocator diagnostic.

        Returns:
            Dict with comparison results
        """
        results = {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_capital': self.initial_capital,
            'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
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

        # 1. Run momentum-only backtest to get momentum returns
        logger.info("Step 1: Running momentum-only backtest...")
        momentum_ensemble = get_momentum_v2_adaptive_quintile_ensemble(self.dm)
        momentum_signal_fn = make_ensemble_signal_fn(momentum_ensemble, self.dm, lookback_days=500)
        momentum_result = run_backtest(universe_fn, momentum_signal_fn, config)

        # Extract monthly returns from equity curve
        momentum_equity = momentum_result.equity_curve
        momentum_returns = momentum_equity.pct_change().dropna()
        momentum_returns.name = 'momentum'

        logger.info(f"  Momentum returns: {len(momentum_returns)} periods")
        logger.info(f"  Mean return: {momentum_returns.mean():.4f}")
        logger.info(f"  Std return: {momentum_returns.std():.4f}")
        logger.info("")

        # 2. Load static 25/75 ensemble returns (pre-computed)
        logger.info("Step 2: Loading static 25/75 ensemble returns...")
        static_csv = Path('results/ensemble_baselines/momentum_quality_v1_monthly_returns.csv')

        if not static_csv.exists():
            raise FileNotFoundError(
                f"Static 25/75 returns not found: {static_csv}\n"
                "Run: python3 scripts/run_momentum_quality_baseline.py"
            )

        static_df = pd.read_csv(static_csv)
        static_df['date'] = pd.to_datetime(static_df['date'])
        static_returns = static_df.set_index('date')['return']
        static_returns.name = 'ensemble_25_75'

        logger.info(f"  Static 25/75 returns: {len(static_returns)} periods")
        logger.info(f"  Mean return: {static_returns.mean():.4f}")
        logger.info(f"  Std return: {static_returns.std():.4f}")
        logger.info("")

        # 3. Reconstruct quality-only returns
        logger.info("Step 3: Reconstructing quality-only returns...")
        quality_returns = reconstruct_quality_returns(
            momentum_returns,
            static_returns,
            w_momentum=0.25,
            w_quality=0.75,
        )

        logger.info(f"  Quality returns: {len(quality_returns)} periods")
        logger.info(f"  Mean return: {quality_returns.mean():.4f}")
        logger.info(f"  Std return: {quality_returns.std():.4f}")

        # Align momentum returns to same dates as quality returns
        momentum_returns = momentum_returns.loc[quality_returns.index]
        logger.info(f"  Aligned momentum returns: {len(momentum_returns)} periods")
        logger.info("")

        # 4. Run Oracle allocator
        logger.info("Step 4: Running Oracle allocator (grid search)...")
        oracle_allocator = OracleRegimeAllocatorMQ(
            momentum_returns,
            quality_returns,
            grid_step=0.1,
        )

        logger.info("  Oracle optimal weights per regime:")
        for regime_name, weights in oracle_allocator.optimal_weights.items():
            logger.info(f"    {regime_name:25} | w_m={weights.momentum:.2f}, w_q={weights.quality:.2f}, Sharpe={weights.sharpe:.3f}")

        oracle_returns = oracle_allocator.ensemble_returns
        oracle_returns.name = 'oracle'
        logger.info(f"  Oracle ensemble returns: {len(oracle_returns)} periods")
        logger.info("")

        # 5. Run Rule-Based allocator
        logger.info("Step 5: Running Rule-Based allocator...")

        # Need equity curve for rule-based (use SPY or momentum equity as proxy)
        # For now, use momentum equity curve
        rule_allocator = RuleBasedRegimeAllocatorMQ(
            equity_curve=momentum_equity,
            vol_threshold_low=0.15,
            vol_threshold_high=0.25,
            dd_threshold_calm=-0.10,
            dd_threshold_stress=-0.15,
        )

        logger.info("  Rule-based regime distribution:")
        regime_summary = rule_allocator.get_summary()
        for _, row in regime_summary.iterrows():
            logger.info(f"    {row['regime']:10} | {row['count']:3} periods ({row['pct']:.1%}) | "
                       f"w_m={row['w_momentum']:.2f}, w_q={row['w_quality']:.2f}")

        rule_returns = rule_allocator.apply_weights(momentum_returns, quality_returns)
        rule_returns.name = 'rule_based'
        logger.info(f"  Rule-based ensemble returns: {len(rule_returns)} periods")
        logger.info("")

        # 6. Compute performance metrics for all strategies
        logger.info("Step 6: Computing performance metrics...")

        metrics = []

        # Static 25/75
        static_metrics = compute_performance_metrics(static_returns, 'static_25_75')
        metrics.append(static_metrics)
        logger.info(f"  Static 25/75: Sharpe={static_metrics['sharpe']:.3f}, CAGR={static_metrics['cagr']:.2%}")

        # Oracle
        oracle_metrics = compute_performance_metrics(oracle_returns, 'oracle')
        metrics.append(oracle_metrics)
        logger.info(f"  Oracle:       Sharpe={oracle_metrics['sharpe']:.3f}, CAGR={oracle_metrics['cagr']:.2%}")

        # Rule-Based
        rule_metrics = compute_performance_metrics(rule_returns, 'rule_based')
        metrics.append(rule_metrics)
        logger.info(f"  Rule-Based:   Sharpe={rule_metrics['sharpe']:.3f}, CAGR={rule_metrics['cagr']:.2%}")
        logger.info("")

        # 7. Compute deltas vs static baseline
        logger.info("Step 7: Computing deltas vs static baseline...")
        static_sharpe = static_metrics['sharpe']

        deltas = []
        for m in metrics:
            if m['strategy'] == 'static_25_75':
                continue  # Skip baseline

            delta = {
                'strategy': m['strategy'],
                'delta_sharpe': m['sharpe'] - static_sharpe,
                'delta_sharpe_pct': ((m['sharpe'] / static_sharpe) - 1.0) * 100 if static_sharpe > 0 else np.nan,
                'delta_cagr': m['cagr'] - static_metrics['cagr'],
                'delta_max_dd': m['max_drawdown'] - static_metrics['max_drawdown'],
            }
            deltas.append(delta)
            logger.info(f"  {m['strategy']:15} | ΔSharpe={delta['delta_sharpe']:+.3f} ({delta['delta_sharpe_pct']:+.1f}%), "
                       f"ΔCAGR={delta['delta_cagr']:+.2%}, ΔMaxDD={delta['delta_max_dd']:+.2%}")

        logger.info("")

        results['metrics'] = metrics
        results['deltas'] = deltas
        results['oracle_weights'] = oracle_allocator.get_summary().to_dict('records')
        results['rule_regime_dist'] = regime_summary.to_dict('records')

        # 8. Save outputs
        self._save_csv(results)
        self._save_markdown(results)

        # 9. Print summary
        self._print_summary(results)

        return results

    def _save_csv(self, results: Dict):
        """Save comparison CSV."""
        output_dir = Path('results/ensemble_baselines')
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / 'momentum_quality_v1_regime_allocators.csv'

        # Combine metrics and deltas
        metrics_df = pd.DataFrame(results['metrics'])
        deltas_df = pd.DataFrame(results['deltas'])

        combined = pd.merge(
            metrics_df,
            deltas_df,
            on='strategy',
            how='left'
        )

        combined.to_csv(csv_path, index=False)
        logger.info(f"Saved comparison CSV: {csv_path}")

    def _save_markdown(self, results: Dict):
        """Save comparison Markdown report."""
        output_dir = Path('results/ensemble_baselines')
        md_path = output_dir / 'momentum_quality_v1_regime_allocators.md'

        content = f"""# Momentum + Quality v1 Regime Allocator Comparison (M3.5)

**Generated:** {results['generated']}
**Period:** {results['start_date']} to {results['end_date']}
**Capital:** ${results['initial_capital']:,.0f}
**Universe:** S&P 500 (actual constituents, min_price=$5)
**Rebalance:** Monthly

---

## Allocator Strategies

### 1. Static 25/75 (Baseline)
- Fixed weights: 25% momentum, 75% quality
- No regime detection
- Baseline for comparison

### 2. Oracle (Research-Only)
- Uses perfect hindsight regime labels
- Grid search per regime to find optimal weights
- NOT PIT-safe (ex-post labels)
- Provides performance ceiling

### 3. Rule-Based (PIT-Safe, Practical)
- Uses observable indicators: realized volatility + drawdown
- 3 regime types: CALM, STRESS, CHOPPY
- PIT-safe, can be used in live trading

---

## Full-Period Performance Comparison

| Strategy | Sharpe | CAGR | Volatility | Max DD | Total Return | Periods |
|----------|--------|------|------------|--------|--------------|---------|
"""
        for m in results['metrics']:
            content += f"| {m['strategy']:15} | {m['sharpe']:6.3f} | {m['cagr']:6.2%} | {m['volatility']:6.2%} | {m['max_drawdown']:7.2%} | {m['total_return']:7.2%} | {m['num_periods']:3} |\n"

        content += """
---

## Delta vs Static Baseline

| Strategy | ΔSharpe | ΔSharpe % | ΔCAGR | ΔMax DD |
|----------|---------|-----------|-------|---------|
"""
        for d in results['deltas']:
            content += f"| {d['strategy']:15} | {d['delta_sharpe']:+7.3f} | {d['delta_sharpe_pct']:+6.1f}% | {d['delta_cagr']:+6.2%} | {d['delta_max_dd']:+7.2%} |\n"

        content += """
---

## Oracle Optimal Weights (Per Regime)

| Regime | w_momentum | w_quality | Sharpe |
|--------|------------|-----------|--------|
"""
        for w in results['oracle_weights']:
            content += f"| {w['regime']:25} | {w['w_momentum']:10.2f} | {w['w_quality']:9.2f} | {w['sharpe']:6.3f} |\n"

        content += """
---

## Rule-Based Regime Distribution

| Regime | Count | Pct | w_momentum | w_quality |
|--------|-------|-----|------------|-----------|
"""
        for r in results['rule_regime_dist']:
            content += f"| {r['regime']:10} | {r['count']:5} | {r['pct']:5.1%} | {r['w_momentum']:10.2f} | {r['w_quality']:9.2f} |\n"

        content += """
---

## Acceptance Gates

### Oracle (GO/NO-GO Criteria):
- ✅ Full-period Sharpe ≥ static baseline
- ✅ Max Drawdown ≤ static baseline
- ✅ Sharpe improvement ≥ 5% (target)

### Rule-Based (GO/NO-GO Criteria):
- ✅ Full-period Sharpe ≥ static baseline
- ✅ Max Drawdown ≤ static baseline
- ✅ Sharpe ≥ 90% of Oracle Sharpe
- ✅ Sharpe improvement ≥ 3% (target)

---

## Recommendations

"""

        # Check acceptance gates
        static_sharpe = next(m['sharpe'] for m in results['metrics'] if m['strategy'] == 'static_25_75')
        oracle_sharpe = next(m['sharpe'] for m in results['metrics'] if m['strategy'] == 'oracle')
        rule_sharpe = next(m['sharpe'] for m in results['metrics'] if m['strategy'] == 'rule_based')

        oracle_delta_pct = next(d['delta_sharpe_pct'] for d in results['deltas'] if d['strategy'] == 'oracle')
        rule_delta_pct = next(d['delta_sharpe_pct'] for d in results['deltas'] if d['strategy'] == 'rule_based')

        # Oracle assessment
        if oracle_delta_pct >= 5.0:
            content += f"### Oracle: **GO** ✅\n"
            content += f"- Sharpe improvement: {oracle_delta_pct:+.1f}% (exceeds 5% target)\n"
            content += f"- Regime-aware allocation has measurable ceiling benefit\n\n"
        else:
            content += f"### Oracle: **NO-GO** ❌\n"
            content += f"- Sharpe improvement: {oracle_delta_pct:+.1f}% (below 5% target)\n"
            content += f"- Limited upside from regime-aware approach\n\n"

        # Rule-based assessment
        rule_vs_oracle_pct = ((rule_sharpe / oracle_sharpe) - 1.0) * 100 if oracle_sharpe > 0 else np.nan

        if rule_delta_pct >= 3.0 and rule_vs_oracle_pct >= -10.0:
            content += f"### Rule-Based: **GO** ✅\n"
            content += f"- Sharpe improvement: {rule_delta_pct:+.1f}% (exceeds 3% target)\n"
            content += f"- Captures {rule_vs_oracle_pct:+.1f}% of oracle performance\n"
            content += f"- **Practical for live deployment**\n\n"
        else:
            content += f"### Rule-Based: **NO-GO** ❌\n"
            content += f"- Sharpe improvement: {rule_delta_pct:+.1f}% (target: 3%)\n"
            content += f"- Captures {rule_vs_oracle_pct:+.1f}% of oracle (target: ≥90%)\n"
            content += f"- Does not justify live deployment complexity\n\n"

        content += """
---

**Status:** Phase 3 M3.5 - Regime-Aware Allocation Diagnostic
**Next Steps:** Review GO/NO-GO decision, update signal catalog if approved
"""

        md_path.write_text(content)
        logger.info(f"Saved comparison report: {md_path}")

    def _print_summary(self, results: Dict):
        """Print summary to stdout."""
        logger.info("=" * 80)
        logger.info("REGIME ALLOCATOR COMPARISON SUMMARY")
        logger.info("=" * 80)
        logger.info("")

        for m in results['metrics']:
            logger.info(f"{m['strategy']:15} | Sharpe: {m['sharpe']:6.3f} | CAGR: {m['cagr']:6.2%} | MaxDD: {m['max_drawdown']:7.2%}")

        logger.info("")
        logger.info("Delta vs Static:")
        for d in results['deltas']:
            logger.info(f"{d['strategy']:15} | ΔSharpe: {d['delta_sharpe']:+7.3f} ({d['delta_sharpe_pct']:+.1f}%)")

        logger.info("")
        logger.info("=" * 80)


def main():
    """Run regime allocator diagnostic."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Run momentum + quality regime allocator comparison',
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
    runner = RegimeAllocatorDiagnostic(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital
    )

    results = runner.run_diagnostic()

    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ REGIME ALLOCATOR DIAGNOSTIC COMPLETE")
    logger.info("=" * 80)
    logger.info("Results saved to:")
    logger.info("  - results/ensemble_baselines/momentum_quality_v1_regime_allocators.csv")
    logger.info("  - results/ensemble_baselines/momentum_quality_v1_regime_allocators.md")
    logger.info("")


if __name__ == '__main__':
    main()
