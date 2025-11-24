"""
Momentum + Quality Weight Sweep

Systematically tests different weight combinations to find optimal momentum/quality mix.

Plan:
- Weight grid: Momentum weights in [0.25, 0.5, 0.75, 1.0], Quality = 1 - momentum_weight
- Universe: S&P 500 actual (same as baseline and regime diagnostics)
- Period: 2015-04-01 to 2024-12-31 (full diagnostic period)
- Metrics: Total return, CAGR, volatility, Sharpe, max drawdown
- Output paths:
  - results/ensemble_baselines/momentum_quality_v1_weight_sweep.csv
  - results/ensemble_baselines/momentum_quality_v1_weight_sweep.md

Usage:
    python3 scripts/run_momentum_quality_weight_sweep.py

    # Quick test with reduced grid
    python3 scripts/run_momentum_quality_weight_sweep.py --quick

Author: Claude Code (Phase 3 Milestone 3.4)
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
from core.signal_adapters import make_multisignal_ensemble_fn
from signals.ml.ensemble_signal import EnsembleSignal, EnsembleMember
from config import get_logger

logger = get_logger(__name__)


# ============================================================================
# WEIGHT GRID DEFINITIONS
# ============================================================================

# Default grid: balanced exploration of weight space
DEFAULT_MOMENTUM_WEIGHTS = [0.25, 0.5, 0.75, 1.0]

# Quick grid: fewer points for smoke testing
QUICK_MOMENTUM_WEIGHTS = [0.5, 1.0]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_weighted_ensemble(
    dm: DataManager,
    momentum_weight: float,
) -> EnsembleSignal:
    """
    Build momentum+quality ensemble with specified weights.

    Args:
        dm: DataManager instance
        momentum_weight: Weight for momentum (0-1), quality gets (1 - momentum_weight)

    Returns:
        EnsembleSignal with weighted members

    Raises:
        ValueError: If weights invalid or sum incorrectly
    """
    if not 0 <= momentum_weight <= 1:
        raise ValueError(f"Momentum weight must be in [0, 1], got {momentum_weight}")

    quality_weight = 1.0 - momentum_weight

    # Momentum v2 params (production)
    momentum_params: Dict = {
        "formation_period": 308,
        "skip_period": 0,
        "winsorize_pct": [0.4, 99.6],
        "rebalance_frequency": "monthly",
        "quintiles": True,
        "quintile_mode": "adaptive",
    }

    # Quality v1 params (production)
    quality_params: Dict = {
        "w_profitability": 0.4,
        "w_growth": 0.3,
        "w_safety": 0.3,
        "winsorize_pct": [5, 95],
        "quintiles": True,
        "quintile_mode": "adaptive",
        "min_coverage": 0.5,
    }

    members = []

    # Add momentum if weight > 0
    if momentum_weight > 0:
        members.append(
            EnsembleMember(
                signal_name="InstitutionalMomentum",
                version="v2",
                weight=momentum_weight,
                normalize="none",
                params=momentum_params,
            )
        )

    # Add quality if weight > 0
    if quality_weight > 0:
        members.append(
            EnsembleMember(
                signal_name="CrossSectionalQuality",
                version="v1",
                weight=quality_weight,
                normalize="none",
                params=quality_params,
            )
        )

    if not members:
        raise ValueError("At least one signal must have positive weight")

    logger.info(f"Building ensemble: momentum={momentum_weight:.2f}, quality={quality_weight:.2f}")

    return EnsembleSignal(
        members=members,
        data_manager=dm,
        enforce_go_only=True,
    )


def compute_custom_score(sharpe: float, max_drawdown: float) -> float:
    """
    Compute custom score balancing risk-adjusted returns and drawdown.

    Score = Sharpe - 0.5 * |MaxDD|

    This penalizes drawdown while rewarding Sharpe. Weights are heuristic.

    Args:
        sharpe: Sharpe ratio (annualized)
        max_drawdown: Max drawdown (negative, e.g., -0.25)

    Returns:
        Custom score (higher is better)
    """
    # Max DD is negative, so abs() makes it positive for penalty
    dd_penalty = 0.5 * abs(max_drawdown)
    return sharpe - dd_penalty


# ============================================================================
# WEIGHT SWEEP RUNNER
# ============================================================================

class WeightSweepRunner:
    """
    Runs systematic weight sweep for momentum+quality ensemble.
    """

    def __init__(
        self,
        momentum_weights: List[float],
        start_date: str = '2015-04-01',
        end_date: str = '2024-12-31',
        initial_capital: float = 100000.0,
    ):
        """
        Initialize weight sweep runner.

        Args:
            momentum_weights: List of momentum weights to test (quality = 1 - momentum)
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital
        """
        self.momentum_weights = sorted(momentum_weights)
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital

        self.dm = DataManager()
        self.um = UniverseManager(self.dm)

        logger.info("=" * 80)
        logger.info("MOMENTUM + QUALITY WEIGHT SWEEP")
        logger.info("=" * 80)
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Capital: ${initial_capital:,.0f}")
        logger.info(f"Universe: sp500_actual (min_price=5.0)")
        logger.info(f"Weight grid: {len(self.momentum_weights)} points")
        for w in self.momentum_weights:
            logger.info(f"  - Momentum: {w:.2f}, Quality: {1-w:.2f}")
        logger.info("=" * 80)
        logger.info("")

    def run_sweep(self) -> Dict:
        """
        Run full weight sweep.

        Returns:
            Dict with sweep results and recommendations
        """
        results = {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_capital': self.initial_capital,
            'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'momentum_weights': self.momentum_weights,
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

        # Run backtest for each weight combination
        sweep_metrics = []

        for i, mom_weight in enumerate(self.momentum_weights, 1):
            qual_weight = 1.0 - mom_weight

            logger.info(f"Weight point {i}/{len(self.momentum_weights)}: "
                       f"Momentum={mom_weight:.2f}, Quality={qual_weight:.2f}")

            # Build ensemble
            ensemble = build_weighted_ensemble(self.dm, mom_weight)

            # Create signal function via cross-sectional adapter
            signal_fn = make_multisignal_ensemble_fn(ensemble, self.dm)

            # Run backtest
            backtest_result = run_backtest(universe_fn, signal_fn, config)

            # Extract metrics
            metrics = {
                'momentum_weight': mom_weight,
                'quality_weight': qual_weight,
                'total_return': backtest_result.total_return,
                'cagr': backtest_result.cagr,
                'volatility': backtest_result.volatility,
                'sharpe': backtest_result.sharpe,
                'max_drawdown': backtest_result.max_drawdown,
                'num_rebalances': backtest_result.num_rebalances,
            }

            # Compute custom score
            metrics['custom_score'] = compute_custom_score(
                backtest_result.sharpe,
                backtest_result.max_drawdown
            )

            sweep_metrics.append(metrics)

            logger.info(f"  Total Return: {metrics['total_return']:.2%}, "
                       f"Sharpe: {metrics['sharpe']:.3f}, "
                       f"MaxDD: {metrics['max_drawdown']:.2%}, "
                       f"Score: {metrics['custom_score']:.3f}")
            logger.info("")

        results['sweep_metrics'] = sweep_metrics

        # Find optimal weights
        best_sharpe = max(sweep_metrics, key=lambda x: x['sharpe'])
        best_score = max(sweep_metrics, key=lambda x: x['custom_score'])
        best_return = max(sweep_metrics, key=lambda x: x['total_return'])
        best_dd = max(sweep_metrics, key=lambda x: x['max_drawdown'])  # Less negative is better

        results['best_weights'] = {
            'sharpe': best_sharpe,
            'custom_score': best_score,
            'total_return': best_return,
            'max_drawdown': best_dd,
        }

        # Save outputs
        self._save_sweep_csv(results)
        self._save_sweep_markdown(results)

        # Print summary
        self._print_summary(results)

        return results

    def _save_sweep_csv(self, results: Dict):
        """Save weight sweep CSV."""
        output_dir = Path('results/ensemble_baselines')
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / 'momentum_quality_v1_weight_sweep.csv'

        df = pd.DataFrame(results['sweep_metrics'])

        # Reorder columns
        col_order = [
            'momentum_weight', 'quality_weight',
            'total_return', 'cagr', 'volatility', 'sharpe', 'max_drawdown',
            'custom_score', 'num_rebalances',
        ]
        df = df[col_order]

        df.to_csv(csv_path, index=False)
        logger.info(f"Saved weight sweep CSV: {csv_path}")

    def _save_sweep_markdown(self, results: Dict):
        """Save weight sweep Markdown report."""
        output_dir = Path('results/ensemble_baselines')
        md_path = output_dir / 'momentum_quality_v1_weight_sweep.md'

        content = f"""# Momentum + Quality v1 Weight Sweep

**Generated:** {results['generated']}
**Period:** {results['start_date']} to {results['end_date']}
**Capital:** ${results['initial_capital']:,.0f}
**Universe:** S&P 500 (actual constituents, min_price=$5)
**Rebalance:** Monthly

---

## Weight Grid Tested

Momentum weights: `{results['momentum_weights']}`
Quality weight = `1 - momentum_weight`

Total configurations tested: **{len(results['sweep_metrics'])}**

---

## Full Sweep Results

| Momentum | Quality | Total Return | CAGR | Volatility | Sharpe | Max Drawdown | Custom Score |
|----------|---------|--------------|------|------------|--------|--------------|--------------|
"""

        for m in results['sweep_metrics']:
            content += (
                f"| {m['momentum_weight']:.2f} | {m['quality_weight']:.2f} | "
                f"{m['total_return']:.2%} | {m['cagr']:.2%} | "
                f"{m['volatility']:.2%} | {m['sharpe']:.3f} | "
                f"{m['max_drawdown']:.2%} | {m['custom_score']:.3f} |\n"
            )

        content += """
---

## Optimal Weights by Metric

### Best Sharpe Ratio
"""
        best_sharpe = results['best_weights']['sharpe']
        content += f"""- **Momentum: {best_sharpe['momentum_weight']:.2f}, Quality: {best_sharpe['quality_weight']:.2f}**
- Sharpe: {best_sharpe['sharpe']:.3f}
- Total Return: {best_sharpe['total_return']:.2%}
- Max Drawdown: {best_sharpe['max_drawdown']:.2%}

"""

        content += """### Best Custom Score (Sharpe - 0.5 * |MaxDD|)
"""
        best_score = results['best_weights']['custom_score']
        content += f"""- **Momentum: {best_score['momentum_weight']:.2f}, Quality: {best_score['quality_weight']:.2f}**
- Custom Score: {best_score['custom_score']:.3f}
- Sharpe: {best_score['sharpe']:.3f}
- Max Drawdown: {best_score['max_drawdown']:.2%}

"""

        content += """### Best Total Return
"""
        best_return = results['best_weights']['total_return']
        content += f"""- **Momentum: {best_return['momentum_weight']:.2f}, Quality: {best_return['quality_weight']:.2f}**
- Total Return: {best_return['total_return']:.2%}
- Sharpe: {best_return['sharpe']:.3f}

"""

        content += """### Best Max Drawdown (least severe)
"""
        best_dd = results['best_weights']['max_drawdown']
        content += f"""- **Momentum: {best_dd['momentum_weight']:.2f}, Quality: {best_dd['quality_weight']:.2f}**
- Max Drawdown: {best_dd['max_drawdown']:.2%}
- Sharpe: {best_dd['sharpe']:.3f}

"""

        # Analyze tradeoff curve
        content += """---

## Observations

"""

        # Check if there's a clear optimum or flat region
        sharpe_values = [m['sharpe'] for m in results['sweep_metrics']]
        sharpe_range = max(sharpe_values) - min(sharpe_values)

        if sharpe_range < 0.2:
            content += f"- Sharpe relatively flat across weight range (range: {sharpe_range:.3f}) → **Weight choice less critical**\n"
        else:
            content += f"- Sharpe varies significantly across weights (range: {sharpe_range:.3f}) → **Weight selection matters**\n"

        # Check if pure momentum or pure quality wins
        pure_momentum = next(m for m in results['sweep_metrics'] if m['momentum_weight'] == 1.0)
        if best_sharpe['momentum_weight'] == 1.0:
            content += "- Pure momentum (no quality) achieves best Sharpe → **Quality may not be needed**\n"
        elif best_sharpe['momentum_weight'] < 0.5:
            content += f"- Quality-heavy allocation (quality={best_sharpe['quality_weight']:.0%}) wins on Sharpe → **Quality adds significant value**\n"
        else:
            content += f"- Balanced allocation (momentum={best_sharpe['momentum_weight']:.0%}) wins on Sharpe → **Diversification benefit**\n"

        # Drawdown analysis
        dd_improvement = best_dd['max_drawdown'] - pure_momentum['max_drawdown']
        if dd_improvement > 0.03:  # 3%+ improvement
            content += f"- Adding quality improves max drawdown by {dd_improvement:.1%} → **Strong defensive value**\n"

        content += """
### Recommendation

"""

        # Recommend based on best custom score (balances Sharpe and DD)
        rec_weight = best_score['momentum_weight']
        content += f"""**Recommended balanced allocation:** Momentum={rec_weight:.0%}, Quality={1-rec_weight:.0%}

This weight balances risk-adjusted returns (Sharpe) with drawdown control.

For more aggressive positioning, consider tilting towards the best Sharpe weight.
For more defensive positioning, tilt towards the best max drawdown weight.

---

**Status:** Phase 3 Milestone 3.4 weight sweep
**Note:** Uses cross-sectional ensemble pathway for all weight combinations
"""

        md_path.write_text(content)
        logger.info(f"Saved weight sweep diagnostic: {md_path}")

    def _print_summary(self, results: Dict):
        """Print summary to stdout."""
        logger.info("=" * 80)
        logger.info("WEIGHT SWEEP SUMMARY")
        logger.info("=" * 80)
        logger.info("")

        logger.info(f"{'Momentum':<10} {'Quality':<10} {'Return':<12} {'Sharpe':<10} {'MaxDD':<12} {'Score':<10}")
        logger.info("-" * 80)

        for m in results['sweep_metrics']:
            logger.info(
                f"{m['momentum_weight']:<10.2f} "
                f"{m['quality_weight']:<10.2f} "
                f"{m['total_return']:<12.2%} "
                f"{m['sharpe']:<10.3f} "
                f"{m['max_drawdown']:<12.2%} "
                f"{m['custom_score']:<10.3f}"
            )

        logger.info("")
        logger.info("Best weights:")
        best_score = results['best_weights']['custom_score']
        logger.info(f"  Custom Score: Momentum={best_score['momentum_weight']:.2f}, "
                   f"Quality={best_score['quality_weight']:.2f}, "
                   f"Score={best_score['custom_score']:.3f}")

        best_sharpe = results['best_weights']['sharpe']
        logger.info(f"  Sharpe:       Momentum={best_sharpe['momentum_weight']:.2f}, "
                   f"Quality={best_sharpe['quality_weight']:.2f}, "
                   f"Sharpe={best_sharpe['sharpe']:.3f}")

        logger.info("")
        logger.info("=" * 80)


def main():
    """Run weight sweep."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Run momentum + quality weight sweep',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--quick', action='store_true',
                       help='Use reduced weight grid for smoke testing')
    parser.add_argument('--start', default='2015-04-01',
                       help='Start date (default: 2015-04-01)')
    parser.add_argument('--end', default='2024-12-31',
                       help='End date (default: 2024-12-31)')
    parser.add_argument('--capital', type=float, default=100000,
                       help='Initial capital (default: $100,000)')

    args = parser.parse_args()

    # Select weight grid
    if args.quick:
        momentum_weights = QUICK_MOMENTUM_WEIGHTS
        logger.info("Using QUICK weight grid (2 points)")
    else:
        momentum_weights = DEFAULT_MOMENTUM_WEIGHTS
        logger.info("Using DEFAULT weight grid (4 points)")

    # Run sweep
    runner = WeightSweepRunner(
        momentum_weights=momentum_weights,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital
    )

    results = runner.run_sweep()

    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ WEIGHT SWEEP COMPLETE")
    logger.info("=" * 80)
    logger.info("Results saved to:")
    logger.info("  - results/ensemble_baselines/momentum_quality_v1_weight_sweep.csv")
    logger.info("  - results/ensemble_baselines/momentum_quality_v1_weight_sweep.md")
    logger.info("")


if __name__ == '__main__':
    main()
