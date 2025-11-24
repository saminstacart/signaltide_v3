"""
Insider Trading Signal - Phase 1 Diagnostics

Run Phase 1 acceptance gate analysis for InstitutionalInsider signal.

Tests 5 acceptance gates:
1. Decile monotonicity (top - bottom ≥ 0.5%/mo)
2. Long-short Sharpe ≥ 0.30 (full sample)
3. t-statistic ≥ 2.0 (statistical significance)
4. Recent regime mean return > 0
5. OOS Sharpe ≥ 0.20 (no catastrophic degradation)

Usage:
    python3 scripts/diagnose_insider_phase1.py [--use-bulk-insiders] [--debug-compare-bulk]

Outputs:
    results/INSIDER_PHASE1_BASELINE.md - Baseline performance summary
    results/INSIDER_PHASE1_REPORT.md - Diagnostic report with GO/NO-GO verdict
    results/INSIDER_PHASE1_DECILES.csv - Decile-level returns
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
import time
from datetime import datetime
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from scripts.run_insider_phase1_baseline import InsiderPhase1Baseline
from validation.simple_validation import compute_basic_metrics, compute_monthly_returns
from config import get_logger

logger = get_logger(__name__)


class InsiderPhase1Diagnostics:
    """
    Phase 1 diagnostic analysis for InstitutionalInsider signal.

    Evaluates 5 acceptance gates and produces GO/NO-GO verdict.
    """

    def __init__(self,
                 start_date: str = '2015-04-01',
                 end_date: str = '2024-12-31',
                 is_cutoff: str = '2022-12-31',
                 oos_start: str = '2023-01-01',
                 use_bulk_insiders: bool = True,
                 debug_compare_bulk: bool = False):
        """
        Initialize diagnostics.

        Args:
            start_date: Analysis start
            end_date: Analysis end
            is_cutoff: In-sample cutoff date
            oos_start: Out-of-sample start date
            use_bulk_insiders: If True, use bulk insider data path (50-100x faster)
            debug_compare_bulk: If True, compare bulk vs legacy signals for consistency
        """
        self.start_date = start_date
        self.end_date = end_date
        self.is_cutoff = is_cutoff
        self.oos_start = oos_start
        self.use_bulk_insiders = use_bulk_insiders
        self.debug_compare_bulk = debug_compare_bulk

        logger.info("=" * 80)
        logger.info("Insider Phase 1 Diagnostics")
        logger.info("=" * 80)
        logger.info(f"Full Period: {self.start_date} to {self.end_date}")
        logger.info(f"In-Sample: {self.start_date} to {self.is_cutoff}")
        logger.info(f"Out-of-Sample: {self.oos_start} to {self.end_date}")
        logger.info(f"Bulk insider mode: {self.use_bulk_insiders}")
        logger.info(f"Debug comparison: {self.debug_compare_bulk}")
        logger.info("=" * 80)

    def run_diagnostics(self) -> Dict:
        """
        Run full Phase 1 diagnostics.

        Returns:
            Dict with diagnostic results and GO/NO-GO verdict
        """
        results = {
            'test_date': datetime.now().isoformat(),
            'period': f"{self.start_date} to {self.end_date}",
            'signal': 'InstitutionalInsider'
        }

        # 1. Run baseline backtest
        logger.info("\n" + "=" * 80)
        logger.info("Step 1: Running Baseline Backtest")
        logger.info("=" * 80)
        baseline = InsiderPhase1Baseline(
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=50000,
            use_bulk_insiders=self.use_bulk_insiders,
            debug_compare_bulk=self.debug_compare_bulk
        )
        baseline_results = baseline.run_baseline()
        results['baseline'] = baseline_results

        # 2. Extract long-short equity curve (primary analysis)
        long_short_equity = baseline_results['long_short_equity']

        # 3. Compute full-sample, IS, OOS metrics
        logger.info("\n" + "=" * 80)
        logger.info("Step 2: Computing IS/OOS Metrics")
        logger.info("=" * 80)

        full_metrics = compute_basic_metrics(long_short_equity)
        is_metrics = compute_basic_metrics(
            long_short_equity[long_short_equity.index <= self.is_cutoff]
        )
        oos_metrics = compute_basic_metrics(
            long_short_equity[long_short_equity.index >= self.oos_start]
        )

        results['full_metrics'] = full_metrics
        results['is_metrics'] = is_metrics
        results['oos_metrics'] = oos_metrics

        logger.info(f"Full Sharpe: {full_metrics['sharpe']:.3f}")
        logger.info(f"IS Sharpe: {is_metrics['sharpe']:.3f}")
        logger.info(f"OOS Sharpe: {oos_metrics['sharpe']:.3f}")

        # 4. Compute monthly returns and t-statistic
        logger.info("\n" + "=" * 80)
        logger.info("Step 3: Computing Statistical Significance")
        logger.info("=" * 80)

        monthly_returns = compute_monthly_returns(long_short_equity)
        mean_monthly = monthly_returns.mean()
        std_monthly = monthly_returns.std()
        n_months = len(monthly_returns)
        t_stat = (mean_monthly / (std_monthly / np.sqrt(n_months))) if std_monthly > 0 else 0

        results['monthly_returns'] = monthly_returns
        results['t_statistic'] = t_stat
        results['mean_monthly_return'] = mean_monthly

        logger.info(f"Mean Monthly Return: {mean_monthly*100:.3f}%")
        logger.info(f"Std Monthly Return: {std_monthly*100:.3f}%")
        logger.info(f"t-statistic: {t_stat:.3f}")

        # 5. Recent regime performance
        logger.info("\n" + "=" * 80)
        logger.info("Step 4: Analyzing Recent Regime")
        logger.info("=" * 80)

        recent_returns = monthly_returns[monthly_returns.index >= self.oos_start]
        recent_mean = recent_returns.mean()
        recent_sharpe = oos_metrics['sharpe']

        results['recent_mean_return'] = recent_mean
        results['recent_sharpe'] = recent_sharpe

        logger.info(f"Recent Mean Return: {recent_mean*100:.3f}%/mo")
        logger.info(f"Recent Sharpe: {recent_sharpe:.3f}")

        # 6. Decile monotonicity
        logger.info("\n" + "=" * 80)
        logger.info("Step 5: Testing Decile Monotonicity")
        logger.info("=" * 80)

        decile_summary = baseline_results['decile_summary']
        d1_annual = decile_summary.iloc[0]['annual_return']
        d10_annual = decile_summary.iloc[-1]['annual_return']
        decile_spread_annual = d1_annual - d10_annual
        decile_spread_monthly = decile_spread_annual / 12

        results['decile_spread_monthly'] = decile_spread_monthly
        results['decile_spread_annual'] = decile_spread_annual

        logger.info(f"D1 Annual Return: {d1_annual*100:.2f}%")
        logger.info(f"D10 Annual Return: {d10_annual*100:.2f}%")
        logger.info(f"Spread (D1 - D10): {decile_spread_annual*100:.2f}%/yr, {decile_spread_monthly*100:.2f}%/mo")

        # 7. Evaluate gates
        logger.info("\n" + "=" * 80)
        logger.info("Step 6: Evaluating Acceptance Gates")
        logger.info("=" * 80)

        gates = self._evaluate_gates(results)
        results['gates'] = gates

        # 8. Determine verdict
        verdict = self._determine_verdict(gates)
        results['verdict'] = verdict

        logger.info(f"\nVerdict: {verdict['decision']}")
        logger.info(f"Passed: {verdict['gates_passed']}/{verdict['total_gates']}")

        # 9. Generate report
        logger.info("\n" + "=" * 80)
        logger.info("Step 7: Generating Diagnostic Report")
        logger.info("=" * 80)

        self._generate_report(results)

        logger.info("\n" + "=" * 80)
        logger.info("Phase 1 Diagnostics Complete")
        logger.info("=" * 80)

        return results

    def _evaluate_gates(self, results: Dict) -> Dict:
        """
        Evaluate all 5 acceptance gates.

        Args:
            results: Dict with computed metrics

        Returns:
            Dict mapping gate names to pass/fail status
        """
        gates = {}

        # Gate 1: Decile Monotonicity
        gate1_pass = results['decile_spread_monthly'] >= 0.005  # 0.5%/mo
        gates['gate1_monotonicity'] = {
            'threshold': '≥ 0.5%/mo',
            'actual': f"{results['decile_spread_monthly']*100:.2f}%/mo",
            'pass': gate1_pass
        }
        logger.info(f"Gate 1 (Monotonicity): {'PASS' if gate1_pass else 'FAIL'} "
                   f"(spread {results['decile_spread_monthly']*100:.2f}%/mo)")

        # Gate 2: Full Sharpe ≥ 0.30
        gate2_pass = results['full_metrics']['sharpe'] >= 0.30
        gates['gate2_sharpe'] = {
            'threshold': '≥ 0.30',
            'actual': f"{results['full_metrics']['sharpe']:.3f}",
            'pass': gate2_pass
        }
        logger.info(f"Gate 2 (Full Sharpe): {'PASS' if gate2_pass else 'FAIL'} "
                   f"(Sharpe {results['full_metrics']['sharpe']:.3f})")

        # Gate 3: t-statistic ≥ 2.0
        gate3_pass = results['t_statistic'] >= 2.0
        gates['gate3_tstat'] = {
            'threshold': '≥ 2.0',
            'actual': f"{results['t_statistic']:.3f}",
            'pass': gate3_pass
        }
        logger.info(f"Gate 3 (t-statistic): {'PASS' if gate3_pass else 'FAIL'} "
                   f"(t-stat {results['t_statistic']:.3f})")

        # Gate 4: Recent regime mean > 0
        gate4_pass = results['recent_mean_return'] > 0
        gates['gate4_recent'] = {
            'threshold': '> 0.0%/mo',
            'actual': f"{results['recent_mean_return']*100:.2f}%/mo",
            'pass': gate4_pass
        }
        logger.info(f"Gate 4 (Recent Return): {'PASS' if gate4_pass else 'FAIL'} "
                   f"(mean {results['recent_mean_return']*100:.2f}%/mo)")

        # Gate 5: OOS Sharpe ≥ 0.20
        gate5_pass = results['oos_metrics']['sharpe'] >= 0.20
        gates['gate5_oos_sharpe'] = {
            'threshold': '≥ 0.20',
            'actual': f"{results['oos_metrics']['sharpe']:.3f}",
            'pass': gate5_pass
        }
        logger.info(f"Gate 5 (OOS Sharpe): {'PASS' if gate5_pass else 'FAIL'} "
                   f"(Sharpe {results['oos_metrics']['sharpe']:.3f})")

        return gates

    def _determine_verdict(self, gates: Dict) -> Dict:
        """
        Determine GO/NO-GO verdict based on gates.

        Args:
            gates: Dict with gate evaluation results

        Returns:
            Dict with verdict details
        """
        total_gates = len(gates)
        gates_passed = sum(1 for g in gates.values() if g['pass'])

        # Decision logic
        if gates_passed == total_gates:
            decision = 'GO'
            reason = 'All acceptance gates passed. Proceed to Phase 2 optimization.'
        elif gates_passed >= 4:
            decision = 'CONDITIONAL GO'
            failed_gates = [k for k, v in gates.items() if not v['pass']]
            reason = f'4/5 gates passed. Failed: {failed_gates[0]}. Review before optimization.'
        else:
            decision = 'NO-GO'
            failed_gates = [k for k, v in gates.items() if not v['pass']]
            reason = f'Only {gates_passed}/5 gates passed. Failed: {failed_gates}. Do not optimize.'

        return {
            'decision': decision,
            'total_gates': total_gates,
            'gates_passed': gates_passed,
            'reason': reason
        }

    def _generate_report(self, results: Dict):
        """
        Generate diagnostic markdown report.

        Args:
            results: Dict with all diagnostic results
        """
        md_path = Path('results/INSIDER_PHASE1_REPORT.md')

        with open(md_path, 'w') as f:
            f.write("# Insider Phase 1 Diagnostic Report\n\n")
            f.write(f"**Date:** {results['test_date'][:10]}\n")
            f.write(f"**Period:** {results['period']}\n")
            f.write(f"**Signal:** {results['signal']}\n\n")
            f.write("---\n\n")

            # Verdict (at top)
            verdict = results['verdict']
            emoji = "✅" if verdict['decision'] == 'GO' else "⚠️" if verdict['decision'] == 'CONDITIONAL GO' else "❌"
            f.write(f"## Verdict: {emoji} {verdict['decision']}\n\n")
            f.write(f"**Gates Passed:** {verdict['gates_passed']}/{verdict['total_gates']}\n\n")
            f.write(f"**Reason:** {verdict['reason']}\n\n")
            f.write("---\n\n")

            # Parameters
            f.write("## Configuration\n\n")
            params = results['baseline']['parameters']
            f.write("**Parameters:**\n")
            for key, val in params.items():
                f.write(f"- {key}: {val}\n")
            f.write("\n**Universe:** S&P 500 PIT (sp500_actual)\n")
            f.write(f"**Rebalancing:** Monthly (end of month)\n")
            f.write(f"**Capital:** $50,000\n\n")
            f.write("---\n\n")

            # Full-sample metrics
            f.write("## Full-Sample Performance (2015-04-01 to 2024-12-31)\n\n")
            f.write("**Long-Short Factor Portfolio:**\n")
            full = results['full_metrics']
            f.write(f"- **Total Return:** {full['total_return']*100:.2f}%\n")
            f.write(f"- **Annual Return:** {full['annual_return']*100:.2f}%\n")
            f.write(f"- **Volatility:** {full['volatility']*100:.2f}%\n")
            f.write(f"- **Sharpe Ratio:** {full['sharpe']:.3f}\n")
            f.write(f"- **Max Drawdown:** {full['max_drawdown']*100:.2f}%\n")
            f.write(f"- **Trading Days:** {full['num_days']}\n\n")

            # IS/OOS metrics
            f.write("## In-Sample vs Out-of-Sample\n\n")
            f.write("| Metric | In-Sample (2015-2022) | Out-of-Sample (2023-2024) |\n")
            f.write("|--------|----------------------|---------------------------|\n")
            is_m = results['is_metrics']
            oos_m = results['oos_metrics']
            f.write(f"| Annual Return | {is_m['annual_return']*100:.2f}% | {oos_m['annual_return']*100:.2f}% |\n")
            f.write(f"| Volatility | {is_m['volatility']*100:.2f}% | {oos_m['volatility']*100:.2f}% |\n")
            f.write(f"| Sharpe Ratio | {is_m['sharpe']:.3f} | {oos_m['sharpe']:.3f} |\n")
            f.write(f"| Max Drawdown | {is_m['max_drawdown']*100:.2f}% | {oos_m['max_drawdown']*100:.2f}% |\n")
            f.write("\n")

            # Statistical significance
            f.write("## Statistical Significance\n\n")
            f.write(f"- **Mean Monthly Return:** {results['mean_monthly_return']*100:.3f}%\n")
            f.write(f"- **Std Monthly Return:** {results['monthly_returns'].std()*100:.3f}%\n")
            f.write(f"- **t-statistic:** {results['t_statistic']:.3f}\n")
            p_value = 2 * (1 - 0.975) if results['t_statistic'] >= 2.0 else 'p > 0.05'
            f.write(f"- **p-value:** {p_value}\n\n")

            # Decile monotonicity
            f.write("## Decile Monotonicity\n\n")
            f.write("| Decile | Annual Return | Sharpe | Max DD |\n")
            f.write("|--------|---------------|--------|--------|\n")
            for _, row in results['baseline']['decile_summary'].iterrows():
                f.write(f"| D{int(row['decile'])} | {row['annual_return']*100:.2f}% | "
                       f"{row['sharpe']:.3f} | {row['max_drawdown']*100:.2f}% |\n")
            f.write("\n")
            f.write(f"**Spread (D1 - D10):** {results['decile_spread_annual']*100:.2f}%/yr "
                   f"({results['decile_spread_monthly']*100:.2f}%/mo)\n\n")

            # Regime performance
            f.write("## Regime Performance\n\n")
            f.write("| Regime | Period | Mean Return | Sharpe | Num Months |\n")
            f.write("|--------|--------|-------------|--------|------------|\n")
            regimes = results['baseline']['long_short_regimes']
            regime_names = {'covid': '2020 COVID', 'bear_2022': '2021-2022 Bear', 'recent': '2023-2024 Recent'}
            for regime_key, metrics in regimes.items():
                regime_label = regime_names.get(regime_key, regime_key)
                f.write(f"| {regime_label} | {regime_key} | {metrics['mean_return']*100:.2f}% | "
                       f"{metrics['sharpe']:.3f} | {metrics['num_months']} |\n")
            f.write("\n")

            # Gate evaluation
            f.write("## Acceptance Gate Evaluation\n\n")
            f.write("| Gate | Metric | Threshold | Actual | Status |\n")
            f.write("|------|--------|-----------|--------|--------|\n")
            gate_labels = {
                'gate1_monotonicity': 'Decile Spread',
                'gate2_sharpe': 'Full Sharpe',
                'gate3_tstat': 't-statistic',
                'gate4_recent': 'Recent Mean Return',
                'gate5_oos_sharpe': 'OOS Sharpe'
            }
            for gate_key, gate_data in results['gates'].items():
                label = gate_labels.get(gate_key, gate_key)
                status = "✅ PASS" if gate_data['pass'] else "❌ FAIL"
                f.write(f"| {label} | - | {gate_data['threshold']} | {gate_data['actual']} | {status} |\n")
            f.write("\n")

            # Recommendation
            f.write("## Recommendation\n\n")
            if verdict['decision'] == 'GO':
                f.write("**Proceed to Phase 2 optimization** with the following focus areas:\n\n")
                f.write("1. Grid search over lookback periods (30-180 days)\n")
                f.write("2. Optimize role weights (CEO/CFO relative importance)\n")
                f.write("3. Test cluster detection parameters\n")
                f.write("4. Evaluate minimum transaction value thresholds\n\n")
            elif verdict['decision'] == 'CONDITIONAL GO':
                f.write("**Proceed with caution.** Address the failed gate before full optimization:\n\n")
                failed_gates = [k for k, v in results['gates'].items() if not v['pass']]
                f.write(f"- Failed gate: {failed_gates[0]}\n")
                f.write("- Consider parameter adjustments or alternative specifications\n")
                f.write("- Rerun diagnostics after modifications\n\n")
            else:
                f.write("**Do not proceed to optimization.** The Insider signal does not meet minimum quality thresholds.\n\n")
                f.write("**Reasons:**\n")
                for gate_key, gate_data in results['gates'].items():
                    if not gate_data['pass']:
                        f.write(f"- {gate_key}: {gate_data['actual']} (threshold: {gate_data['threshold']})\n")
                f.write("\n**Next Steps:**\n")
                f.write("- Document failure in ERROR_PREVENTION_ARCHITECTURE.md\n")
                f.write("- Consider alternative insider methodologies\n")
                f.write("- Archive results for reference\n\n")

            f.write("---\n\n")
            f.write(f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        logger.info(f"Saved diagnostic report: {md_path}")


def main():
    """Run Insider Phase 1 diagnostics with optional bulk mode."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Insider Phase 1 Diagnostics with optional bulk data mode',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with bulk mode (default, recommended)
  python3 scripts/diagnose_insider_phase1.py

  # Run with legacy per-ticker mode
  python3 scripts/diagnose_insider_phase1.py --no-bulk-insiders

  # Run with bulk mode and debug comparison
  python3 scripts/diagnose_insider_phase1.py --debug-compare-bulk
        """
    )

    parser.add_argument(
        '--use-bulk-insiders',
        dest='use_bulk_insiders',
        action='store_true',
        help='Use bulk insider data path (preferred for large runs, 50-100x faster)'
    )
    parser.add_argument(
        '--no-bulk-insiders',
        dest='use_bulk_insiders',
        action='store_false',
        help='Disable bulk insider data path (fallback to per-ticker queries)'
    )
    parser.set_defaults(use_bulk_insiders=True)

    parser.add_argument(
        '--debug-compare-bulk',
        action='store_true',
        help='Compare bulk vs legacy insider signals on small subset and assert they match'
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Insider Phase 1 Diagnostics")
    logger.info(f"Bulk mode: {args.use_bulk_insiders}")
    logger.info(f"Debug comparison: {args.debug_compare_bulk}")
    logger.info("=" * 80)

    # Start timer
    start_time = time.perf_counter()

    # Run diagnostics (bulk mode handled internally by passing args)
    diagnostics = InsiderPhase1Diagnostics(
        start_date='2015-04-01',
        end_date='2024-12-31',
        is_cutoff='2022-12-31',
        oos_start='2023-01-01',
        use_bulk_insiders=args.use_bulk_insiders,
        debug_compare_bulk=args.debug_compare_bulk
    )

    results = diagnostics.run_diagnostics()

    # End timer
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # Calculate metrics
    num_rebalances = len(results.get('baseline', {}).get('decile_portfolios', {}).get(1, pd.Series()))
    avg_per_rebalance = elapsed_time / num_rebalances if num_rebalances > 0 else 0

    logger.info("=" * 80)
    logger.info(f"Insider Phase 1 runtime (use_bulk_insiders={args.use_bulk_insiders}): {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")
    if num_rebalances > 0:
        logger.info(f"Average runtime per rebalance: {avg_per_rebalance:.2f} seconds")
    logger.info("=" * 80)

    # Print console summary
    print("\n" + "=" * 80)
    print("INSIDER PHASE 1 DIAGNOSTICS - SUMMARY")
    print("=" * 80)
    print(f"\nRuntime: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
    print(f"Bulk mode: {'ENABLED' if args.use_bulk_insiders else 'DISABLED'}")
    print(f"\nLong-Short Factor Portfolio:")
    print(f"  Full Sharpe: {results['full_metrics']['sharpe']:.3f}")
    print(f"  OOS Sharpe: {results['oos_metrics']['sharpe']:.3f}")
    print(f"  t-statistic: {results['t_statistic']:.3f}")
    print(f"\nDecile Analysis:")
    print(f"  Spread (D1 - D10): {results['decile_spread_monthly']*100:.2f}%/mo")
    print(f"\nAcceptance Gates:")
    for gate_key, gate_data in results['gates'].items():
        status = "✅ PASS" if gate_data['pass'] else "❌ FAIL"
        print(f"  {gate_key}: {status}")
    print(f"\n{'='*80}")
    verdict = results['verdict']
    emoji = "✅" if verdict['decision'] == 'GO' else "⚠️" if verdict['decision'] == 'CONDITIONAL GO' else "❌"
    print(f"VERDICT: {emoji} {verdict['decision']}")
    print(f"Gates Passed: {verdict['gates_passed']}/{verdict['total_gates']}")
    print(f"Reason: {verdict['reason']}")
    print("=" * 80)

    # Speedup estimate (baseline: 2.5 hours for 116 rebalances with legacy mode)
    if args.use_bulk_insiders:
        legacy_runtime_hours = 2.5
        legacy_runtime_seconds = legacy_runtime_hours * 3600
        speedup = legacy_runtime_seconds / elapsed_time if elapsed_time > 0 else 0
        print(f"\nSpeedup vs Legacy Mode:")
        print(f"  Legacy runtime: ~{legacy_runtime_hours} hours")
        print(f"  Bulk runtime: {elapsed_time/60:.1f} minutes")
        print(f"  Approximate speedup: ~{speedup:.0f}x faster")
        print("=" * 80)


if __name__ == '__main__':
    main()
