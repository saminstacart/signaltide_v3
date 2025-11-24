"""
Momentum + Quality + Insider (M+Q+I) Three-Signal Ensemble Diagnostic Runner

M3.6 Priority 3.2: Smoke test for three-signal ensemble baseline.

Runs side-by-side comparison of:
1. Momentum + Quality v1 ensemble (two-signal baseline)
2. Momentum + Quality + Insider v1 ensemble (three-signal, M3.6)

Outputs:
    results/ensemble_baselines/mqi_three_signal_v1_diagnostic.md
    results/ensemble_baselines/mqi_three_signal_v1_comparison.csv

Usage:
    # Smoke test (5 years, smaller universe)
    python3 scripts/run_mqi_three_signal_baseline.py

    # Full 10-year backtest
    python3 scripts/run_mqi_three_signal_baseline.py --start 2015-04-01 --full-universe
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import argparse
import warnings
warnings.filterwarnings('ignore')

from data.data_manager import DataManager
from core.universe_manager import UniverseManager
from core.backtest_engine import BacktestConfig, run_backtest, BacktestResult
from core.signal_adapters import make_multisignal_ensemble_fn
from signals.ml.ensemble_configs import get_momentum_quality_v1_ensemble, get_momentum_quality_insider_v1_ensemble
from config import get_logger

logger = get_logger(__name__)


class MQIThreeSignalDiagnosticRunner:
    """
    Runs diagnostic comparison of M+Q (two-signal) vs M+Q+I (three-signal) ensembles.
    """

    def __init__(self,
                 start_date: str = '2020-01-01',
                 end_date: str = '2024-12-31',
                 initial_capital: float = 100000.0,
                 smoke_test: bool = True):
        """
        Initialize diagnostic runner.

        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital
            smoke_test: If True, use smaller universe (~40 tickers) for faster smoke test
        """
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.smoke_test = smoke_test

        self.dm = DataManager()
        self.um = UniverseManager(self.dm)

        logger.info("=" * 80)
        logger.info("MOMENTUM + QUALITY + INSIDER (M+Q+I) THREE-SIGNAL ENSEMBLE DIAGNOSTIC")
        logger.info("=" * 80)
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Capital: ${initial_capital:,.0f}")
        logger.info(f"Mode: {'SMOKE TEST (small universe)' if smoke_test else 'FULL BACKTEST (S&P 500)'}")
        logger.info("=" * 80)
        logger.info("")

    def run_diagnostic(self) -> Dict:
        """
        Run full diagnostic comparison.

        Returns:
            Dict with both results and comparison metrics
        """
        results = {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_capital': self.initial_capital,
            'smoke_test': self.smoke_test,
            'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Define universe function
        if self.smoke_test:
            # Smoke test: Use manual list of ~40 large-cap tickers
            # These are large, liquid stocks with good insider/fundamental data coverage
            smoke_test_tickers = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM',
                'JNJ', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV',
                'KO', 'AVGO', 'COST', 'WMT', 'LLY', 'TMO', 'DIS', 'CSCO',
                'ADBE', 'ACN', 'NFLX', 'CRM', 'NKE', 'ABT', 'MCD', 'VZ',
                'PM', 'INTC', 'AMD', 'TXN', 'UNP', 'NEE', 'UNH', 'RTX'
            ]

            def universe_fn(rebal_date: str) -> List[str]:
                """Static universe for smoke test."""
                return smoke_test_tickers

            universe_desc = f"Manual smoke test universe ({len(smoke_test_tickers)} tickers)"
        else:
            # Full backtest: Use S&P 500 PIT universe
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

            universe_desc = "S&P 500 (actual constituents, min_price=$5)"

        results['universe_desc'] = universe_desc

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

        # 1. Run M+Q ensemble (two-signal baseline)
        logger.info("Running M+Q (Momentum + Quality) two-signal baseline...")
        mq_ensemble = get_momentum_quality_v1_ensemble(self.dm)
        mq_signal_fn = make_multisignal_ensemble_fn(mq_ensemble, self.dm)
        mq_result = run_backtest(universe_fn, mq_signal_fn, config)

        logger.info(f"  M+Q: Total Return={mq_result.total_return:.2%}, "
                   f"Sharpe={mq_result.sharpe:.3f}, MaxDD={mq_result.max_drawdown:.2%}")
        logger.info("")

        # 2. Run M+Q+I ensemble (three-signal, M3.6)
        logger.info("Running M+Q+I (Momentum + Quality + Insider) three-signal ensemble...")
        mqi_ensemble = get_momentum_quality_insider_v1_ensemble(self.dm)
        mqi_signal_fn = make_multisignal_ensemble_fn(mqi_ensemble, self.dm)
        mqi_result = run_backtest(universe_fn, mqi_signal_fn, config)

        logger.info(f"  M+Q+I: Total Return={mqi_result.total_return:.2%}, "
                   f"Sharpe={mqi_result.sharpe:.3f}, MaxDD={mqi_result.max_drawdown:.2%}")
        logger.info("")

        # Store results
        results['mq_two_signal'] = self._extract_metrics(mq_result, "M+Q (Two-Signal)")
        results['mqi_three_signal'] = self._extract_metrics(mqi_result, "M+Q+I (Three-Signal)")

        # 3. Compute return correlation
        logger.info("Computing return correlation...")
        mq_returns = mq_result.equity_curve.pct_change().dropna()
        mqi_returns = mqi_result.equity_curve.pct_change().dropna()

        # Align indices
        common_dates = mq_returns.index.intersection(mqi_returns.index)
        correlation = mq_returns.loc[common_dates].corr(mqi_returns.loc[common_dates])

        results['return_correlation'] = correlation
        logger.info(f"  Return correlation: {correlation:.4f}")
        logger.info("")

        # 4. Compute incremental metrics (M+Q+I vs M+Q)
        logger.info("Computing incremental impact of insider signal...")
        delta_return = mqi_result.total_return - mq_result.total_return
        delta_sharpe = mqi_result.sharpe - mq_result.sharpe
        delta_drawdown = mqi_result.max_drawdown - mq_result.max_drawdown

        results['delta_return'] = delta_return
        results['delta_sharpe'] = delta_sharpe
        results['delta_drawdown'] = delta_drawdown

        logger.info(f"  Δ Total Return: {delta_return:+.2%}")
        logger.info(f"  Δ Sharpe Ratio: {delta_sharpe:+.3f}")
        logger.info(f"  Δ Max Drawdown: {delta_drawdown:+.2%} (negative is better)")
        logger.info("")

        # 5. Save diagnostic files
        self._save_diagnostic_markdown(results)
        self._save_comparison_csv(results)

        # 6. Print summary
        self._print_summary(results)

        return results

    def _extract_metrics(self, result: BacktestResult, label: str) -> Dict:
        """Extract key metrics from BacktestResult."""
        return {
            'label': label,
            'total_return': result.total_return,
            'cagr': result.cagr,
            'volatility': result.volatility,
            'sharpe': result.sharpe,
            'max_drawdown': result.max_drawdown,
            'num_rebalances': result.num_rebalances,
            'equity_curve': result.equity_curve,
        }

    def _save_diagnostic_markdown(self, results: Dict):
        """Save diagnostic markdown file."""
        output_dir = Path('results/ensemble_baselines')
        output_dir.mkdir(parents=True, exist_ok=True)

        md_path = output_dir / 'mqi_three_signal_v1_diagnostic.md'

        mq = results['mq_two_signal']
        mqi = results['mqi_three_signal']

        mode_str = "SMOKE TEST" if results['smoke_test'] else "FULL BACKTEST"

        content = f"""# Momentum + Quality + Insider (M+Q+I) v1 Three-Signal Ensemble Diagnostic

**Generated:** {results['generated']}
**Mode:** {mode_str}
**Period:** {results['start_date']} to {results['end_date']}
**Capital:** ${results['initial_capital']:,.0f}
**Universe:** {results['universe_desc']}
**Rebalance:** Monthly

---

## Configuration Summary

### M+Q Two-Signal Baseline (Control)
- **Ensemble:** `momentum_quality_v1`
- **Signals:**
  - InstitutionalMomentum v2 (weight=0.25)
  - CrossSectionalQuality v1 (weight=0.75)
- **Parameters:**
  - Momentum: 308-day formation, 0-day skip, adaptive quintiles
  - Quality: w_profitability=0.4, w_growth=0.3, w_safety=0.3, adaptive quintiles
- **Normalization:** none (signals return quintiles)

### M+Q+I Three-Signal (M3.6)
- **Ensemble:** `momentum_quality_insider_v1`
- **Signals:**
  - InstitutionalMomentum v2 (weight=0.25)
  - CrossSectionalQuality v1 (weight=0.50)
  - InstitutionalInsider v1 (weight=0.25)
- **Parameters:**
  - Momentum: 308-day formation, 0-day skip, adaptive quintiles
  - Quality: w_profitability=0.4, w_growth=0.3, w_safety=0.3, adaptive quintiles
  - Insider: lookback=90 days, min_transactions=3, value_threshold=$100K, adaptive quintiles
- **Normalization:** none (signals return quintiles)
- **Data Coverage:** 98.1% S&P 500 sample (validated in Priority 3.1)

---

## Performance Comparison

### M+Q Two-Signal Baseline
```
Total Return:      {mq['total_return']:>10.2%}
CAGR:              {mq['cagr']:>10.2%}
Volatility:        {mq['volatility']:>10.2%}
Sharpe Ratio:      {mq['sharpe']:>10.3f}
Max Drawdown:      {mq['max_drawdown']:>10.2%}
Num Rebalances:    {mq['num_rebalances']:>10d}
```

### M+Q+I Three-Signal (M3.6)
```
Total Return:      {mqi['total_return']:>10.2%}
CAGR:              {mqi['cagr']:>10.2%}
Volatility:        {mqi['volatility']:>10.2%}
Sharpe Ratio:      {mqi['sharpe']:>10.3f}
Max Drawdown:      {mqi['max_drawdown']:>10.2%}
Num Rebalances:    {mqi['num_rebalances']:>10d}
```

### Incremental Impact of Insider Signal
```
Δ Total Return:    {results['delta_return']:>10.2%}
Δ Sharpe Ratio:    {results['delta_sharpe']:>10.3f}
Δ Max Drawdown:    {results['delta_drawdown']:>10.2%}  (negative is better)

Return Correlation: {results['return_correlation']:>9.4f}
```

---

## Interpretation

### Insider Signal Contribution
- **Return Impact:** {"Positive" if results['delta_return'] > 0 else "Negative" if results['delta_return'] < 0 else "Neutral"} ({results['delta_return']:+.2%})
- **Risk-Adjusted Impact:** {"Positive" if results['delta_sharpe'] > 0 else "Negative" if results['delta_sharpe'] < 0 else "Neutral"} ({results['delta_sharpe']:+.3f} Sharpe)
- **Drawdown Impact:** {"Improved" if results['delta_drawdown'] > 0 else "Worse" if results['delta_drawdown'] < 0 else "Neutral"} ({results['delta_drawdown']:+.2%})

### Correlation Analysis
- **Strategy Correlation:** {results['return_correlation']:.2%}
- **Diversification:** {"High" if results['return_correlation'] > 0.9 else "Moderate" if results['return_correlation'] > 0.7 else "Low"} correlation suggests {"limited" if results['return_correlation'] > 0.9 else "moderate" if results['return_correlation'] > 0.7 else "good"} diversification benefit

---

## Next Steps

{"### ✅ SMOKE TEST PASSED" if results['smoke_test'] else "### FULL BACKTEST COMPLETE"}

**Recommendations:**
1. {"Run full 10-year backtest on complete S&P 500 universe" if results['smoke_test'] else "Analyze regime-specific performance"}
2. {"Verify insider signal contribution is consistent across regimes" if results['smoke_test'] else "Consider weight tuning if insider signal shows consistent alpha"}
3. {"Check signal coverage (what % of universe has insider data)" if results['smoke_test'] else "Validate transaction cost sensitivity"}

**Status:** M3.6 Priority 3.2 - Three-signal ensemble wiring {"(smoke test)" if results['smoke_test'] else "(full diagnostic)"} ✅

---

**Generated:** {results['generated']}
**Report:** results/ensemble_baselines/mqi_three_signal_v1_diagnostic.md
"""

        with open(md_path, 'w') as f:
            f.write(content)

        logger.info(f"Saved diagnostic markdown: {md_path}")

    def _save_comparison_csv(self, results: Dict):
        """Save comparison CSV with key metrics."""
        output_dir = Path('results/ensemble_baselines')
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / 'mqi_three_signal_v1_comparison.csv'

        mq = results['mq_two_signal']
        mqi = results['mqi_three_signal']

        comparison_df = pd.DataFrame({
            'Metric': [
                'Total Return',
                'CAGR',
                'Volatility',
                'Sharpe Ratio',
                'Max Drawdown',
                'Num Rebalances'
            ],
            'M+Q (Two-Signal)': [
                f"{mq['total_return']:.2%}",
                f"{mq['cagr']:.2%}",
                f"{mq['volatility']:.2%}",
                f"{mq['sharpe']:.3f}",
                f"{mq['max_drawdown']:.2%}",
                mq['num_rebalances']
            ],
            'M+Q+I (Three-Signal)': [
                f"{mqi['total_return']:.2%}",
                f"{mqi['cagr']:.2%}",
                f"{mqi['volatility']:.2%}",
                f"{mqi['sharpe']:.3f}",
                f"{mqi['max_drawdown']:.2%}",
                mqi['num_rebalances']
            ],
            'Delta': [
                f"{results['delta_return']:+.2%}",
                f"{(mqi['cagr'] - mq['cagr']):+.2%}",
                f"{(mqi['volatility'] - mq['volatility']):+.2%}",
                f"{results['delta_sharpe']:+.3f}",
                f"{results['delta_drawdown']:+.2%}",
                mqi['num_rebalances'] - mq['num_rebalances']
            ]
        })

        comparison_df.to_csv(csv_path, index=False)
        logger.info(f"Saved comparison CSV: {csv_path}")

    def _print_summary(self, results: Dict):
        """Print summary to console."""
        logger.info("=" * 80)
        logger.info("M+Q+I THREE-SIGNAL DIAGNOSTIC SUMMARY")
        logger.info("=" * 80)
        logger.info("")

        mq = results['mq_two_signal']
        mqi = results['mqi_three_signal']

        logger.info(f"M+Q (Two-Signal):    Return={mq['total_return']:>7.2%}, "
                   f"Sharpe={mq['sharpe']:>6.3f}, MaxDD={mq['max_drawdown']:>7.2%}")
        logger.info(f"M+Q+I (Three-Signal): Return={mqi['total_return']:>7.2%}, "
                   f"Sharpe={mqi['sharpe']:>6.3f}, MaxDD={mqi['max_drawdown']:>7.2%}")
        logger.info("")
        logger.info(f"Insider Signal Impact:")
        logger.info(f"  Δ Return:     {results['delta_return']:+7.2%}")
        logger.info(f"  Δ Sharpe:     {results['delta_sharpe']:+7.3f}")
        logger.info(f"  Δ Drawdown:   {results['delta_drawdown']:+7.2%}")
        logger.info(f"  Correlation:  {results['return_correlation']:7.4f}")
        logger.info("")

        if results['smoke_test']:
            logger.info("✅ SMOKE TEST COMPLETE")
            logger.info("   Next: Run full backtest with --full-universe flag")
        else:
            logger.info("✅ FULL DIAGNOSTIC COMPLETE")
            logger.info("   Next: Analyze regime-specific performance")

        logger.info("")
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='M+Q+I Three-Signal Ensemble Diagnostic')
    parser.add_argument('--start', default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default='2024-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100000.0, help='Initial capital')
    parser.add_argument('--full-universe', action='store_true',
                       help='Use full S&P 500 universe (default: smoke test with ~40 tickers)')

    args = parser.parse_args()

    smoke_test = not args.full_universe

    runner = MQIThreeSignalDiagnosticRunner(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        smoke_test=smoke_test
    )

    try:
        results = runner.run_diagnostic()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
