"""
Momentum + Quality Ensemble Diagnostic Runner

Runs side-by-side comparison of:
1. Momentum-only ensemble (production baseline)
2. Momentum + Quality v1 ensemble (multi-signal, Phase 3)

Outputs:
    results/ensemble_baselines/momentum_quality_v1_diagnostic.md
    results/ensemble_baselines/momentum_quality_v1_comparison.csv

Usage:
    python3 scripts/run_momentum_quality_baseline.py

    # Shorter date range for testing
    python3 scripts/run_momentum_quality_baseline.py --start 2020-01-01
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
from core.backtest_engine import BacktestConfig, run_backtest, BacktestResult
from core.signal_adapters import make_ensemble_signal_fn, make_multisignal_ensemble_fn
from signals.ml.ensemble_configs import get_momentum_v2_ensemble, get_momentum_quality_v1_ensemble
from config import get_logger

logger = get_logger(__name__)


class MomentumQualityDiagnosticRunner:
    """
    Runs diagnostic comparison of momentum-only vs momentum+quality ensembles.
    """

    def __init__(self,
                 start_date: str = '2015-04-01',
                 end_date: str = '2024-12-31',
                 initial_capital: float = 100000.0):
        """
        Initialize diagnostic runner.

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
        logger.info("MOMENTUM + QUALITY ENSEMBLE DIAGNOSTIC")
        logger.info("=" * 80)
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Capital: ${initial_capital:,.0f}")
        logger.info(f"Universe: sp500_actual (min_price=5.0)")
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
            'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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
        momentum_ensemble = get_momentum_v2_ensemble(self.dm)
        momentum_signal_fn = make_ensemble_signal_fn(momentum_ensemble, self.dm, lookback_days=500)
        momentum_result = run_backtest(universe_fn, momentum_signal_fn, config)

        logger.info(f"  Momentum-only: Total Return={momentum_result.total_return:.2%}, "
                   f"Sharpe={momentum_result.sharpe:.3f}, MaxDD={momentum_result.max_drawdown:.2%}")
        logger.info("")

        # 2. Run momentum + quality ensemble (cross-sectional path)
        logger.info("Running momentum + quality ensemble...")
        multi_ensemble = get_momentum_quality_v1_ensemble(self.dm)
        multi_signal_fn = make_multisignal_ensemble_fn(multi_ensemble, self.dm)
        multi_result = run_backtest(universe_fn, multi_signal_fn, config)

        logger.info(f"  Momentum+Quality: Total Return={multi_result.total_return:.2%}, "
                   f"Sharpe={multi_result.sharpe:.3f}, MaxDD={multi_result.max_drawdown:.2%}")
        logger.info("")

        # Store results
        results['momentum_only'] = self._extract_metrics(momentum_result, "Momentum-Only")
        results['momentum_quality'] = self._extract_metrics(multi_result, "Momentum+Quality")

        # 3. Compute return correlation
        logger.info("Computing return correlation...")
        momentum_returns = momentum_result.equity_curve.pct_change().dropna()
        multi_returns = multi_result.equity_curve.pct_change().dropna()

        # Align indices (should already be aligned, but just in case)
        common_dates = momentum_returns.index.intersection(multi_returns.index)
        correlation = momentum_returns.loc[common_dates].corr(multi_returns.loc[common_dates])

        results['return_correlation'] = correlation
        logger.info(f"  Return correlation: {correlation:.4f}")
        logger.info("")

        # 4. Fetch SPY benchmark and compute metrics
        logger.info("Processing SPY benchmark comparison...")
        spy_equity = self._fetch_spy_benchmark()

        if not spy_equity.empty:
            # Compute benchmark metrics
            benchmark_metrics = self._compute_benchmark_metrics(
                multi_result.equity_curve,
                spy_equity
            )
            results['benchmark'] = benchmark_metrics

            # Save SPY comparison CSV
            self._save_spy_comparison_csv(multi_result.equity_curve, spy_equity)
            logger.info("")
        else:
            logger.warning("SPY data not available, skipping benchmark comparison")
            results['benchmark'] = None

        # 5. Compute portfolio turnover metrics
        logger.info("Computing portfolio turnover...")
        turnover_data = self._compute_turnover_metrics(
            universe_fn,
            multi_signal_fn,
            multi_result.rebalance_dates
        )
        results['turnover'] = turnover_data

        if turnover_data:
            self._save_turnover_csv(turnover_data)
            logger.info("")

        # 6. Compute sector/size exposure snapshots
        logger.info("Computing sector/size exposure snapshots...")
        # Select representative snapshot dates from rebalance dates
        rebal_dates = multi_result.rebalance_dates
        if len(rebal_dates) >= 5:
            # Pick: first, 25%, 50%, 75%, last
            indices = [0, len(rebal_dates)//4, len(rebal_dates)//2, 3*len(rebal_dates)//4, -1]
            snapshot_dates = [rebal_dates[i] for i in indices]
        else:
            snapshot_dates = rebal_dates  # Use all if short period

        # Remove duplicates while preserving order
        snapshot_dates = list(dict.fromkeys(snapshot_dates))

        exposure_data = self._compute_exposure_snapshot(
            universe_fn,
            multi_signal_fn,
            snapshot_dates
        )
        results['exposure'] = exposure_data

        if exposure_data:
            self._save_exposure_csv(exposure_data)
            logger.info("")

        # 7. Compute bootstrap Sharpe confidence intervals
        logger.info("Computing bootstrap robustness check...")
        bootstrap_data = self._compute_bootstrap_sharpe(multi_result.equity_curve)
        results['bootstrap'] = bootstrap_data

        if bootstrap_data:
            self._save_bootstrap_csv(bootstrap_data)
            logger.info("")

        # 8. Save diagnostic files
        self._save_diagnostic_markdown(results)
        self._save_comparison_csv(results)

        # 9. Print summary
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

    def _fetch_spy_benchmark(self) -> pd.Series:
        """
        Fetch SPY prices and compute equity curve for benchmark.

        Returns:
            Series with SPY equity curve (normalized to initial_capital)
        """
        logger.info("Fetching SPY benchmark data...")

        # Fetch SPY prices from database
        spy_prices = self.dm.get_prices(
            symbols=['SPY'],
            start_date=self.start_date,
            end_date=self.end_date
        )

        if spy_prices.empty:
            logger.warning("No SPY data found in database!")
            return pd.Series(dtype=float)

        # Get adjusted close prices
        if 'adj_close' in spy_prices.columns:
            spy_close = spy_prices['adj_close']
        elif 'close' in spy_prices.columns:
            spy_close = spy_prices['close']
        else:
            logger.error(f"No price column found. Available: {spy_prices.columns.tolist()}")
            return pd.Series(dtype=float)

        # Normalize to initial capital (buy-and-hold)
        spy_equity = spy_close / spy_close.iloc[0] * self.initial_capital

        logger.info(f"  SPY data: {len(spy_equity)} days")
        logger.info(f"  SPY total return: {(spy_equity.iloc[-1] / spy_equity.iloc[0] - 1):.2%}")

        return spy_equity

    def _compute_benchmark_metrics(self, mq_equity: pd.Series, spy_equity: pd.Series) -> Dict:
        """
        Compute benchmark comparison metrics.

        Args:
            mq_equity: M+Q equity curve
            spy_equity: SPY equity curve

        Returns:
            Dict with tracking error, information ratio, excess return
        """
        logger.info("Computing benchmark metrics...")

        # Compute monthly returns
        mq_monthly = mq_equity.resample('ME').last().pct_change().dropna()
        spy_monthly = spy_equity.resample('ME').last().pct_change().dropna()

        # Align dates
        common_dates = mq_monthly.index.intersection(spy_monthly.index)
        mq_monthly = mq_monthly.loc[common_dates]
        spy_monthly = spy_monthly.loc[common_dates]

        # Excess returns
        excess_returns = mq_monthly - spy_monthly

        # Tracking error (annualized std of excess returns)
        tracking_error = excess_returns.std() * np.sqrt(12)

        # Information ratio (annualized excess return / tracking error)
        annualized_excess = excess_returns.mean() * 12
        information_ratio = annualized_excess / tracking_error if tracking_error > 0 else 0.0

        # Total returns
        mq_total = (mq_equity.iloc[-1] / mq_equity.iloc[0]) - 1
        spy_total = (spy_equity.iloc[-1] / spy_equity.iloc[0]) - 1
        excess_total = mq_total - spy_total

        logger.info(f"  Tracking Error: {tracking_error:.2%}")
        logger.info(f"  Information Ratio: {information_ratio:.3f}")
        logger.info(f"  Excess Return: {excess_total:.2%}")

        return {
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'annualized_excess': annualized_excess,
            'excess_total': excess_total,
            'num_periods': len(common_dates)
        }

    def _save_spy_comparison_csv(self, mq_equity: pd.Series, spy_equity: pd.Series):
        """
        Save monthly returns comparison CSV.

        Args:
            mq_equity: M+Q equity curve
            spy_equity: SPY equity curve
        """
        logger.info("Generating SPY comparison CSV...")

        # Compute monthly equity values (end of month)
        mq_monthly_equity = mq_equity.resample('ME').last()
        spy_monthly_equity = spy_equity.resample('ME').last()

        # Align dates
        common_dates = mq_monthly_equity.index.intersection(spy_monthly_equity.index)

        # Compute returns
        mq_returns = mq_monthly_equity.loc[common_dates].pct_change()
        spy_returns = spy_monthly_equity.loc[common_dates].pct_change()

        # Compute cumulative returns (starting at 0%)
        mq_cum = (mq_monthly_equity.loc[common_dates] / mq_monthly_equity.loc[common_dates].iloc[0]) - 1
        spy_cum = (spy_monthly_equity.loc[common_dates] / spy_monthly_equity.loc[common_dates].iloc[0]) - 1

        # Build DataFrame
        comparison_df = pd.DataFrame({
            'date': common_dates,
            'mq_return': mq_returns.values,
            'spy_return': spy_returns.values,
            'mq_cumulative': mq_cum.values,
            'spy_cumulative': spy_cum.values,
            'excess_return': (mq_returns - spy_returns).values
        })

        # Save to CSV
        output_dir = Path('results/ensemble_baselines')
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / 'momentum_quality_v1_vs_spy_monthly.csv'

        comparison_df.to_csv(csv_path, index=False)
        logger.info(f"Saved SPY comparison: {csv_path}")
        logger.info(f"  Rows: {len(comparison_df)}")
        logger.info(f"  Final M+Q cumulative: {mq_cum.iloc[-1]:.2%}")
        logger.info(f"  Final SPY cumulative: {spy_cum.iloc[-1]:.2%}")

    def _compute_turnover_metrics(self,
                                   universe_fn,
                                   signal_fn,
                                   rebalance_dates: List[str]) -> Dict:
        """
        Compute portfolio turnover at each rebalance.

        Turnover formula: turnover_t = 0.5 * sum_i |w_i_t - w_i_{t-1}|

        Args:
            universe_fn: Function to get universe at date
            signal_fn: Function to generate signals
            rebalance_dates: List of rebalance dates

        Returns:
            Dict with turnover statistics and time series
        """
        logger.info("Computing portfolio turnover metrics...")

        turnover_series = []
        prev_weights = {}

        for i, rebal_date in enumerate(rebalance_dates):
            # Get universe and signals for this rebalance
            universe = universe_fn(rebal_date)
            if not universe:
                continue

            # Get signals (note: signal_fn signature is (rebal_date, universe))
            signals = signal_fn(rebal_date, universe)
            if signals.empty:
                continue

            # Determine which tickers are long positions (equal-weight)
            # Mirror backtest_engine.py logic: signals > 0 (all positive signals go long)
            long_tickers = signals[signals > 0].index.tolist()

            # Equal-weight allocation
            if len(long_tickers) > 0:
                weight_per_ticker = 1.0 / len(long_tickers)
                current_weights = {ticker: weight_per_ticker for ticker in long_tickers}
            else:
                current_weights = {}

            # Calculate turnover if not first rebalance
            if i > 0 and prev_weights:
                # Get all unique tickers from both periods
                all_tickers = set(current_weights.keys()) | set(prev_weights.keys())

                # Sum absolute weight changes
                weight_change_sum = sum(
                    abs(current_weights.get(ticker, 0.0) - prev_weights.get(ticker, 0.0))
                    for ticker in all_tickers
                )

                # Turnover = 0.5 * sum of absolute weight changes
                turnover = 0.5 * weight_change_sum
                turnover_series.append({
                    'date': rebal_date,
                    'turnover': turnover,
                    'num_positions': len(long_tickers),
                    'new_positions': len(set(long_tickers) - set(prev_weights.keys())),
                    'exited_positions': len(set(prev_weights.keys()) - set(long_tickers))
                })

            # Update previous weights
            prev_weights = current_weights.copy()

        # Compute statistics
        if turnover_series:
            turnover_values = [t['turnover'] for t in turnover_series]
            turnover_stats = {
                'mean': np.mean(turnover_values),
                'median': np.percentile(turnover_values, 50),
                'p25': np.percentile(turnover_values, 25),
                'p75': np.percentile(turnover_values, 75),
                'p95': np.percentile(turnover_values, 95),
                'min': np.min(turnover_values),
                'max': np.max(turnover_values),
                'num_rebalances': len(turnover_series)
            }

            logger.info(f"  Turnover statistics computed over {len(turnover_series)} rebalances")
            logger.info(f"  Mean turnover: {turnover_stats['mean']:.2%}")
            logger.info(f"  Median turnover: {turnover_stats['median']:.2%}")
            logger.info(f"  95th percentile: {turnover_stats['p95']:.2%}")

            return {
                'stats': turnover_stats,
                'series': turnover_series
            }
        else:
            logger.warning("  No turnover data computed")
            return None

    def _save_turnover_csv(self, turnover_data: Dict):
        """
        Save turnover metrics CSV.

        Args:
            turnover_data: Dict with turnover stats and series
        """
        if not turnover_data:
            logger.warning("No turnover data to save")
            return

        logger.info("Generating turnover metrics CSV...")

        # Create DataFrame from turnover series
        df = pd.DataFrame(turnover_data['series'])

        # Save to CSV
        output_dir = Path('results/ensemble_baselines')
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / 'momentum_quality_v1_turnover_metrics.csv'

        df.to_csv(csv_path, index=False)
        logger.info(f"Saved turnover metrics: {csv_path}")
        logger.info(f"  Rows: {len(df)}")

    def _compute_exposure_snapshot(self,
                                    universe_fn,
                                    signal_fn,
                                    snapshot_dates: List[str]) -> Dict:
        """
        Compute sector and size exposure at representative snapshot dates.

        Args:
            universe_fn: Function to get universe at date
            signal_fn: Function to generate signals
            snapshot_dates: List of dates to analyze

        Returns:
            Dict with exposure analysis by date
        """
        import sqlite3

        logger.info("Computing sector/size exposure snapshots...")

        # Mag-7 tickers
        MAG_7 = {'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA'}

        # Connect to database for sector/size lookups
        conn = sqlite3.connect(self.dm.db_path)

        exposure_data = []

        for snapshot_date in snapshot_dates:
            logger.info(f"  Analyzing {snapshot_date}...")

            # Get universe and signals
            universe = universe_fn(snapshot_date)
            if not universe:
                logger.warning(f"    No universe for {snapshot_date}")
                continue

            signals = signal_fn(snapshot_date, universe)
            if signals.empty:
                logger.warning(f"    No signals for {snapshot_date}")
                continue

            # Get long positions (signal > 0)
            long_tickers = signals[signals > 0].index.tolist()
            if not long_tickers:
                continue

            # Equal-weight per ticker
            weight_per_ticker = 1.0 / len(long_tickers)

            # Fetch sector/size for long tickers
            placeholders = ','.join(['?' for _ in long_tickers])
            query = f"""
                SELECT DISTINCT ticker, sector, scalemarketcap
                FROM sharadar_tickers
                WHERE ticker IN ({placeholders})
            """
            ticker_info = pd.read_sql_query(query, conn, params=long_tickers)
            ticker_info = ticker_info.drop_duplicates(subset=['ticker'])

            # Build holdings DataFrame
            holdings = pd.DataFrame({
                'ticker': long_tickers,
                'weight': weight_per_ticker
            })
            holdings = holdings.merge(ticker_info, on='ticker', how='left')

            # --- Top 10 Holdings ---
            top_10 = holdings.nlargest(10, 'weight')[['ticker', 'weight', 'sector', 'scalemarketcap']]
            for _, row in top_10.iterrows():
                exposure_data.append({
                    'snapshot_date': snapshot_date,
                    'analysis_type': 'top_10_holdings',
                    'item': row['ticker'],
                    'mq_weight': row['weight'],
                    'spy_weight': None,
                    'delta': None,
                    'sector': row['sector'],
                    'size_bucket': row['scalemarketcap']
                })

            # --- Sector Breakdown ---
            sector_weights = holdings.groupby('sector')['weight'].sum().sort_values(ascending=False)
            for sector, weight in sector_weights.items():
                exposure_data.append({
                    'snapshot_date': snapshot_date,
                    'analysis_type': 'sector_breakdown',
                    'item': sector,
                    'mq_weight': weight,
                    'spy_weight': None,  # Could add SPY sector weights if available
                    'delta': None,
                    'sector': sector,
                    'size_bucket': None
                })

            # --- Size Distribution ---
            size_weights = holdings.groupby('scalemarketcap')['weight'].sum().sort_values(ascending=False)
            for size, weight in size_weights.items():
                exposure_data.append({
                    'snapshot_date': snapshot_date,
                    'analysis_type': 'size_distribution',
                    'item': size if size else 'Unknown',
                    'mq_weight': weight,
                    'spy_weight': None,
                    'delta': None,
                    'sector': None,
                    'size_bucket': size
                })

            # --- Mag-7 Exposure ---
            mag7_holdings = holdings[holdings['ticker'].isin(MAG_7)]
            mag7_weight = mag7_holdings['weight'].sum()
            mag7_count = len(mag7_holdings)
            exposure_data.append({
                'snapshot_date': snapshot_date,
                'analysis_type': 'mag7_exposure',
                'item': f"Mag-7 ({mag7_count} of 8)",
                'mq_weight': mag7_weight,
                'spy_weight': 0.30,  # Approximate SPY Mag-7 weight (~30%)
                'delta': mag7_weight - 0.30,
                'sector': 'Mixed',
                'size_bucket': '6 - Mega'
            })

            logger.info(f"    Positions: {len(long_tickers)}, Mag-7: {mag7_weight:.1%} vs SPY ~30%")

        conn.close()

        return exposure_data

    def _save_exposure_csv(self, exposure_data: List[Dict]):
        """Save exposure snapshot CSV."""
        if not exposure_data:
            logger.warning("No exposure data to save")
            return

        logger.info("Generating exposure snapshot CSV...")

        df = pd.DataFrame(exposure_data)

        output_dir = Path('results/ensemble_baselines')
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / 'momentum_quality_v1_exposure_snapshot.csv'

        df.to_csv(csv_path, index=False)
        logger.info(f"Saved exposure snapshot: {csv_path}")
        logger.info(f"  Rows: {len(df)}")

    def _compute_bootstrap_sharpe(self, equity_curve: pd.Series, n_bootstrap: int = 5000) -> Dict:
        """
        Compute bootstrap confidence intervals for Sharpe ratio.

        Args:
            equity_curve: Portfolio equity curve
            n_bootstrap: Number of bootstrap resamples

        Returns:
            Dict with bootstrap statistics
        """
        logger.info(f"Computing bootstrap Sharpe ratio ({n_bootstrap} resamples)...")

        # Compute monthly returns
        monthly_returns = equity_curve.resample('ME').last().pct_change().dropna()

        if len(monthly_returns) < 12:
            logger.warning("Not enough monthly returns for meaningful bootstrap")
            return None

        bootstrap_sharpes = []
        np.random.seed(42)  # Reproducibility

        for _ in range(n_bootstrap):
            # Resample with replacement
            resampled = np.random.choice(monthly_returns.values, size=len(monthly_returns), replace=True)
            # Compute Sharpe (annualized)
            if np.std(resampled) > 0:
                sharpe = np.mean(resampled) / np.std(resampled) * np.sqrt(12)
                bootstrap_sharpes.append(sharpe)

        bootstrap_sharpes = np.array(bootstrap_sharpes)

        results = {
            'mean_sharpe': np.mean(bootstrap_sharpes),
            'std_sharpe': np.std(bootstrap_sharpes),
            'ci_5': np.percentile(bootstrap_sharpes, 5),
            'ci_95': np.percentile(bootstrap_sharpes, 95),
            'prob_positive': np.mean(bootstrap_sharpes > 0),
            'n_bootstrap': n_bootstrap,
            'n_periods': len(monthly_returns)
        }

        logger.info(f"  Bootstrap Sharpe: {results['mean_sharpe']:.3f} (std: {results['std_sharpe']:.3f})")
        logger.info(f"  95% CI: [{results['ci_5']:.3f}, {results['ci_95']:.3f}]")
        logger.info(f"  P(Sharpe > 0): {results['prob_positive']:.1%}")

        return results

    def _save_bootstrap_csv(self, bootstrap_data: Dict):
        """Save bootstrap results CSV."""
        if not bootstrap_data:
            logger.warning("No bootstrap data to save")
            return

        logger.info("Generating bootstrap CSV...")

        # Create single-row DataFrame with results
        df = pd.DataFrame([bootstrap_data])

        output_dir = Path('results/ensemble_baselines')
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / 'momentum_quality_v1_bootstrap_sharpe.csv'

        df.to_csv(csv_path, index=False)
        logger.info(f"Saved bootstrap results: {csv_path}")

    def _save_diagnostic_markdown(self, results: Dict):
        """Save diagnostic markdown file."""
        output_dir = Path('results/ensemble_baselines')
        output_dir.mkdir(parents=True, exist_ok=True)

        md_path = output_dir / 'momentum_quality_v1_diagnostic.md'

        momentum = results['momentum_only']
        multi = results['momentum_quality']

        content = f"""# Momentum + Quality v1 Ensemble Diagnostic

**Generated:** {results['generated']}
**Period:** {results['start_date']} to {results['end_date']}
**Capital:** ${results['initial_capital']:,.0f}
**Universe:** S&P 500 (actual constituents, min_price=$5)
**Rebalance:** Monthly

---

## Configuration Summary

### Momentum-Only Baseline
- **Ensemble:** `momentum_v2_adaptive_quintile`
- **Signal:** InstitutionalMomentum v2
- **Parameters:** 308-day formation, 0-day skip, adaptive quintiles
- **Normalization:** none (signal already returns quintiles)
- **Pathway:** Price-based (via `make_ensemble_signal_fn`)

### Momentum + Quality v1
- **Ensemble:** `momentum_quality_v1`
- **Signals:**
  - InstitutionalMomentum v2 (weight=0.5)
  - CrossSectionalQuality v1 (weight=0.5)
- **Parameters:**
  - Momentum: 308-day formation, 0-day skip, adaptive quintiles
  - Quality: QMJ methodology (profitability=0.4, growth=0.3, safety=0.3), adaptive quintiles
- **Normalization:** none (both signals return quintiles)
- **Pathway:** Cross-sectional (via `make_multisignal_ensemble_fn`)

---

## Performance Comparison

| Metric | Momentum-Only | Momentum+Quality v1 | Difference |
|--------|--------------|-------------------|-----------|
| **Total Return** | {momentum['total_return']:.2%} | {multi['total_return']:.2%} | {(multi['total_return'] - momentum['total_return']):.2%} |
| **CAGR** | {momentum['cagr']:.2%} | {multi['cagr']:.2%} | {(multi['cagr'] - momentum['cagr']):.2%} |
| **Volatility** | {momentum['volatility']:.2%} | {multi['volatility']:.2%} | {(multi['volatility'] - momentum['volatility']):.2%} |
| **Sharpe Ratio** | {momentum['sharpe']:.3f} | {multi['sharpe']:.3f} | {(multi['sharpe'] - momentum['sharpe']):.3f} |
| **Max Drawdown** | {momentum['max_drawdown']:.2%} | {multi['max_drawdown']:.2%} | {(multi['max_drawdown'] - momentum['max_drawdown']):.2%} |
| **Rebalances** | {momentum['num_rebalances']} | {multi['num_rebalances']} | {multi['num_rebalances'] - momentum['num_rebalances']} |

---

## Return Correlation

**Periodic Return Correlation:** {results['return_correlation']:.4f}

The correlation of {results['return_correlation']:.4f} indicates that the two strategies have {'highly correlated' if results['return_correlation'] > 0.8 else 'moderately correlated' if results['return_correlation'] > 0.5 else 'low correlation'} returns.

---

## Observations

### Total Return Impact
"""

        # Add qualitative observations based on metrics
        ret_diff = multi['total_return'] - momentum['total_return']
        if abs(ret_diff) < 0.05:
            content += f"- Adding quality has minimal impact on total return ({ret_diff:+.2%})\n"
        elif ret_diff > 0:
            content += f"- Adding quality **improves** total return by {ret_diff:.2%}\n"
        else:
            content += f"- Adding quality **reduces** total return by {abs(ret_diff):.2%}\n"

        # Sharpe comparison
        sharpe_diff = multi['sharpe'] - momentum['sharpe']
        if abs(sharpe_diff) < 0.1:
            content += f"- Sharpe ratio roughly unchanged ({sharpe_diff:+.3f})\n"
        elif sharpe_diff > 0:
            content += f"- Sharpe ratio **improves** by {sharpe_diff:.3f}\n"
        else:
            content += f"- Sharpe ratio **deteriorates** by {abs(sharpe_diff):.3f}\n"

        # Drawdown comparison
        # Note: Max drawdown is negative. Less negative = better (smaller loss)
        dd_diff = multi['max_drawdown'] - momentum['max_drawdown']
        if abs(dd_diff) < 0.02:
            content += f"- Max drawdown similar ({dd_diff:+.2%})\n"
        elif dd_diff > 0:
            # Positive diff means less negative = improvement
            content += f"- Max drawdown **improved** by {abs(dd_diff):.2%} (less severe loss)\n"
        else:
            # Negative diff means more negative = deterioration
            content += f"- Max drawdown **deteriorated** by {abs(dd_diff):.2%} (more severe loss)\n"

        # Correlation interpretation
        if results['return_correlation'] > 0.8:
            content += f"- High return correlation ({results['return_correlation']:.4f}) suggests quality adds limited diversification\n"
        elif results['return_correlation'] > 0.5:
            content += f"- Moderate correlation ({results['return_correlation']:.4f}) suggests partial diversification benefit\n"
        else:
            content += f"- Low correlation ({results['return_correlation']:.4f}) suggests strong diversification from quality factor\n"

        # Add SPY benchmark section if available
        if results.get('benchmark'):
            bench = results['benchmark']
            content += f"""
### SPY Benchmark Comparison

**vs S&P 500 ETF (SPY):**

| Metric | Value |
|--------|-------|
| **Tracking Error** | {bench['tracking_error']:.2%} |
| **Information Ratio** | {bench['information_ratio']:.3f} |
| **Annualized Excess Return** | {bench['annualized_excess']:.2%} |
| **Total Excess Return** | {bench['excess_total']:.2%} |
| **Periods Analyzed** | {bench['num_periods']} months |

**Interpretation:**
"""

            # Add interpretation based on metrics
            if bench['information_ratio'] > 0.5:
                content += f"- Strong information ratio ({bench['information_ratio']:.3f}) indicates consistent outperformance vs SPY\n"
            elif bench['information_ratio'] > 0.0:
                content += f"- Positive information ratio ({bench['information_ratio']:.3f}) suggests modest alpha generation\n"
            else:
                content += f"- Negative information ratio ({bench['information_ratio']:.3f}) indicates underperformance vs SPY\n"

            if bench['tracking_error'] < 0.05:
                content += f"- Low tracking error ({bench['tracking_error']:.2%}) suggests high correlation with benchmark\n"
            elif bench['tracking_error'] < 0.10:
                content += f"- Moderate tracking error ({bench['tracking_error']:.2%}) indicates some divergence from SPY\n"
            else:
                content += f"- High tracking error ({bench['tracking_error']:.2%}) indicates significant active management\n"

            content += f"\n**Monthly comparison saved to:** `results/ensemble_baselines/momentum_quality_v1_vs_spy_monthly.csv`\n"

        # Add turnover section if available
        if results.get('turnover'):
            turnover = results['turnover']
            stats = turnover['stats']
            content += f"""
### Portfolio Turnover

**Turnover Statistics** (computed over {stats['num_rebalances']} rebalances):

| Statistic | Value |
|-----------|-------|
| **Mean Turnover** | {stats['mean']:.2%} |
| **Median Turnover** | {stats['median']:.2%} |
| **25th Percentile** | {stats['p25']:.2%} |
| **75th Percentile** | {stats['p75']:.2%} |
| **95th Percentile** | {stats['p95']:.2%} |
| **Min Turnover** | {stats['min']:.2%} |
| **Max Turnover** | {stats['max']:.2%} |

**Interpretation:**
"""

            # Add interpretation
            if stats['mean'] < 0.20:
                content += f"- Low mean turnover ({stats['mean']:.2%}) suggests stable, low-churn portfolio\n"
            elif stats['mean'] < 0.50:
                content += f"- Moderate mean turnover ({stats['mean']:.2%}) indicates balanced rebalancing activity\n"
            else:
                content += f"- High mean turnover ({stats['mean']:.2%}) suggests active portfolio management\n"

            if stats['p95'] < 0.50:
                content += f"- 95th percentile ({stats['p95']:.2%}) shows turnover rarely exceeds 50% of portfolio\n"
            elif stats['p95'] < 0.75:
                content += f"- 95th percentile ({stats['p95']:.2%}) indicates occasional significant rebalancing\n"
            else:
                content += f"- 95th percentile ({stats['p95']:.2%}) shows some rebalances involve major position changes\n"

            content += f"\n**Time series saved to:** `results/ensemble_baselines/momentum_quality_v1_turnover_metrics.csv`\n"

        content += """
### Next Steps

- If Sharpe improved and drawdown reduced: consider increasing quality weight
- If correlation is high (>0.8): quality may be redundant with momentum
- If total return declined significantly: investigate quality signal performance in isolation
- Consider regime-specific analysis (COVID, 2022 bear market, recent period)

---

**Status:** Phase 3 Milestone 3.2 baseline diagnostic
**Note:** This uses the new cross-sectional ensemble pathway for momentum+quality
"""

        md_path.write_text(content)
        logger.info(f"Saved diagnostic: {md_path}")

    def _save_comparison_csv(self, results: Dict):
        """Save comparison CSV with key metrics."""
        output_dir = Path('results/ensemble_baselines')
        csv_path = output_dir / 'momentum_quality_v1_comparison.csv'

        momentum = results['momentum_only']
        multi = results['momentum_quality']

        comparison_data = {
            'Metric': [
                'Total Return',
                'CAGR',
                'Volatility',
                'Sharpe Ratio',
                'Max Drawdown',
                'Num Rebalances',
                'Return Correlation'
            ],
            'Momentum-Only': [
                f"{momentum['total_return']:.4f}",
                f"{momentum['cagr']:.4f}",
                f"{momentum['volatility']:.4f}",
                f"{momentum['sharpe']:.4f}",
                f"{momentum['max_drawdown']:.4f}",
                f"{momentum['num_rebalances']}",
                "1.0000"
            ],
            'Momentum+Quality': [
                f"{multi['total_return']:.4f}",
                f"{multi['cagr']:.4f}",
                f"{multi['volatility']:.4f}",
                f"{multi['sharpe']:.4f}",
                f"{multi['max_drawdown']:.4f}",
                f"{multi['num_rebalances']}",
                f"{results['return_correlation']:.4f}"
            ]
        }

        df = pd.DataFrame(comparison_data)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved comparison CSV: {csv_path}")

    def _print_summary(self, results: Dict):
        """Print summary to stdout."""
        momentum = results['momentum_only']
        multi = results['momentum_quality']

        logger.info("=" * 80)
        logger.info("DIAGNOSTIC SUMMARY")
        logger.info("=" * 80)
        logger.info("")
        logger.info(f"{'Metric':<20} {'Momentum-Only':<20} {'Momentum+Quality':<20} {'Difference':<15}")
        logger.info("-" * 80)
        logger.info(f"{'Total Return':<20} {momentum['total_return']:>19.2%} {multi['total_return']:>19.2%} {(multi['total_return'] - momentum['total_return']):>14.2%}")
        logger.info(f"{'CAGR':<20} {momentum['cagr']:>19.2%} {multi['cagr']:>19.2%} {(multi['cagr'] - momentum['cagr']):>14.2%}")
        logger.info(f"{'Volatility':<20} {momentum['volatility']:>19.2%} {multi['volatility']:>19.2%} {(multi['volatility'] - momentum['volatility']):>14.2%}")
        logger.info(f"{'Sharpe':<20} {momentum['sharpe']:>19.3f} {multi['sharpe']:>19.3f} {(multi['sharpe'] - momentum['sharpe']):>14.3f}")
        logger.info(f"{'Max Drawdown':<20} {momentum['max_drawdown']:>19.2%} {multi['max_drawdown']:>19.2%} {(multi['max_drawdown'] - momentum['max_drawdown']):>14.2%}")
        logger.info("")
        logger.info(f"Return Correlation: {results['return_correlation']:.4f}")

        # Add benchmark metrics if available
        if results.get('benchmark'):
            bench = results['benchmark']
            logger.info("")
            logger.info("-" * 80)
            logger.info("SPY BENCHMARK COMPARISON")
            logger.info("-" * 80)
            logger.info(f"{'Tracking Error':<30} {bench['tracking_error']:>19.2%}")
            logger.info(f"{'Information Ratio':<30} {bench['information_ratio']:>19.3f}")
            logger.info(f"{'Annualized Excess Return':<30} {bench['annualized_excess']:>19.2%}")
            logger.info(f"{'Total Excess Return':<30} {bench['excess_total']:>19.2%}")

        # Add turnover metrics if available
        if results.get('turnover'):
            stats = results['turnover']['stats']
            logger.info("")
            logger.info("-" * 80)
            logger.info("PORTFOLIO TURNOVER")
            logger.info("-" * 80)
            logger.info(f"{'Mean Turnover':<30} {stats['mean']:>19.2%}")
            logger.info(f"{'Median Turnover':<30} {stats['median']:>19.2%}")
            logger.info(f"{'95th Percentile':<30} {stats['p95']:>19.2%}")

        logger.info("")
        logger.info("=" * 80)


def main():
    """Run momentum + quality diagnostic."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Run momentum + quality ensemble diagnostic',
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
    runner = MomentumQualityDiagnosticRunner(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital
    )

    results = runner.run_diagnostic()

    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ DIAGNOSTIC COMPLETE")
    logger.info("=" * 80)
    logger.info("Results saved to:")
    logger.info("  - results/ensemble_baselines/momentum_quality_v1_diagnostic.md")
    logger.info("  - results/ensemble_baselines/momentum_quality_v1_comparison.csv")
    logger.info("")


if __name__ == '__main__':
    main()
