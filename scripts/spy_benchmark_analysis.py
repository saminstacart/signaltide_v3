"""
SPY Benchmark Analysis - Comprehensive Framework

Proves (or disproves) that our strategy beats SPY with institutional rigor.

Key Metrics:
1. Information Ratio (IR > 1.0 = excellent)
2. Alpha/Beta decomposition
3. Risk-adjusted performance (Sharpe, Sortino, Calmar)
4. Drawdown comparison
5. Regime-specific performance
6. Rolling performance windows
7. Statistical significance tests

Usage:
    python3 scripts/spy_benchmark_analysis.py --signals institutional --period 2020-2024

Output:
    - results/spy_benchmark_report.md
    - results/spy_performance_charts.png
    - results/spy_metrics.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse
from scipy import stats
from config import get_logger

logger = get_logger(__name__)


class SPYBenchmarkAnalysis:
    """
    Comprehensive SPY benchmark comparison framework.

    Answers THE question: Do we beat SPY?
    """

    def __init__(self,
                 strategy_returns: pd.Series,
                 spy_returns: pd.Series,
                 rf_rate: float = 0.02):
        """
        Initialize benchmark analysis.

        Args:
            strategy_returns: Daily strategy returns
            spy_returns: Daily SPY returns
            rf_rate: Risk-free rate (annualized, default 2%)
        """
        # Align indices
        common_index = strategy_returns.index.intersection(spy_returns.index)
        self.strategy_returns = strategy_returns.loc[common_index]
        self.spy_returns = spy_returns.loc[common_index]

        self.rf_rate = rf_rate
        self.rf_daily = (1 + rf_rate) ** (1/252) - 1

        logger.info(f"Initialized SPY benchmark analysis")
        logger.info(f"Period: {common_index.min().date()} to {common_index.max().date()}")
        logger.info(f"Trading days: {len(common_index)}")

        # Results storage
        self.metrics = {}

    # ==================== CORE METRICS ====================

    def information_ratio(self) -> float:
        """
        Information Ratio = (Strategy Return - SPY Return) / Tracking Error

        THE primary metric for active management.

        Interpretation:
        IR > 0.5: Good (beat SPY with reasonable consistency)
        IR > 1.0: Excellent (institutional quality)
        IR > 2.0: World-class (top 1% of managers)

        Returns:
            Annualized Information Ratio
        """
        excess_returns = self.strategy_returns - self.spy_returns

        # Annualize
        mean_excess = excess_returns.mean() * 252
        tracking_error = excess_returns.std() * np.sqrt(252)

        if tracking_error == 0:
            return 0.0

        ir = mean_excess / tracking_error

        self.metrics['information_ratio'] = ir
        self.metrics['tracking_error'] = tracking_error

        logger.info(f"Information Ratio: {ir:.3f}")
        logger.info(f"Tracking Error: {tracking_error*100:.2f}%")

        return ir

    def alpha_beta_analysis(self) -> Dict[str, float]:
        """
        Regression: Strategy = α + β*SPY + ε

        Alpha: Excess return NOT explained by market
        Beta: Market exposure (sensitivity to SPY)

        Questions:
        1. Is alpha > 0? (we add value)
        2. Is alpha > transaction costs?
        3. Is beta ≈ 1.0? (market-neutral, not levered SPY)
        4. Is alpha statistically significant?

        Returns:
            Dict with alpha, beta, r_squared, p_value
        """
        # Remove NaN
        df = pd.DataFrame({
            'strategy': self.strategy_returns,
            'spy': self.spy_returns
        }).dropna()

        # Regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df['spy'], df['strategy']
        )

        # Annualize alpha
        alpha_daily = intercept
        alpha_annual = alpha_daily * 252

        beta = slope
        r_squared = r_value ** 2

        self.metrics['alpha'] = alpha_annual
        self.metrics['beta'] = beta
        self.metrics['r_squared'] = r_squared
        self.metrics['alpha_pvalue'] = p_value

        logger.info(f"Alpha (annualized): {alpha_annual*100:.2f}%")
        logger.info(f"Beta: {beta:.3f}")
        logger.info(f"R-squared: {r_squared:.3f}")
        logger.info(f"Alpha p-value: {p_value:.4f}")

        # Interpretation
        if p_value < 0.05:
            logger.info("✅ Alpha is statistically significant!")
        else:
            logger.warning("⚠️ Alpha is NOT statistically significant")

        if abs(beta - 1.0) < 0.1:
            logger.info("✅ Beta ≈ 1.0 (market-neutral exposure)")
        elif beta > 1.1:
            logger.warning(f"⚠️ Beta = {beta:.2f} (levered SPY exposure)")
        else:
            logger.warning(f"⚠️ Beta = {beta:.2f} (defensive exposure)")

        return {
            'alpha_annual': alpha_annual,
            'beta': beta,
            'r_squared': r_squared,
            'p_value': p_value
        }

    def sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe Ratio."""
        excess = returns - self.rf_daily
        sharpe = (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() > 0 else 0
        return sharpe

    def sortino_ratio(self, returns: pd.Series) -> float:
        """
        Sortino Ratio = (Return - RiskFree) / Downside Deviation

        Only penalizes DOWNSIDE volatility.
        Better metric than Sharpe for asymmetric returns.
        """
        excess = returns - self.rf_daily
        downside = excess[excess < 0]

        if len(downside) == 0 or downside.std() == 0:
            return 0.0

        downside_dev = downside.std() * np.sqrt(252)
        sortino = (excess.mean() * 252) / downside_dev

        return sortino

    def calmar_ratio(self, returns: pd.Series) -> float:
        """
        Calmar Ratio = Annual Return / Max Drawdown

        Measures return per unit of worst loss.
        """
        annual_return = returns.mean() * 252
        max_dd = self.max_drawdown(returns)

        if max_dd == 0:
            return 0.0

        calmar = annual_return / abs(max_dd)
        return calmar

    def risk_adjusted_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compare all risk-adjusted metrics.

        Returns:
            Dict with strategy and SPY metrics
        """
        strategy_sharpe = self.sharpe_ratio(self.strategy_returns)
        spy_sharpe = self.sharpe_ratio(self.spy_returns)

        strategy_sortino = self.sortino_ratio(self.strategy_returns)
        spy_sortino = self.sortino_ratio(self.spy_returns)

        strategy_calmar = self.calmar_ratio(self.strategy_returns)
        spy_calmar = self.calmar_ratio(self.spy_returns)

        self.metrics['strategy_sharpe'] = strategy_sharpe
        self.metrics['spy_sharpe'] = spy_sharpe
        self.metrics['strategy_sortino'] = strategy_sortino
        self.metrics['spy_sortino'] = spy_sortino
        self.metrics['strategy_calmar'] = strategy_calmar
        self.metrics['spy_calmar'] = spy_calmar

        logger.info(f"\n{'='*60}")
        logger.info("RISK-ADJUSTED PERFORMANCE")
        logger.info(f"{'='*60}")
        logger.info(f"{'Metric':<20} {'Strategy':>15} {'SPY':>15} {'Diff':>10}")
        logger.info(f"{'-'*60}")
        logger.info(f"{'Sharpe Ratio':<20} {strategy_sharpe:>15.3f} {spy_sharpe:>15.3f} {strategy_sharpe-spy_sharpe:>+10.3f}")
        logger.info(f"{'Sortino Ratio':<20} {strategy_sortino:>15.3f} {spy_sortino:>15.3f} {strategy_sortino-spy_sortino:>+10.3f}")
        logger.info(f"{'Calmar Ratio':<20} {strategy_calmar:>15.3f} {spy_calmar:>15.3f} {strategy_calmar-spy_calmar:>+10.3f}")

        return {
            'strategy': {
                'sharpe': strategy_sharpe,
                'sortino': strategy_sortino,
                'calmar': strategy_calmar
            },
            'spy': {
                'sharpe': spy_sharpe,
                'sortino': spy_sortino,
                'calmar': spy_calmar
            }
        }

    # ==================== DRAWDOWN ANALYSIS ====================

    def max_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate maximum drawdown.

        Returns:
            Max drawdown as negative decimal (e.g., -0.25 = -25%)
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        return drawdown.min()

    def drawdown_analysis(self) -> Dict[str, any]:
        """
        Comprehensive drawdown comparison.

        Metrics:
        - Max drawdown
        - Average drawdown
        - Max drawdown duration
        - Recovery time
        """
        def analyze_drawdowns(returns):
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max

            max_dd = drawdown.min()
            avg_dd = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0

            # Find max drawdown period
            max_dd_idx = drawdown.idxmin()

            # Duration: time from peak to trough
            peak_idx = running_max[:max_dd_idx].idxmax()
            dd_duration = (max_dd_idx - peak_idx).days

            # Recovery: time from trough back to peak
            recovery_idx = cumulative[max_dd_idx:][cumulative >= running_max.loc[max_dd_idx]].index
            recovery_days = (recovery_idx[0] - max_dd_idx).days if len(recovery_idx) > 0 else None

            return {
                'max_drawdown': max_dd,
                'avg_drawdown': avg_dd,
                'max_dd_date': max_dd_idx,
                'dd_duration_days': dd_duration,
                'recovery_days': recovery_days
            }

        strategy_dd = analyze_drawdowns(self.strategy_returns)
        spy_dd = analyze_drawdowns(self.spy_returns)

        self.metrics['strategy_max_dd'] = strategy_dd['max_drawdown']
        self.metrics['spy_max_dd'] = spy_dd['max_drawdown']

        logger.info(f"\n{'='*60}")
        logger.info("DRAWDOWN ANALYSIS")
        logger.info(f"{'='*60}")
        logger.info(f"{'Metric':<25} {'Strategy':>15} {'SPY':>15}")
        logger.info(f"{'-'*60}")
        logger.info(f"{'Max Drawdown':<25} {strategy_dd['max_drawdown']*100:>14.2f}% {spy_dd['max_drawdown']*100:>14.2f}%")
        logger.info(f"{'Avg Drawdown':<25} {strategy_dd['avg_drawdown']*100:>14.2f}% {spy_dd['avg_drawdown']*100:>14.2f}%")
        logger.info(f"{'Max DD Duration (days)':<25} {strategy_dd['dd_duration_days']:>15} {spy_dd['dd_duration_days']:>15}")

        if strategy_dd['recovery_days']:
            logger.info(f"{'Recovery Days':<25} {strategy_dd['recovery_days']:>15} {spy_dd['recovery_days'] or 'N/A':>15}")

        # Comparison
        if strategy_dd['max_drawdown'] > spy_dd['max_drawdown']:
            logger.warning(f"⚠️ Strategy has WORSE drawdown than SPY")
        else:
            logger.info(f"✅ Strategy has BETTER drawdown protection than SPY")

        return {
            'strategy': strategy_dd,
            'spy': spy_dd
        }

    # ==================== REGIME ANALYSIS ====================

    def regime_performance(self) -> Dict[str, Dict]:
        """
        Performance by market regime.

        Regimes:
        1. Bull (SPY > 200 MA)
        2. Bear (SPY < 200 MA)
        3. High Volatility (21-day rolling vol > 20% annualized)
        """
        # Calculate SPY 200-day MA
        spy_prices = (1 + self.spy_returns).cumprod()
        ma_200 = spy_prices.rolling(200).mean()

        # Calculate 21-day rolling volatility
        rolling_vol = self.spy_returns.rolling(21).std() * np.sqrt(252)

        # Define regimes
        bull = spy_prices > ma_200
        bear = spy_prices <= ma_200
        high_vol = rolling_vol > 0.20  # 20% annualized

        def regime_metrics(mask, name):
            if mask.sum() == 0:
                return None

            strat_ret = self.strategy_returns[mask]
            spy_ret = self.spy_returns[mask]

            return {
                'name': name,
                'days': mask.sum(),
                'strategy_return': strat_ret.mean() * 252,
                'spy_return': spy_ret.mean() * 252,
                'strategy_sharpe': self.sharpe_ratio(strat_ret),
                'spy_sharpe': self.sharpe_ratio(spy_ret),
                'outperformance': (strat_ret.mean() - spy_ret.mean()) * 252
            }

        regimes = {
            'bull': regime_metrics(bull, 'Bull Market'),
            'bear': regime_metrics(bear, 'Bear Market'),
            'high_vol': regime_metrics(high_vol, 'High Volatility')
        }

        logger.info(f"\n{'='*60}")
        logger.info("REGIME-SPECIFIC PERFORMANCE")
        logger.info(f"{'='*60}")

        for regime_name, metrics in regimes.items():
            if metrics:
                logger.info(f"\n{metrics['name']} ({metrics['days']} days):")
                logger.info(f"  Strategy Return: {metrics['strategy_return']*100:>6.2f}%")
                logger.info(f"  SPY Return: {metrics['spy_return']*100:>11.2f}%")
                logger.info(f"  Outperformance: {metrics['outperformance']*100:>+6.2f}%")
                logger.info(f"  Strategy Sharpe: {metrics['strategy_sharpe']:>5.3f}")
                logger.info(f"  SPY Sharpe: {metrics['spy_sharpe']:>10.3f}")

        self.metrics['regime_performance'] = regimes

        return regimes

    # ==================== CONSISTENCY ANALYSIS ====================

    def rolling_performance(self, window_days: int = 252) -> Dict:
        """
        Rolling window performance analysis.

        Args:
            window_days: Window size in days (default 252 = 1 year)

        Returns:
            Dict with rolling statistics
        """
        # Calculate rolling returns
        strategy_rolling = self.strategy_returns.rolling(window_days).apply(
            lambda x: (1 + x).prod() - 1
        )
        spy_rolling = self.spy_returns.rolling(window_days).apply(
            lambda x: (1 + x).prod() - 1
        )

        # Outperformance
        outperformance = strategy_rolling - spy_rolling

        # Win rate
        wins = (outperformance > 0).sum()
        total = (~outperformance.isna()).sum()
        win_rate = wins / total if total > 0 else 0

        # Average win/loss
        avg_win = outperformance[outperformance > 0].mean() if (outperformance > 0).any() else 0
        avg_loss = outperformance[outperformance < 0].mean() if (outperformance < 0).any() else 0

        self.metrics['rolling_win_rate'] = win_rate
        self.metrics['rolling_avg_win'] = avg_win
        self.metrics['rolling_avg_loss'] = avg_loss

        logger.info(f"\n{'='*60}")
        logger.info(f"ROLLING {window_days}-DAY PERFORMANCE")
        logger.info(f"{'='*60}")
        logger.info(f"Win Rate: {win_rate*100:.1f}% ({wins}/{total} periods)")
        logger.info(f"Avg Win: {avg_win*100:+.2f}%")
        logger.info(f"Avg Loss: {avg_loss*100:+.2f}%")
        logger.info(f"Win/Loss Ratio: {abs(avg_win/avg_loss):.2f}x" if avg_loss != 0 else "N/A")

        if win_rate >= 0.60:
            logger.info("✅ Win rate > 60% (excellent consistency)")
        elif win_rate >= 0.50:
            logger.info("✅ Win rate > 50% (good consistency)")
        else:
            logger.warning("⚠️ Win rate < 50% (inconsistent outperformance)")

        return {
            'win_rate': win_rate,
            'total_periods': total,
            'wins': wins,
            'losses': total - wins,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }

    def calendar_year_performance(self) -> pd.DataFrame:
        """
        Annual performance by calendar year.

        Returns:
            DataFrame with yearly metrics
        """
        yearly_results = []

        for year in self.strategy_returns.index.year.unique():
            year_mask = self.strategy_returns.index.year == year

            strat_ret = self.strategy_returns[year_mask]
            spy_ret = self.spy_returns[year_mask]

            yearly_results.append({
                'year': year,
                'strategy_return': (1 + strat_ret).prod() - 1,
                'spy_return': (1 + spy_ret).prod() - 1,
                'outperformance': ((1 + strat_ret).prod() - 1) - ((1 + spy_ret).prod() - 1),
                'strategy_sharpe': self.sharpe_ratio(strat_ret),
                'spy_sharpe': self.sharpe_ratio(spy_ret)
            })

        df = pd.DataFrame(yearly_results)

        logger.info(f"\n{'='*60}")
        logger.info("CALENDAR YEAR PERFORMANCE")
        logger.info(f"{'='*60}")
        logger.info(f"{'Year':<10} {'Strategy':>12} {'SPY':>12} {'Outperf':>12}")
        logger.info(f"{'-'*60}")

        for _, row in df.iterrows():
            logger.info(
                f"{row['year']:<10} {row['strategy_return']*100:>11.2f}% "
                f"{row['spy_return']*100:>11.2f}% {row['outperformance']*100:>+11.2f}%"
            )

        return df

    # ==================== SUMMARY ====================

    def run_full_analysis(self) -> Dict:
        """
        Run complete benchmark analysis.

        Returns:
            Dict with all metrics
        """
        logger.info("\n" + "="*60)
        logger.info("SPY BENCHMARK ANALYSIS - FULL REPORT")
        logger.info("="*60)

        # Core metrics
        self.information_ratio()
        self.alpha_beta_analysis()
        self.risk_adjusted_metrics()

        # Risk analysis
        self.drawdown_analysis()

        # Regime analysis
        self.regime_performance()

        # Consistency
        self.rolling_performance()
        self.calendar_year_performance()

        # Overall assessment
        logger.info("\n" + "="*60)
        logger.info("OVERALL ASSESSMENT")
        logger.info("="*60)

        ir = self.metrics.get('information_ratio', 0)
        alpha = self.metrics.get('alpha', 0)
        alpha_pval = self.metrics.get('alpha_pvalue', 1)

        passed = 0
        total = 0

        # Check 1: IR
        total += 1
        if ir > 1.0:
            logger.info("✅ Information Ratio > 1.0 (EXCELLENT)")
            passed += 1
        elif ir > 0.5:
            logger.info("✅ Information Ratio > 0.5 (GOOD)")
            passed += 1
        else:
            logger.warning(f"❌ Information Ratio = {ir:.2f} (< 0.5)")

        # Check 2: Alpha
        total += 1
        if alpha > 0 and alpha_pval < 0.05:
            logger.info(f"✅ Positive alpha ({alpha*100:.2f}%, p={alpha_pval:.3f})")
            passed += 1
        else:
            logger.warning(f"❌ Alpha not significant")

        # Check 3: Sharpe
        total += 1
        if self.metrics['strategy_sharpe'] > self.metrics['spy_sharpe']:
            logger.info("✅ Sharpe Ratio beats SPY")
            passed += 1
        else:
            logger.warning("❌ Sharpe Ratio below SPY")

        # Check 4: Drawdown
        total += 1
        # Max DD is negative (e.g., -0.28). Less negative = better performance
        if self.metrics['strategy_max_dd'] > self.metrics['spy_max_dd']:
            logger.info("✅ Max drawdown better than SPY")
            passed += 1
        else:
            logger.warning("❌ Max drawdown worse than SPY")

        # Check 5: Win rate
        total += 1
        if self.metrics['rolling_win_rate'] >= 0.60:
            logger.info(f"✅ Win rate = {self.metrics['rolling_win_rate']*100:.1f}% (> 60%)")
            passed += 1
        elif self.metrics['rolling_win_rate'] >= 0.50:
            logger.info(f"✅ Win rate = {self.metrics['rolling_win_rate']*100:.1f}% (> 50%)")
            passed += 1
        else:
            logger.warning(f"❌ Win rate = {self.metrics['rolling_win_rate']*100:.1f}% (< 50%)")

        logger.info(f"\nFinal Score: {passed}/{total} checks passed")

        if passed == total:
            logger.info("\n🎉 EXCELLENT: All checks passed! Strategy beats SPY convincingly.")
        elif passed >= total * 0.6:
            logger.info("\n✅ GOOD: Most checks passed. Strategy shows promise.")
        else:
            logger.warning("\n⚠️ NEEDS WORK: Strategy does not convincingly beat SPY.")

        return self.metrics


def main():
    parser = argparse.ArgumentParser(description='SPY Benchmark Analysis')
    parser.add_argument('--signals', default='institutional', help='Signal type (simple/institutional)')
    parser.add_argument('--period', default='2020-2024', help='Test period')
    args = parser.parse_args()

    # TODO: Load actual strategy and SPY returns
    # For now, create dummy data
    logger.info("SPY Benchmark Analysis - IN DEVELOPMENT")
    logger.info("Next: Integrate with actual backtest results")

    # Example usage (will be replaced with real data):
    """
    from data.data_manager import DataManager
    dm = DataManager()

    # Get SPY returns
    spy_data = dm.get_prices('SPY', '2020-01-01', '2024-12-31')
    spy_returns = spy_data['close'].pct_change().dropna()

    # Get strategy returns (from backtest)
    strategy_returns = ...  # Load from backtest results

    # Run analysis
    analyzer = SPYBenchmarkAnalysis(strategy_returns, spy_returns)
    results = analyzer.run_full_analysis()
    """


if __name__ == '__main__':
    main()
