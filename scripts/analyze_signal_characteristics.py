"""
Analyze Signal Trading Characteristics

Understand how many trading opportunities each optimized signal produces,
then recommend appropriate testing methodology.

Usage:
    python scripts/analyze_signal_characteristics.py
"""

import sys
sys.path.insert(0, '/Users/samuelksherman/signaltide_v3')

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from data.data_manager import DataManager
from signals.momentum.simple_momentum import SimpleMomentum
from signals.quality.simple_quality import SimpleQuality
from signals.insider.simple_insider import SimpleInsider


class SignalCharacteristicsAnalyzer:
    """Analyze trading characteristics of optimized signals."""

    def __init__(self):
        self.dm = DataManager()

        # Optimized parameters from Optuna
        self.best_params = {
            'momentum': {
                'lookback': 25,
                'rank_window': 62,
                'long_threshold': 0.2538,
                'short_threshold': -0.8770
            },
            'quality': {
                'rank_window': 284,
                'long_threshold': 0.0833,
                'short_threshold': -0.2538
            },
            'insider': {
                'lookback_days': 35,
                'rank_window': 102,
                'long_threshold': 0.1435,
                'short_threshold': -0.8688
            }
        }

        # Test universe
        self.tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
                       'NVDA', 'TSLA', 'JPM', 'JNJ', 'XOM']

        # Test period
        self.start_date = '2020-01-01'
        self.end_date = '2023-12-31'

    def analyze_signal(self, signal_type: str) -> dict:
        """Analyze trading characteristics for a signal type."""
        print(f"\n{'='*80}")
        print(f"Analyzing {signal_type.upper()} Signal")
        print(f"{'='*80}")

        params = self.best_params[signal_type]
        long_thresh = params['long_threshold']
        short_thresh = params['short_threshold']

        # Create signal instance
        if signal_type == 'momentum':
            signal = SimpleMomentum({
                'lookback': params['lookback'],
                'rank_window': params['rank_window']
            })
        elif signal_type == 'quality':
            signal = SimpleQuality({
                'rank_window': params['rank_window']
            }, data_manager=self.dm)
        elif signal_type == 'insider':
            signal = SimpleInsider({
                'lookback_days': params['lookback_days'],
                'rank_window': params['rank_window']
            }, data_manager=self.dm)

        # Analyze each ticker
        results = []
        all_positions = []

        for ticker in self.tickers:
            prices = self.dm.get_prices(ticker, self.start_date, self.end_date)
            if 'ticker' in prices.columns:
                prices = prices.drop(columns=['ticker'])
            prices['ticker'] = ticker

            # Generate signals
            signals = signal.generate_signals(prices)

            # Generate positions
            positions = pd.Series(0, index=signals.index)
            positions[signals > long_thresh] = 1
            positions[signals < short_thresh] = -1

            # Count trades (position changes)
            position_changes = positions.diff().fillna(0)
            n_trades = (position_changes != 0).sum()

            # Days in each position
            n_long = (positions == 1).sum()
            n_short = (positions == -1).sum()
            n_flat = (positions == 0).sum()

            # Longest consecutive position
            longest_long = self._longest_streak(positions, 1)
            longest_short = self._longest_streak(positions, -1)
            longest_flat = self._longest_streak(positions, 0)

            # Signal statistics
            signal_mean = signals.mean()
            signal_std = signals.std()
            signal_min = signals.min()
            signal_max = signals.max()

            # Days above/below thresholds
            days_above_long = (signals > long_thresh).sum()
            days_below_short = (signals < short_thresh).sum()

            results.append({
                'ticker': ticker,
                'n_days': len(signals),
                'n_trades': n_trades,
                'n_long_days': n_long,
                'n_short_days': n_short,
                'n_flat_days': n_flat,
                'longest_long': longest_long,
                'longest_short': longest_short,
                'longest_flat': longest_flat,
                'signal_mean': signal_mean,
                'signal_std': signal_std,
                'signal_min': signal_min,
                'signal_max': signal_max,
                'days_above_long': days_above_long,
                'days_below_short': days_below_short
            })

            all_positions.append(positions)

        # Aggregate statistics
        df = pd.DataFrame(results)

        print(f"\nOptimized Parameters:")
        for key, value in params.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        print(f"\nTrading Activity Across {len(self.tickers)} Stocks:")
        print(f"  Total trading days: {df['n_days'].sum():,}")
        print(f"  Total trades: {df['n_trades'].sum():,}")
        print(f"  Trades per stock: {df['n_trades'].mean():.1f} ± {df['n_trades'].std():.1f}")
        print(f"  Trades per year per stock: {df['n_trades'].mean() / 4:.1f}")

        print(f"\nPosition Distribution:")
        total_days = df['n_days'].sum()
        print(f"  Long days: {df['n_long_days'].sum():,} ({100*df['n_long_days'].sum()/total_days:.1f}%)")
        print(f"  Short days: {df['n_short_days'].sum():,} ({100*df['n_short_days'].sum()/total_days:.1f}%)")
        print(f"  Flat days: {df['n_flat_days'].sum():,} ({100*df['n_flat_days'].sum()/total_days:.1f}%)")

        print(f"\nLongest Consecutive Positions (days):")
        print(f"  Long: {df['longest_long'].max()}")
        print(f"  Short: {df['longest_short'].max()}")
        print(f"  Flat: {df['longest_flat'].max()}")

        print(f"\nSignal Statistics:")
        print(f"  Mean: {df['signal_mean'].mean():.4f} ± {df['signal_mean'].std():.4f}")
        print(f"  Range: [{df['signal_min'].min():.4f}, {df['signal_max'].max():.4f}]")
        print(f"  Std: {df['signal_std'].mean():.4f}")

        print(f"\nThreshold Crossings:")
        print(f"  Days above long threshold ({long_thresh:.3f}): {df['days_above_long'].sum():,}")
        print(f"  Days below short threshold ({short_thresh:.3f}): {df['days_below_short'].sum():,}")

        # Per-ticker breakdown
        print(f"\nPer-Ticker Breakdown:")
        print(f"{'Ticker':<8} {'Trades':<8} {'Long%':<8} {'Short%':<8} {'Flat%':<8}")
        print("-" * 50)
        for _, row in df.iterrows():
            long_pct = 100 * row['n_long_days'] / row['n_days']
            short_pct = 100 * row['n_short_days'] / row['n_days']
            flat_pct = 100 * row['n_flat_days'] / row['n_days']
            print(f"{row['ticker']:<8} {row['n_trades']:<8} {long_pct:<8.1f} {short_pct:<8.1f} {flat_pct:<8.1f}")

        return {
            'signal_type': signal_type,
            'params': params,
            'df': df,
            'total_trades': df['n_trades'].sum(),
            'trades_per_year': df['n_trades'].sum() / 4,
            'long_pct': 100 * df['n_long_days'].sum() / total_days,
            'short_pct': 100 * df['n_short_days'].sum() / total_days,
            'flat_pct': 100 * df['n_flat_days'].sum() / total_days
        }

    def _longest_streak(self, series: pd.Series, value: int) -> int:
        """Find longest consecutive streak of a value."""
        if len(series) == 0:
            return 0

        is_value = (series == value).astype(int)
        streaks = is_value.groupby((is_value != is_value.shift()).cumsum()).sum()
        return streaks.max() if len(streaks) > 0 else 0

    def generate_recommendations(self, analyses: dict):
        """Generate testing recommendations based on signal characteristics."""
        print(f"\n{'='*80}")
        print("TESTING RECOMMENDATIONS")
        print(f"{'='*80}\n")

        for signal_type, analysis in analyses.items():
            total_trades = analysis['total_trades']
            trades_per_year = analysis['trades_per_year']

            print(f"{signal_type.upper()} Signal:")
            print(f"  Total trades (4 years, 10 stocks): {total_trades}")
            print(f"  Trades per year: {trades_per_year:.1f}")

            # Recommendations based on sample size
            if total_trades < 100:
                print(f"  ⚠️  VERY SPARSE - Only {total_trades} trades")
                print(f"     Recommendation: Extend test period to 10+ years OR expand universe to 50+ stocks")
                print(f"     Statistical significance will be challenging to prove")
                print(f"     Consider this a 'rare event' strategy")
            elif total_trades < 500:
                print(f"  ⚠️  SPARSE - {total_trades} trades")
                print(f"     Recommendation: Extend test period to 7-10 years OR expand universe to 30+ stocks")
                print(f"     Monte Carlo with 1000+ permutations recommended")
            else:
                print(f"  ✅ SUFFICIENT - {total_trades} trades")
                print(f"     Recommendation: Standard Monte Carlo validation (500 permutations)")
                print(f"     Should have good statistical power")

            # Activity level
            long_pct = analysis['long_pct']
            short_pct = analysis['short_pct']
            flat_pct = analysis['flat_pct']

            if long_pct + short_pct < 10:
                print(f"  ⚠️  Very selective (only {long_pct + short_pct:.1f}% of days have positions)")
            elif long_pct + short_pct < 30:
                print(f"  📊 Moderately selective ({long_pct + short_pct:.1f}% of days have positions)")
            else:
                print(f"  📊 Active ({long_pct + short_pct:.1f}% of days have positions)")

            print()

        # Overall recommendations
        print(f"\nOVERALL RECOMMENDATIONS:\n")

        # Find the sparsest signal
        min_trades = min(a['total_trades'] for a in analyses.values())

        if min_trades < 100:
            print("1. **Extend Test Period**")
            print("   - Current: 2020-2023 (4 years)")
            print("   - Recommended: 2015-2024 (10 years) to get 2.5x more trades")
            print()
            print("2. **Expand Universe**")
            print("   - Current: 10 stocks")
            print("   - Recommended: 30-50 stocks (Russell 1000 or S&P 500 subset)")
            print()
            print("3. **Alternative Validation Approach**")
            print("   - Use out-of-sample testing (2024 data)")
            print("   - Cross-validation with longer embargo periods")
            print("   - Accept that rare signals may have low trade counts")
            print()
            print("4. **Signal Ensemble**")
            print("   - Combine multiple signals to increase opportunity count")
            print("   - Test portfolio-level performance rather than individual signals")
        else:
            print("1. **Current Test Period is Adequate**")
            print("   - 4 years with 10 stocks provides sufficient trades")
            print()
            print("2. **Standard Validation Approach**")
            print("   - Monte Carlo validation with 500-1000 permutations")
            print("   - Out-of-sample testing on 2024 data")
            print("   - Portfolio-level backtesting")

    def save_report(self, analyses: dict):
        """Save analysis report to markdown."""
        output_file = Path('results/optimization/signal_characteristics.md')

        with open(output_file, 'w') as f:
            f.write("# Signal Trading Characteristics Analysis\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Test Period:** {self.start_date} to {self.end_date} (4 years)\n")
            f.write(f"**Universe:** {len(self.tickers)} stocks\n\n")
            f.write("---\n\n")

            f.write("## Summary Table\n\n")
            f.write("| Signal | Total Trades | Trades/Year | Long% | Short% | Flat% | Assessment |\n")
            f.write("|--------|--------------|-------------|-------|--------|-------|------------|\n")

            for signal_type, analysis in analyses.items():
                total_trades = analysis['total_trades']
                trades_per_year = analysis['trades_per_year']
                long_pct = analysis['long_pct']
                short_pct = analysis['short_pct']
                flat_pct = analysis['flat_pct']

                if total_trades < 100:
                    assessment = "⚠️ Very Sparse"
                elif total_trades < 500:
                    assessment = "⚠️ Sparse"
                else:
                    assessment = "✅ Sufficient"

                f.write(f"| {signal_type.capitalize()} | {total_trades} | {trades_per_year:.0f} | "
                       f"{long_pct:.1f}% | {short_pct:.1f}% | {flat_pct:.1f}% | {assessment} |\n")

            f.write("\n---\n\n")

            # Detailed sections for each signal
            for signal_type, analysis in analyses.items():
                f.write(f"## {signal_type.capitalize()} Signal\n\n")

                f.write("**Optimized Parameters:**\n")
                for key, value in analysis['params'].items():
                    if isinstance(value, float):
                        f.write(f"- `{key}`: {value:.4f}\n")
                    else:
                        f.write(f"- `{key}`: {value}\n")

                f.write("\n**Trading Activity:**\n")
                f.write(f"- Total trades (4 years, 10 stocks): {analysis['total_trades']}\n")
                f.write(f"- Trades per year: {analysis['trades_per_year']:.1f}\n")
                f.write(f"- Long positions: {analysis['long_pct']:.1f}% of days\n")
                f.write(f"- Short positions: {analysis['short_pct']:.1f}% of days\n")
                f.write(f"- Flat positions: {analysis['flat_pct']:.1f}% of days\n\n")

                f.write("---\n\n")

        print(f"\n✓ Report saved: {output_file}")


def main():
    """Main entry point."""
    analyzer = SignalCharacteristicsAnalyzer()

    print("="*80)
    print("SignalTide v3 - Signal Characteristics Analysis")
    print("="*80)
    print(f"\nTest Period: {analyzer.start_date} to {analyzer.end_date}")
    print(f"Universe: {len(analyzer.tickers)} stocks")

    # Analyze each signal
    analyses = {}
    for signal_type in ['momentum', 'quality', 'insider']:
        analysis = analyzer.analyze_signal(signal_type)
        analyses[signal_type] = analysis

    # Generate recommendations
    analyzer.generate_recommendations(analyses)

    # Save report
    analyzer.save_report(analyses)

    print(f"\n{'='*80}")
    print("✓ Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()
