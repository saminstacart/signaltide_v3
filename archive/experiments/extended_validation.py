"""
Extended Validation with 10 Years and 50 Stocks

Run optimized signals on extended dataset for high-confidence validation.

Usage:
    python scripts/extended_validation.py
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


class ExtendedValidator:
    """Validate signals on extended dataset (10 years, 50 stocks)."""

    def __init__(self):
        self.dm = DataManager()

        # Extended test period: 10 years
        self.start_date = '2015-01-01'
        self.end_date = '2024-12-31'

        # Expanded universe: 50 stocks across sectors
        # Diversified selection from S&P 500
        self.universe = {
            # Technology (15)
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AVGO',
            'ORCL', 'CSCO', 'ADBE', 'CRM', 'INTC', 'AMD', 'QCOM',

            # Healthcare (8)
            'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'MRK', 'ABT', 'LLY',

            # Financials (8)
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW',

            # Consumer Discretionary (5)
            'HD', 'NKE', 'MCD', 'SBUX', 'TGT',

            # Consumer Staples (4)
            'PG', 'KO', 'PEP', 'WMT',

            # Industrials (4)
            'BA', 'CAT', 'GE', 'UPS',

            # Energy (3)
            'XOM', 'CVX', 'COP',

            # Utilities & Real Estate (3)
            'NEE', 'DUK', 'AMT'
        }

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

        self.output_dir = Path('results/extended_validation')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def backtest_signal(self, signal, prices, long_threshold, short_threshold):
        """Simple backtest for a single stock."""
        try:
            # Generate signals
            signals = signal.generate_signals(prices)

            # Calculate returns
            price_returns = prices['close'].pct_change()

            # Generate positions
            positions = pd.Series(0.0, index=signals.index)
            positions[signals > long_threshold] = 1.0
            positions[signals < short_threshold] = -1.0

            # Calculate strategy returns (shift positions to avoid lookahead)
            strategy_returns = positions.shift(1) * price_returns
            strategy_returns = strategy_returns.dropna()

            if len(strategy_returns) == 0 or strategy_returns.std() == 0:
                return None

            # Calculate metrics
            total_return = (1 + strategy_returns).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
            annual_vol = strategy_returns.std() * np.sqrt(252)
            sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0

            # Max drawdown
            cum_returns = (1 + strategy_returns).cumprod()
            cum_max = cum_returns.cummax()
            drawdown = (cum_returns - cum_max) / cum_max
            max_dd = drawdown.min()

            # Number of trades
            n_trades = (positions.diff() != 0).sum()

            # Win rate
            winning_days = (strategy_returns > 0).sum()
            total_trading_days = (strategy_returns != 0).sum()
            win_rate = winning_days / total_trading_days if total_trading_days > 0 else 0

            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe': sharpe,
                'volatility': annual_vol,
                'max_dd': max_dd,
                'n_trades': n_trades,
                'win_rate': win_rate,
                'n_days': len(strategy_returns)
            }

        except Exception as e:
            print(f"  Error: {str(e)}")
            return None

    def validate_signal(self, signal_type: str):
        """Validate a signal on extended dataset."""
        print(f"\n{'='*80}")
        print(f"Validating {signal_type.upper()} Signal")
        print(f"{'='*80}")

        params = self.best_params[signal_type]
        long_thresh = params['long_threshold']
        short_thresh = params['short_threshold']

        print(f"\nOptimized Parameters:")
        for key, value in params.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        print(f"\nTest Period: {self.start_date} to {self.end_date} (10 years)")
        print(f"Universe: {len(self.universe)} stocks\n")

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

        # Test on each ticker
        results = []
        successful = 0
        failed = 0

        for i, ticker in enumerate(sorted(self.universe), 1):
            print(f"[{i:2d}/{len(self.universe)}] Testing {ticker}...", end=' ')

            try:
                prices = self.dm.get_prices(ticker, self.start_date, self.end_date)
                if len(prices) < 252:  # Need at least 1 year of data
                    print("SKIP (insufficient data)")
                    failed += 1
                    continue

                if 'ticker' in prices.columns:
                    prices = prices.drop(columns=['ticker'])
                prices['ticker'] = ticker

                metrics = self.backtest_signal(signal, prices, long_thresh, short_thresh)

                if metrics is None:
                    print("SKIP (no valid returns)")
                    failed += 1
                    continue

                results.append({
                    'ticker': ticker,
                    **metrics
                })

                print(f"Sharpe: {metrics['sharpe']:6.3f}, Trades: {metrics['n_trades']:4d}")
                successful += 1

            except Exception as e:
                print(f"FAILED ({str(e)[:50]})")
                failed += 1

        print(f"\n{'='*80}")
        print(f"Results: {successful} successful, {failed} failed")
        print(f"{'='*80}")

        if len(results) == 0:
            print("❌ No valid results")
            return None

        # Aggregate statistics
        df = pd.DataFrame(results)

        print(f"\n{'='*80}")
        print(f"AGGREGATE STATISTICS - {signal_type.upper()}")
        print(f"{'='*80}\n")

        print(f"Returns:")
        print(f"  Mean Annual Return: {df['annual_return'].mean()*100:6.2f}%")
        print(f"  Median Annual Return: {df['annual_return'].median()*100:6.2f}%")
        print(f"  Std Annual Return: {df['annual_return'].std()*100:6.2f}%")

        print(f"\nSharpe Ratio:")
        print(f"  Mean: {df['sharpe'].mean():6.3f}")
        print(f"  Median: {df['sharpe'].median():6.3f}")
        print(f"  Std: {df['sharpe'].std():6.3f}")
        print(f"  Min: {df['sharpe'].min():6.3f}")
        print(f"  Max: {df['sharpe'].max():6.3f}")

        print(f"\nRisk Metrics:")
        print(f"  Mean Volatility: {df['volatility'].mean()*100:6.2f}%")
        print(f"  Mean Max Drawdown: {df['max_dd'].mean()*100:6.2f}%")
        print(f"  Worst Max Drawdown: {df['max_dd'].min()*100:6.2f}%")

        print(f"\nTrading Activity:")
        print(f"  Total Trades: {df['n_trades'].sum():,}")
        print(f"  Trades per Stock: {df['n_trades'].mean():.1f} ± {df['n_trades'].std():.1f}")
        print(f"  Mean Win Rate: {df['win_rate'].mean()*100:.1f}%")

        # Win/Loss analysis
        positive_sharpe = (df['sharpe'] > 0).sum()
        print(f"\nWin/Loss Analysis:")
        print(f"  Positive Sharpe: {positive_sharpe}/{len(df)} ({100*positive_sharpe/len(df):.1f}%)")
        print(f"  Negative Sharpe: {len(df)-positive_sharpe}/{len(df)} ({100*(len(df)-positive_sharpe)/len(df):.1f}%)")

        # Top and bottom performers
        print(f"\nTop 5 Performers (by Sharpe):")
        top5 = df.nlargest(5, 'sharpe')
        for _, row in top5.iterrows():
            print(f"  {row['ticker']:5s}: Sharpe={row['sharpe']:6.3f}, Return={row['annual_return']*100:6.2f}%, Trades={int(row['n_trades'])}")

        print(f"\nBottom 5 Performers (by Sharpe):")
        bottom5 = df.nsmallest(5, 'sharpe')
        for _, row in bottom5.iterrows():
            print(f"  {row['ticker']:5s}: Sharpe={row['sharpe']:6.3f}, Return={row['annual_return']*100:6.2f}%, Trades={int(row['n_trades'])}")

        # Save detailed results
        output_file = self.output_dir / f'{signal_type}_extended_results.csv'
        df.to_csv(output_file, index=False)
        print(f"\n✓ Detailed results saved: {output_file}")

        return {
            'signal_type': signal_type,
            'df': df,
            'mean_sharpe': df['sharpe'].mean(),
            'median_sharpe': df['sharpe'].median(),
            'total_trades': df['n_trades'].sum(),
            'positive_sharpe_pct': 100 * positive_sharpe / len(df)
        }

    def generate_report(self, analyses: dict):
        """Generate comprehensive markdown report."""
        output_file = self.output_dir / 'extended_validation_report.md'

        with open(output_file, 'w') as f:
            f.write("# Extended Validation Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Test Period:** {self.start_date} to {self.end_date} (10 years)\n")
            f.write(f"**Universe:** {len(self.universe)} stocks\n\n")
            f.write("---\n\n")

            f.write("## Summary Table\n\n")
            f.write("| Signal | Mean Sharpe | Median Sharpe | Total Trades | Win Rate | Status |\n")
            f.write("|--------|-------------|---------------|--------------|----------|--------|\n")

            for signal_type, analysis in analyses.items():
                if analysis is None:
                    continue

                mean_sharpe = analysis['mean_sharpe']
                median_sharpe = analysis['median_sharpe']
                total_trades = analysis['total_trades']
                win_rate = analysis['positive_sharpe_pct']

                status = "✅ High Confidence" if total_trades > 500 else "⚠️ Low Confidence"

                f.write(f"| {signal_type.capitalize()} | {mean_sharpe:.3f} | {median_sharpe:.3f} | "
                       f"{total_trades:,} | {win_rate:.1f}% | {status} |\n")

            f.write("\n---\n\n")

            # Detailed sections
            for signal_type, analysis in analyses.items():
                if analysis is None:
                    continue

                df = analysis['df']

                f.write(f"## {signal_type.capitalize()} Signal\n\n")

                f.write("**Performance Metrics:**\n")
                f.write(f"- Mean Sharpe: {df['sharpe'].mean():.3f}\n")
                f.write(f"- Median Sharpe: {df['sharpe'].median():.3f}\n")
                f.write(f"- Mean Annual Return: {df['annual_return'].mean()*100:.2f}%\n")
                f.write(f"- Mean Volatility: {df['volatility'].mean()*100:.2f}%\n")
                f.write(f"- Mean Max Drawdown: {df['max_dd'].mean()*100:.2f}%\n\n")

                f.write("**Trading Statistics:**\n")
                f.write(f"- Total Trades: {df['n_trades'].sum():,}\n")
                f.write(f"- Trades per Stock: {df['n_trades'].mean():.1f}\n")
                f.write(f"- Win Rate: {df['win_rate'].mean()*100:.1f}%\n\n")

                f.write("**Distribution:**\n")
                positive = (df['sharpe'] > 0).sum()
                f.write(f"- Positive Sharpe: {positive}/{len(df)} ({100*positive/len(df):.1f}%)\n")
                f.write(f"- Negative Sharpe: {len(df)-positive}/{len(df)} ({100*(len(df)-positive)/len(df):.1f}%)\n\n")

                f.write("---\n\n")

        print(f"\n✓ Report saved: {output_file}")


def main():
    """Main entry point."""
    validator = ExtendedValidator()

    print("="*80)
    print("SignalTide v3 - Extended Validation")
    print("="*80)
    print(f"\nTest Period: {validator.start_date} to {validator.end_date} (10 years)")
    print(f"Universe: {len(validator.universe)} stocks")
    print(f"\nThis will test optimized parameters on extended dataset for high confidence.")

    # Validate each signal
    analyses = {}
    for signal_type in ['momentum', 'quality', 'insider']:
        analysis = validator.validate_signal(signal_type)
        if analysis is not None:
            analyses[signal_type] = analysis

    # Generate report
    if analyses:
        validator.generate_report(analyses)

        print(f"\n{'='*80}")
        print("✓ Extended validation complete!")
        print("="*80)
        print(f"\nResults saved to: {validator.output_dir}/")
    else:
        print("\n❌ No valid results to report")


if __name__ == '__main__':
    main()
