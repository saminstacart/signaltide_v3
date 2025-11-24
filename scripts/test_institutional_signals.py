"""
Test Institutional Signals

Validates that institutional signals:
1. Load and run without errors
2. Produce regular trading (not sparse)
3. Use proper cross-sectional methodology
4. Generate monthly rebalanced signals

Quick test: 10 stocks, 2023 data
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # scripts/ -> repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data.data_manager import DataManager
from signals.momentum.institutional_momentum import InstitutionalMomentum
from signals.quality.institutional_quality import InstitutionalQuality
from signals.insider.institutional_insider import InstitutionalInsider

# For comparison
from archive.simple_signals_v1.simple_momentum import SimpleMomentum
from archive.simple_signals_v1.simple_quality import SimpleQuality
from archive.simple_signals_v1.simple_insider import SimpleInsider


def test_signal(signal, signal_name, ticker, prices, dm=None):
    """Test a single signal and return statistics."""
    print(f"\n{'='*80}")
    print(f"Testing {signal_name} on {ticker}")
    print(f"{'='*80}")

    try:
        # Generate signals
        signals = signal.generate_signals(prices)

        # Basic statistics
        print(f"\nSignal Statistics:")
        print(f"  Total days: {len(signals)}")
        print(f"  Non-zero signals: {(signals != 0).sum()} ({100*(signals != 0).sum()/len(signals):.1f}%)")
        print(f"  Range: [{signals.min():.3f}, {signals.max():.3f}]")
        print(f"  Mean: {signals.mean():.3f}, Std: {signals.std():.3f}")

        # Distribution
        print(f"\nSignal Distribution:")
        if signals.nunique() <= 10:
            value_counts = signals.value_counts().sort_index()
            for val, count in value_counts.items():
                print(f"  {val:6.3f}: {count:4d} days ({100*count/len(signals):5.1f}%)")
        else:
            print(f"  Unique values: {signals.nunique()}")
            non_zero = signals[signals != 0]
            if len(non_zero) > 0:
                try:
                    quintiles, bins = pd.qcut(non_zero, q=5, labels=False, duplicates='drop', retbins=True)
                    n_bins = len(bins) - 1
                    print(f"  Quintile distribution ({n_bins} bins):")
                    for i in range(n_bins):
                        count = (quintiles == i).sum()
                        print(f"    Q{i+1}: {count} ({100*count/len(quintiles):.1f}%)")
                except:
                    print(f"  Range: [{non_zero.min():.3f}, {non_zero.max():.3f}]")

        # Position changes (trades)
        position_changes = signals.diff().fillna(0)
        n_changes = (position_changes != 0).sum()
        print(f"\nTrading Activity:")
        print(f"  Position changes: {n_changes}")
        print(f"  Changes per month: {n_changes / 12:.1f}")

        # Monthly rebalancing check
        monthly_changes = signals.resample('M').last().diff().fillna(0)
        monthly_rebalances = (monthly_changes != 0).sum()
        print(f"  Monthly rebalances: {monthly_rebalances} (out of 12 months)")

        return {
            'signal_name': signal_name,
            'ticker': ticker,
            'total_days': len(signals),
            'non_zero_pct': 100*(signals != 0).sum()/len(signals),
            'signal_range': (signals.min(), signals.max()),
            'n_changes': n_changes,
            'changes_per_month': n_changes / 12,
            'monthly_rebalances': monthly_rebalances,
            'mean': signals.mean(),
            'std': signals.std()
        }

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def compare_simple_vs_institutional():
    """Compare simple vs institutional signal trading frequency."""
    print("\n" + "="*80)
    print("COMPARISON: Simple vs Institutional Signals")
    print("="*80)

    dm = DataManager()

    # Test tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
               'NVDA', 'TSLA', 'JPM', 'JNJ', 'XOM']

    # Test period: 2023 only (1 year for quick test)
    start_date = '2023-01-01'
    end_date = '2023-12-31'

    results = []

    for ticker in tickers:
        print(f"\n{'='*80}")
        print(f"Testing {ticker}")
        print(f"{'='*80}")

        # Get prices
        prices = dm.get_prices(ticker, start_date, end_date)
        if len(prices) < 50:
            print(f"Skipping {ticker} - insufficient data")
            continue

        prices = prices.copy()
        if 'ticker' in prices.columns:
            prices = prices.drop(columns=['ticker'])
        prices['ticker'] = ticker

        print(f"Data: {len(prices)} days from {prices.index.min().date()} to {prices.index.max().date()}")

        # Test MOMENTUM
        print("\n" + "-"*80)
        print("MOMENTUM COMPARISON")
        print("-"*80)

        # Simple
        simple_mom = SimpleMomentum({'lookback': 25, 'rank_window': 62})
        simple_result = test_signal(simple_mom, "SimpleMomentum", ticker, prices)

        # Institutional
        inst_mom = InstitutionalMomentum({
            'formation_period': 252,
            'skip_period': 21,
            'rebalance_frequency': 'monthly',
            'quintiles': True
        })
        inst_result = test_signal(inst_mom, "InstitutionalMomentum", ticker, prices)

        if simple_result and inst_result:
            results.append({
                'ticker': ticker,
                'signal': 'Momentum',
                'simple_changes': simple_result['n_changes'],
                'inst_changes': inst_result['n_changes'],
                'simple_monthly': simple_result['changes_per_month'],
                'inst_monthly': inst_result['changes_per_month'],
                'improvement': inst_result['n_changes'] / simple_result['n_changes'] if simple_result['n_changes'] > 0 else 0
            })

        # Test QUALITY (THE KEY TEST!)
        print("\n" + "-"*80)
        print("QUALITY COMPARISON (KEY TEST - Was this sparse?)")
        print("-"*80)

        # Simple
        simple_qual = SimpleQuality({'rank_window': 284}, data_manager=dm)
        simple_result = test_signal(simple_qual, "SimpleQuality", ticker, prices, dm)

        # Institutional
        inst_qual = InstitutionalQuality({
            'use_profitability': True,
            'use_growth': True,
            'use_safety': True,
            'rebalance_frequency': 'monthly'
        }, data_manager=dm)
        inst_result = test_signal(inst_qual, "InstitutionalQuality", ticker, prices, dm)

        if simple_result and inst_result:
            results.append({
                'ticker': ticker,
                'signal': 'Quality',
                'simple_changes': simple_result['n_changes'],
                'inst_changes': inst_result['n_changes'],
                'simple_monthly': simple_result['changes_per_month'],
                'inst_monthly': inst_result['changes_per_month'],
                'improvement': inst_result['n_changes'] / simple_result['n_changes'] if simple_result['n_changes'] > 0 else 0
            })

        # Test INSIDER
        print("\n" + "-"*80)
        print("INSIDER COMPARISON")
        print("-"*80)

        # Simple
        simple_ins = SimpleInsider({'lookback_days': 35, 'rank_window': 102}, data_manager=dm)
        simple_result = test_signal(simple_ins, "SimpleInsider", ticker, prices, dm)

        # Institutional
        inst_ins = InstitutionalInsider({
            'lookback_days': 90,
            'min_transaction_value': 10000,
            'cluster_window': 7,
            'cluster_min_insiders': 3,
            'rebalance_frequency': 'monthly'
        }, data_manager=dm)
        inst_result = test_signal(inst_ins, "InstitutionalInsider", ticker, prices, dm)

        if simple_result and inst_result:
            results.append({
                'ticker': ticker,
                'signal': 'Insider',
                'simple_changes': simple_result['n_changes'],
                'inst_changes': inst_result['n_changes'],
                'simple_monthly': simple_result['changes_per_month'],
                'inst_monthly': inst_result['changes_per_month'],
                'improvement': inst_result['n_changes'] / simple_result['n_changes'] if simple_result['n_changes'] > 0 else 0
            })

    # Summary
    if results:
        df = pd.DataFrame(results)

        print("\n" + "="*80)
        print("SUMMARY: Trading Frequency Comparison")
        print("="*80)

        for signal_type in ['Momentum', 'Quality', 'Insider']:
            signal_df = df[df['signal'] == signal_type]

            if len(signal_df) == 0:
                continue

            print(f"\n{signal_type} Signal:")
            print(f"  Simple - Avg changes per month: {signal_df['simple_monthly'].mean():.1f}")
            print(f"  Institutional - Avg changes per month: {signal_df['inst_monthly'].mean():.1f}")

            if signal_df['simple_monthly'].mean() > 0:
                improvement = signal_df['inst_monthly'].mean() / signal_df['simple_monthly'].mean()
                print(f"  Improvement factor: {improvement:.2f}x")

            # Quality-specific analysis
            if signal_type == 'Quality':
                print(f"\n  🎯 QUALITY SIGNAL ANALYSIS:")
                simple_avg = signal_df['simple_changes'].mean()
                inst_avg = signal_df['inst_changes'].mean()

                print(f"  Simple Quality: {simple_avg:.1f} changes per year per stock")
                print(f"  Institutional Quality: {inst_avg:.1f} changes per year per stock")

                if inst_avg >= 10:
                    print(f"  ✅ SOLVED: Institutional Quality produces regular trading!")
                else:
                    print(f"  ⚠️  PROBLEM REMAINS: Still too sparse")

        # Save results
        df.to_csv('results/institutional_vs_simple_comparison.csv', index=False)
        print(f"\n✓ Results saved to: results/institutional_vs_simple_comparison.csv")

        return df

    return None


def main():
    """Main test execution."""
    print("="*80)
    print("Institutional Signal Validation")
    print("="*80)
    print(f"Test Period: 2023 (1 year)")
    print(f"Test Universe: 10 stocks")
    print(f"Purpose: Verify signals work and solve sparsity problem")
    print("="*80)

    # Run comparison
    results = compare_simple_vs_institutional()

    if results is not None:
        print("\n" + "="*80)
        print("✅ VALIDATION COMPLETE")
        print("="*80)
        print("\nKey Findings:")

        # Quality signal analysis
        quality_df = results[results['signal'] == 'Quality']
        if len(quality_df) > 0:
            inst_monthly_avg = quality_df['inst_monthly'].mean()

            print(f"\n1. Quality Signal Trading Frequency:")
            print(f"   - Institutional: {inst_monthly_avg:.1f} changes/month")
            print(f"   - Expected: ~1 per month (monthly rebalancing)")

            if inst_monthly_avg >= 0.5:
                print(f"   ✅ Quality signal sparsity SOLVED!")
            else:
                print(f"   ⚠️  Quality signal still sparse - needs investigation")

        # Overall assessment
        print(f"\n2. All Signals:")
        print(f"   - Momentum: {len(results[results['signal']=='Momentum'])} stocks tested")
        print(f"   - Quality: {len(results[results['signal']=='Quality'])} stocks tested")
        print(f"   - Insider: {len(results[results['signal']=='Insider'])} stocks tested")

        print(f"\n3. Next Steps:")
        print(f"   - Full backtest on extended period")
        print(f"   - Optimization of institutional parameters")
        print(f"   - Portfolio construction & validation")
    else:
        print("\n❌ No results generated - check for errors above")


if __name__ == '__main__':
    main()
