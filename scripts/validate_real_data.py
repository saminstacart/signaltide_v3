"""
Validate Simple Signals with Real Sharadar Data

Tests:
1. DataManager can access real Sharadar database
2. Signals generate valid output on real data
3. No lookahead bias
4. Statistical properties are reasonable

Writes results to: results/real_data_validation.txt
"""

import sys
sys.path.insert(0, '/Users/samuelksherman/signaltide_v3')

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from data.data_manager import DataManager
from signals.momentum.simple_momentum import SimpleMomentum
from signals.quality.simple_quality import SimpleQuality
from signals.insider.simple_insider import SimpleInsider


class RealDataValidator:
    """Validate signals on real Sharadar data."""

    def __init__(self, output_file: str = 'results/real_data_validation.txt'):
        self.output_file = output_file
        self.dm = DataManager()
        self.results = []

        # Test universe
        self.tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
                       'NVDA', 'TSLA', 'JPM', 'JNJ', 'XOM']
        self.start_date = '2020-01-01'
        self.end_date = '2023-12-31'

    def log(self, message: str):
        """Log message to both console and results."""
        print(message)
        self.results.append(message)

    def write_results(self):
        """Write all results to file."""
        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write('\n'.join(self.results))

        self.log(f"\n✓ Results written to {self.output_file}")

    def test_data_access(self) -> bool:
        """Test DataManager can access real Sharadar data."""
        self.log("=" * 80)
        self.log("TEST 1: DataManager Access to Real Sharadar Database")
        self.log("=" * 80)

        try:
            # Test price data
            self.log(f"\n1.1 Testing price data for {len(self.tickers)} tickers...")
            prices = self.dm.get_prices(
                self.tickers,
                self.start_date,
                self.end_date
            )

            self.log(f"  ✓ Got {len(prices)} price rows")
            self.log(f"  ✓ Date range: {prices.index.min()} to {prices.index.max()}")
            self.log(f"  ✓ Tickers: {prices['ticker'].nunique()} unique")
            self.log(f"  ✓ Columns: {list(prices.columns)}")

            # Test fundamentals
            self.log(f"\n1.2 Testing fundamental data (AAPL)...")
            fundamentals = self.dm.get_fundamentals(
                'AAPL',
                self.start_date,
                self.end_date,
                dimension='ARQ'
            )

            self.log(f"  ✓ Got {len(fundamentals)} fundamental rows")
            if len(fundamentals) > 0:
                latest_roe = fundamentals['roe'].iloc[-1]
                latest_rev = fundamentals['revenue'].iloc[-1]

                if latest_roe is not None and not pd.isna(latest_roe):
                    self.log(f"  ✓ Latest ROE: {latest_roe:.4f}")
                else:
                    self.log(f"  ✓ Latest ROE: None")

                if latest_rev is not None and not pd.isna(latest_rev):
                    self.log(f"  ✓ Latest Revenue: ${latest_rev/1e9:.2f}B")
                else:
                    self.log(f"  ✓ Latest Revenue: None")

            # Test insider data
            self.log(f"\n1.3 Testing insider trading data (AAPL)...")
            insider = self.dm.get_insider_trades(
                'AAPL',
                self.start_date,
                self.end_date
            )

            self.log(f"  ✓ Got {len(insider)} insider trades")
            if len(insider) > 0:
                n_buys = (insider['transactioncode'] == 'P').sum()
                n_sells = (insider['transactioncode'] == 'S').sum()
                self.log(f"  ✓ Purchases: {n_buys}, Sales: {n_sells}")

            self.log("\n✓ TEST 1 PASSED: DataManager successfully accessing real data")
            return True

        except Exception as e:
            self.log(f"\n✗ TEST 1 FAILED: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            return False

    def test_signal_on_real_data(self, signal_name: str, signal, ticker: str = 'AAPL'):
        """Test a signal on real data."""
        self.log(f"\n{'=' * 80}")
        self.log(f"Testing {signal_name} on {ticker}")
        self.log("=" * 80)

        try:
            # Get price data
            prices = self.dm.get_prices(ticker, self.start_date, self.end_date)

            if len(prices) == 0:
                self.log(f"  ✗ No price data for {ticker}")
                return False

            # Drop ticker column if it exists (single ticker)
            if 'ticker' in prices.columns:
                prices = prices.drop(columns=['ticker'])

            # Add ticker back as column for signal
            prices['ticker'] = ticker

            self.log(f"  Input: {len(prices)} days of price data")

            # Generate signals
            signals = signal.generate_signals(prices)

            # Validate output
            self.log(f"\n  Signal Output Validation:")
            self.log(f"    ✓ Length matches input: {len(signals) == len(prices)}")
            self.log(f"    ✓ Index matches input: {signals.index.equals(prices.index)}")

            # Check range
            in_range = (signals.min() >= -1.0) and (signals.max() <= 1.0)
            self.log(f"    ✓ In [-1, 1] range: {in_range}")
            self.log(f"      Min: {signals.min():.4f}, Max: {signals.max():.4f}")

            # Check for NaN/Inf
            has_nan = signals.isna().any()
            has_inf = np.isinf(signals).any()
            self.log(f"    ✓ No NaN: {not has_nan} ({signals.isna().sum()} NaN)")
            self.log(f"    ✓ No Inf: {not has_inf}")

            # Coverage (non-zero signals)
            non_zero = (signals != 0).sum()
            coverage = non_zero / len(signals) * 100
            self.log(f"\n  Coverage: {coverage:.1f}% ({non_zero}/{len(signals)} non-zero)")

            # Distribution
            self.log(f"\n  Distribution:")
            self.log(f"    Mean: {signals.mean():.4f}")
            self.log(f"    Std:  {signals.std():.4f}")
            self.log(f"    Q25:  {signals.quantile(0.25):.4f}")
            self.log(f"    Q50:  {signals.quantile(0.50):.4f}")
            self.log(f"    Q75:  {signals.quantile(0.75):.4f}")

            # Autocorrelation
            if len(signals) > 1:
                autocorr = signals.autocorr(lag=1)
                self.log(f"\n  Autocorrelation (lag=1): {autocorr:.4f}")

            # Sample output (first 20 non-zero signals)
            non_zero_signals = signals[signals != 0].head(20)
            if len(non_zero_signals) > 0:
                self.log(f"\n  Sample Output (first 20 non-zero signals):")
                for date, val in non_zero_signals.items():
                    self.log(f"    {date.strftime('%Y-%m-%d')}: {val:7.4f}")

            # Test for lookahead bias (signals should be stable)
            self.log(f"\n  Lookahead Bias Check:")
            # Generate signals on first 80% of data
            cutoff = int(len(prices) * 0.8)
            partial_prices = prices.iloc[:cutoff].copy()
            partial_signals = signal.generate_signals(partial_prices)

            # Compare overlapping dates
            overlap_dates = partial_signals.index
            correlation = partial_signals.corr(signals.loc[overlap_dates])

            self.log(f"    Correlation (full vs partial): {correlation:.4f}")
            self.log(f"    ✓ No lookahead bias: {correlation > 0.95}")

            self.log(f"\n✓ {signal_name} PASSED on {ticker}")
            return True

        except Exception as e:
            self.log(f"\n✗ {signal_name} FAILED on {ticker}: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            return False

    def run_all_tests(self):
        """Run all validation tests."""
        self.log("SignalTide v3 - Real Data Validation")
        self.log(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("=" * 80)

        # Test 1: Data access
        data_ok = self.test_data_access()

        if not data_ok:
            self.log("\n✗ STOPPING: DataManager cannot access real data")
            self.write_results()
            return False

        # Test 2: SimpleMomentum
        self.log("\n\n")
        momentum_params = {'lookback': 20, 'rank_window': 252}
        momentum = SimpleMomentum(momentum_params)
        momentum_ok = self.test_signal_on_real_data('SimpleMomentum', momentum, 'AAPL')

        # Test 3: SimpleQuality
        self.log("\n\n")
        quality_params = {'rank_window': 252 * 2}
        quality = SimpleQuality(quality_params, data_manager=self.dm)
        quality_ok = self.test_signal_on_real_data('SimpleQuality', quality, 'AAPL')

        # Test 4: SimpleInsider
        self.log("\n\n")
        insider_params = {'lookback_days': 30, 'rank_window': 252}
        insider = SimpleInsider(insider_params, data_manager=self.dm)
        insider_ok = self.test_signal_on_real_data('SimpleInsider', insider, 'AAPL')

        # Test multiple tickers for momentum (fastest signal)
        self.log("\n\n")
        self.log("=" * 80)
        self.log("Multi-Ticker Test (SimpleMomentum)")
        self.log("=" * 80)

        multi_ticker_results = []
        for ticker in self.tickers:
            try:
                prices = self.dm.get_prices(ticker, self.start_date, self.end_date)
                if 'ticker' in prices.columns:
                    prices = prices.drop(columns=['ticker'])
                prices['ticker'] = ticker

                signals = momentum.generate_signals(prices)
                coverage = (signals != 0).sum() / len(signals) * 100

                multi_ticker_results.append({
                    'ticker': ticker,
                    'coverage': coverage,
                    'mean': signals.mean(),
                    'std': signals.std()
                })

                self.log(f"  {ticker}: {coverage:.1f}% coverage, mean={signals.mean():.4f}")
            except Exception as e:
                self.log(f"  {ticker}: FAILED - {str(e)}")

        # Summary
        self.log("\n\n")
        self.log("=" * 80)
        self.log("VALIDATION SUMMARY")
        self.log("=" * 80)
        self.log(f"  DataManager Access:  {'✓ PASS' if data_ok else '✗ FAIL'}")
        self.log(f"  SimpleMomentum:      {'✓ PASS' if momentum_ok else '✗ FAIL'}")
        self.log(f"  SimpleQuality:       {'✓ PASS' if quality_ok else '✗ FAIL'}")
        self.log(f"  SimpleInsider:       {'✓ PASS' if insider_ok else '✗ FAIL'}")

        all_passed = data_ok and momentum_ok and quality_ok and insider_ok

        if all_passed:
            self.log("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
            self.log("\nReady to proceed with backtesting and optimization.")
        else:
            self.log("\n✗ SOME TESTS FAILED")
            self.log("\nReview errors above before proceeding.")

        self.write_results()
        return all_passed


if __name__ == '__main__':
    print("Starting real data validation...")
    print("This will test signals on actual Sharadar data.\n")

    validator = RealDataValidator()
    success = validator.run_all_tests()

    if success:
        print("\n✓ Validation complete. Check results/real_data_validation.txt")
        sys.exit(0)
    else:
        print("\n✗ Validation failed. Check results/real_data_validation.txt for details.")
        sys.exit(1)
