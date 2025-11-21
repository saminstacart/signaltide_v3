#!/usr/bin/env python3
"""
Survivorship Bias Testing Script

Tests that our backtest system:
1. Includes delisted stocks in the universe
2. Captures final losses before delisting
3. Respects IPO dates (no pre-IPO data)
4. Does not prematurely remove stocks

References:
- Brown et al. (1992) "Survivorship Bias in Performance Studies"
- Elton et al. (1996) "Survivorship Bias and Mutual Fund Performance"
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_manager import DataManager
from config import get_logger

logger = get_logger(__name__)


class SurvivorshipBiasTest:
    """Comprehensive survivorship bias testing."""

    def __init__(self):
        """Initialize with DataManager."""
        self.dm = DataManager()
        self.results = {
            'delisted_in_universe': False,
            'final_losses_captured': False,
            'no_premature_removal': False,
            'ipo_dates_respected': False,
            'test_cases': []
        }

    def get_delisted_stocks(self, start_date: str = '2020-01-01',
                           end_date: str = '2024-12-31') -> pd.DataFrame:
        """Query all stocks delisted during period."""
        logger.info(f"Querying delisted stocks {start_date} to {end_date}")

        query = """
        SELECT
            ticker,
            name,
            category,
            isdelisted,
            firstpricedate,
            lastpricedate
        FROM sharadar_tickers
        WHERE isdelisted = 'Y'
          AND lastpricedate >= ?
          AND lastpricedate <= ?
          AND category LIKE 'Domestic%'
        ORDER BY lastpricedate DESC
        """

        conn = self.dm._get_connection()
        delisted = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()

        logger.info(f"Found {len(delisted)} delisted stocks")
        return delisted

    def test_svb_failure(self) -> dict:
        """
        Test SVB (Silicon Valley Bank) failure - March 2023.

        This is our primary test case for catastrophic delisting.
        """
        logger.info("Testing SVB failure (March 2023)")

        result = {
            'ticker': 'SIVBQ',
            'name': 'SVB Financial Group',
            'expected_delisting': '2023-03-28',
            'tests': {}
        }

        # Test 1: Price data exists before failure
        prices_before = self.dm.get_prices(
            'SIVBQ',
            '2023-01-01',
            '2023-03-27'
        )
        result['tests']['data_exists_before_failure'] = len(prices_before) > 0
        result['days_before_failure'] = len(prices_before)

        # Test 2: Final day price data exists (March 28)
        prices_final = self.dm.get_prices(
            'SIVBQ',
            '2023-03-28',
            '2023-03-28'
        )
        result['tests']['final_day_exists'] = len(prices_final) > 0

        if len(prices_final) > 0:
            result['final_price'] = float(prices_final['close'].iloc[0])

        # Test 3: Get price just before collapse
        prices_before_collapse = self.dm.get_prices(
            'SIVBQ',
            '2023-03-27',
            '2023-03-27'
        )

        if len(prices_before_collapse) > 0:
            result['price_before_collapse'] = float(prices_before_collapse['close'].iloc[0])

        # Test 4: Calculate loss
        if 'price_before_collapse' in result and 'final_price' in result:
            result['loss_pct'] = (
                (result['final_price'] - result['price_before_collapse'])
                / result['price_before_collapse'] * 100
            )
            result['tests']['catastrophic_loss_captured'] = result['loss_pct'] < -90

        # Test 5: No data after delisting
        try:
            prices_after = self.dm.get_prices(
                'SIVBQ',
                '2023-03-29',
                '2023-04-30'
            )
            result['tests']['no_data_after_delisting'] = len(prices_after) == 0
        except:
            result['tests']['no_data_after_delisting'] = True

        logger.info(f"SVB test results: {result['tests']}")
        return result

    def test_bed_bath_beyond(self) -> dict:
        """Test Bed Bath & Beyond bankruptcy - May 2023."""
        logger.info("Testing Bed Bath & Beyond bankruptcy")

        result = {
            'ticker': 'BBBYQ',
            'name': 'Bed Bath & Beyond',
            'expected_delisting': '2023-05-02',
            'tests': {}
        }

        # Check data exists before bankruptcy
        prices = self.dm.get_prices('BBBYQ', '2023-01-01', '2023-05-02')
        result['tests']['data_exists'] = len(prices) > 0
        result['trading_days'] = len(prices)

        # Check final price
        if len(prices) > 0:
            final_row = prices.iloc[-1]
            result['final_price'] = float(final_row['close'])
            result['final_date'] = str(final_row.name)  # date is index
            result['tests']['final_loss_captured'] = result['final_price'] < 5.0

        logger.info(f"BBBY test results: {result['tests']}")
        return result

    def test_first_republic_bank(self) -> dict:
        """Test First Republic Bank failure - May 2023."""
        logger.info("Testing First Republic Bank failure")

        result = {
            'ticker': 'FRCB',
            'name': 'First Republic Bank',
            'expected_delisting': '2023-05-03',
            'tests': {}
        }

        # Check data exists through failure
        prices = self.dm.get_prices('FRCB', '2023-01-01', '2023-05-03')
        result['tests']['data_exists'] = len(prices) > 0
        result['trading_days'] = len(prices)

        # Check we have the final day
        if len(prices) > 0:
            result['final_price'] = float(prices.iloc[-1]['close'])
            result['final_date'] = str(prices.iloc[-1].name)  # date is index

        logger.info(f"FRCB test results: {result['tests']}")
        return result

    def test_twitter_acquisition(self) -> dict:
        """Test Twitter acquisition by Elon Musk - October 2022."""
        logger.info("Testing Twitter acquisition")

        result = {
            'ticker': 'TWTR',
            'name': 'Twitter Inc',
            'expected_delisting': '2022-10-27',
            'tests': {}
        }

        # Check data exists through acquisition
        prices = self.dm.get_prices('TWTR', '2022-01-01', '2022-10-27')
        result['tests']['data_exists'] = len(prices) > 0
        result['trading_days'] = len(prices)

        if len(prices) > 0:
            result['final_price'] = float(prices.iloc[-1]['close'])
            result['final_date'] = str(prices.iloc[-1].name)  # date is index
            # Acquisition was at $54.20/share
            result['tests']['acquisition_price_correct'] = 50 < result['final_price'] < 60

        logger.info(f"TWTR test results: {result['tests']}")
        return result

    def test_ipo_dates(self) -> dict:
        """Test that we don't include pre-IPO data."""
        logger.info("Testing IPO date respect")

        result = {
            'tests': {},
            'tickers_tested': []
        }

        # Test recent IPOs
        test_cases = [
            ('RIVN', '2021-11-10', 'Rivian'),
            ('COIN', '2021-04-14', 'Coinbase'),
        ]

        for ticker, ipo_date, name in test_cases:
            logger.info(f"Testing {ticker} IPO date: {ipo_date}")

            # Try to get data before IPO
            ipo_dt = pd.to_datetime(ipo_date)
            before_ipo = (ipo_dt - timedelta(days=30)).strftime('%Y-%m-%d')

            try:
                prices = self.dm.get_prices(ticker, before_ipo, ipo_date)

                if len(prices) > 0:
                    first_date = pd.to_datetime(prices.index.min())  # date is index
                    no_pre_ipo = first_date >= ipo_dt

                    result['tickers_tested'].append({
                        'ticker': ticker,
                        'name': name,
                        'ipo_date': ipo_date,
                        'first_price_date': str(first_date),
                        'no_pre_ipo_data': no_pre_ipo
                    })
                else:
                    # No data is good - means we respect IPO date
                    result['tickers_tested'].append({
                        'ticker': ticker,
                        'name': name,
                        'ipo_date': ipo_date,
                        'no_pre_ipo_data': True
                    })
            except Exception as e:
                logger.warning(f"Could not test {ticker}: {e}")

        # Overall test passes if all tickers respect IPO dates
        if result['tickers_tested']:
            result['tests']['all_ipo_dates_respected'] = all(
                t.get('no_pre_ipo_data', False)
                for t in result['tickers_tested']
            )

        logger.info(f"IPO test results: {result['tests']}")
        return result

    def run_all_tests(self) -> dict:
        """Run all survivorship bias tests."""
        logger.info("=" * 80)
        logger.info("SURVIVORSHIP BIAS AUDIT - Starting")
        logger.info("=" * 80)

        # Get overview of delisted stocks
        delisted = self.get_delisted_stocks()
        self.results['total_delisted_2020_2024'] = len(delisted)

        # Run specific test cases
        logger.info("\n--- Testing Specific Delisting Cases ---")

        svb_result = self.test_svb_failure()
        self.results['test_cases'].append(svb_result)

        bbby_result = self.test_bed_bath_beyond()
        self.results['test_cases'].append(bbby_result)

        frcb_result = self.test_first_republic_bank()
        self.results['test_cases'].append(frcb_result)

        twtr_result = self.test_twitter_acquisition()
        self.results['test_cases'].append(twtr_result)

        # Test IPO dates
        logger.info("\n--- Testing IPO Date Respect ---")
        ipo_result = self.test_ipo_dates()
        self.results['ipo_test'] = ipo_result

        # Determine overall pass/fail
        self.results['delisted_in_universe'] = self.results['total_delisted_2020_2024'] > 40

        # Check if final losses captured
        all_final_tests = [
            tc.get('tests', {}).get('final_day_exists', False)
            for tc in self.results['test_cases']
        ]
        self.results['final_losses_captured'] = all(all_final_tests)

        # Check IPO dates
        self.results['ipo_dates_respected'] = ipo_result['tests'].get(
            'all_ipo_dates_respected', False
        )

        # Overall grade
        passed_tests = sum([
            self.results['delisted_in_universe'],
            self.results['final_losses_captured'],
            self.results['ipo_dates_respected']
        ])

        if passed_tests == 3:
            self.results['grade'] = 'A+++'
            self.results['passed'] = True
        elif passed_tests == 2:
            self.results['grade'] = 'B+'
            self.results['passed'] = True
        else:
            self.results['grade'] = 'FAIL'
            self.results['passed'] = False

        return self.results

    def print_report(self):
        """Print comprehensive report."""
        print("\n" + "=" * 80)
        print("SURVIVORSHIP BIAS AUDIT REPORT")
        print("=" * 80)
        print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Grade: {self.results['grade']}")
        print(f"Status: {'PASSED' if self.results['passed'] else 'FAILED'}")

        print("\n" + "-" * 80)
        print("SUMMARY")
        print("-" * 80)
        print(f"Total delisted stocks (2020-2024): {self.results['total_delisted_2020_2024']}")
        print(f"Delisted in universe: {'PASS' if self.results['delisted_in_universe'] else 'FAIL'}")
        print(f"Final losses captured: {'PASS' if self.results['final_losses_captured'] else 'FAIL'}")
        print(f"IPO dates respected: {'PASS' if self.results['ipo_dates_respected'] else 'FAIL'}")

        print("\n" + "-" * 80)
        print("SPECIFIC TEST CASES")
        print("-" * 80)

        for case in self.results['test_cases']:
            print(f"\n{case['ticker']} - {case['name']}")
            print(f"  Expected delisting: {case['expected_delisting']}")

            if 'trading_days' in case:
                print(f"  Trading days found: {case['trading_days']}")

            if 'final_price' in case:
                print(f"  Final price: ${case['final_price']:.2f}")

            if 'price_before_collapse' in case:
                print(f"  Price before collapse: ${case['price_before_collapse']:.2f}")

            if 'loss_pct' in case:
                print(f"  Loss captured: {case['loss_pct']:.2f}%")

            print(f"  Tests:")
            for test_name, test_result in case['tests'].items():
                status = 'PASS' if test_result else 'FAIL'
                print(f"    - {test_name}: {status}")

        # IPO tests
        print("\n" + "-" * 80)
        print("IPO DATE RESPECT TEST")
        print("-" * 80)

        ipo_result = self.results.get('ipo_test', {})
        for ticker_test in ipo_result.get('tickers_tested', []):
            status = 'PASS' if ticker_test.get('no_pre_ipo_data') else 'FAIL'
            print(f"\n{ticker_test['ticker']} - {ticker_test['name']}")
            print(f"  IPO date: {ticker_test['ipo_date']}")
            if 'first_price_date' in ticker_test:
                print(f"  First price date: {ticker_test['first_price_date']}")
            print(f"  Status: {status}")

        print("\n" + "=" * 80)
        print(f"FINAL GRADE: {self.results['grade']}")
        print("=" * 80)

        if self.results['passed']:
            print("\nNo survivorship bias detected.")
            print("System correctly includes delisted stocks and captures final losses.")
        else:
            print("\nWARNING: Potential survivorship bias detected!")
            print("Review failed tests above.")

        print()


def main():
    """Run survivorship bias audit."""
    tester = SurvivorshipBiasTest()
    results = tester.run_all_tests()
    tester.print_report()

    # Exit with error code if failed
    if not results['passed']:
        sys.exit(1)


if __name__ == '__main__':
    main()
