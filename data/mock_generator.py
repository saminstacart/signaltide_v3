"""
Mock Data Generator for Testing

Generates realistic fake data for testing signals without requiring
actual Sharadar database. Useful for rapid prototyping and unit tests.

Generates:
- OHLCV price data (5 years, 50 stocks)
- Fundamental metrics (quarterly)
- Insider trading transactions

Data is deterministic (uses fixed seed) for reproducible tests.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional


class MockDataGenerator:
    """Generate mock financial data for testing."""

    def __init__(self, seed: int = 42):
        """
        Initialize mock data generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Default universe
        self.tickers = [f'TICK{i:02d}' for i in range(1, 51)]  # TICK01 to TICK50

    def generate_price_data(self,
                           ticker: str,
                           start_date: datetime,
                           end_date: datetime) -> pd.DataFrame:
        """
        Generate mock OHLCV price data.

        Args:
            ticker: Stock ticker
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV columns, datetime index
        """
        # Generate daily dates
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(dates)

        # Generate realistic price movements (geometric Brownian motion)
        ticker_seed = hash(ticker) % 10000
        local_rng = np.random.RandomState(self.seed + ticker_seed)

        # Starting price
        initial_price = 50 + local_rng.rand() * 100

        # Daily returns (mean ~0.05%/day = ~13%/year)
        daily_returns = local_rng.normal(0.0005, 0.02, n_days)

        # Price series
        price_multipliers = np.exp(np.cumsum(daily_returns))
        close = initial_price * price_multipliers

        # Generate OHLC from close
        daily_range = local_rng.uniform(0.005, 0.03, n_days)  # 0.5-3% daily range

        high = close * (1 + daily_range * local_rng.uniform(0.3, 1.0, n_days))
        low = close * (1 - daily_range * local_rng.uniform(0.3, 1.0, n_days))
        open_price = low + (high - low) * local_rng.uniform(0.2, 0.8, n_days)

        # Volume (with realistic variation)
        avg_volume = 1_000_000 + local_rng.randint(0, 5_000_000)
        volume = avg_volume * local_rng.uniform(0.5, 2.0, n_days)

        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume.astype(int),
            'ticker': ticker
        }, index=dates)

        return df

    def generate_fundamentals(self,
                             ticker: str,
                             start_date: datetime,
                             end_date: datetime) -> pd.DataFrame:
        """
        Generate mock fundamental data (quarterly).

        Args:
            ticker: Stock ticker
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with fundamental metrics, datetime index
        """
        # Generate quarterly dates
        dates = pd.date_range(start=start_date, end=end_date, freq='QE')  # 'QE' = quarter end (replaces deprecated 'Q')
        n_quarters = len(dates)

        ticker_seed = hash(ticker) % 10000
        local_rng = np.random.RandomState(self.seed + ticker_seed)

        # Revenue (growing over time with noise)
        base_revenue = 1e9 * (1 + local_rng.rand())
        growth_rate = local_rng.uniform(0.03, 0.15)  # 3-15% quarterly growth
        revenue = base_revenue * np.power(1 + growth_rate, np.arange(n_quarters))
        revenue *= (1 + local_rng.normal(0, 0.1, n_quarters))  # Add noise

        # Profit margins
        gross_margin = local_rng.uniform(0.30, 0.60)
        operating_margin = local_rng.uniform(0.10, 0.30)
        net_margin = local_rng.uniform(0.05, 0.20)

        gross_profit = revenue * gross_margin
        operating_income = revenue * operating_margin
        net_income = revenue * net_margin

        # Balance sheet
        total_assets = revenue * local_rng.uniform(2.0, 4.0)
        equity = total_assets * local_rng.uniform(0.4, 0.7)
        debt = total_assets - equity

        # Cash flow
        operating_cash_flow = net_income * local_rng.uniform(1.0, 1.3)

        # Ratios
        roe = net_income / equity
        roa = net_income / total_assets
        debt_to_equity = debt / equity
        current_ratio = local_rng.uniform(1.2, 2.5, n_quarters)
        quick_ratio = current_ratio * local_rng.uniform(0.6, 0.9, n_quarters)

        df = pd.DataFrame({
            'revenue': revenue,
            'gross_profit': gross_profit,
            'operating_income': operating_income,
            'net_income': net_income,
            'operating_cash_flow': operating_cash_flow,
            'total_assets': total_assets,
            'equity': equity,
            'debt': debt,
            'roe': roe,
            'roa': roa,
            'debt_to_equity': debt_to_equity,
            'current_ratio': current_ratio,
            'quick_ratio': quick_ratio,
            'ticker': ticker
        }, index=dates)

        return df

    def generate_insider_trades(self,
                               ticker: str,
                               start_date: datetime,
                               end_date: datetime,
                               n_trades: Optional[int] = None) -> pd.DataFrame:
        """
        Generate mock insider trading data.

        Args:
            ticker: Stock ticker
            start_date: Start date
            end_date: End date
            n_trades: Number of trades (default: random 5-50)

        Returns:
            DataFrame with insider trades
        """
        ticker_seed = hash(ticker) % 10000
        local_rng = np.random.RandomState(self.seed + ticker_seed)

        # Number of insider trades
        if n_trades is None:
            n_trades = local_rng.randint(5, 50)

        if n_trades == 0:
            return pd.DataFrame()

        # Random trade dates
        days_range = (end_date - start_date).days
        random_days = local_rng.randint(0, days_range, n_trades)
        trade_dates = [start_date + timedelta(days=int(d)) for d in random_days]
        filing_dates = [td + timedelta(days=local_rng.randint(1, 5)) for td in trade_dates]

        # Insider details
        titles = ['CEO', 'CFO', 'COO', 'Director', 'VP', 'Officer']
        insiders = [f'Insider_{i}' for i in range(10)]

        trades = []
        for i in range(n_trades):
            # 60% buys, 40% sells
            transaction_type = 'P' if local_rng.rand() < 0.6 else 'S'

            trade = {
                'filing_date': filing_dates[i],
                'trade_date': trade_dates[i],
                'insider_name': local_rng.choice(insiders),
                'insider_title': local_rng.choice(titles),
                'transaction_type': transaction_type,
                'transactioncode': transaction_type,  # Alias for consistency
                'shares': local_rng.randint(1000, 100000),
                'price_per_share': 50 + local_rng.rand() * 100,
                'shares_owned_after': local_rng.randint(10000, 1000000),
                'ticker': ticker
            }
            trades.append(trade)

        df = pd.DataFrame(trades)
        df['filing_date'] = pd.to_datetime(df['filing_date'])
        df['trade_date'] = pd.to_datetime(df['trade_date'])

        return df.sort_values('filing_date').reset_index(drop=True)

    def generate_universe(self,
                         start_date: datetime,
                         end_date: datetime,
                         tickers: Optional[List[str]] = None) -> dict:
        """
        Generate complete mock dataset for multiple tickers.

        Args:
            start_date: Start date
            end_date: End date
            tickers: List of tickers (default: self.tickers)

        Returns:
            Dict with keys 'prices', 'fundamentals', 'insider_trades'
        """
        if tickers is None:
            tickers = self.tickers

        prices = []
        fundamentals = []
        insider_trades = []

        for ticker in tickers:
            prices.append(self.generate_price_data(ticker, start_date, end_date))
            fundamentals.append(self.generate_fundamentals(ticker, start_date, end_date))
            insider_trades.append(self.generate_insider_trades(ticker, start_date, end_date))

        return {
            'prices': pd.concat(prices, ignore_index=False),
            'fundamentals': pd.concat(fundamentals, ignore_index=False),
            'insider_trades': pd.concat(insider_trades, ignore_index=True)
        }


def create_test_data(n_tickers: int = 5,
                    n_years: int = 5,
                    seed: int = 42) -> dict:
    """
    Convenience function to create test data.

    Args:
        n_tickers: Number of stocks
        n_years: Years of history
        seed: Random seed

    Returns:
        Dict with prices, fundamentals, insider_trades
    """
    generator = MockDataGenerator(seed=seed)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * n_years)

    tickers = [f'TICK{i:02d}' for i in range(1, n_tickers + 1)]

    return generator.generate_universe(start_date, end_date, tickers)


if __name__ == '__main__':
    # Demo usage
    print("Generating mock data...")

    generator = MockDataGenerator()

    # Single ticker
    start = datetime(2020, 1, 1)
    end = datetime(2024, 12, 31)

    prices = generator.generate_price_data('TICK01', start, end)
    print(f"\nPrice data shape: {prices.shape}")
    print(prices.head())

    fundamentals = generator.generate_fundamentals('TICK01', start, end)
    print(f"\nFundamentals shape: {fundamentals.shape}")
    print(fundamentals.head())

    insider = generator.generate_insider_trades('TICK01', start, end)
    print(f"\nInsider trades shape: {insider.shape}")
    print(insider.head())

    # Full universe
    data = create_test_data(n_tickers=5, n_years=5)
    print(f"\nUniverse data:")
    print(f"  Prices: {data['prices'].shape}")
    print(f"  Fundamentals: {data['fundamentals'].shape}")
    print(f"  Insider trades: {data['insider_trades'].shape}")
