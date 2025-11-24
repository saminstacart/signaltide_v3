"""
Institutional Momentum Signal

Implementation of Jegadeesh-Titman (1993) momentum strategy
with professional cross-sectional methodology.

Strategy:
- Formation period: 12 months (252 trading days)
- Skip period: 1 month (21 days) to avoid short-term reversals
- Monthly rebalancing
- Cross-sectional quintile ranking

This is the "12-1" momentum that forms the basis of:
- AQR's Momentum Everywhere
- Fama-French MOM factor
- Every professional momentum strategy

References:
- Jegadeesh, Titman (1993) "Returns to Buying Winners and Selling Losers"
- Asness, Moskowitz, Pedersen (2013) "Value and Momentum Everywhere"
"""

from typing import Dict, Any, List
from datetime import timedelta
import pandas as pd
import numpy as np
from core.institutional_base import InstitutionalSignal


class InstitutionalMomentum(InstitutionalSignal):
    """
    Jegadeesh-Titman 12-1 Momentum Signal.

    Professional implementation with:
    - 12-month formation, 1-month skip
    - Cross-sectional quintile ranking
    - Monthly rebalancing
    - Winsorized returns

    Parameters:
        formation_period: Days for momentum calculation (default: 252 = 12 months)
        skip_period: Days to skip (default: 21 = 1 month)
        rebalance_frequency: 'monthly' (professional standard)
        winsorize_pct: [lower, upper] percentiles (default: [5, 95])
        quintiles: True to use quintile signals (default: True)
    """

    def __init__(self,
                 params: Dict[str, Any],
                 name: str = 'InstitutionalMomentum'):
        super().__init__(params, name)

        # Jegadeesh-Titman standard: 12-1
        self.formation_period = params.get('formation_period', 252)  # 12 months
        self.skip_period = params.get('skip_period', 21)  # 1 month

        # Total lookback = formation + skip
        self.total_lookback = self.formation_period + self.skip_period

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate momentum signals using Jegadeesh-Titman methodology.

        Args:
            data: DataFrame with 'close' prices, datetime index

        Returns:
            Series with signals in [-1, 1] range
        """
        if 'close' not in data.columns:
            return pd.Series(0, index=data.index)

        if len(data) < self.total_lookback:
            return pd.Series(0, index=data.index)

        # Calculate momentum: return from t-273 to t-21 (12-1)
        # Skip most recent month to avoid short-term reversal
        prices = data['close'].copy()

        # Calculate formation period returns
        # From (t - formation - skip) to (t - skip)
        momentum = prices.pct_change(periods=self.formation_period, fill_method=None).shift(self.skip_period)

        # Winsorize to handle outliers (professional standard)
        momentum_winsorized = self.winsorize(
            momentum,
            lower_pct=self.winsorize_pct[0],
            upper_pct=self.winsorize_pct[1]
        )

        # Convert to signals
        if self.quintiles:
            # Professional standard: Quintile signals
            signals = self._to_quintile_signals(momentum_winsorized)
        else:
            # Alternative: Continuous z-score signals
            signals = self._to_continuous_signals(momentum_winsorized)

        # Monthly rebalancing: Hold signal for entire month
        if self.rebalance_frequency == 'monthly':
            signals = self._apply_monthly_rebalancing(signals)

        # Ensure proper range
        signals = signals.clip(-1, 1).fillna(0)

        return signals

    def _to_quintile_signals(self, momentum: pd.Series) -> pd.Series:
        """
        Convert momentum to quintile signals [-1, -0.5, 0, 0.5, 1].

        Professional standard: Discrete quintiles for clear interpretation
        and reduced turnover.
        """
        # Create quintiles on non-null values
        momentum_clean = momentum.dropna()

        if len(momentum_clean) == 0:
            return pd.Series(0.0, index=momentum.index)

        # Ensure unique index before quintile assignment
        if momentum_clean.index.duplicated().any():
            # Keep last value for each duplicate date
            momentum_clean = momentum_clean[~momentum_clean.index.duplicated(keep='last')]

        quintiles = self.to_quintiles(momentum_clean)

        # Map to full index
        signals = pd.Series(0.0, index=momentum.index)
        signals.loc[quintiles.index] = quintiles.astype(float)

        return signals

    def _to_continuous_signals(self, momentum: pd.Series) -> pd.Series:
        """
        Convert momentum to continuous signals using cross-sectional ranking.

        Alternative to quintiles: Smoother signal evolution.
        """
        # Rank to [0, 1]
        ranked = momentum.rank(pct=True)

        # Convert to [-1, 1]
        signals = (ranked - 0.5) * 2

        return signals.fillna(0)

    def _apply_monthly_rebalancing(self, signals: pd.Series) -> pd.Series:
        """
        Apply monthly rebalancing: Hold signal constant within each month.

        Professional practice: Reduces turnover and transaction costs.
        Signal calculated at month-end, held for entire following month.
        """
        # Get month-end signals
        month_ends = signals.resample('ME').last()  # 'ME' = month end (replaces deprecated 'M')

        # Forward-fill to all days
        # Each month uses the signal from the previous month-end
        rebalanced = month_ends.reindex(signals.index, method='ffill')

        return rebalanced.fillna(0)

    def generate_cross_sectional_scores(
        self,
        rebal_date: pd.Timestamp,
        universe: List[str],
        data_manager: "DataManager",
    ) -> pd.Series:
        """
        Generate InstitutionalMomentum scores for universe at rebalance date.

        Copied literally from test_backtest_engine.py direct_signal_fn (lines 304-320).

        Args:
            rebal_date: Rebalance date (PIT cutoff)
            universe: List of ticker symbols to score
            data_manager: DataManager instance for fetching prices

        Returns:
            pd.Series indexed by ticker with momentum scores
        """
        # Same lookback as equivalence test
        lookback_start = (rebal_date - timedelta(days=500)).strftime('%Y-%m-%d')
        rebal_date_str = rebal_date.strftime('%Y-%m-%d')

        scores = {}
        for ticker in universe:
            try:
                prices = data_manager.get_prices(ticker, lookback_start, rebal_date_str)
                if len(prices) > 0 and 'close' in prices.columns:
                    data = pd.DataFrame({'close': prices['close'], 'ticker': ticker})
                    sig_series = self.generate_signals(data)
                    if len(sig_series) > 0:
                        signal_value = sig_series.iloc[-1]
                        if pd.notna(signal_value) and signal_value != 0:
                            scores[ticker] = signal_value
            except Exception:
                # Skip tickers with data issues (matches equivalence test pattern)
                continue

        return pd.Series(scores)

    def get_parameter_space(self) -> Dict[str, tuple]:
        """
        Define parameter space for optimization.

        Returns:
            Dict with parameter specifications for Optuna
        """
        return {
            'formation_period': ('int', 126, 378),  # 6-18 months
            'skip_period': ('int', 5, 42),  # 1 week to 2 months
            'winsorize_pct': ('categorical', [[1, 99], [5, 95], [10, 90]]),
            'quintiles': ('categorical', [True, False])
        }

    def __repr__(self) -> str:
        return (f"InstitutionalMomentum("
                f"formation={self.formation_period}, "
                f"skip={self.skip_period}, "
                f"rebalance={self.rebalance_frequency})")


class CrossSectionalMomentum(InstitutionalSignal):
    """
    Cross-sectional momentum for multi-stock portfolios.

    When used with multiple stocks, ranks momentum cross-sectionally
    at each rebalance date. This is the institutional standard.

    Parameters:
        formation_period: Days for momentum calculation
        skip_period: Days to skip before ranking
        rebalance_frequency: Rebalancing frequency
    """

    def __init__(self,
                 params: Dict[str, Any],
                 name: str = 'CrossSectionalMomentum'):
        super().__init__(params, name)

        self.formation_period = params.get('formation_period', 252)
        self.skip_period = params.get('skip_period', 21)

    def generate_signals_cross_sectional(self,
                                        prices_dict: Dict[str, pd.DataFrame],
                                        rebalance_dates: pd.DatetimeIndex) -> Dict[str, pd.Series]:
        """
        Generate cross-sectional momentum signals for multiple stocks.

        Professional implementation:
        1. Calculate momentum for each stock
        2. At each rebalance date, rank stocks cross-sectionally
        3. Assign quintile signals
        4. Hold until next rebalance

        Args:
            prices_dict: Dict mapping ticker -> price DataFrame
            rebalance_dates: Dates to rebalance (typically month-ends)

        Returns:
            Dict mapping ticker -> signal Series
        """
        # Calculate momentum for all stocks
        momentum_dict = {}
        for ticker, prices in prices_dict.items():
            if 'close' not in prices.columns:
                continue

            # Calculate formation period momentum, skipping recent period
            mom = prices['close'].pct_change(periods=self.formation_period, fill_method=None).shift(self.skip_period)
            momentum_dict[ticker] = mom

        # Combine into DataFrame for cross-sectional ranking
        momentum_df = pd.DataFrame(momentum_dict)

        # Rank cross-sectionally at each rebalance date
        signals_dict = {ticker: pd.Series(0, index=prices.index)
                       for ticker, prices in prices_dict.items()}

        for date in rebalance_dates:
            if date not in momentum_df.index:
                continue

            # Get momentum at this date
            mom_values = momentum_df.loc[date].dropna()

            if len(mom_values) < 5:  # Need at least 5 stocks for quintiles
                continue

            # Winsorize cross-sectionally
            mom_winsorized = self.winsorize(mom_values)

            # Create quintiles
            quintiles = self.to_quintiles(mom_winsorized)

            # Assign signals (hold until next rebalance)
            next_rebal_idx = rebalance_dates.get_loc(date) + 1
            if next_rebal_idx < len(rebalance_dates):
                next_rebal_date = rebalance_dates[next_rebal_idx]
                for ticker in quintiles.index:
                    if ticker in signals_dict:
                        mask = (prices_dict[ticker].index >= date) & \
                               (prices_dict[ticker].index < next_rebal_date)
                        signals_dict[ticker].loc[mask] = float(quintiles[ticker])

        return signals_dict

    def __repr__(self) -> str:
        return f"CrossSectionalMomentum({self.formation_period}-{self.skip_period})"


if __name__ == '__main__':
    print("InstitutionalMomentum - Jegadeesh-Titman 12-1 Strategy")
    print("Professional cross-sectional implementation")
    print()
    print("Standard parameters:")
    print("  Formation: 252 days (12 months)")
    print("  Skip: 21 days (1 month)")
    print("  Rebalance: Monthly")
    print("  Quintile signals: [-1, -0.5, 0, 0.5, 1]")
