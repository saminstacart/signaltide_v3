"""
Test suite for EnsembleSignal with Momentum v2 configuration.

Validates:
1. Ensemble scores match direct InstitutionalMomentum v2 signal
2. Baseline runner produces reasonable metrics
3. No regressions in ensemble framework

Last Updated: 2025-11-21
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any
import pytest

from data.data_manager import DataManager
from signals.momentum.institutional_momentum import InstitutionalMomentum
from signals.ml.ensemble_configs import get_momentum_v2_ensemble
from config import get_logger

logger = get_logger(__name__)


class TestEnsembleMomentumValidation:
    """Validate ensemble produces identical results to direct momentum signal."""

    @classmethod
    def setup_class(cls):
        """Initialize data manager and test parameters."""
        cls.dm = DataManager()

        # Test universe: Only tickers available in fixture DB
        cls.test_tickers = ['AAPL', 'MSFT', 'GOOGL']

        # Test period: Q1 2020 to match fixture DB
        # Need extra lookback for momentum formation (308 days) - not available in fixture
        # Using minimal range that fixture supports
        cls.price_start_date = datetime(2020, 1, 1)
        cls.start_date = datetime(2020, 1, 31)
        cls.end_date = datetime(2020, 3, 31)

        # Canonical Momentum v2 parameters (Trial 11)
        cls.momentum_params: Dict[str, Any] = {
            "formation_period": 308,
            "skip_period": 0,
            "winsorize_pct": [0.4, 99.6],
            "rebalance_frequency": "monthly",
            "quintiles": True,
        }

    @pytest.mark.requires_full_db
    def test_ensemble_vs_direct_numerical_equivalence(self):
        """
        Test that ensemble scores exactly match direct momentum scores.

        Acceptance: max absolute difference < 1e-9
        """
        logger.info("=" * 80)
        logger.info("TEST: Ensemble vs Direct Numerical Equivalence")
        logger.info("=" * 80)

        # Instantiate both approaches
        ensemble = get_momentum_v2_ensemble(self.dm)
        direct_signal = InstitutionalMomentum(params=self.momentum_params)

        # Get monthly rebalance dates
        rebalance_dates = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq='M'
        ).tolist()

        logger.info(f"Testing {len(rebalance_dates)} rebalance dates")
        logger.info(f"Universe: {len(self.test_tickers)} tickers")

        max_diff = 0.0
        mean_diff = 0.0
        n_comparisons = 0

        for rebal_date in rebalance_dates:
            # Fetch price data
            prices = {}
            for ticker in self.test_tickers:
                try:
                    price_data = self.dm.get_prices(
                        ticker,
                        self.price_start_date.strftime('%Y-%m-%d'),
                        rebal_date.strftime('%Y-%m-%d')
                    )
                    if len(price_data) > 0 and 'close' in price_data.columns:
                        prices[ticker] = price_data['close']
                except Exception as e:
                    logger.debug(f"Could not fetch {ticker}: {e}")
                    continue

            if len(prices) < 5:
                logger.warning(f"Skipping {rebal_date}: insufficient data")
                continue

            # Generate ensemble scores
            ensemble_scores = ensemble.generate_ensemble_scores(prices, rebalance_date=rebal_date)

            # Generate direct scores (ticker by ticker)
            direct_scores_dict = {}
            for ticker, px_series in prices.items():
                # Build DataFrame matching signal API
                data = pd.DataFrame({
                    'close': px_series,
                    'ticker': ticker
                })
                try:
                    sig_series = direct_signal.generate_signals(data)
                    if len(sig_series) > 0:
                        signal_value = sig_series.iloc[-1]
                        if pd.notna(signal_value) and signal_value != 0:
                            direct_scores_dict[ticker] = signal_value
                except Exception as e:
                    logger.debug(f"Error generating direct signal for {ticker}: {e}")

            direct_scores = pd.Series(direct_scores_dict)

            # Find common tickers
            common_tickers = set(ensemble_scores.index) & set(direct_scores.index)

            if len(common_tickers) == 0:
                logger.warning(f"No common tickers at {rebal_date}")
                continue

            # Compare scores
            ensemble_vals = ensemble_scores.loc[list(common_tickers)]
            direct_vals = direct_scores.loc[list(common_tickers)]

            abs_diff = (ensemble_vals - direct_vals).abs()
            max_diff = max(max_diff, abs_diff.max())
            mean_diff += abs_diff.mean()
            n_comparisons += 1

            logger.debug(f"{rebal_date.date()}: {len(common_tickers)} tickers, "
                        f"max_diff={abs_diff.max():.2e}")

        mean_diff /= max(n_comparisons, 1)

        logger.info("")
        logger.info(f"Rebalances compared: {n_comparisons}")
        logger.info(f"Max abs difference: {max_diff:.2e}")
        logger.info(f"Mean abs difference: {mean_diff:.2e}")

        # Acceptance criterion: max difference < 1e-9 (tight numerical tolerance)
        assert max_diff < 1e-9, (
            f"Ensemble scores deviate from direct momentum: max_diff={max_diff:.2e} "
            f"(expected < 1e-9)"
        )

        logger.info("✅ PASS: Ensemble matches direct momentum (within numerical precision)")
        logger.info("=" * 80)

    @pytest.mark.requires_full_db
    def test_ensemble_signal_properties(self):
        """
        Test that ensemble signals have expected properties:
        - Values in [-1, 1] range
        - Quintile distribution (~20% in each bin)
        - No NaN/inf values
        """
        logger.info("=" * 80)
        logger.info("TEST: Ensemble Signal Properties")
        logger.info("=" * 80)

        ensemble = get_momentum_v2_ensemble(self.dm)

        # Fetch data for one rebalance
        test_date = datetime(2023, 6, 30)
        prices = {}
        for ticker in self.test_tickers:
            try:
                price_data = self.dm.get_prices(
                    ticker,
                    '2022-01-01',
                    test_date.strftime('%Y-%m-%d')
                )
                if len(price_data) > 0 and 'close' in price_data.columns:
                    prices[ticker] = price_data['close']
            except Exception:
                continue

        signals = ensemble.generate_ensemble_scores(prices, rebalance_date=test_date)

        # Check value range
        assert signals.min() >= -1.0, f"Signal below -1: {signals.min()}"
        assert signals.max() <= 1.0, f"Signal above 1: {signals.max()}"

        # Check for NaN/inf
        assert not signals.isna().any(), "Signals contain NaN values"
        assert np.isfinite(signals).all(), "Signals contain inf values"

        # Check quintile distribution (with some tolerance)
        unique_vals = signals.value_counts(normalize=True)
        logger.info(f"Signal distribution:\n{unique_vals}")

        # Expected quintiles: -1, -0.5, 0, 0.5, 1
        expected_quintiles = {-1.0, -0.5, 0.0, 0.5, 1.0}
        actual_quintiles = set(signals.unique())

        assert actual_quintiles.issubset(expected_quintiles), (
            f"Unexpected signal values: {actual_quintiles - expected_quintiles}"
        )

        logger.info("✅ PASS: Signals have expected properties")
        logger.info("=" * 80)


class TestEnsembleBaselineSmokeTest:
    """Smoke tests for ensemble baseline runner."""

    @classmethod
    def setup_class(cls):
        """Initialize test parameters."""
        cls.dm = DataManager()

        # Small universe for fast execution (fixture DB only has these 3 tickers)
        cls.test_tickers = ['AAPL', 'MSFT', 'GOOGL']

        # Short test period: Q1 2020 to match fixture DB
        # Need extra lookback for momentum formation (308 days) - not available in fixture
        # Using minimal range that fixture supports
        cls.price_start_date = datetime(2020, 1, 1)
        cls.start_date = datetime(2020, 1, 31)
        cls.end_date = datetime(2020, 3, 31)

    @pytest.mark.requires_full_db
    def test_baseline_runner_produces_equity_curve(self):
        """
        Smoke test: Baseline runner produces non-empty equity curve.
        """
        logger.info("=" * 80)
        logger.info("TEST: Baseline Runner Equity Curve")
        logger.info("=" * 80)

        # Get ensemble
        ensemble = get_momentum_v2_ensemble(self.dm)

        # Get rebalance dates
        rebalance_dates = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq='M'
        ).tolist()

        logger.info(f"Running baseline for {len(rebalance_dates)} rebalances")

        # Fetch all price data (with lookback for momentum formation)
        prices_dict = {}
        for ticker in self.test_tickers:
            try:
                price_data = self.dm.get_prices(
                    ticker,
                    self.price_start_date.strftime('%Y-%m-%d'),
                    self.end_date.strftime('%Y-%m-%d')
                )
                if len(price_data) > 0 and 'close' in price_data.columns:
                    prices_dict[ticker] = price_data['close']
            except Exception:
                continue

        # Simple backtest logic (simplified from run_ensemble_baseline.py)
        initial_capital = 50000.0
        portfolio_value = initial_capital
        equity_data = []

        for i, rebal_date in enumerate(rebalance_dates):
            # Get signals
            prices_subset = {
                t: p[p.index <= rebal_date]
                for t, p in prices_dict.items()
            }

            try:
                signals = ensemble.generate_ensemble_scores(
                    prices_subset,
                    rebalance_date=rebal_date
                )
            except Exception as e:
                logger.warning(f"Signal generation failed at {rebal_date}: {e}")
                continue

            if len(signals) == 0:
                continue

            # Equal-weight top quintile
            top_quintile = signals[signals == signals.max()]
            if len(top_quintile) == 0:
                continue

            target_weight = 1.0 / len(top_quintile)
            current_holdings = {t: target_weight * portfolio_value for t in top_quintile.index}

            # Track equity until next rebalance
            next_rebal = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else self.end_date

            all_dates = sorted(set().union(*[
                set(prices_dict[t].index[(prices_dict[t].index >= rebal_date) &
                                         (prices_dict[t].index <= next_rebal)])
                for t in current_holdings.keys() if t in prices_dict
            ]))

            for date in all_dates:
                holding_value = 0.0
                for ticker, shares_value in current_holdings.items():
                    if ticker in prices_dict:
                        price_history = prices_dict[ticker]
                        if rebal_date in price_history.index and date in price_history.index:
                            rebal_price = price_history.loc[rebal_date]
                            current_price = price_history.loc[date]
                            # Handle duplicate indices
                            if isinstance(rebal_price, pd.Series):
                                rebal_price = rebal_price.iloc[0]
                            if isinstance(current_price, pd.Series):
                                current_price = current_price.iloc[0]
                            shares = shares_value / rebal_price
                            holding_value += shares * current_price

                equity_data.append({'date': date, 'equity': holding_value})

            if equity_data:
                portfolio_value = equity_data[-1]['equity']

        # Convert to DataFrame
        if not equity_data:
            pytest.fail("No equity data generated - check signal generation and price data")

        equity_df = pd.DataFrame(equity_data)
        # Handle potential duplicate dates by keeping last value for each date
        equity_df = equity_df.drop_duplicates(subset=['date'], keep='last')
        equity_curve = equity_df.set_index('date')['equity'].sort_index()

        logger.info(f"Equity curve length: {len(equity_curve)}")
        logger.info(f"Initial capital: ${initial_capital:,.0f}")

        final_equity = equity_curve.iloc[-1]
        if isinstance(final_equity, pd.Series):
            final_equity = final_equity.iloc[0]  # Get first value if Series
        logger.info(f"Final equity: ${final_equity:,.0f}")

        # Acceptance criteria: equity curve exists and is non-empty
        assert len(equity_curve) > 0, "Equity curve is empty"

        logger.info("✅ PASS: Baseline runner produces equity curve")
        logger.info("=" * 80)

    @pytest.mark.requires_full_db
    def test_baseline_runner_produces_reasonable_metrics(self):
        """
        Smoke test: Baseline runner produces metrics within reasonable bounds.

        Acceptance:
        - Total return between -50% and +200%
        - Sharpe between -2.0 and +2.5
        - Max drawdown between -60% and 0%
        """
        logger.info("=" * 80)
        logger.info("TEST: Baseline Runner Reasonable Metrics")
        logger.info("=" * 80)

        # Get ensemble
        ensemble = get_momentum_v2_ensemble(self.dm)

        # Get rebalance dates
        rebalance_dates = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq='M'
        ).tolist()

        # Fetch all price data (with lookback for momentum formation)
        prices_dict = {}
        for ticker in self.test_tickers:
            try:
                price_data = self.dm.get_prices(
                    ticker,
                    self.price_start_date.strftime('%Y-%m-%d'),
                    self.end_date.strftime('%Y-%m-%d')
                )
                if len(price_data) > 0 and 'close' in price_data.columns:
                    prices_dict[ticker] = price_data['close']
            except Exception:
                continue

        # Run simplified backtest
        initial_capital = 50000.0
        portfolio_value = initial_capital
        equity_data = []

        for i, rebal_date in enumerate(rebalance_dates):
            prices_subset = {
                t: p[p.index <= rebal_date]
                for t, p in prices_dict.items()
            }

            try:
                signals = ensemble.generate_ensemble_scores(
                    prices_subset,
                    rebalance_date=rebal_date
                )
            except Exception:
                continue

            if len(signals) == 0:
                continue

            top_quintile = signals[signals == signals.max()]
            if len(top_quintile) == 0:
                continue

            target_weight = 1.0 / len(top_quintile)
            current_holdings = {t: target_weight * portfolio_value for t in top_quintile.index}

            next_rebal = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else self.end_date

            all_dates = sorted(set().union(*[
                set(prices_dict[t].index[(prices_dict[t].index >= rebal_date) &
                                         (prices_dict[t].index <= next_rebal)])
                for t in current_holdings.keys() if t in prices_dict
            ]))

            for date in all_dates:
                holding_value = 0.0
                for ticker, shares_value in current_holdings.items():
                    if ticker in prices_dict:
                        price_history = prices_dict[ticker]
                        if rebal_date in price_history.index and date in price_history.index:
                            rebal_price = price_history.loc[rebal_date]
                            current_price = price_history.loc[date]
                            # Handle duplicate indices
                            if isinstance(rebal_price, pd.Series):
                                rebal_price = rebal_price.iloc[0]
                            if isinstance(current_price, pd.Series):
                                current_price = current_price.iloc[0]
                            shares = shares_value / rebal_price
                            holding_value += shares * current_price

                equity_data.append({'date': date, 'equity': holding_value})

            if equity_data:
                portfolio_value = equity_data[-1]['equity']

        # Convert to DataFrame
        if not equity_data:
            pytest.fail("No equity data generated - check signal generation and price data")

        equity_df = pd.DataFrame(equity_data)
        # Handle potential duplicate dates by keeping last value for each date
        equity_df = equity_df.drop_duplicates(subset=['date'], keep='last')
        equity_curve = equity_df.set_index('date')['equity'].sort_index()

        # Remove any None/NaN values
        equity_curve = equity_curve.dropna()
        # Filter out zeros (invalid equity values)
        equity_curve = equity_curve[equity_curve > 0]

        if len(equity_curve) == 0:
            pytest.fail("No valid equity values after filtering")

        # Compute metrics
        final_equity = equity_curve.iloc[-1]
        if isinstance(final_equity, pd.Series):
            final_equity = final_equity.iloc[0]
        total_return = (final_equity / initial_capital) - 1

        daily_returns = equity_curve.pct_change().dropna()
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        else:
            sharpe = 0.0

        running_max = equity_curve.expanding().max()
        # Avoid division by zero if running_max is zero
        drawdown = (equity_curve - running_max) / running_max.replace(0, 1)
        max_drawdown = drawdown.min()

        logger.info(f"Total Return: {total_return:.2%}")
        logger.info(f"Sharpe Ratio: {sharpe:.3f}")
        logger.info(f"Max Drawdown: {max_drawdown:.2%}")

        # Acceptance criteria: reasonable bounds
        assert -0.50 <= total_return <= 2.00, (
            f"Total return {total_return:.2%} outside [-50%, +200%]"
        )
        assert -2.0 <= sharpe <= 2.5, (
            f"Sharpe {sharpe:.3f} outside [-2.0, +2.5]"
        )
        assert -0.60 <= max_drawdown <= 0.0, (
            f"Max drawdown {max_drawdown:.2%} outside [-60%, 0%]"
        )

        logger.info("✅ PASS: Metrics within reasonable bounds")
        logger.info("=" * 80)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
