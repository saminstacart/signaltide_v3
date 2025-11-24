"""
Tests for regime-aware allocators (M3.5).

Tests both Oracle and Rule-Based allocators on synthetic data.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from signals.ml.regime_allocators import (
    OracleRegimeAllocatorMQ,
    RuleBasedRegimeAllocatorMQ,
    get_regime_label,
    assign_regime_labels,
)


class TestRegimeLabelFunctions:
    """Test regime label mapping functions."""

    def test_get_regime_label_pre_covid(self):
        """Test regime label for pre-COVID dates."""
        date = pd.Timestamp('2018-06-15')
        assert get_regime_label(date) == 'pre_covid_expansion'

    def test_get_regime_label_covid_crash(self):
        """Test regime label for COVID crash."""
        date = pd.Timestamp('2020-03-15')
        assert get_regime_label(date) == 'covid_crash'

    def test_get_regime_label_boundaries(self):
        """Test regime label at boundaries."""
        # Just before COVID crash
        assert get_regime_label(pd.Timestamp('2020-01-31')) == 'pre_covid_expansion'
        # Start of COVID crash
        assert get_regime_label(pd.Timestamp('2020-02-01')) == 'covid_crash'
        # Start of recovery
        assert get_regime_label(pd.Timestamp('2020-05-01')) == 'covid_recovery'

    def test_assign_regime_labels(self):
        """Test assigning labels to multiple dates."""
        dates = pd.DatetimeIndex([
            '2018-01-31',
            '2020-03-31',
            '2022-06-30',
        ])
        labels = assign_regime_labels(dates)

        assert len(labels) == 3
        assert labels.iloc[0] == 'pre_covid_expansion'
        assert labels.iloc[1] == 'covid_crash'
        assert labels.iloc[2] == 'bear_2022'


class TestOracleRegimeAllocatorMQ:
    """Test Oracle allocator on synthetic data."""

    @pytest.fixture
    def synthetic_returns(self):
        """Create synthetic monthly return series."""
        np.random.seed(42)

        # 60 months of data (5 years)
        dates = pd.date_range('2020-01-31', periods=60, freq='M')

        # Momentum: high vol, positive drift
        mom_returns = pd.Series(
            np.random.normal(0.01, 0.05, 60),
            index=dates,
            name='momentum'
        )

        # Quality: low vol, lower drift
        qual_returns = pd.Series(
            np.random.normal(0.008, 0.03, 60),
            index=dates,
            name='quality'
        )

        return mom_returns, qual_returns

    def test_oracle_initialization(self, synthetic_returns):
        """Test oracle allocator initializes correctly."""
        mom_returns, qual_returns = synthetic_returns

        allocator = OracleRegimeAllocatorMQ(mom_returns, qual_returns)

        # Should have optimal weights for each regime
        assert len(allocator.optimal_weights) == 5

        # Check each regime has required attributes (use tolerance for float precision)
        for regime_name, weights in allocator.optimal_weights.items():
            assert weights.momentum >= 0.1 - 1e-9, f"{regime_name}: momentum {weights.momentum} < 0.1"
            assert weights.momentum <= 0.9 + 1e-9, f"{regime_name}: momentum {weights.momentum} > 0.9"
            assert weights.quality >= 0.1 - 1e-9, f"{regime_name}: quality {weights.quality} < 0.1"
            assert weights.quality <= 0.9 + 1e-9, f"{regime_name}: quality {weights.quality} > 0.9"
            assert abs(weights.momentum + weights.quality - 1.0) < 1e-6

    def test_oracle_weights_sum_to_one(self, synthetic_returns):
        """Test that all weights sum to 1.0."""
        mom_returns, qual_returns = synthetic_returns

        allocator = OracleRegimeAllocatorMQ(mom_returns, qual_returns)

        for date_idx, row in allocator.weight_series.iterrows():
            total = row['momentum'] + row['quality']
            assert abs(total - 1.0) < 1e-6, f"Weights at {date_idx} sum to {total}"

    def test_oracle_respects_bounds(self, synthetic_returns):
        """Test that oracle respects min/max weight bounds."""
        mom_returns, qual_returns = synthetic_returns

        allocator = OracleRegimeAllocatorMQ(mom_returns, qual_returns)

        for date_idx, row in allocator.weight_series.iterrows():
            assert row['momentum'] >= 0.1 - 1e-9, f"Momentum weight {row['momentum']} < 0.1"
            assert row['momentum'] <= 0.9 + 1e-9, f"Momentum weight {row['momentum']} > 0.9"
            assert row['quality'] >= 0.1 - 1e-9, f"Quality weight {row['quality']} < 0.1"
            assert row['quality'] <= 0.9 + 1e-9, f"Quality weight {row['quality']} > 0.9"

    def test_oracle_ensemble_returns_shape(self, synthetic_returns):
        """Test that ensemble returns have correct shape."""
        mom_returns, qual_returns = synthetic_returns

        allocator = OracleRegimeAllocatorMQ(mom_returns, qual_returns)

        # Ensemble returns should have same length as inputs
        assert len(allocator.ensemble_returns) == len(mom_returns)
        assert allocator.ensemble_returns.index.equals(mom_returns.index)

    def test_oracle_get_summary(self, synthetic_returns):
        """Test oracle summary generation."""
        mom_returns, qual_returns = synthetic_returns

        allocator = OracleRegimeAllocatorMQ(mom_returns, qual_returns)
        summary = allocator.get_summary()

        # Should have 5 regimes
        assert len(summary) == 5
        assert 'regime' in summary.columns
        assert 'w_momentum' in summary.columns
        assert 'w_quality' in summary.columns
        assert 'sharpe' in summary.columns


class TestRuleBasedRegimeAllocatorMQ:
    """Test Rule-Based allocator on synthetic data."""

    @pytest.fixture
    def synthetic_equity_curve(self):
        """Create synthetic equity curve with varying regimes."""
        np.random.seed(42)

        dates = pd.date_range('2020-01-31', periods=60, freq='M')

        # Start at 100, create equity curve with different regimes
        equity = np.ones(60) * 100.0

        for i in range(1, 60):
            # First 20 months: CALM (low vol, uptrend)
            if i < 20:
                equity[i] = equity[i-1] * (1 + np.random.normal(0.01, 0.02))

            # Next 10 months: STRESS (high vol, downtrend)
            elif i < 30:
                equity[i] = equity[i-1] * (1 + np.random.normal(-0.02, 0.08))

            # Last 30 months: CHOPPY (medium vol, mixed)
            else:
                equity[i] = equity[i-1] * (1 + np.random.normal(0.005, 0.04))

        return pd.Series(equity, index=dates, name='equity')

    def test_rule_based_initialization(self, synthetic_equity_curve):
        """Test rule-based allocator initializes correctly."""
        allocator = RuleBasedRegimeAllocatorMQ(synthetic_equity_curve)

        # Should have computed indicators
        assert len(allocator.indicators) > 0

        # Should have classified regimes
        assert len(allocator.regime_series) > 0

        # Should have weight series
        assert len(allocator.weight_series) > 0

    def test_rule_based_regime_classification(self, synthetic_equity_curve):
        """Test that regimes are classified into CALM/STRESS/CHOPPY."""
        allocator = RuleBasedRegimeAllocatorMQ(synthetic_equity_curve)

        regimes = allocator.regime_series.unique()

        # All regimes should be one of the three types
        for regime in regimes:
            assert regime in ['CALM', 'STRESS', 'CHOPPY']

    def test_rule_based_weights_sum_to_one(self, synthetic_equity_curve):
        """Test that weights sum to 1.0."""
        allocator = RuleBasedRegimeAllocatorMQ(synthetic_equity_curve)

        for date_idx, row in allocator.weight_series.iterrows():
            total = row['momentum'] + row['quality']
            assert abs(total - 1.0) < 1e-6, f"Weights at {date_idx} sum to {total}"

    def test_rule_based_respects_bounds(self, synthetic_equity_curve):
        """Test that rule-based respects min/max weight bounds."""
        allocator = RuleBasedRegimeAllocatorMQ(synthetic_equity_curve)

        for date_idx, row in allocator.weight_series.iterrows():
            assert row['momentum'] >= 0.1, f"Momentum weight {row['momentum']} < 0.1"
            assert row['momentum'] <= 0.9, f"Momentum weight {row['momentum']} > 0.9"
            assert row['quality'] >= 0.1, f"Quality weight {row['quality']} < 0.1"
            assert row['quality'] <= 0.9, f"Quality weight {row['quality']} > 0.9"

    def test_rule_based_weight_presets(self, synthetic_equity_curve):
        """Test that rule-based uses correct weight presets per regime."""
        allocator = RuleBasedRegimeAllocatorMQ(synthetic_equity_curve)

        for date_idx, row in allocator.weight_series.iterrows():
            regime = row['regime']

            # Check weights match presets
            if regime == 'CALM':
                assert row['momentum'] == 0.35
                assert row['quality'] == 0.65
            elif regime == 'STRESS':
                assert row['momentum'] == 0.15
                assert row['quality'] == 0.85
            elif regime == 'CHOPPY':
                assert row['momentum'] == 0.25
                assert row['quality'] == 0.75

    def test_rule_based_apply_weights(self, synthetic_equity_curve):
        """Test applying weights to construct ensemble returns."""
        allocator = RuleBasedRegimeAllocatorMQ(synthetic_equity_curve)

        # Create synthetic returns
        np.random.seed(42)
        mom_returns = pd.Series(
            np.random.normal(0.01, 0.05, len(synthetic_equity_curve)),
            index=synthetic_equity_curve.index
        )
        qual_returns = pd.Series(
            np.random.normal(0.008, 0.03, len(synthetic_equity_curve)),
            index=synthetic_equity_curve.index
        )

        ensemble_returns = allocator.apply_weights(mom_returns, qual_returns)

        # Should have returns (though may be shorter due to lookback)
        assert len(ensemble_returns) > 0
        assert ensemble_returns.index.isin(synthetic_equity_curve.index).all()

    def test_rule_based_get_summary(self, synthetic_equity_curve):
        """Test rule-based summary generation."""
        allocator = RuleBasedRegimeAllocatorMQ(synthetic_equity_curve)
        summary = allocator.get_summary()

        # Should have 3 regime types
        assert len(summary) == 3
        assert 'regime' in summary.columns
        assert 'count' in summary.columns
        assert 'pct' in summary.columns

        # Counts should sum to total regime classifications
        total_count = summary['count'].sum()
        assert total_count == len(allocator.regime_series)

    def test_rule_based_extreme_volatility_triggers_stress(self):
        """Test that very high volatility triggers STRESS regime."""
        # Create equity curve with extreme volatility
        np.random.seed(42)
        dates = pd.date_range('2020-01-31', periods=30, freq='M')
        equity = np.ones(30) * 100.0

        for i in range(1, 30):
            # Extreme volatility
            equity[i] = equity[i-1] * (1 + np.random.normal(0, 0.15))

        equity_series = pd.Series(equity, index=dates)

        allocator = RuleBasedRegimeAllocatorMQ(equity_series)

        # Should have at least some STRESS classifications
        assert 'STRESS' in allocator.regime_series.values

    def test_rule_based_deep_drawdown_triggers_stress(self):
        """Test that deep drawdown triggers STRESS regime."""
        # Create equity curve with severe drawdown
        dates = pd.date_range('2020-01-31', periods=30, freq='M')
        equity = np.ones(30) * 100.0

        # Create 30% drawdown
        for i in range(1, 15):
            equity[i] = equity[i-1] * 0.98  # Gradual decline

        # Then flat
        for i in range(15, 30):
            equity[i] = equity[14]

        equity_series = pd.Series(equity, index=dates)

        allocator = RuleBasedRegimeAllocatorMQ(equity_series)

        # After drawdown, should see STRESS
        assert 'STRESS' in allocator.regime_series.values


class TestAllocatorEdgeCases:
    """Test edge cases and error conditions."""

    def test_oracle_mismatched_indices_raises(self):
        """Test that mismatched indices raise error."""
        dates1 = pd.date_range('2020-01-31', periods=30, freq='M')
        dates2 = pd.date_range('2020-02-29', periods=30, freq='M')

        mom_returns = pd.Series(np.random.normal(0.01, 0.05, 30), index=dates1)
        qual_returns = pd.Series(np.random.normal(0.008, 0.03, 30), index=dates2)

        with pytest.raises(ValueError, match="matching indices"):
            OracleRegimeAllocatorMQ(mom_returns, qual_returns)

    def test_oracle_single_regime_period_handled(self):
        """Test oracle handles regimes with very few periods gracefully."""
        # Create returns spanning only COVID crash (2 months)
        dates = pd.date_range('2020-02-29', '2020-04-30', freq='M')
        mom_returns = pd.Series(np.random.normal(-0.1, 0.3, len(dates)), index=dates)
        qual_returns = pd.Series(np.random.normal(-0.05, 0.2, len(dates)), index=dates)

        # Should not crash
        allocator = OracleRegimeAllocatorMQ(mom_returns, qual_returns)

        # Should have some optimal weights (may use defaults for regimes with <2 periods)
        assert len(allocator.optimal_weights) == 5

    def test_rule_based_short_equity_curve_handled(self):
        """Test rule-based handles short equity curves gracefully."""
        # Very short equity curve
        dates = pd.date_range('2020-01-31', periods=10, freq='M')
        equity = pd.Series(np.linspace(100, 110, 10), index=dates)

        # Should not crash (may have no classifications due to lookback requirements)
        allocator = RuleBasedRegimeAllocatorMQ(equity)

        # May have empty or short regime series due to lookback
        assert allocator.regime_series is not None
