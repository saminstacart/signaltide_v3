"""
Unit tests for Deflated Sharpe Ratio implementation.
"""

import pytest
import numpy as np
from scipy import stats

from core.deflated_sharpe import (
    compute_deflated_sharpe,
    compute_sharpe_std_error,
    expected_max_sharpe,
    apply_dsr_to_trials,
    compute_minimum_track_record_length,
    dsr_summary_report
)


class TestDeflatedSharpe:
    """Tests for DSR calculation."""

    def test_basic_dsr_calculation(self):
        """Test basic DSR calculation returns valid values."""
        dsr, pval = compute_deflated_sharpe(
            observed_sharpe=0.628,
            num_trials=50,
            returns_skewness=-0.2,
            returns_kurtosis=1.5,
            T=116
        )

        assert isinstance(dsr, float), "DSR should be float"
        assert isinstance(pval, float), "P-value should be float"
        assert 0 <= pval <= 1, "P-value should be in [0, 1]"

    def test_more_trials_deflates_more(self):
        """More trials should increase expected max Sharpe, reducing DSR."""
        dsr_few, _ = compute_deflated_sharpe(
            observed_sharpe=0.5,
            num_trials=10,
            returns_skewness=0,
            returns_kurtosis=0,
            T=100
        )

        dsr_many, _ = compute_deflated_sharpe(
            observed_sharpe=0.5,
            num_trials=100,
            returns_skewness=0,
            returns_kurtosis=0,
            T=100
        )

        assert dsr_few > dsr_many, "More trials should deflate Sharpe more"

    def test_higher_sharpe_more_significant(self):
        """Higher observed Sharpe should be more significant (lower p-value)."""
        _, pval_low = compute_deflated_sharpe(
            observed_sharpe=0.3,
            num_trials=50,
            returns_skewness=0,
            returns_kurtosis=0,
            T=100
        )

        _, pval_high = compute_deflated_sharpe(
            observed_sharpe=0.8,
            num_trials=50,
            returns_skewness=0,
            returns_kurtosis=0,
            T=100
        )

        assert pval_high < pval_low, "Higher Sharpe should be more significant"

    def test_longer_track_record_more_significant(self):
        """Longer track record should increase significance."""
        _, pval_short = compute_deflated_sharpe(
            observed_sharpe=0.5,
            num_trials=50,
            returns_skewness=0,
            returns_kurtosis=0,
            T=50
        )

        _, pval_long = compute_deflated_sharpe(
            observed_sharpe=0.5,
            num_trials=50,
            returns_skewness=0,
            returns_kurtosis=0,
            T=500
        )

        assert pval_long < pval_short, "Longer track record should be more significant"

    def test_production_mq_significant(self):
        """M+Q production Sharpe (0.628) should be significant after ~50 trials."""
        dsr, pval = compute_deflated_sharpe(
            observed_sharpe=0.628,
            num_trials=50,
            returns_skewness=-0.2,
            returns_kurtosis=1.5,
            T=116  # 116 monthly returns
        )

        assert pval < 0.05, f"M+Q should be significant, got p={pval}"
        assert dsr > 0, f"DSR should be positive, got {dsr}"

    def test_invalid_inputs_raise(self):
        """Invalid inputs should raise errors."""
        with pytest.raises(ValueError):
            compute_deflated_sharpe(0.5, 0, 0, 0, 100)  # num_trials < 1

        with pytest.raises(ValueError):
            compute_deflated_sharpe(0.5, 10, 0, 0, 1)  # T < 2


class TestSharpeStdError:
    """Tests for Sharpe standard error calculation."""

    def test_std_error_positive(self):
        """Standard error should always be positive."""
        se = compute_sharpe_std_error(0.5, 0, 0, 100)
        assert se > 0, "Standard error should be positive"

    def test_more_observations_lower_se(self):
        """More observations should reduce standard error."""
        se_few = compute_sharpe_std_error(0.5, 0, 0, 50)
        se_many = compute_sharpe_std_error(0.5, 0, 0, 500)
        assert se_many < se_few, "More observations should reduce SE"

    def test_higher_kurtosis_higher_se(self):
        """Fat tails (high kurtosis) should increase standard error."""
        se_normal = compute_sharpe_std_error(0.5, 0, 0, 100)  # Normal kurtosis
        se_fat = compute_sharpe_std_error(0.5, 0, 5, 100)  # Fat tails
        assert se_fat > se_normal, "Fat tails should increase SE"


class TestExpectedMaxSharpe:
    """Tests for expected maximum Sharpe calculation."""

    def test_single_trial_zero(self):
        """With 1 trial, expected max should be 0."""
        exp_max = expected_max_sharpe(1, 0.1)
        assert exp_max == 0.0, "Expected max with 1 trial should be 0"

    def test_more_trials_higher_expected_max(self):
        """More trials should increase expected maximum."""
        exp_10 = expected_max_sharpe(10, 0.1)
        exp_100 = expected_max_sharpe(100, 0.1)
        assert exp_100 > exp_10, "More trials should increase expected max"


class TestApplyDSRToTrials:
    """Tests for applying DSR to optimization trials."""

    def test_finds_best_trial(self):
        """Should correctly identify best trial."""
        np.random.seed(42)
        trial_sharpes = [0.3, 0.5, 0.8, 0.4]
        trial_returns = [np.random.normal(0.01, 0.05, 100) for _ in range(4)]

        results = apply_dsr_to_trials(trial_sharpes, trial_returns, T=100)

        assert results['best_trial_idx'] == 2, "Should find trial with 0.8 Sharpe"
        assert results['observed_sharpe'] == 0.8, "Should report best Sharpe"

    def test_returns_required_fields(self):
        """Should return all required fields."""
        np.random.seed(42)
        trial_sharpes = [0.3, 0.5]
        trial_returns = [np.random.normal(0.01, 0.05, 100) for _ in range(2)]

        results = apply_dsr_to_trials(trial_sharpes, trial_returns, T=100)

        required_fields = [
            'best_trial_idx', 'observed_sharpe', 'deflated_sharpe',
            'p_value', 'num_trials', 'is_significant'
        ]
        for field in required_fields:
            assert field in results, f"Missing field: {field}"

    def test_empty_trials_raises(self):
        """Empty trials should raise error."""
        with pytest.raises(ValueError):
            apply_dsr_to_trials([], [], T=100)


class TestMinimumTrackRecordLength:
    """Tests for MinTRL calculation."""

    def test_higher_sharpe_shorter_min_trl(self):
        """Higher Sharpe should require shorter track record."""
        min_trl_low = compute_minimum_track_record_length(0.3)
        min_trl_high = compute_minimum_track_record_length(1.0)

        assert min_trl_high < min_trl_low, "Higher Sharpe needs shorter track record"

    def test_zero_sharpe_infinite_trl(self):
        """Zero Sharpe should require infinite track record."""
        min_trl = compute_minimum_track_record_length(0.0)
        assert min_trl == float('inf'), "Zero Sharpe needs infinite track record"


class TestSummaryReport:
    """Tests for summary report generation."""

    def test_report_contains_key_info(self):
        """Report should contain key information."""
        np.random.seed(42)
        returns = np.random.normal(0.008, 0.04, 100)

        report = dsr_summary_report(0.628, 50, returns, "Test Strategy")

        assert "Test Strategy" in report
        assert "0.628" in report
        assert "50" in report
        assert "P-value" in report
        assert "SIGNIFICANT" in report
