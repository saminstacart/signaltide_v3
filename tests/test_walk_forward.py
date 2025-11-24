"""
Unit tests for Walk-Forward Validation Framework.
"""

import pytest
import numpy as np

from core.walk_forward import (
    generate_expanding_window_folds,
    generate_rolling_window_folds,
    run_walk_forward_validation,
    WalkForwardFold,
    WalkForwardResult,
    analyze_walk_forward_stability
)


class TestFoldGeneration:
    """Tests for fold generation functions."""

    def test_expanding_window_basic(self):
        """Test basic expanding window fold generation."""
        folds = generate_expanding_window_folds(
            '2015-01-01', '2024-12-31',
            min_train_years=3, test_years=1, step_years=1
        )

        assert len(folds) == 7, f"Expected 7 folds, got {len(folds)}"

        # First fold should be 2015-2017 train, 2018 test
        assert folds[0][0] == '2015-01-01'  # train_start
        assert folds[0][1] == '2017-12-31'  # train_end
        assert folds[0][2] == '2018-01-01'  # test_start
        assert folds[0][3] == '2018-12-31'  # test_end

    def test_expanding_window_expands(self):
        """Test that training window actually expands."""
        folds = generate_expanding_window_folds(
            '2015-01-01', '2024-12-31',
            min_train_years=3, test_years=1, step_years=1
        )

        # All folds should start at 2015
        for fold in folds:
            assert fold[0] == '2015-01-01', "All folds should start at same date"

        # Training end should increase
        for i in range(1, len(folds)):
            prev_year = int(folds[i-1][1][:4])
            curr_year = int(folds[i][1][:4])
            assert curr_year > prev_year, "Training window should expand"

    def test_rolling_window_basic(self):
        """Test basic rolling window fold generation."""
        folds = generate_rolling_window_folds(
            '2015-01-01', '2024-12-31',
            train_years=3, test_years=1
        )

        assert len(folds) > 0, "Should generate at least one fold"

        # Training window should be exactly 3 years
        train_start_year = int(folds[0][0][:4])
        train_end_year = int(folds[0][1][:4])
        assert train_end_year - train_start_year == 2, "Training should span 3 years"

    def test_rolling_window_slides(self):
        """Test that rolling window slides forward."""
        folds = generate_rolling_window_folds(
            '2015-01-01', '2024-12-31',
            train_years=3, test_years=1
        )

        # Each fold should start one year later
        for i in range(1, len(folds)):
            prev_start = int(folds[i-1][0][:4])
            curr_start = int(folds[i][0][:4])
            assert curr_start == prev_start + 1, "Rolling window should slide by 1 year"

    def test_empty_folds_for_short_period(self):
        """Short period should produce fewer folds."""
        folds = generate_expanding_window_folds(
            '2020-01-01', '2022-12-31',
            min_train_years=3, test_years=1, step_years=1
        )

        # 3 years of data with 3 year min train = 0 folds possible
        assert len(folds) == 0, "Should produce no folds for too-short period"


class TestWalkForwardResult:
    """Tests for WalkForwardResult class."""

    def test_is_robust_positive_case(self):
        """Test robustness check with good results."""
        folds = [
            WalkForwardFold(1, '2015-01-01', '2017-12-31', '2018-01-01', '2018-12-31',
                           train_sharpe=0.8, test_sharpe=0.5),
            WalkForwardFold(2, '2015-01-01', '2018-12-31', '2019-01-01', '2019-12-31',
                           train_sharpe=0.9, test_sharpe=0.6),
        ]

        result = WalkForwardResult(
            folds=folds,
            mean_test_sharpe=0.55,
            std_test_sharpe=0.05,
            mean_train_sharpe=0.85,
            test_train_ratio=0.65,
            all_folds_positive=True,
            num_positive_folds=2,
            num_folds=2
        )

        assert result.is_robust(), "Should be robust with good metrics"

    def test_is_robust_negative_case(self):
        """Test robustness check with poor results."""
        result = WalkForwardResult(
            folds=[],
            mean_test_sharpe=-0.1,
            std_test_sharpe=0.5,
            mean_train_sharpe=0.8,
            test_train_ratio=-0.125,
            all_folds_positive=False,
            num_positive_folds=1,
            num_folds=3
        )

        assert not result.is_robust(), "Should not be robust with negative mean Sharpe"

    def test_summary_output(self):
        """Test that summary generates valid output."""
        result = WalkForwardResult(
            folds=[],
            mean_test_sharpe=0.5,
            std_test_sharpe=0.1,
            mean_train_sharpe=0.8,
            test_train_ratio=0.625,
            all_folds_positive=True,
            num_positive_folds=3,
            num_folds=3
        )

        summary = result.summary()
        assert "Walk-Forward" in summary
        assert "0.5" in summary  # mean test sharpe
        assert "0.8" in summary  # mean train sharpe


class TestRunWalkForward:
    """Tests for walk-forward validation execution."""

    def test_mock_validation_runs(self):
        """Test that validation completes with mock functions."""
        folds = [
            ('2015-01-01', '2017-12-31', '2018-01-01', '2018-12-31'),
            ('2015-01-01', '2018-12-31', '2019-01-01', '2019-12-31'),
        ]

        def mock_optimize(train_start, train_end, n_trials):
            return {'param': 0.5}, 0.7

        def mock_evaluate(test_start, test_end, params):
            return 0.5, 0.1

        result = run_walk_forward_validation(
            mock_optimize, mock_evaluate, folds, n_trials_per_fold=5
        )

        assert result.num_folds == 2
        assert result.mean_train_sharpe == 0.7
        assert result.mean_test_sharpe == 0.5

    def test_handles_variable_performance(self):
        """Test with variable train/test performance."""
        folds = [
            ('2015-01-01', '2017-12-31', '2018-01-01', '2018-12-31'),
            ('2015-01-01', '2018-12-31', '2019-01-01', '2019-12-31'),
            ('2015-01-01', '2019-12-31', '2020-01-01', '2020-12-31'),
        ]

        train_sharpes = [0.6, 0.8, 0.7]
        test_sharpes = [0.4, 0.6, 0.3]
        idx = [0]

        def mock_optimize(train_start, train_end, n_trials):
            sharpe = train_sharpes[idx[0]]
            return {'param': 0.5}, sharpe

        def mock_evaluate(test_start, test_end, params):
            sharpe = test_sharpes[idx[0]]
            idx[0] += 1
            return sharpe, 0.1

        result = run_walk_forward_validation(
            mock_optimize, mock_evaluate, folds, n_trials_per_fold=5
        )

        assert result.num_folds == 3
        assert result.num_positive_folds == 3
        assert abs(result.mean_test_sharpe - np.mean(test_sharpes)) < 0.01


class TestStabilityAnalysis:
    """Tests for stability analysis."""

    def test_stable_results(self):
        """Test stability analysis with consistent results."""
        folds = [
            WalkForwardFold(i, '', '', '', '', train_sharpe=0.8, test_sharpe=0.5 + 0.02*i)
            for i in range(5)
        ]

        result = WalkForwardResult(
            folds=folds,
            mean_test_sharpe=0.55,
            std_test_sharpe=0.03,
            mean_train_sharpe=0.8,
            test_train_ratio=0.69,
            all_folds_positive=True,
            num_positive_folds=5,
            num_folds=5
        )

        stability = analyze_walk_forward_stability(result)

        assert 'coefficient_of_variation' in stability
        assert stability['num_folds'] == 5

    def test_insufficient_folds(self):
        """Test with too few folds."""
        folds = [WalkForwardFold(1, '', '', '', '', train_sharpe=0.8, test_sharpe=0.5)]

        result = WalkForwardResult(
            folds=folds,
            mean_test_sharpe=0.5,
            std_test_sharpe=0,
            mean_train_sharpe=0.8,
            test_train_ratio=0.625,
            all_folds_positive=True,
            num_positive_folds=1,
            num_folds=1
        )

        stability = analyze_walk_forward_stability(result)
        assert 'error' in stability, "Should return error for single fold"
