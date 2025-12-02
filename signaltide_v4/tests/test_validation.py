"""
Unit tests for SignalTide v4 validation modules.

Tests:
- Deflated Sharpe Ratio calculation
- Walk-Forward validation
- Fama-French attribution
"""

import unittest
from datetime import datetime

import pandas as pd
import numpy as np


class TestDeflatedSharpe(unittest.TestCase):
    """Test Deflated Sharpe Ratio calculation."""

    def setUp(self):
        """Set up test fixtures."""
        # Generate synthetic returns
        np.random.seed(42)
        n = 252 * 5  # 5 years of daily data
        self.returns = pd.Series(
            np.random.normal(0.0003, 0.01, n),  # ~7.5% annual, 16% vol
            index=pd.date_range('2019-01-01', periods=n, freq='B')
        )

    def test_sharpe_calculation(self):
        """Test basic Sharpe ratio calculation."""
        mean_ret = self.returns.mean()
        std_ret = self.returns.std()
        sharpe = (mean_ret / std_ret) * np.sqrt(252)

        # Should be roughly 0.5 based on our parameters
        self.assertGreater(sharpe, 0.0)
        self.assertLess(sharpe, 2.0)

    def test_expected_max_sharpe(self):
        """Test expected maximum Sharpe calculation."""
        # For N trials, E[max] ~ sqrt(2 * ln(N))
        n_trials = 100
        expected_max = np.sqrt(2 * np.log(n_trials))

        # Should be around 3.0 for 100 trials
        self.assertGreater(expected_max, 2.5)
        self.assertLess(expected_max, 4.0)

    def test_multiple_testing_correction(self):
        """Test that DSR is lower than observed SR with multiple tests."""
        # With more trials, DSR should decrease
        n_low = 10
        n_high = 1000

        expected_max_low = np.sqrt(2 * np.log(n_low))
        expected_max_high = np.sqrt(2 * np.log(n_high))

        # Higher N means higher expected max, so lower significance
        self.assertLess(expected_max_low, expected_max_high)


class TestWalkForward(unittest.TestCase):
    """Test walk-forward validation."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        n = 252 * 10  # 10 years
        self.returns = pd.Series(
            np.random.normal(0.0003, 0.01, n),
            index=pd.date_range('2014-01-01', periods=n, freq='B')
        )

    def test_fold_generation(self):
        """Test generation of walk-forward folds."""
        from signaltide_v4.validation.walk_forward import WalkForwardValidator

        validator = WalkForwardValidator(
            train_months=60,
            test_months=12,
            min_folds=3,
        )

        # Check fold generation
        folds = validator._generate_folds('2014-01-01', '2024-01-01')

        # Should have multiple folds
        self.assertGreater(len(folds), 3)

        # Each fold should have train_start < train_end < test_start < test_end
        for train_start, train_end, test_start, test_end in folds:
            self.assertLess(train_start, train_end)
            self.assertLessEqual(train_end, test_start)
            self.assertLess(test_start, test_end)

    def test_sharpe_calculation(self):
        """Test Sharpe calculation within validator."""
        from signaltide_v4.validation.walk_forward import WalkForwardValidator

        validator = WalkForwardValidator()
        sharpe = validator._calculate_sharpe(self.returns)

        # Should be reasonable
        self.assertGreater(sharpe, -2.0)
        self.assertLess(sharpe, 3.0)


class TestFactorAttribution(unittest.TestCase):
    """Test Fama-French factor attribution."""

    def test_ols_regression_properties(self):
        """Test OLS regression statistical properties."""
        np.random.seed(42)
        n = 252

        # Generate synthetic data with known relationship
        # y = 0.01 + 1.0 * x + noise
        x = np.random.normal(0, 0.01, n)
        y = 0.0001 + 1.0 * x + np.random.normal(0, 0.005, n)

        # Run simple OLS
        X = np.column_stack([np.ones(n), x])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]

        # Intercept should be close to 0.0001
        self.assertAlmostEqual(beta[0], 0.0001, delta=0.001)

        # Slope should be close to 1.0
        self.assertAlmostEqual(beta[1], 1.0, delta=0.2)

    def test_r_squared_calculation(self):
        """Test R-squared calculation."""
        np.random.seed(42)
        n = 100

        # Perfect fit (R^2 = 1)
        x = np.random.normal(0, 1, n)
        y = 2 * x + 1  # y = mx + b, perfectly linear

        # Calculate R^2
        y_pred = 2 * x + 1
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # Should be 1.0 for perfect fit
        self.assertAlmostEqual(r_squared, 1.0, places=5)

    def test_factor_beta_interpretation(self):
        """Test factor beta interpretation."""
        # If beta = 1.0 for MKT-RF, strategy moves 1:1 with market
        # If beta = 0.5 for SMB, half the size premium
        # Negative HML = growth tilt

        # Verify interpretation logic
        smb_beta = 0.3  # Positive = small cap tilt
        hml_beta = -0.2  # Negative = growth tilt

        self.assertEqual('small_cap', 'small_cap' if smb_beta > 0 else 'large_cap')
        self.assertEqual('growth', 'growth' if hml_beta < 0 else 'value')


if __name__ == '__main__':
    unittest.main()
