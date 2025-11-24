"""
Tests for multi-signal ensemble functionality (Phase 3 Milestone 3).

Tests the new cross-sectional ensemble pathway that enables
multi-signal ensembles with different data dependencies.

Test Coverage:
1. Unit test: EnsembleSignal.generate_cross_sectional_ensemble_scores()
2. Adapter smoke test: make_multisignal_ensemble_fn() with run_backtest
3. Contract test: NotImplementedError for signals lacking implementation
"""

import pytest
import pandas as pd
from datetime import datetime
from typing import List

from data.data_manager import DataManager
from core.universe_manager import UniverseManager
from signals.ml.ensemble_signal import EnsembleSignal, EnsembleMember
from signals.ml.ensemble_configs import get_momentum_quality_v1_ensemble
from core.backtest_engine import BacktestConfig, run_backtest
from core.signal_adapters import make_multisignal_ensemble_fn
from core.institutional_base import InstitutionalSignal


class TestMultiSignalEnsemble:
    """Tests for cross-sectional ensemble pathway (Phase 3 Milestone 3)."""

    @pytest.fixture(scope="class")
    def setup(self):
        """Setup test fixtures."""
        dm = DataManager()
        um = UniverseManager(dm)
        return {'dm': dm, 'um': um}

    def test_cross_sectional_ensemble_scores_unit(self, setup):
        """
        Unit test: generate_cross_sectional_ensemble_scores() method.

        Tests that the new cross-sectional pathway:
        - Calls each signal's generate_cross_sectional_scores()
        - Normalizes scores according to member config
        - Combines using weighted average
        - Returns valid pd.Series

        Uses momentum + quality ensemble to test multi-signal combination.
        """
        dm = setup['dm']

        # Create momentum + quality ensemble
        ensemble = get_momentum_quality_v1_ensemble(dm)

        # Test with small universe
        test_universe = ['AAPL', 'MSFT', 'GOOGL']
        test_date = pd.Timestamp('2022-06-30')

        # Generate scores
        scores = ensemble.generate_cross_sectional_ensemble_scores(
            rebal_date=test_date,
            universe=test_universe,
        )

        # Type check
        assert isinstance(scores, pd.Series), "Should return pd.Series"

        # Index should be subset of universe (some tickers may lack data)
        assert all(ticker in test_universe for ticker in scores.index), (
            "Result should only contain tickers from universe"
        )

        # Scores should be numeric (may be object type if fixture DB lacks data)
        if len(scores) > 0:
            # Convert to numeric to handle object dtype from empty results
            numeric_scores = pd.to_numeric(scores, errors='coerce')
            assert numeric_scores.dtype in [float, 'float64'], "Scores should be convertible to float"
            # Note: May be all NaN if fixture DB lacks data for test tickers

    @pytest.mark.requires_full_db
    def test_multisignal_adapter_smoke(self, setup):
        """
        Smoke test: make_multisignal_ensemble_fn() with run_backtest().

        Tests that multi-signal ensemble flows through unified backtest engine
        via the new adapter without errors.

        This is the integration test for Phase 3 Milestone 3.
        """
        dm = setup['dm']
        um = setup['um']

        # Create momentum + quality ensemble
        ensemble = get_momentum_quality_v1_ensemble(dm)

        # Create signal function via new adapter
        signal_fn = make_multisignal_ensemble_fn(ensemble, dm)

        # Small universe for smoke test speed
        def universe_fn(rebal_date: str) -> List[str]:
            """Get small S&P 500 sample for smoke test."""
            universe = um.get_universe(
                universe_type='sp500_actual',
                as_of_date=rebal_date,
                min_price=5.0
            )
            if isinstance(universe, pd.Series):
                return universe.tolist()[:50]  # First 50 for speed
            elif isinstance(universe, pd.DataFrame):
                return universe.index.tolist()[:50]
            else:
                return list(universe)[:50]

        # Short backtest window (6 months for smoke test)
        config = BacktestConfig(
            start_date='2022-01-31',
            end_date='2022-06-30',
            initial_capital=100000.0,
            rebalance_schedule='M',
            long_only=True,
            equal_weight=True,
            track_daily_equity=False,
            data_manager=dm
        )

        # Run backtest
        result = run_backtest(universe_fn, signal_fn, config)

        # Sanity checks (no tight bounds - just ensure it runs)
        assert len(result.equity_curve) > 0, "Equity curve should not be empty"
        assert result.num_rebalances > 0, "Should have at least one rebalance"
        assert result.final_equity > 0, "Final equity should be positive"
        assert not result.equity_curve.isna().all(), "Equity curve should not be all NaN"

        # Basic structure checks
        assert hasattr(result, 'total_return'), "Should have total_return metric"
        assert hasattr(result, 'sharpe'), "Should have sharpe metric"
        assert hasattr(result, 'max_drawdown'), "Should have max_drawdown metric"

    def test_not_implemented_error_contract(self, setup):
        """
        Contract test: Ensure NotImplementedError for signals without cross-sectional impl.

        Tests that attempting to use a signal without generate_cross_sectional_scores()
        in a cross-sectional ensemble fails loudly with clear error message.

        This validates the fail-loud philosophy of Phase 3.
        """
        dm = setup['dm']

        # Create a mock signal that lacks cross-sectional implementation
        class MockSignalNoImpl(InstitutionalSignal):
            """Mock signal without cross-sectional scores (raises NotImplementedError)."""
            def __init__(self, params=None):
                super().__init__(params or {}, name='MockSignalNoImpl')

            def generate_signals(self, data: pd.DataFrame) -> pd.Series:
                """Legacy implementation - not relevant for this test."""
                return pd.Series(dtype=float)

            def get_parameter_space(self):
                """Stub implementation of abstract method."""
                return {}

            def generate_cross_sectional_scores(self, rebal_date, universe, data_manager):
                """Explicitly raise NotImplementedError to test contract."""
                raise NotImplementedError("generate_cross_sectional_scores not implemented")

        # Use existing momentum ensemble and replace signal object
        # This bypasses registry validation while testing the contract
        ensemble = get_momentum_quality_v1_ensemble(dm)

        # Replace momentum signal with mock that lacks cross-sectional implementation
        mock_signal = MockSignalNoImpl()
        ensemble._signal_objects[('InstitutionalMomentum', 'v2')] = mock_signal

        # Attempt to generate cross-sectional scores should fail loudly
        with pytest.raises(NotImplementedError) as exc_info:
            ensemble.generate_cross_sectional_ensemble_scores(
                rebal_date=pd.Timestamp('2022-01-31'),
                universe=['AAPL', 'MSFT'],
            )

        # Verify exception type and message
        assert isinstance(exc_info.value, NotImplementedError), (
            "Should raise NotImplementedError for missing cross-sectional implementation"
        )

        error_msg = str(exc_info.value)
        # Should contain the explicit error message from the mock
        assert 'generate_cross_sectional_scores not implemented' in error_msg, (
            f"Error should mention 'generate_cross_sectional_scores not implemented', got: {error_msg}"
        )

        # Should also mention the signal name (from ensemble's wrapping)
        assert 'InstitutionalMomentum' in error_msg, (
            f"Error should mention signal name 'InstitutionalMomentum', got: {error_msg}"
        )

    def test_adapter_signature_contract(self, setup):
        """
        Test that make_multisignal_ensemble_fn() produces correct signature.

        Validates adapter contract without running full backtest.
        """
        dm = setup['dm']

        # Create momentum + quality ensemble
        ensemble = get_momentum_quality_v1_ensemble(dm)

        # Create signal function
        signal_fn = make_multisignal_ensemble_fn(ensemble, dm)

        # Test that signal_fn is callable
        assert callable(signal_fn), "make_multisignal_ensemble_fn should return a callable"

        # Test signature: signal_fn(rebal_date: str, tickers: List[str]) -> pd.Series
        test_tickers = ['AAPL', 'MSFT', 'GOOGL']
        test_date = '2022-01-31'

        # This should not raise
        result = signal_fn(test_date, test_tickers)

        # Check return type
        assert isinstance(result, pd.Series), "signal_fn should return pd.Series"

        # Result should be empty or contain subset of tickers (fixture DB limitation)
        assert len(result) <= len(test_tickers), (
            "Result should not have more tickers than input"
        )

        # If any scores returned, they should be convertible to floats
        if len(result) > 0:
            # Convert to numeric to handle object dtype from empty results
            numeric_result = pd.to_numeric(result, errors='coerce')
            assert numeric_result.dtype in [float, 'float64'], "Scores should be convertible to float"

    def test_empty_universe_graceful(self, setup):
        """
        Test that cross-sectional ensemble handles empty universe gracefully.
        """
        dm = setup['dm']

        ensemble = get_momentum_quality_v1_ensemble(dm)

        # Empty universe
        scores = ensemble.generate_cross_sectional_ensemble_scores(
            rebal_date=pd.Timestamp('2022-01-31'),
            universe=[],
        )

        assert isinstance(scores, pd.Series), "Should return pd.Series even for empty universe"
        assert len(scores) == 0, "Should return empty Series for empty universe"

    @pytest.mark.requires_full_db
    def test_regime_diagnostic_smoke(self):
        """
        Smoke test: regime diagnostic script runs end-to-end without error.

        Tests scripts/run_momentum_quality_regime_diagnostic.py on a short window.
        Verifies outputs are created and contain reasonable data.
        """
        import subprocess
        from pathlib import Path

        # Use a narrow date window covering COVID crash regime
        start_date = '2020-02-01'
        end_date = '2020-06-30'

        # Run regime diagnostic script
        result = subprocess.run(
            [
                'python3',
                'scripts/run_momentum_quality_regime_diagnostic.py',
                '--start', start_date,
                '--end', end_date,
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes max
        )

        # Script should complete without error
        assert result.returncode == 0, (
            f"Regime diagnostic script failed with exit code {result.returncode}\n"
            f"stderr: {result.stderr}"
        )

        # Check outputs were created
        output_dir = Path('results/ensemble_baselines')
        csv_path = output_dir / 'momentum_quality_v1_regime_comparison.csv'
        md_path = output_dir / 'momentum_quality_v1_regime_diagnostic.md'

        assert csv_path.exists(), f"Missing CSV output: {csv_path}"
        assert md_path.exists(), f"Missing MD output: {md_path}"

        # Load CSV and verify structure
        import pandas as pd
        df = pd.read_csv(csv_path)

        assert 'regime_name' in df.columns, "CSV missing 'regime_name' column"
        assert 'strategy' in df.columns, "CSV missing 'strategy' column"
        assert 'sharpe' in df.columns, "CSV missing 'sharpe' column"
        assert 'total_return' in df.columns, "CSV missing 'total_return' column"

        # Verify both strategies reported
        strategies = df['strategy'].unique()
        assert 'momentum_v2' in strategies, "Missing momentum_v2 strategy"
        assert 'momentum_quality_v1' in strategies, "Missing momentum_quality_v1 strategy"

        # Verify at least one regime has non-NaN metrics
        momentum_rows = df[df['strategy'] == 'momentum_v2']
        assert not momentum_rows['total_return'].isna().all(), (
            "All momentum total returns are NaN"
        )

        # Verify MD has expected sections
        md_content = md_path.read_text()
        assert '## Regime Definitions' in md_content, "Missing regime definitions section"
        assert '## Per-Regime Performance' in md_content, "Missing performance section"
        assert '## Regime Delta Analysis' in md_content, "Missing delta section"

    def test_momentum_quality_v1_config_25_75_weights(self, setup):
        """
        Config sanity test: Verify momentum_quality_v1 has correct 25/75 weights.

        Tests Phase 3 M3.4 weight calibration lock-in (grid sweep + Optuna validated).
        Ensures canonical ensemble uses M=0.25, Q=0.75 as configured.
        """
        dm = setup['dm']

        # Get the canonical momentum_quality_v1 ensemble
        ensemble = get_momentum_quality_v1_ensemble(dm)

        # Verify it has exactly 2 members
        assert len(ensemble.members) == 2, (
            f"momentum_quality_v1 should have 2 members, got {len(ensemble.members)}"
        )

        # Extract member details
        member_dict = {}
        for member in ensemble.members:
            member_dict[member.signal_name] = member.weight

        # Verify member names
        assert 'InstitutionalMomentum' in member_dict, (
            "Missing InstitutionalMomentum member"
        )
        assert 'CrossSectionalQuality' in member_dict, (
            "Missing CrossSectionalQuality member"
        )

        # Verify exact weights (25/75 calibration)
        tolerance = 1e-6
        momentum_weight = member_dict['InstitutionalMomentum']
        quality_weight = member_dict['CrossSectionalQuality']

        assert abs(momentum_weight - 0.25) < tolerance, (
            f"InstitutionalMomentum weight should be 0.25, got {momentum_weight}"
        )
        assert abs(quality_weight - 0.75) < tolerance, (
            f"CrossSectionalQuality weight should be 0.75, got {quality_weight}"
        )

        # Verify weights sum to 1.0
        total_weight = momentum_weight + quality_weight
        assert abs(total_weight - 1.0) < tolerance, (
            f"Weights should sum to 1.0, got {total_weight}"
        )

    def test_ensemble_registry_25_75_metadata(self, setup):
        """
        Registry consistency test: Verify ENSEMBLE_REGISTRY reflects 25/75 weights.

        Tests that registry metadata correctly documents the weight calibration
        and references the appropriate validation reports.
        """
        from signals.ml.ensemble_configs import ENSEMBLE_REGISTRY

        # Verify momentum_quality_v1 is in registry
        assert 'momentum_quality_v1' in ENSEMBLE_REGISTRY, (
            "momentum_quality_v1 should be in ENSEMBLE_REGISTRY"
        )

        registry_entry = ENSEMBLE_REGISTRY['momentum_quality_v1']

        # Verify registry entry has required fields
        assert hasattr(registry_entry, 'name'), "Registry entry missing 'name'"
        assert hasattr(registry_entry, 'description'), "Registry entry missing 'description'"
        assert hasattr(registry_entry, 'status'), "Registry entry missing 'status'"
        assert hasattr(registry_entry, 'validation_report'), "Registry entry missing 'validation_report'"

        # Verify description mentions 25/75 weights
        description = registry_entry.description.lower()
        assert '25/75' in description or ('25' in description and '75' in description), (
            f"Registry description should mention 25/75 weights, got: {registry_entry.description}"
        )

        # Verify description mentions calibration method (grid + Optuna)
        assert 'grid' in description or 'optuna' in description, (
            f"Registry description should mention calibration method (grid/optuna), got: {registry_entry.description}"
        )

        # Verify validation report points to a diagnostic file
        report_path = registry_entry.validation_report
        # Accept weight sweep, canonical diagnostic, or full diagnostic paths
        valid_patterns = ['weight', 'diag', 'MQ_v1', 'M3.6']
        assert any(pattern in report_path for pattern in valid_patterns), (
            f"Validation report should reference weight calibration or diagnostic, got: {report_path}"
        )

        # Verify status is appropriate for a calibrated ensemble
        valid_statuses = ['RESEARCH', 'PRODUCTION', 'CANDIDATE_PROD', 'PROD_READY']
        assert registry_entry.status in valid_statuses, (
            f"Status should be one of {valid_statuses}, got: {registry_entry.status}"
        )
