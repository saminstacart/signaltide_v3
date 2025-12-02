#!/usr/bin/env python3
"""
Quality Control (QC) Script for SignalTide v4.

Runs comprehensive checks at multiple levels:
- Level 1: Import Tests
- Level 2: Dependency Chain
- Level 3: Data Infrastructure
- Level 4: Signal Smoke Tests
- Level 5: Integration Smoke Test

Usage:
    python -m signaltide_v4.scripts.qc_check
    python signaltide_v4/scripts/qc_check.py
"""

import os
import sys
import importlib
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Add parent directories to path
script_dir = Path(__file__).parent
v4_dir = script_dir.parent
project_dir = v4_dir.parent
sys.path.insert(0, str(project_dir))
sys.path.insert(0, str(v4_dir.parent))

import warnings
warnings.filterwarnings('ignore')


@dataclass
class CheckResult:
    """Result of a single check."""
    name: str
    level: int
    passed: bool
    message: str = ""
    error: Optional[str] = None
    skipped: bool = False


@dataclass
class QCReport:
    """Complete QC report."""
    results: List[CheckResult] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    def add(self, result: CheckResult):
        self.results.append(result)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed and not r.skipped)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed and not r.skipped)

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.skipped)

    @property
    def is_ready(self) -> bool:
        # Ready if no critical failures (Levels 1-2 must pass)
        critical_failures = sum(
            1 for r in self.results
            if not r.passed and not r.skipped and r.level <= 2
        )
        return critical_failures == 0

    def print_report(self):
        """Print formatted report."""
        self.end_time = datetime.now()

        print("\n" + "=" * 80)
        print("SIGNALTIDE V4 QUALITY CONTROL REPORT")
        print("=" * 80)
        print(f"Started:  {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Finished: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {(self.end_time - self.start_time).total_seconds():.2f}s")
        print()

        # Results by level
        for level in range(1, 6):
            level_results = [r for r in self.results if r.level == level]
            if not level_results:
                continue

            level_names = {
                1: "Import Tests",
                2: "Dependency Chain",
                3: "Data Infrastructure",
                4: "Signal Smoke Tests",
                5: "Integration Smoke Test",
            }

            print(f"\n{'─' * 80}")
            print(f"LEVEL {level}: {level_names[level]}")
            print(f"{'─' * 80}")

            passed = sum(1 for r in level_results if r.passed and not r.skipped)
            failed = sum(1 for r in level_results if not r.passed and not r.skipped)
            skipped = sum(1 for r in level_results if r.skipped)

            print(f"Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
            print()

            for r in level_results:
                if r.skipped:
                    status = "⊘ SKIP"
                elif r.passed:
                    status = "✓ PASS"
                else:
                    status = "✗ FAIL"

                print(f"  {status}  {r.name}")
                if r.message:
                    print(f"         {r.message}")
                if r.error and not r.passed:
                    # Truncate long errors
                    error_lines = r.error.strip().split('\n')
                    if len(error_lines) > 3:
                        error_lines = error_lines[-3:]
                    for line in error_lines:
                        print(f"         ERROR: {line[:70]}")

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total Checks: {self.total}")
        print(f"Passed:       {self.passed}")
        print(f"Failed:       {self.failed}")
        print(f"Skipped:      {self.skipped}")
        print()

        if self.is_ready:
            print("╔═══════════════════════════════════════════════════════════════════════════╗")
            print("║                          VERDICT: READY                                   ║")
            print("║                    SignalTide v4 passed critical checks                   ║")
            print("╚═══════════════════════════════════════════════════════════════════════════╝")
        else:
            print("╔═══════════════════════════════════════════════════════════════════════════╗")
            print("║                        VERDICT: NOT READY                                 ║")
            print("║                   Critical failures detected (Levels 1-2)                 ║")
            print("╚═══════════════════════════════════════════════════════════════════════════╝")

        print()


class QCChecker:
    """Quality control checker for SignalTide v4."""

    def __init__(self):
        self.report = QCReport()
        self.v4_path = Path(__file__).parent.parent

    def run_all(self):
        """Run all QC levels."""
        print("\nRunning SignalTide v4 Quality Control...\n")

        self.level1_import_tests()
        self.level2_dependency_chain()
        self.level3_data_infrastructure()
        self.level4_signal_smoke_tests()
        self.level5_integration_test()

        self.report.print_report()

        return self.report.is_ready

    # =========================================================================
    # LEVEL 1: Import Tests
    # =========================================================================

    def level1_import_tests(self):
        """Test that all .py files can be imported."""
        print("Level 1: Import Tests...")

        # Find all Python files
        py_files = list(self.v4_path.rglob("*.py"))

        for py_file in py_files:
            # Skip __pycache__
            if "__pycache__" in str(py_file):
                continue

            # Convert path to module name
            rel_path = py_file.relative_to(self.v4_path.parent)
            module_name = str(rel_path).replace("/", ".").replace("\\", ".")[:-3]

            # Skip __init__ for cleaner output
            display_name = module_name.replace("signaltide_v4.", "")

            try:
                importlib.import_module(module_name)
                self.report.add(CheckResult(
                    name=f"Import {display_name}",
                    level=1,
                    passed=True,
                    message="OK",
                ))
            except Exception as e:
                self.report.add(CheckResult(
                    name=f"Import {display_name}",
                    level=1,
                    passed=False,
                    error=str(e),
                ))

    # =========================================================================
    # LEVEL 2: Dependency Chain
    # =========================================================================

    def level2_dependency_chain(self):
        """Test key import chains and class existence."""
        print("Level 2: Dependency Chain...")

        checks = [
            ("config.settings", "get_settings", "function"),
            ("data.base", "PITDataManager", "class"),
            ("data.market_data", "MarketDataProvider", "class"),
            ("data.factor_data", "FactorDataProvider", "class"),
            ("data.fundamental_data", "FundamentalDataProvider", "class"),
            ("data.sharadar_adapter", "SharadarAdapter", "class"),
            ("data.integration", "DataIntegration", "class"),
            ("signals.base", "BaseSignal", "class"),
            ("signals.residual_momentum", "ResidualMomentumSignal", "class"),
            ("signals.quality", "QualitySignal", "class"),
            ("signals.insider", "OpportunisticInsiderSignal", "class"),
            ("portfolio.scoring", "SignalAggregator", "class"),
            ("portfolio.construction", "PortfolioConstructor", "class"),
            ("backtest.engine", "BacktestEngine", "class"),
            ("backtest.metrics", "MetricsCalculator", "class"),
            ("backtest.costs", "EnhancedCostModel", "class"),
            ("validation.deflated_sharpe", "DeflatedSharpeCalculator", "class"),
            ("validation.walk_forward", "WalkForwardValidator", "class"),
            ("validation.factor_attribution", "FactorAttributor", "class"),
        ]

        for module_suffix, obj_name, obj_type in checks:
            module_name = f"signaltide_v4.{module_suffix}"
            display_name = f"{module_suffix}.{obj_name}"

            try:
                module = importlib.import_module(module_name)
                obj = getattr(module, obj_name, None)

                if obj is None:
                    self.report.add(CheckResult(
                        name=display_name,
                        level=2,
                        passed=False,
                        error=f"{obj_name} not found in {module_name}",
                    ))
                elif obj_type == "class" and not isinstance(obj, type):
                    self.report.add(CheckResult(
                        name=display_name,
                        level=2,
                        passed=False,
                        error=f"{obj_name} is not a class",
                    ))
                elif obj_type == "function" and not callable(obj):
                    self.report.add(CheckResult(
                        name=display_name,
                        level=2,
                        passed=False,
                        error=f"{obj_name} is not callable",
                    ))
                else:
                    self.report.add(CheckResult(
                        name=display_name,
                        level=2,
                        passed=True,
                        message=f"{obj_type} exists",
                    ))

            except Exception as e:
                self.report.add(CheckResult(
                    name=display_name,
                    level=2,
                    passed=False,
                    error=str(e),
                ))

    # =========================================================================
    # LEVEL 3: Data Infrastructure
    # =========================================================================

    def level3_data_infrastructure(self):
        """Test data connections."""
        print("Level 3: Data Infrastructure...")

        # Check 1: Settings load
        try:
            from signaltide_v4.config.settings import get_settings
            settings = get_settings()
            self.report.add(CheckResult(
                name="Settings load",
                level=3,
                passed=True,
                message=f"capital=${settings.initial_capital:,.0f}",
            ))
        except Exception as e:
            self.report.add(CheckResult(
                name="Settings load",
                level=3,
                passed=False,
                error=str(e),
            ))

        # Check 2: Sharadar adapter connection
        try:
            from signaltide_v4.data.sharadar_adapter import SharadarAdapter

            # Check if database path exists
            db_path = os.environ.get(
                'SIGNALTIDE_DB_PATH',
                '/Users/samuelksherman/signaltide/data/signaltide.db'
            )

            if os.path.exists(db_path):
                adapter = SharadarAdapter(db_path=db_path)
                adapter.close()
                self.report.add(CheckResult(
                    name="Sharadar DB connection",
                    level=3,
                    passed=True,
                    message=f"Connected to {os.path.basename(db_path)}",
                ))
            else:
                self.report.add(CheckResult(
                    name="Sharadar DB connection",
                    level=3,
                    passed=False,
                    skipped=True,
                    message=f"DB not found at {db_path}",
                ))
        except Exception as e:
            self.report.add(CheckResult(
                name="Sharadar DB connection",
                level=3,
                passed=False,
                error=str(e),
            ))

        # Check 3: Get universe
        try:
            from signaltide_v4.data.sharadar_adapter import SharadarAdapter

            db_path = os.environ.get(
                'SIGNALTIDE_DB_PATH',
                '/Users/samuelksherman/signaltide/data/signaltide.db'
            )

            if os.path.exists(db_path):
                adapter = SharadarAdapter(db_path=db_path)
                tickers = adapter.get_sp500_constituents()
                adapter.close()

                if len(tickers) > 0:
                    self.report.add(CheckResult(
                        name="Get universe",
                        level=3,
                        passed=True,
                        message=f"{len(tickers)} tickers",
                    ))
                else:
                    self.report.add(CheckResult(
                        name="Get universe",
                        level=3,
                        passed=False,
                        message="Empty universe returned",
                    ))
            else:
                self.report.add(CheckResult(
                    name="Get universe",
                    level=3,
                    passed=False,
                    skipped=True,
                    message="DB not available",
                ))
        except Exception as e:
            self.report.add(CheckResult(
                name="Get universe",
                level=3,
                passed=False,
                error=str(e),
            ))

        # Check 4: Get price data
        try:
            from signaltide_v4.data.sharadar_adapter import SharadarAdapter

            db_path = os.environ.get(
                'SIGNALTIDE_DB_PATH',
                '/Users/samuelksherman/signaltide/data/signaltide.db'
            )

            if os.path.exists(db_path):
                adapter = SharadarAdapter(db_path=db_path)
                prices = adapter.get_prices(
                    ['AAPL', 'MSFT'],
                    '2024-01-01',
                    '2024-01-31',
                )
                adapter.close()

                if not prices.empty:
                    self.report.add(CheckResult(
                        name="Get price data",
                        level=3,
                        passed=True,
                        message=f"{len(prices)} rows",
                    ))
                else:
                    self.report.add(CheckResult(
                        name="Get price data",
                        level=3,
                        passed=False,
                        message="Empty price data returned",
                    ))
            else:
                self.report.add(CheckResult(
                    name="Get price data",
                    level=3,
                    passed=False,
                    skipped=True,
                    message="DB not available",
                ))
        except Exception as e:
            self.report.add(CheckResult(
                name="Get price data",
                level=3,
                passed=False,
                error=str(e),
            ))

        # Check 5: Factor data provider instantiation
        try:
            from signaltide_v4.data.factor_data import FactorDataProvider
            provider = FactorDataProvider()
            self.report.add(CheckResult(
                name="Factor data provider",
                level=3,
                passed=True,
                message="Instantiated OK",
            ))
        except Exception as e:
            self.report.add(CheckResult(
                name="Factor data provider",
                level=3,
                passed=False,
                error=str(e),
            ))

    # =========================================================================
    # LEVEL 4: Signal Smoke Tests
    # =========================================================================

    def level4_signal_smoke_tests(self):
        """Test signal instantiation and basic operation."""
        print("Level 4: Signal Smoke Tests...")

        test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        test_date = '2024-01-15'

        signals_to_test = [
            ("ResidualMomentumSignal", "signaltide_v4.signals.residual_momentum"),
            ("QualitySignal", "signaltide_v4.signals.quality"),
            ("OpportunisticInsiderSignal", "signaltide_v4.signals.insider"),
            ("ToneChangeSignal", "signaltide_v4.signals.tone_change"),
        ]

        for signal_name, module_name in signals_to_test:
            # Test instantiation
            try:
                module = importlib.import_module(module_name)
                signal_class = getattr(module, signal_name)
                signal = signal_class()

                self.report.add(CheckResult(
                    name=f"{signal_name} instantiation",
                    level=4,
                    passed=True,
                    message="Created OK",
                ))

                # Test generate_signals method exists (the standard signal API)
                if hasattr(signal, 'generate_signals'):
                    self.report.add(CheckResult(
                        name=f"{signal_name}.generate_signals exists",
                        level=4,
                        passed=True,
                        message="Method found",
                    ))
                else:
                    self.report.add(CheckResult(
                        name=f"{signal_name}.generate_signals exists",
                        level=4,
                        passed=False,
                        error="generate_signals method not found",
                    ))

            except Exception as e:
                self.report.add(CheckResult(
                    name=f"{signal_name} instantiation",
                    level=4,
                    passed=False,
                    error=str(e),
                ))

    # =========================================================================
    # LEVEL 5: Integration Smoke Test
    # =========================================================================

    def level5_integration_test(self):
        """Run minimal end-to-end test."""
        print("Level 5: Integration Smoke Test...")

        # Check 1: Portfolio constructor
        try:
            from signaltide_v4.portfolio.construction import PortfolioConstructor
            import pandas as pd

            constructor = PortfolioConstructor()

            # Create mock scores
            mock_scores = pd.Series({
                'AAPL': 0.8,
                'MSFT': 0.6,
                'GOOGL': 0.4,
                'AMZN': 0.2,
                'META': 0.1,
            })

            self.report.add(CheckResult(
                name="PortfolioConstructor instantiation",
                level=5,
                passed=True,
                message="Created OK",
            ))

        except Exception as e:
            self.report.add(CheckResult(
                name="PortfolioConstructor instantiation",
                level=5,
                passed=False,
                error=str(e),
            ))

        # Check 2: Signal aggregator
        try:
            from signaltide_v4.portfolio.scoring import SignalAggregator

            aggregator = SignalAggregator()

            self.report.add(CheckResult(
                name="SignalAggregator instantiation",
                level=5,
                passed=True,
                message="Created OK",
            ))

        except Exception as e:
            self.report.add(CheckResult(
                name="SignalAggregator instantiation",
                level=5,
                passed=False,
                error=str(e),
            ))

        # Check 3: Backtest engine
        try:
            from signaltide_v4.backtest.engine import BacktestEngine

            # Just test instantiation with minimal params
            engine = BacktestEngine(initial_capital=50000)

            self.report.add(CheckResult(
                name="BacktestEngine instantiation",
                level=5,
                passed=True,
                message="Created OK",
            ))

        except Exception as e:
            self.report.add(CheckResult(
                name="BacktestEngine instantiation",
                level=5,
                passed=False,
                error=str(e),
            ))

        # Check 4: Metrics calculator with mock data
        try:
            from signaltide_v4.backtest.metrics import MetricsCalculator
            import pandas as pd
            import numpy as np

            calculator = MetricsCalculator()

            # Create mock returns
            np.random.seed(42)
            mock_returns = pd.Series(
                np.random.normal(0.0003, 0.01, 252),
                index=pd.date_range('2024-01-01', periods=252, freq='B'),
            )

            metrics = calculator.calculate_metrics(mock_returns)

            if metrics.sharpe_ratio is not None:
                self.report.add(CheckResult(
                    name="MetricsCalculator.calculate_metrics",
                    level=5,
                    passed=True,
                    message=f"Sharpe={metrics.sharpe_ratio:.3f}",
                ))
            else:
                self.report.add(CheckResult(
                    name="MetricsCalculator.calculate_metrics",
                    level=5,
                    passed=False,
                    error="Sharpe is None",
                ))

        except Exception as e:
            self.report.add(CheckResult(
                name="MetricsCalculator.calculate_metrics",
                level=5,
                passed=False,
                error=str(e),
            ))

        # Check 5: Validation modules
        try:
            from signaltide_v4.validation.deflated_sharpe import DeflatedSharpeCalculator
            import pandas as pd
            import numpy as np

            calculator = DeflatedSharpeCalculator(n_trials=50)

            np.random.seed(42)
            mock_returns = pd.Series(
                np.random.normal(0.0005, 0.01, 252),
                index=pd.date_range('2024-01-01', periods=252, freq='B'),
            )

            result = calculator.calculate(mock_returns)

            if result.deflated_sharpe is not None:
                self.report.add(CheckResult(
                    name="DeflatedSharpeCalculator.calculate",
                    level=5,
                    passed=True,
                    message=f"DSR={result.deflated_sharpe:.3f}",
                ))
            else:
                self.report.add(CheckResult(
                    name="DeflatedSharpeCalculator.calculate",
                    level=5,
                    passed=False,
                    error="Deflated Sharpe is None",
                ))

        except Exception as e:
            self.report.add(CheckResult(
                name="DeflatedSharpeCalculator.calculate",
                level=5,
                passed=False,
                error=str(e),
            ))

        # Check 6: Walk-forward validator
        try:
            from signaltide_v4.validation.walk_forward import WalkForwardValidator
            import pandas as pd
            import numpy as np

            validator = WalkForwardValidator(
                train_months=12,
                test_months=3,
                min_folds=2,
            )

            # Create 3 years of mock returns
            np.random.seed(42)
            mock_returns = pd.Series(
                np.random.normal(0.0003, 0.01, 756),
                index=pd.date_range('2021-01-01', periods=756, freq='B'),
            )

            result = validator.validate_returns(mock_returns)

            if result.n_folds > 0:
                self.report.add(CheckResult(
                    name="WalkForwardValidator.validate_returns",
                    level=5,
                    passed=True,
                    message=f"{result.n_folds} folds, valid={result.is_valid}",
                ))
            else:
                self.report.add(CheckResult(
                    name="WalkForwardValidator.validate_returns",
                    level=5,
                    passed=False,
                    error="No folds generated",
                ))

        except Exception as e:
            self.report.add(CheckResult(
                name="WalkForwardValidator.validate_returns",
                level=5,
                passed=False,
                error=str(e),
            ))


def main():
    """Main entry point."""
    checker = QCChecker()
    is_ready = checker.run_all()

    # Return appropriate exit code
    return 0 if is_ready else 1


if __name__ == '__main__':
    sys.exit(main())
