"""
Signal Registry - Centralized tracking of signal validation status.

This module provides programmatic access to the signal registry documented in
docs/core/SIGNAL_REGISTRY.md. It tracks which signals have passed Phase 1
validation and are ready for ensemble construction.

Usage:
    from core.signal_registry import get_signal_status, list_signals, list_go_signals

    # Get specific signal
    momentum = get_signal_status('InstitutionalMomentum', 'v2')
    print(f"Status: {momentum.status}, OOS Sharpe: {momentum.oos_sharpe}")

    # List all GO signals
    go_signals = list_go_signals()
"""

from dataclasses import dataclass
from typing import List, Optional, Literal


@dataclass
class SignalStatus:
    """
    Status record for a signal that has completed Phase 1 validation.

    Attributes:
        signal_name: Name of the signal (e.g., 'InstitutionalMomentum')
        version: Version identifier (e.g., 'v2', 'Trial 11')
        universe: Universe used for validation (e.g., 'sp500_actual')
        status: Validation outcome ('GO' or 'NO_GO')
        full_sharpe: Full sample Sharpe ratio (2015-2024)
        oos_sharpe: Out-of-sample Sharpe ratio (2023-2024)
        recent_sharpe: Recent regime Sharpe ratio (2023-2024)
        verdict_notes: Brief summary of decision rationale
        report_path: Path to detailed diagnostic report
    """
    signal_name: str
    version: str
    universe: str
    status: Literal['GO', 'NO_GO']
    full_sharpe: Optional[float]
    oos_sharpe: Optional[float]
    recent_sharpe: Optional[float]
    verdict_notes: str
    report_path: str

    def __repr__(self) -> str:
        return (f"SignalStatus(name='{self.signal_name}', version='{self.version}', "
                f"status='{self.status}', oos_sharpe={self.oos_sharpe})")


# Hard-coded signal registry matching docs/core/SIGNAL_REGISTRY.md
_SIGNAL_REGISTRY = [
    SignalStatus(
        signal_name='InstitutionalMomentum',
        version='v2',
        universe='sp500_actual',
        status='GO',
        full_sharpe=0.245,
        oos_sharpe=0.742,
        recent_sharpe=0.309,
        verdict_notes=(
            'Ready for ensemble. Canonical config: 308d formation, 0d skip, 9.2% winsor. '
            'Strong OOS performance (Sharpe 0.742). Passes all Phase 2 acceptance gates.'
        ),
        report_path='results/MOMENTUM_PHASE2_TRIAL11_DIAGNOSTIC.md'
    ),
    SignalStatus(
        signal_name='CrossSectionalQuality',
        version='v1',
        universe='sp500_actual',
        status='GO',
        full_sharpe=None,  # TBD - pending full diagnostic
        oos_sharpe=None,  # TBD - pending full diagnostic
        recent_sharpe=None,  # TBD - pending full diagnostic
        verdict_notes=(
            'Backtest-ready as of Phase 3 Milestone 2. QMJ methodology with proper cross-sectional '
            'implementation. Passed smoke tests. Ready for multi-signal ensembles. Full diagnostic pending.'
        ),
        report_path='results/quality_v1_phase1_report.md'
    ),
    SignalStatus(
        signal_name='InstitutionalInsider',
        version='v1',
        universe='sp500_actual',
        status='NO_GO',
        full_sharpe=0.034,
        oos_sharpe=0.374,
        recent_sharpe=0.132,
        verdict_notes=(
            'Failed 3/5 gates. No decile monotonicity (D1-D10 spread only 0.28%/yr). '
            'Statistically insignificant (t-stat 0.180). Insider activity not predictive '
            'of returns on S&P 500 large caps. Archived.'
        ),
        report_path='results/INSIDER_PHASE1_REPORT.md'
    ),
]


def get_signal_status(signal_name: str, version: str) -> Optional[SignalStatus]:
    """
    Get status record for a specific signal by name and version.

    Args:
        signal_name: Signal name (e.g., 'InstitutionalMomentum')
        version: Version identifier (e.g., 'v2')

    Returns:
        SignalStatus if found, None otherwise

    Example:
        >>> momentum = get_signal_status('InstitutionalMomentum', 'v2')
        >>> if momentum and momentum.status == 'GO':
        ...     print(f"Momentum v2 ready for ensemble: {momentum.oos_sharpe}")
    """
    for signal in _SIGNAL_REGISTRY:
        if signal.signal_name == signal_name and signal.version == version:
            return signal
    return None


def list_signals() -> List[SignalStatus]:
    """
    List all signals in the registry.

    Returns:
        List of all SignalStatus records

    Example:
        >>> all_signals = list_signals()
        >>> for signal in all_signals:
        ...     print(f"{signal.signal_name} {signal.version}: {signal.status}")
    """
    return _SIGNAL_REGISTRY.copy()


def list_go_signals() -> List[SignalStatus]:
    """
    List only signals that passed validation (status='GO').

    Returns:
        List of SignalStatus records with status='GO'

    Example:
        >>> go_signals = list_go_signals()
        >>> print(f"Signals ready for ensemble: {len(go_signals)}")
        >>> for signal in go_signals:
        ...     print(f"  - {signal.signal_name} {signal.version} (OOS Sharpe: {signal.oos_sharpe})")
    """
    return [s for s in _SIGNAL_REGISTRY if s.status == 'GO']


def list_no_go_signals() -> List[SignalStatus]:
    """
    List signals that failed validation (status='NO_GO').

    Returns:
        List of SignalStatus records with status='NO_GO'

    Example:
        >>> failed_signals = list_no_go_signals()
        >>> for signal in failed_signals:
        ...     print(f"{signal.signal_name}: {signal.verdict_notes}")
    """
    return [s for s in _SIGNAL_REGISTRY if s.status == 'NO_GO']


def to_markdown_table() -> str:
    """
    Generate markdown table representation of the registry.

    Returns:
        Markdown-formatted table string

    Example:
        >>> table = to_markdown_table()
        >>> print(table)
    """
    header = (
        "| Signal Name | Version | Universe | Status | Full Sharpe | OOS Sharpe | "
        "Recent Sharpe | Verdict Notes |\n"
    )
    separator = (
        "|-------------|---------|----------|--------|-------------|------------|"
        "---------------|---------------|\n"
    )

    rows = []
    for signal in _SIGNAL_REGISTRY:
        full_sharpe_str = f"{signal.full_sharpe:.3f}" if signal.full_sharpe is not None else "N/A"
        oos_sharpe_str = f"{signal.oos_sharpe:.3f}" if signal.oos_sharpe is not None else "N/A"
        recent_sharpe_str = f"{signal.recent_sharpe:.3f}" if signal.recent_sharpe is not None else "N/A"

        # Truncate verdict notes for table
        notes = signal.verdict_notes[:60] + "..." if len(signal.verdict_notes) > 60 else signal.verdict_notes

        row = (
            f"| {signal.signal_name} | {signal.version} | {signal.universe} | "
            f"**{signal.status}** | {full_sharpe_str} | {oos_sharpe_str} | "
            f"{recent_sharpe_str} | {notes} |\n"
        )
        rows.append(row)

    return header + separator + ''.join(rows)


def get_ensemble_ready_signals() -> List[SignalStatus]:
    """
    Alias for list_go_signals() for clarity in ensemble context.

    Returns:
        List of GO signals ready for ensemble construction
    """
    return list_go_signals()


if __name__ == '__main__':
    # Demo usage
    print("=" * 80)
    print("Signal Registry - Stage 1 Status")
    print("=" * 80)
    print()

    print("All Signals:")
    for signal in list_signals():
        status_emoji = "✅" if signal.status == 'GO' else "❌"
        print(f"  {status_emoji} {signal.signal_name} {signal.version}: {signal.status}")
        if signal.oos_sharpe is not None:
            print(f"      OOS Sharpe: {signal.oos_sharpe:.3f}")

    print()
    print("=" * 80)
    print(f"Ensemble-Ready Signals: {len(list_go_signals())}")
    print("=" * 80)

    for signal in list_go_signals():
        print(f"\n{signal.signal_name} {signal.version}:")
        print(f"  Full Sharpe: {signal.full_sharpe:.3f}")
        print(f"  OOS Sharpe: {signal.oos_sharpe:.3f}")
        print(f"  Recent Sharpe: {signal.recent_sharpe:.3f}")
        print(f"  Report: {signal.report_path}")

    print()
    print("=" * 80)
    print("Markdown Table:")
    print("=" * 80)
    print(to_markdown_table())
