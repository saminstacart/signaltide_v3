"""
Backtest Manifest

Structured representation of a backtest run for reproducibility and auditability.

A BacktestManifest captures all parameters, data sources, and code versions
used in a backtest run, enabling:
- Full reproducibility of results
- Audit trails for production runs
- Debugging of unexpected behavior
- Comparison across runs

Usage:
    manifest = BacktestManifest.from_context(
        dm=data_manager,
        start_date='2023-01-01',
        end_date='2023-12-31',
        universe_type='manual',
        universe_params={'tickers': ['AAPL', 'MSFT']},
        signals=[{
            'name': 'InstitutionalMomentum',
            'module': 'signals.momentum.institutional_momentum',
            'params': {'formation_period': 252, ...}
        }],
        initial_capital=50000.0,
        rebalance_schedule='monthly',
        transaction_costs={'commission_pct': 0.0, ...}
    )

    # Serialize to JSON-safe dict
    manifest_dict = manifest.to_dict()

    # Log or save
    logger.info(f"Backtest manifest: {manifest.run_id}")
    with open('manifest.json', 'w') as f:
        json.dump(manifest_dict, f, indent=2)
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
import subprocess
from config import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestManifest:
    """
    Structured manifest describing a backtest run.

    Captures all inputs, configuration, and versioning information
    needed to reproduce a backtest.

    Fields are organized into logical groups:
    - Identity: Unique identifiers and timestamps
    - Data slice: Date ranges and trading calendar version
    - Universe: Universe definition and parameters
    - Signals: Signal specifications and parameters
    - Execution: Portfolio and trading parameters
    - Code versioning: Git SHA and other version metadata
    """

    # ===== Identity =====
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """Unique identifier for this backtest run"""

    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + 'Z')
    """ISO 8601 timestamp of manifest creation (UTC)"""

    # ===== Data Slice =====
    start_date: str = ""
    """Backtest start date (YYYY-MM-DD)"""

    end_date: str = ""
    """Backtest end date (YYYY-MM-DD)"""

    trading_calendar_version: Optional[str] = None
    """Trading calendar version identifier (e.g., row count or date range)"""

    # ===== Universe =====
    universe_type: str = ""
    """Universe type (manual, sp500_proxy, sp500_actual, etc.)"""

    universe_params: Dict[str, Any] = field(default_factory=dict)
    """Universe construction parameters (tickers, filters, etc.)"""

    universe_size_snapshot: Optional[int] = None
    """Number of tickers in universe at start (if cheap to compute)"""

    # ===== Signals =====
    signals: List[Dict[str, Any]] = field(default_factory=list)
    """
    Signal specifications. Each entry has:
    - name: Signal class name (e.g., 'InstitutionalMomentum')
    - module: Full module path (e.g., 'signals.momentum.institutional_momentum')
    - params: Dict of actual parameters used
    """

    # ===== Execution / Portfolio =====
    initial_capital: float = 0.0
    """Initial portfolio capital"""

    rebalance_schedule: str = ""
    """Rebalance frequency (daily, weekly, monthly)"""

    transaction_costs: Dict[str, Any] = field(default_factory=dict)
    """Transaction cost model parameters (commission_pct, slippage_pct, spread_pct)"""

    # ===== Code Versioning =====
    git_sha: Optional[str] = None
    """Git commit SHA if available"""

    code_version_meta: Dict[str, Any] = field(default_factory=dict)
    """Additional version metadata (module versions, Python version, etc.)"""

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert manifest to JSON-safe dictionary.

        Returns:
            Dict with all manifest fields, using only basic types
            (str, int, float, list, dict). All nested objects are
            converted to their dict representations.
        """
        return asdict(self)

    @classmethod
    def from_context(
        cls,
        dm,
        start_date: str,
        end_date: str,
        universe_type: str,
        universe_params: Dict[str, Any],
        signals: List[Dict[str, Any]],
        initial_capital: float,
        rebalance_schedule: str,
        transaction_costs: Dict[str, Any],
        run_id: Optional[str] = None,
        created_at: Optional[str] = None
    ) -> 'BacktestManifest':
        """
        Create a manifest from backtest context.

        Args:
            dm: DataManager instance (used to probe trading calendar version)
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            universe_type: Type of universe (manual, sp500_actual, etc.)
            universe_params: Universe parameters (tickers, filters, etc.)
            signals: List of signal specifications, each with:
                - name: Signal class name
                - module: Full module path
                - params: Dict of parameters
            initial_capital: Initial portfolio capital
            rebalance_schedule: Rebalance frequency string
            transaction_costs: Transaction cost model parameters
            run_id: Optional custom run ID (generates UUID if not provided)
            created_at: Optional custom timestamp (uses UTC now if not provided)

        Returns:
            Populated BacktestManifest instance
        """
        # Probe trading calendar version (safe, returns None on failure)
        calendar_version = cls._probe_trading_calendar_version(dm)

        # Probe git SHA (safe, returns None on failure)
        git_sha = cls._probe_git_sha()

        # Calculate universe size snapshot if it's a list of tickers
        universe_size = None
        if universe_type == 'manual' and 'tickers' in universe_params:
            tickers = universe_params['tickers']
            universe_size = len(tickers) if isinstance(tickers, list) else None

        return cls(
            run_id=run_id or str(uuid.uuid4()),
            created_at=created_at or (datetime.utcnow().isoformat() + 'Z'),
            start_date=start_date,
            end_date=end_date,
            trading_calendar_version=calendar_version,
            universe_type=universe_type,
            universe_params=universe_params,
            universe_size_snapshot=universe_size,
            signals=signals,
            initial_capital=initial_capital,
            rebalance_schedule=rebalance_schedule,
            transaction_costs=transaction_costs,
            git_sha=git_sha,
            code_version_meta={}
        )

    @staticmethod
    def _probe_trading_calendar_version(dm) -> Optional[str]:
        """
        Probe trading calendar version from DataManager.

        Uses calendar size and date range as a proxy for version.
        Safe - returns None on any failure.

        Args:
            dm: DataManager instance

        Returns:
            Version string like "9049_trading_days__2001-01-02_to_2050-12-30"
            or None if probe fails
        """
        try:
            # Query calendar for row count and date range
            query = """
                SELECT
                    COUNT(*) as total_days,
                    SUM(CASE WHEN is_trading_day = 1 THEN 1 ELSE 0 END) as trading_days,
                    MIN(date) as min_date,
                    MAX(date) as max_date
                FROM dim_trading_calendar
            """
            result = dm.execute_query(query)
            if result and len(result) > 0:
                row = result[0]
                trading_days = row.get('trading_days', 0)
                min_date = row.get('min_date', 'unknown')
                max_date = row.get('max_date', 'unknown')
                return f"{trading_days}_trading_days__{min_date}_to_{max_date}"
        except Exception as e:
            logger.debug(f"Could not probe trading calendar version: {e}")

        return None

    @staticmethod
    def _probe_git_sha() -> Optional[str]:
        """
        Attempt to read current git commit SHA.

        Safe - returns None if git is not available or repo is not a git repo.

        Returns:
            Git SHA (first 12 chars) or None if unavailable
        """
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--short=12', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=2.0
            )
            if result.returncode == 0:
                sha = result.stdout.strip()
                if sha:
                    logger.debug(f"Captured git SHA: {sha}")
                    return sha
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.debug(f"Could not probe git SHA: {e}")

        return None

    def __repr__(self) -> str:
        """Compact string representation."""
        return (
            f"BacktestManifest(run_id={self.run_id[:8]}..., "
            f"period={self.start_date}..{self.end_date}, "
            f"universe={self.universe_type}, "
            f"signals={len(self.signals)})"
        )
