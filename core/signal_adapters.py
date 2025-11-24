"""
Signal-to-backtest adapters for centralized wiring.

Converts InstitutionalSignal and EnsembleSignal objects into
signal_fn callables compatible with run_backtest().

All DataManager usage for backtest wiring lives here.

Exception Handling Philosophy:
- Fail loudly rather than silently returning empty results
- Let type errors, signature mismatches, and unexpected failures propagate
- Only catch exceptions where the existing baseline scripts already do
- Per-ticker data fetch failures are OK to catch (data issues are expected)
- Top-level failures should crash tests so we detect breakage immediately
"""

from typing import Callable, List, TYPE_CHECKING
from datetime import timedelta
import pandas as pd
from config import get_logger

if TYPE_CHECKING:
    from core.institutional_base import InstitutionalSignal
    from signals.ml.ensemble_signal import EnsembleSignal
    from data.data_manager import DataManager

logger = get_logger(__name__)

# Type alias for backtest signal function
SignalFn = Callable[[str, List[str]], pd.Series]


def make_signal_fn(
    signal: "InstitutionalSignal",
    data_manager: "DataManager",
) -> SignalFn:
    """
    Create backtest-compatible signal_fn from InstitutionalSignal.

    Uses signal.generate_cross_sectional_scores() internally.

    Args:
        signal: InstitutionalSignal instance
        data_manager: DataManager instance for data fetching

    Returns:
        Callable compatible with run_backtest(signal_fn=...)

    Example:
        >>> from signals.momentum.institutional_momentum import InstitutionalMomentum
        >>> signal = InstitutionalMomentum({'formation_period': 308})
        >>> signal_fn = make_signal_fn(signal, dm)
        >>> scores = signal_fn('2023-01-31', ['AAPL', 'MSFT'])

    Notes:
        - No top-level exception handling - failures propagate to caller
        - This ensures tests fail loudly on signature mismatches or type errors
        - Per-ticker data issues are handled inside generate_cross_sectional_scores
    """
    def signal_fn(rebal_date: str, tickers: List[str]) -> pd.Series:
        """Generate signals for tickers at rebalance date."""
        rebal_ts = pd.Timestamp(rebal_date)
        scores = signal.generate_cross_sectional_scores(
            rebal_date=rebal_ts,
            universe=tickers,
            data_manager=data_manager,
        )

        # Type safety check - fail loudly if contract violated
        if not isinstance(scores, pd.Series):
            raise TypeError(
                f"Expected pd.Series from {signal.__class__.__name__}.generate_cross_sectional_scores(), "
                f"got {type(scores).__name__}"
            )

        return scores

    return signal_fn


def make_ensemble_signal_fn(
    ensemble: "EnsembleSignal",
    data_manager: "DataManager",
    lookback_days: int = 500,
) -> SignalFn:
    """
    Create backtest-compatible signal_fn from EnsembleSignal.

    Handles data fetching and calls ensemble.generate_ensemble_scores().

    Copied literally from test_backtest_engine.py ensemble_signal_fn (lines 323-343).

    Args:
        ensemble: EnsembleSignal instance
        data_manager: DataManager instance for data fetching
        lookback_days: Calendar days to look back for price data (default 500)

    Returns:
        Callable compatible with run_backtest(signal_fn=...)

    Example:
        >>> from signals.ml.ensemble_signal import EnsembleSignal
        >>> ensemble = EnsembleSignal(...)
        >>> signal_fn = make_ensemble_signal_fn(ensemble, dm, lookback_days=500)
        >>> scores = signal_fn('2023-01-31', ['AAPL', 'MSFT'])

    Notes:
        - Per-ticker try/except matches existing ensemble baseline pattern
        - No top-level blanket exception handling - fails loudly on unexpected errors
        - Returns empty Series if no valid price data (matches existing semantics)
    """
    def signal_fn(rebal_date: str, tickers: List[str]) -> pd.Series:
        """Generate ensemble signals for tickers at rebalance date."""
        rebal_ts = pd.Timestamp(rebal_date)
        lookback_start = (rebal_ts - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

        # Build prices_dict for ensemble API
        # Per-ticker try/except matches existing baseline scripts
        prices_dict = {}
        for ticker in tickers:
            try:
                prices = data_manager.get_prices(ticker, lookback_start, rebal_date)
                if len(prices) > 0 and 'close' in prices.columns:
                    # PIT slice (matches equivalence test line 330)
                    px_slice = prices['close'][prices.index <= rebal_ts]

                    # Minimum history filter (matches equivalence test line 331)
                    if len(px_slice) >= 90:
                        prices_dict[ticker] = px_slice
            except Exception as e:
                # Per-ticker data issues are expected - log and continue
                logger.debug(f"Could not fetch prices for {ticker}: {e}")
                continue

        # Early exit if no valid tickers (matches equivalence test lines 337-338)
        if len(prices_dict) == 0:
            return pd.Series(dtype=float)

        # Call ensemble - let any failures here propagate
        ensemble_scores = ensemble.generate_ensemble_scores(
            prices_by_ticker=prices_dict,
            rebalance_date=rebal_ts,
        )

        # Type safety check - fail loudly if contract violated
        if not isinstance(ensemble_scores, pd.Series):
            raise TypeError(
                f"Expected pd.Series from ensemble.generate_ensemble_scores(), "
                f"got {type(ensemble_scores).__name__}"
            )

        return ensemble_scores

    return signal_fn


def make_multisignal_ensemble_fn(
    ensemble: "EnsembleSignal",
    data_manager: "DataManager",
) -> SignalFn:
    """
    Create backtest-compatible signal_fn from EnsembleSignal using cross-sectional path.

    Uses ensemble.generate_cross_sectional_ensemble_scores() which supports
    multi-signal ensembles with different data dependencies (prices, fundamentals, etc.).

    Args:
        ensemble: EnsembleSignal instance with multiple signals
        data_manager: DataManager instance (passed to signal implementations)

    Returns:
        Callable compatible with run_backtest(signal_fn=...)

    Raises:
        NotImplementedError: If any signal in ensemble lacks generate_cross_sectional_scores()
        TypeError: If ensemble returns wrong type

    Example:
        >>> from signals.ml.ensemble_signal import EnsembleSignal
        >>> ensemble = get_momentum_quality_v1_ensemble(dm)
        >>> signal_fn = make_multisignal_ensemble_fn(ensemble, dm)
        >>> scores = signal_fn('2023-01-31', ['AAPL', 'MSFT'])

    Notes:
        - No data fetching here - each signal handles its own data via DataManager
        - Fails loudly on NotImplementedError (signal missing cross-sectional implementation)
        - No top-level exception handling - errors propagate to caller for visibility
    """
    def signal_fn(rebal_date: str, tickers: List[str]) -> pd.Series:
        """Generate multi-signal ensemble scores for tickers at rebalance date."""
        rebal_ts = pd.Timestamp(rebal_date)

        # Call cross-sectional ensemble path
        # Any NotImplementedError or type errors will propagate (fail loud)
        ensemble_scores = ensemble.generate_cross_sectional_ensemble_scores(
            rebal_date=rebal_ts,
            universe=tickers,
        )

        # Type safety check - fail loudly if contract violated
        if not isinstance(ensemble_scores, pd.Series):
            raise TypeError(
                f"Expected pd.Series from ensemble.generate_cross_sectional_ensemble_scores(), "
                f"got {type(ensemble_scores).__name__}"
            )

        return ensemble_scores

    return signal_fn
