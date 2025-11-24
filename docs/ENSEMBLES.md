# Ensemble Framework - SignalTide v3

**Last Updated**: 2025-11-22
**Status**: Production (price-based path) + Multi-signal pathway (Phase 3)

---

## Overview

An **ensemble** in SignalTide combines multiple signal instances into a single composite score using configurable weights and normalization methods.

**Key benefits**:
- Diversify across factor exposures (momentum, quality, value, etc.)
- Reduce single-signal volatility through combination
- Leverage signals with different data dependencies (prices, fundamentals, insider data)
- Centralized configuration and registry validation

**Difference from single-signal runs**:
- Single signal: One `InstitutionalSignal` → `make_signal_fn()` → `run_backtest()`
- Ensemble: Multiple `InstitutionalSignal` members → `EnsembleSignal` → adapter → `run_backtest()`

All ensembles are defined in `signals/ml/ensemble_configs.py` and registered in `ENSEMBLE_REGISTRY`.

---

## Two Ensemble Pathways

SignalTide supports two pathways for ensemble score generation, each optimized for different use cases:

### 1. Price-Based Path (Legacy, Momentum-Only)

**Method**: `EnsembleSignal.generate_ensemble_scores(prices_by_ticker, rebalance_date, bulk_insider_data=None)`

**Use case**: Single-signal ensembles where all members use only price data (e.g., momentum-only with different parameters).

**How it works**:
1. Caller provides `prices_by_ticker` dict: `{ticker: pd.Series[close prices]}`
2. For each ensemble member, loop over tickers and call `signal.generate_signals(data)` where `data` is a per-ticker DataFrame
3. Collect per-signal scores, normalize according to member config (zscore, rank, none)
4. Combine using weighted average via `_combine_normalized_scores()`

**Data responsibility**: Caller fetches and provides all price data via `prices_by_ticker`.

**Current usage**:
- `get_momentum_v2_adaptive_quintile_ensemble()` (production default)
- `get_momentum_v1_legacy_quintile_ensemble()` (archived reference)

**Adapter**: `make_ensemble_signal_fn(ensemble, data_manager, lookback_days=500)`
- Fetches price data for each ticker with lookback window
- Builds `prices_by_ticker` dict
- Calls `ensemble.generate_ensemble_scores(prices_by_ticker, rebalance_date)`
- Returns `signal_fn(rebal_date: str, tickers: List[str]) -> pd.Series`

**Example**:
```python
from signals.ml.ensemble_configs import get_momentum_v2_ensemble
from core.signal_adapters import make_ensemble_signal_fn

ensemble = get_momentum_v2_ensemble(dm)
signal_fn = make_ensemble_signal_fn(ensemble, dm, lookback_days=500)
result = run_backtest(universe_fn, signal_fn, config)
```

---

### 2. Cross-Sectional Path (Multi-Signal, Phase 3+)

**Method**: `EnsembleSignal.generate_cross_sectional_ensemble_scores(rebal_date, universe)`

**Use case**: Multi-signal ensembles where members need different data types (prices, fundamentals, insider transactions, alternative data).

**How it works**:
1. Caller provides rebalance date and universe (list of tickers)
2. For each ensemble member, call `signal.generate_cross_sectional_scores(rebal_date, universe, data_manager)`
3. Each signal handles its own data fetching via `DataManager` (prices, fundamentals, etc.)
4. Collect per-signal scores, normalize according to member config
5. Combine using weighted average via `_combine_normalized_scores()`

**Data responsibility**: Each signal fetches its own required data via `data_manager`.

**Current usage**:
- `get_momentum_quality_v1_ensemble()` (research, Phase 3 Milestone 3)
  - 0.5 InstitutionalMomentum v2 (prices only)
  - 0.5 CrossSectionalQuality v1 (fundamentals with 33-day lag)

**Adapter**: `make_multisignal_ensemble_fn(ensemble, data_manager)`
- No data fetching (signals do it internally)
- Calls `ensemble.generate_cross_sectional_ensemble_scores(rebal_date, universe)`
- Returns `signal_fn(rebal_date: str, tickers: List[str]) -> pd.Series`

**Example**:
```python
from signals.ml.ensemble_configs import get_momentum_quality_v1_ensemble
from core.signal_adapters import make_multisignal_ensemble_fn

ensemble = get_momentum_quality_v1_ensemble(dm)
signal_fn = make_multisignal_ensemble_fn(ensemble, dm)
result = run_backtest(universe_fn, signal_fn, config)
```

**Advantages**:
- Supports signals with heterogeneous data needs
- Each signal controls its own PIT lag and lookback logic
- Cleaner separation of concerns (no caller-side data fetching)
- Easier to add new signal types (insider, alternative data, etc.)

---

## Adapters: Wiring Ensembles to `run_backtest()`

Both pathways require an adapter to convert `EnsembleSignal` into the `signal_fn` signature expected by `run_backtest()`.

### `make_ensemble_signal_fn(ensemble, data_manager, lookback_days=500)`

**Purpose**: Wire price-based ensemble to backtest engine.

**Signature**:
```python
def make_ensemble_signal_fn(
    ensemble: EnsembleSignal,
    data_manager: DataManager,
    lookback_days: int = 500,
) -> SignalFn
```

**Returns**: `signal_fn(rebal_date: str, tickers: List[str]) -> pd.Series`

**When to use**:
- Momentum-only ensembles (all members use price data)
- Legacy price-based pathway
- Existing production momentum configs

**What it does**:
1. For each ticker, fetch prices with `lookback_days` calendar days of history
2. Build `prices_by_ticker` dict
3. Call `ensemble.generate_ensemble_scores(prices_by_ticker, rebalance_date)`
4. Type-check and return scores

**Notes**:
- Includes per-ticker try/except for data fetch failures (expected)
- No top-level blanket exception handling (fail loud on unexpected errors)
- Minimum history filter: 90 trading days per ticker

---

### `make_multisignal_ensemble_fn(ensemble, data_manager)`

**Purpose**: Wire cross-sectional ensemble to backtest engine.

**Signature**:
```python
def make_multisignal_ensemble_fn(
    ensemble: EnsembleSignal,
    data_manager: DataManager,
) -> SignalFn
```

**Returns**: `signal_fn(rebal_date: str, tickers: List[str]) -> pd.Series`

**When to use**:
- Multi-signal ensembles with different data dependencies
- Signals that need fundamentals, insider data, or alternative data
- Cross-sectional pathway (Phase 3+)

**What it does**:
1. Convert `rebal_date` string to `pd.Timestamp`
2. Call `ensemble.generate_cross_sectional_ensemble_scores(rebal_date, universe)`
3. Type-check and return scores

**Notes**:
- No data fetching here (signals handle internally via `data_manager`)
- Fails loudly on `NotImplementedError` if signal lacks cross-sectional implementation
- No top-level exception handling (fail loud)

---

## Making a Signal Ensemble-Ready

To use a signal in multi-signal ensembles (cross-sectional path), follow this checklist:

### 1. Implement `generate_cross_sectional_scores()`

Add this method to your `InstitutionalSignal` subclass:

```python
def generate_cross_sectional_scores(
    self,
    rebal_date: pd.Timestamp,
    universe: Sequence[str],
    data_manager: "DataManager",
) -> pd.Series:
    """
    Generate cross-sectional signal scores for a universe at a single rebalance date.

    Args:
        rebal_date: Rebalance timestamp
        universe: List of ticker symbols in current universe
        data_manager: DataManager instance for fetching required data

    Returns:
        pd.Series indexed by ticker with signal scores (subset of universe allowed)

    Notes:
        - Fetch your own data via data_manager (prices, fundamentals, etc.)
        - Enforce PIT correctness (only use data available as of rebal_date)
        - Return empty Series if insufficient data for any reason
        - Scores should be cross-sectionally comparable (not time-series scaled)
    """
    # Your implementation here
    pass
```

**Key requirements**:
- Returns `pd.Series` indexed by ticker (fail loud if wrong type)
- May return subset of universe (some tickers lacking data is OK)
- Handles all data fetching internally via `data_manager`
- Enforces point-in-time correctness (no lookahead bias)
- Scores are cross-sectionally comparable at each rebalance date

**Example** (from `InstitutionalMomentum`):
```python
def generate_cross_sectional_scores(
    self,
    rebal_date: pd.Timestamp,
    universe: Sequence[str],
    data_manager: "DataManager",
) -> pd.Series:
    lookback_days = 500
    lookback_start = (rebal_date - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    rebal_date_str = rebal_date.strftime('%Y-%m-%d')

    scores = {}
    for ticker in universe:
        try:
            prices = data_manager.get_prices(ticker, lookback_start, rebal_date_str)
            if len(prices) > 0 and 'close' in prices.columns:
                data = pd.DataFrame({'close': prices['close'], 'ticker': ticker})
                sig_series = self.generate_signals(data)
                if len(sig_series) > 0:
                    signal_value = sig_series.iloc[-1]
                    if pd.notna(signal_value) and signal_value != 0:
                        scores[ticker] = signal_value
        except Exception:
            continue  # Missing data is OK

    return pd.Series(scores, dtype=float)
```

---

### 2. Add to Signal Class Map

In `signals/ml/ensemble_signal.py`, add your signal to `_SIGNAL_CLASS_MAP`:

```python
_SIGNAL_CLASS_MAP = {
    ("InstitutionalMomentum", "v2"): InstitutionalMomentum,
    ("CrossSectionalQuality", "v1"): CrossSectionalQuality,
    ("YourNewSignal", "v1"): YourNewSignal,  # <-- Add here
}
```

Don't forget to import your signal class at the top of the file.

---

### 3. Register in Signal Registry

In `core/signal_registry.py`, add a `SignalStatus` entry:

```python
SignalStatus(
    signal_name='YourNewSignal',
    version='v1',
    universe='sp500_actual',
    status='GO',  # or 'NO_GO', 'RESEARCH'
    full_sharpe=0.XX,  # From diagnostic report
    oos_sharpe=0.XX,
    recent_sharpe=0.XX,
    verdict_notes=(
        'Brief description of signal methodology and validation status.'
    ),
    report_path='results/your_signal_diagnostic.md'
)
```

**Status guidelines**:
- `GO`: Passed validation, ready for production ensembles
- `RESEARCH`: Experimental, use only with `allow_no_go=True`
- `NO_GO`: Failed validation or archived

---

### 4. Add Smoke Test

Create or update `tests/test_your_signal_integration.py`:

```python
def test_your_signal_smoke(setup):
    """Smoke test: YourSignal flows through run_backtest via adapter."""
    dm = setup['dm']

    signal = YourNewSignal(params, data_manager=dm)
    signal_fn = make_signal_fn(signal, dm)

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

    result = run_backtest(universe_fn, signal_fn, config)

    assert len(result.equity_curve) > 0
    assert result.num_rebalances > 0
    assert result.final_equity > 0
```

Run:
```bash
python3 -m pytest tests/test_your_signal_integration.py -v
```

---

### 5. Normalization Compatibility

Ensure your signal's score scale is compatible with ensemble normalization modes:

**If `normalize="none"` (no normalization)**:
- Signal should already return cross-sectionally comparable scores
- Typical: quintile values `[-1, -0.5, 0, 0.5, 1]` or similar bounded range
- Used when signal already has built-in cross-sectional ranking

**If `normalize="zscore"` (z-score normalization)**:
- Signal returns raw scores (any scale)
- Ensemble normalizes to `(score - mean) / std` across universe
- Use when signal returns continuous values without inherent ranking

**If `normalize="rank"` (percentile rank)**:
- Signal returns raw scores
- Ensemble converts to percentile ranks, centered at 0: `rank(pct=True) - 0.5`
- Output range: `[-0.5, 0.5]`
- Use when only relative ordering matters, not magnitude

**Example ensemble config**:
```python
members = [
    EnsembleMember(
        signal_name="InstitutionalMomentum",
        version="v2",
        weight=0.5,
        normalize="none",  # Already returns quintiles
        params=momentum_params,
    ),
    EnsembleMember(
        signal_name="YourNewSignal",
        version="v1",
        weight=0.5,
        normalize="zscore",  # Normalize raw scores
        params=your_params,
    ),
]
```

---

## Production Ensemble Configs

All production ensemble configurations live in `signals/ml/ensemble_configs.py`.

**Current configs**:

| Config Name | Status | Pathway | Members | Notes |
|------------|--------|---------|---------|-------|
| `get_momentum_v2_adaptive_quintile_ensemble()` | Production | Price-based | 1 (InstitutionalMomentum v2) | Adaptive quintiles, 308d formation |
| `get_momentum_v1_legacy_quintile_ensemble()` | Archived | Price-based | 1 (InstitutionalMomentum v2) | Hard 20% quintiles, Trial 11 reference |
| `get_momentum_quality_v1_ensemble()` | Research | Cross-sectional | 2 (Momentum 0.25, Quality 0.75) | First multi-signal config, Phase 3 |

**Adding a new ensemble config**:

1. Define a function in `ensemble_configs.py`:
```python
def get_your_ensemble(dm: DataManager) -> EnsembleSignal:
    """Your ensemble description."""
    members = [
        EnsembleMember(
            signal_name="SignalA",
            version="v1",
            weight=0.6,
            normalize="none",
            params={...},
        ),
        EnsembleMember(
            signal_name="SignalB",
            version="v2",
            weight=0.4,
            normalize="zscore",
            params={...},
        ),
    ]

    return EnsembleSignal(
        members=members,
        data_manager=dm,
        enforce_go_only=True,  # Reject NO_GO signals
    )
```

2. Register in `ENSEMBLE_REGISTRY`:
```python
ENSEMBLE_REGISTRY = {
    "your_ensemble_name": EnsembleDefinition(
        name="your_ensemble_name",
        description="Brief description",
        status="RESEARCH",  # or "PRODUCTION"
        validation_report="results/your_diagnostic.md"
    ),
}
```

3. Run smoke test and full diagnostic

---

## Ensemble Diagnostics

Production ensembles should have both **baseline** and **regime** diagnostics:

**Baseline diagnostic** (full-period metrics):
- Script: `scripts/run_<ensemble>_baseline.py`
- Outputs: `results/ensemble_baselines/<ensemble>_diagnostic.md` and `*_comparison.csv`
- Shows overall performance vs component signals

**Regime diagnostic** (breakdown by market regime):
- Script: `scripts/run_<ensemble>_regime_diagnostic.py`
- Outputs: `results/ensemble_baselines/<ensemble>_regime_diagnostic.md` and `*_regime_comparison.csv`
- Shows where ensemble adds value (crisis, bear, bull, etc.)
- Helps inform weight tuning and regime-conditional allocation

**Example: `momentum_quality_v1` (canonical template for future ensembles)**

Baseline:
- Script: `scripts/run_momentum_quality_baseline.py`
- MD: `results/ensemble_baselines/momentum_quality_v1_diagnostic.md`
- Shows M+Q vs momentum-only over full period (2015-2024)

Regime:
- Script: `scripts/run_momentum_quality_regime_diagnostic.py`
- MD: `results/ensemble_baselines/momentum_quality_v1_regime_diagnostic.md`
- Breaks down performance across 5 macro regimes
- Key finding: Quality helps in 4/5 regimes (crisis, bear, recent); drags in steady bull

Weight Calibration (Phase 3 M3.4):
- Grid sweep: `scripts/run_momentum_quality_weight_sweep.py` → `momentum_quality_v1_weight_sweep.md`
- Optuna validation: `scripts/run_momentum_quality_weight_optuna.py` → `momentum_quality_v1_weight_optuna.md`
- Grid sweep: M=0.25/Q=0.75 best across all metrics (Sharpe 2.876, total return 135.98%, max DD -23.89%)
- Optuna: Continuous search converged to M≈0.20/Q≈0.80 (32 trials, TPE sampler)
- **Selected: M=0.25, Q=0.75** - Plateau center, grid sweep winner, Optuna-validated, non-overfitted
- Quality-heavy allocation outperforms pure momentum by +19% Sharpe (2.876 vs 2.413)

---

## See Also

- `signals/ml/ensemble_signal.py` - EnsembleSignal implementation
- `core/signal_adapters.py` - Adapter functions for backtest wiring
- `signals/ml/ensemble_configs.py` - Production ensemble configurations
- `core/signal_registry.py` - Signal validation and status tracking
- `docs/signal_catalog.md` - Individual signal documentation
- `core/backtest_engine.py` - Unified backtest harness

---

**Maintenance**: Update this doc when:
- New ensemble pathway added
- New adapter function created
- Ensemble contract changes
- New production config added
