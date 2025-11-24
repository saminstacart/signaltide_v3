# Signal Catalog - Institutional Signals

**Last Updated**: 2025-11-22
**Purpose**: Comprehensive catalog of all `InstitutionalSignal` subclasses in SignalTide v3

---

## Overview

### What is an InstitutionalSignal?

An `InstitutionalSignal` is a base class (`core/institutional_base.py`) providing professional-grade utilities for factor signal construction:

- Cross-sectional ranking and z-scoring
- Winsorization for outlier handling
- Quintile construction (hard 20% vs adaptive modes)
- Sector neutralization
- Monthly rebalancing alignment

All institutional signals inherit from this base and implement the signal generation contract.

### Backtest Integration Contract

**New in Phase 3**: Signals implement `generate_cross_sectional_scores()` to enable seamless backtest integration:

```python
def generate_cross_sectional_scores(
    self,
    rebal_date: pd.Timestamp,
    universe: Sequence[str],
    data_manager: "DataManager",
) -> pd.Series:
    """
    Generate cross-sectional signal scores for a universe at a single rebalance date.

    Returns:
        pd.Series indexed by ticker with signal scores (subset of universe allowed)
    """
```

**Backtest wiring via adapters**:
```python
from core.signal_adapters import make_signal_fn

signal = InstitutionalMomentum(params)
signal_fn = make_signal_fn(signal, data_manager)
result = run_backtest(universe_fn, signal_fn, config)
```

This contract enables:
- Unified backtest engine (`core/backtest_engine.py`)
- Consistent PIT correctness
- Easy multi-signal ensembles
- Centralized data fetching via adapters

---

## Signal Implementations

### 1. InstitutionalMomentum

**Module**: `signals/momentum/institutional_momentum.py`
**Status**: ✅ **Production** (backtest-ready)
**Academic Reference**: Jegadeesh & Titman (1993), Asness et al. (2013)

**Methodology**: 12-1 momentum (12-month formation, 1-month skip)

**Parameters**:
- `formation_period` (default: 252 days = 12 months)
- `skip_period` (default: 21 days = 1 month)
- `winsorize_pct` (default: [5, 95])
- `quintiles` (default: True)
- `quintile_mode` (default: 'adaptive')

**Score Scale**:
- Quintile values: `[-1, -0.5, 0, 0.5, 1]` when `quintiles=True`
- Continuous `[-1, 1]` when `quintiles=False`
- Uses `to_quintiles(mode='adaptive')` by default

**Data Dependencies**:
- Prices only (close prices)
- Lookback: 500 calendar days (hardcoded for ~308 trading days)

**Backtest Integration**: ✅ Complete
- Has `generate_cross_sectional_scores()` implementation
- Works with `make_signal_fn()` adapter
- Full test coverage in `test_backtest_engine.py`

**Current Usage**:
- `scripts/run_momentum_direct_baseline.py` (uses adapter)
- `scripts/run_ensemble_baseline.py` (via EnsembleSignal)
- Production ensemble: `get_momentum_v2_adaptive_quintile_ensemble()`

**Notes**:
- Canonical production implementation
- Proven to pass tight equivalence tests (1e-6, 1e-8 tolerances)
- Used as reference for all other signal integrations

---

### 2. CrossSectionalMomentum

**Module**: `signals/momentum/institutional_momentum.py`
**Status**: ⚠️ **Deprecated** (superseded by InstitutionalMomentum)

**Methodology**: Same as InstitutionalMomentum but with different API

**Parameters**: Same as InstitutionalMomentum

**Score Scale**: Quintiles via cross-sectional ranking

**Data Dependencies**: Prices only

**Backtest Integration**: ❌ Not compatible
- Has custom `generate_signals_cross_sectional(prices_dict, rebalance_dates)` API
- Different signature than new contract
- NOT recommended for new work

**Notes**:
- Legacy implementation from earlier architecture
- InstitutionalMomentum supersedes this with better API
- Kept for backward compatibility only
- **Do not use for new backtests or ensembles**

---

### 3. CrossSectionalQuality

**Module**: `signals/quality/cross_sectional_quality.py`
**Status**: ✅ **v1 Production** (backtest-ready as of Phase 3 Milestone 2)
**Academic Reference**: Asness, Frazzini & Pedersen (2018) "Quality Minus Junk"
**Specification**: `docs/QUALITY_SPEC.md`

**Methodology**: Composite quality score from profitability + growth + safety

**Parameters**:
- `w_profitability` (default: 0.4) - Weight on profitability component
- `w_growth` (default: 0.3) - Weight on growth component
- `w_safety` (default: 0.3) - Weight on safety component
- Constraint: `w_profitability + w_growth + w_safety == 1.0`
- `winsorize_pct` (default: [5, 95])
- `quintiles` (default: True)
- `min_coverage` (default: 0.5) - Minimum 50% of universe must have valid scores

**Score Scale**:
- Continuous `[-1, 1]` when `quintiles=False`
- Quintile values when `quintiles=True`
- Cross-sectional ranking at each rebalance

**Data Dependencies**:
- **Fundamentals** (quarterly/annual):
  - Profitability: ROE, ROA, GP/A (Gross Profit / Assets)
  - Growth: Revenue growth, earnings growth
  - Safety: Leverage ratios, volatility metrics
- **PIT-safe**: 33-day filing lag enforced
- Accessed via `DataManager.get_fundamentals()`

**Backtest Integration**: ✅ Complete (Milestone 2)
- Has `generate_cross_sectional_scores()` implementation
- Works with `make_signal_fn()` adapter
- Smoke test in `test_quality_integration.py`

**Current Usage**:
- `scripts/test_cross_sectional_quality_v1.py` (acceptance tests)
- `scripts/diagnose_quality_v1_phase1.py` (diagnostics)

**Notes**:
- **Proper cross-sectional implementation** (v1)
- Supersedes `InstitutionalQuality` v0
- Well-documented with complete mathematical spec
- Good candidate for Quality + Momentum ensembles
- Complexity: Requires fundamental data with proper PIT lag handling

---

### 4. InstitutionalQuality

**Module**: `signals/quality/institutional_quality.py`
**Status**: ⚠️ **v0 Deprecated** (superseded by CrossSectionalQuality v1)

**Methodology**: Quality composite with time-series scaling (problematic)

**Parameters**: Similar to CrossSectionalQuality

**Score Scale**: Time-series scaled (each stock vs its own history)

**Data Dependencies**: Fundamentals

**Backtest Integration**: ❌ Not recommended

**Notes**:
- **Time-series ranking is incorrect for factor investing**
- CrossSectionalQuality v1 fixes this with proper cross-sectional methodology
- Kept for historical reference only
- **Do not use for new backtests or ensembles**
- See `docs/QUALITY_SPEC.md` for why v0 was superseded

---

### 5. InstitutionalInsider

**Module**: `signals/insider/institutional_insider.py`
**Status**: ✅ **Implemented** (needs backtest integration)
**Academic Reference**: Cohen, Malloy & Pomorski (2012) "Decoding Inside Information"

**Methodology**: Dollar-weighted insider transactions with role hierarchy and cluster detection

**Parameters**:
- `lookback_days` (default: 90 days = 3 months)
- `min_transaction_value` (default: $10,000)
- `cluster_window` (default: 7 days)
- `cluster_min_insiders` (default: 3) - For cluster detection
- `role_weights` (default: CEO=3.0, CFO=2.5, Director=1.5, Officer=1.0, Other=0.5)

**Score Scale**:
- Dollar-weighted net insider buying/selling
- Winsorized and cross-sectionally ranked
- Higher scores = more insider buying

**Data Dependencies**:
- **Insider transaction data** (Form 4 filings via `sharadar_insiders`)
- **Price data** for dollar value scaling
- **PIT-safe**: Uses filing dates from regulatory filings
- Supports bulk mode (pre-fetched) or legacy mode (per-ticker queries)

**Backtest Integration**: ⏳ Planned (future milestone)
- Needs `generate_cross_sectional_scores()` implementation
- Bulk insider data fetching complicates adapter pattern
- Good candidate for Phase 3 Milestone 3+

**Current Usage**:
- `scripts/run_insider_phase1_baseline.py` (custom baseline)
- `scripts/diagnose_insider_phase1.py` (diagnostics)

**Notes**:
- More complex data dependency than momentum/quality
- Requires bulk insider data fetching for efficiency
- Role weighting and cluster detection add sophistication
- Good diversification from momentum/quality (low correlation)
- **Future work**: Standardize bulk data pattern in adapters

---

## Signal Maturity Matrix

| Signal | Status | Backtest-Ready | Data Complexity | Ensemble-Ready | Recommended Use |
|--------|--------|---------------|-----------------|----------------|-----------------|
| InstitutionalMomentum | Production | ✅ Yes | Low (prices) | ✅ Yes | **Use for all momentum needs** |
| CrossSectionalMomentum | Deprecated | ❌ No | Low (prices) | ❌ No | Skip (superseded) |
| CrossSectionalQuality | v1 Production | ✅ Yes | Medium (fundamentals) | ✅ Yes | **Use for quality factors** |
| InstitutionalQuality | v0 Deprecated | ❌ No | Medium (fundamentals) | ❌ No | Skip (superseded) |
| InstitutionalInsider | Implemented | ⏳ Planned | High (insider txns) | ⏳ Planned | Future integration |

---

## Adding New Signals

To add a new `InstitutionalSignal` to the catalog:

1. **Inherit from `InstitutionalSignal`**:
   ```python
   from core.institutional_base import InstitutionalSignal

   class MyNewSignal(InstitutionalSignal):
       def __init__(self, params, name='MyNewSignal'):
           super().__init__(params, name)
   ```

2. **Implement the contract**:
   ```python
   def generate_cross_sectional_scores(
       self, rebal_date, universe, data_manager
   ) -> pd.Series:
       # Your implementation
   ```

3. **Add to this catalog** with:
   - Academic reference
   - Parameters and defaults
   - Score scale and methodology
   - Data dependencies
   - PIT safety considerations

4. **Add smoke test** in `tests/test_*_integration.py`

5. **Document in** `docs/` if methodology is non-trivial

---

## Ensembles

Signals can be combined into **ensembles** for diversified factor exposure. Ensemble configurations are defined in `signals/ml/ensemble_configs.py`.

**Current multi-signal ensembles**:
- `get_momentum_quality_v1_ensemble()` - Momentum + Quality (25/75 weights, Phase 3, research status)
  - **Weights:** M=0.25, Q=0.75 (calibrated via grid sweep + Optuna, quality-heavy plateau)
  - **Performance (2015-2024):** Sharpe 2.876, Total Return 136%, Max DD -24%
  - Quality-heavy allocation outperforms pure momentum by +19% Sharpe (2.876 vs 2.413)
  - Grid sweep best: M=0.25/Q=0.75 (Sharpe 2.876); Optuna converged to M≈0.20/Q≈0.80
  - Regime finding: Quality adds value in 4/5 regimes (crisis, bear, recent); modest drag in steady bull markets
  - Diagnostics: Baseline, regime breakdown, weight sweep, and Optuna validation in `results/ensemble_baselines/`

**For details on**:
- Ensemble pathways (price-based vs cross-sectional)
- Adapter functions for backtest wiring
- Making a signal ensemble-ready

See **[`docs/ENSEMBLES.md`](ENSEMBLES.md)** for complete ensemble framework documentation.

---

## See Also

- `core/institutional_base.py` - Base class implementation
- `core/signal_adapters.py` - Backtest wiring utilities
- `core/backtest_engine.py` - Unified backtest harness
- `docs/ENSEMBLES.md` - Ensemble framework and multi-signal configurations
- `docs/QUALITY_SPEC.md` - CrossSectionalQuality mathematical specification
- `docs/ARCHITECTURE.md` - Overall system architecture

---

**Maintenance**: Update this catalog when:
- New signals are added
- Signal status changes (experimental → production, deprecated, etc.)
- Backtest integration status changes
- Major parameter defaults change
