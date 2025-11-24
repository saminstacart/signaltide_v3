# Ensemble Signal Architecture

**Last Updated:** 2025-11-21
**Status:** Production-Ready
**Current Version:** v1.0

## Overview

The ensemble layer combines multiple validated signals into a single portfolio score using configurable weights and normalization. This enables:

- **Multi-signal portfolios**: Combine Momentum, Quality, Insider, etc.
- **Signal diversification**: Reduce single-signal risk
- **Registry enforcement**: Only GO signals allowed in production (by default)
- **Flexible research**: Can include NO_GO signals for testing with `allow_no_go=True`

## Architecture

```
Signal Registry (GO/NO_GO validation)
           ↓
EnsembleSignal (orchestrator)
    ├─→ InstitutionalMomentum v2 (weight=1.0, normalize=zscore)
    ├─→ CrossSectionalQuality v1 (weight=0.5, normalize=rank)  # if GO
    └─→ InstitutionalInsider v1 (weight=0.0, allow_no_go=True) # research only
           ↓
Cross-sectional scores per ticker
           ↓
Weighted combination → Portfolio scores
```

## Key Components

### 1. EnsembleMember

Configuration for a single signal within an ensemble:

```python
@dataclass
class EnsembleMember:
    signal_name: str      # From signal registry (e.g., 'InstitutionalMomentum')
    version: str          # Version identifier (e.g., 'v2')
    weight: float         # Contribution weight (normalized by total)
    allow_no_go: bool     # If True, allows NO_GO signals (default: False)
    normalize: str        # Normalization method: 'zscore', 'rank', 'none'
    params: dict | None   # Optional signal-specific parameters
```

### 2. EnsembleSignal

Main ensemble class that:
1. **Validates** members against signal registry
2. **Instantiates** signal objects
3. **Generates** cross-sectional scores at each rebalance
4. **Normalizes** scores within universe
5. **Combines** using weighted average

## Usage Examples

### Example 1: Momentum-Only (Current Production)

```python
from signals.ml.ensemble_signal import EnsembleSignal, EnsembleMember

# Single-signal ensemble (validates against registry)
# NOTE: Use normalize="none" because InstitutionalMomentum already
# outputs quintile-mapped values [-1, -0.5, 0, 0.5, 1]
ensemble = EnsembleSignal(
    members=[
        EnsembleMember(
            signal_name="InstitutionalMomentum",
            version="v2",
            weight=1.0,
            normalize="none"  # Signal already normalized to quintiles
        )
    ],
    data_manager=dm,
    enforce_go_only=True  # Rejects NO_GO signals
)

# Generate scores at rebalance
scores = ensemble.generate_ensemble_scores(
    prices_by_ticker=prices_dict,  # {ticker: price_series}
    rebalance_date=rebalance_date
)
```

### Example 2: Multi-Signal Production (Future)

```python
# When Quality v2 passes validation...
ensemble = EnsembleSignal(
    members=[
        EnsembleMember("InstitutionalMomentum", "v2", weight=0.6, normalize="zscore"),
        EnsembleMember("CrossSectionalQuality", "v2", weight=0.4, normalize="rank"),
    ],
    enforce_go_only=True
)
```

### Example 3: Research with NO_GO Signals

```python
# Test Insider v1 in sandbox alongside Momentum v2
ensemble = EnsembleSignal(
    members=[
        EnsembleMember("InstitutionalMomentum", "v2", weight=0.7),
        EnsembleMember("InstitutionalInsider", "v1", weight=0.3, allow_no_go=True),
    ],
    enforce_go_only=True  # Still enforced, but allow_no_go=True overrides
)
```

## Normalization Methods

**CRITICAL:** Choose normalization based on whether signals already output normalized values:

- **Signals with internal normalization** (e.g., quintiles, percentiles): Use `normalize="none"`
  - Example: `InstitutionalMomentum` with `quintiles=True` → outputs [-1, -0.5, 0, 0.5, 1]
  - Ensemble should pass through these values unchanged

- **Signals with raw outputs** (e.g., raw returns, Z-scores): Use `normalize="zscore"` or `"rank"`
  - Example: A signal that outputs raw 12-month returns
  - Ensemble normalizes cross-sectionally before combining

**Validation:** Always run `scripts/validate_ensemble_momentum.py --debug` after adding new signals to verify correct normalization.

### Z-Score

Cross-sectional standardization within universe:

```
normalized = (raw_score - mean) / std
```

**Pros:**
- Preserves relative magnitude
- Standard statistical approach
- Works well when scores are normally distributed

**Cons:**
- Sensitive to outliers
- Assumes signal scores have meaningful scale

### Rank

Percentile ranking, scaled to [-0.5, 0.5]:

```
normalized = percentile_rank(raw_score) - 0.5
```

**Pros:**
- Robust to outliers
- Uniform distribution by construction
- Good when signals have different scales

**Cons:**
- Loses magnitude information
- Treats small differences as equal to large ones

### None

Pass through raw scores unchanged:

```
normalized = raw_score
```

**Pros:**
- Preserves original signal scale
- Useful when signals already normalized

**Cons:**
- Requires signals to be on comparable scales
- Can be dominated by high-variance signals

## Signal Registry Integration

### GO Signal (Production)

Signals that passed Phase 1 validation:

```python
# Automatically allowed
EnsembleMember("InstitutionalMomentum", "v2", weight=1.0)
```

### NO_GO Signal (Research Only)

Signals that failed validation but you want to test:

```python
# Requires explicit allow_no_go=True
EnsembleMember(
    signal_name="InstitutionalInsider",
    version="v1",
    weight=0.3,
    allow_no_go=True  # REQUIRED for NO_GO signals
)
```

### Enforcement Modes

**Production (enforce_go_only=True):**
- Rejects NO_GO signals unless `allow_no_go=True`
- Use for live trading
- Prevents accidental use of unvalidated signals

**Research (enforce_go_only=False):**
- Allows any signal from registry
- Use for rapid prototyping
- Still validates signal exists in registry

## Adding New Signals

### 1. Pass Phase 1 Validation

Signal must complete Phase 1 diagnostic and receive GO verdict.

### 2. Add to Signal Registry

Update `core/signal_registry.py`:

```python
SignalStatus(
    signal_name='NewSignal',
    version='v1',
    universe='sp500_actual',
    status='GO',
    full_sharpe=0.45,
    oos_sharpe=0.52,
    verdict_notes='Passed all gates...'
)
```

### 3. Map Implementation Class

Update `signals/ml/ensemble_signal.py`:

```python
_SIGNAL_CLASS_MAP = {
    ("InstitutionalMomentum", "v2"): InstitutionalMomentum,
    ("NewSignal", "v1"): NewSignalClass,  # <-- Add here
}
```

### 4. Use in Ensemble

```python
EnsembleMember("NewSignal", "v1", weight=0.5)
```

## Weighting Guidelines

### Equal Weight (Default)

Simple diversification when signals have similar Sharpe ratios:

```python
members=[
    EnsembleMember("Signal1", "v1", weight=1.0),
    EnsembleMember("Signal2", "v1", weight=1.0),
]
# Effective weights: 50% / 50%
```

### Sharpe-Weighted

Weight by out-of-sample Sharpe ratios:

```python
# Momentum OOS Sharpe: 0.742
# Quality OOS Sharpe: 0.520
members=[
    EnsembleMember("InstitutionalMomentum", "v2", weight=0.742),
    EnsembleMember("CrossSectionalQuality", "v2", weight=0.520),
]
# Effective weights: 59% / 41%
```

### Conservative Allocation

Give more weight to higher-conviction signals:

```python
members=[
    EnsembleMember("InstitutionalMomentum", "v2", weight=0.7),  # proven
    EnsembleMember("NewSignal", "v1", weight=0.3),  # less proven
]
```

## Testing & Validation

### Unit Test: Momentum-Only Equivalence

The ensemble with Momentum-only should match standalone Momentum v2 results:

```python
# Ensemble path
ensemble_scores = ensemble.generate_ensemble_scores(...)

# Direct path
momentum_scores = InstitutionalMomentum(params).generate_signals(...)

# Should match within numerical tolerance
assert (ensemble_scores - momentum_scores).abs().max() < 1e-9
```

### Integration Test: Multi-Signal Baseline

Run full backtest with multi-signal ensemble:
1. Use known GO signals only
2. Compare Sharpe ratio to single-signal baselines
3. Check for diversification benefit (ensemble volatility < single signals)
4. Verify no lookahead bias (use same point-in-time universe)

## Common Patterns

### Pattern 1: Single Signal Wrapper

Use ensemble even for single signal to maintain consistent API:

```python
# Production: Momentum-only
ensemble = EnsembleSignal(
    members=[EnsembleMember("InstitutionalMomentum", "v2", weight=1.0)]
)
```

### Pattern 2: Progressive Roll-out

Start with proven signal, add new signals incrementally:

```python
# Month 1-3: Momentum only (weight=1.0)
# Month 4-6: Add Quality at 20% (Momentum=0.8, Quality=0.2)
# Month 7+:   Equal weight (Momentum=0.5, Quality=0.5)
```

### Pattern 3: Research Sandbox

Test NO_GO signal alongside GO signals:

```python
ensemble = EnsembleSignal(
    members=[
        EnsembleMember("InstitutionalMomentum", "v2", weight=0.5),
        EnsembleMember("ExperimentalSignal", "v1", weight=0.5, allow_no_go=True),
    ]
)
```

## Current Production Config

**As of 2025-11-21:**

Only **InstitutionalMomentum v2** has passed Phase 1 validation (GO status).

Production ensemble config:

```python
EnsembleSignal(
    members=[
        EnsembleMember(
            signal_name="InstitutionalMomentum",
            version="v2",
            weight=1.0,
            normalize="none"  # Signal already outputs quintiles
        )
    ],
    enforce_go_only=True
)
```

## Validation Status

### ✅ Ensemble Layer Validated (2025-11-21)

The ensemble framework has been validated and is now the **canonical path** for all signal generation:

1. **Numerical Validation**: `scripts/validate_ensemble_momentum.py`
   - Ensemble scores match direct InstitutionalMomentum v2 signal
   - Max difference: 0.00e+00 (perfect match)
   - Tested across 6 rebalances (2022-2024)

2. **Production Config**: `signals/ml/ensemble_configs.py`
   - `get_momentum_v2_ensemble(dm)` provides centralized config
   - Trial 11 canonical parameters (308d formation, 0d skip, 9.2% winsor)
   - Registry-validated (GO status only)

3. **Baseline Integration**: `scripts/run_ensemble_baseline.py`
   - Full 2015-2024 backtest: 129.89% total return, Sharpe 0.601
   - Produces reasonable metrics and equity curves
   - Debug mode confirms ensemble == direct at each rebalance

### Usage

All future momentum-based strategies should use the ensemble path:

```python
from signals.ml.ensemble_configs import get_momentum_v2_ensemble

ensemble = get_momentum_v2_ensemble(dm)
scores = ensemble.generate_ensemble_scores(prices_by_ticker, rebalance_date)
```

## Next Steps

1. **~~Validate ensemble layer~~**: ✅ **COMPLETE** (2025-11-21)
2. **Develop Quality v2**: Improve Quality signal to pass Phase 1
3. **Test multi-signal**: Once 2+ GO signals, run ensemble backtests
4. **Optimize weights**: Use walk-forward optimization for weight allocation
5. **Production deployment**: Wire ensemble into live trading system

## References

- Signal Registry: `core/signal_registry.py`
- Implementation: `signals/ml/ensemble_signal.py`
- Momentum v2 Spec: `results/MOMENTUM_PHASE2_TRIAL11_DIAGNOSTIC.md`
- Quality v1 Postmortem: `results/quality_v1_phase1_report.md`
- Insider v1 Postmortem: `results/INSIDER_PHASE1_REPORT.md`

---

**Document Version:** 1.0
**Last Review:** 2025-11-21
**Next Review:** 2025-12-21
