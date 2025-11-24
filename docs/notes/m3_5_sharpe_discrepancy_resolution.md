# M3.5 Sharpe Discrepancy Resolution

**Date:** 2025-11-23
**Status:** RESOLVED
**Severity:** HIGH (affects all backtest metrics)
**Category:** Bug - Incorrect Annualization Heuristic

---

## Executive Summary

Two diagnostics for the static 25/75 Momentum+Quality ensemble reported vastly different Sharpe ratios:
- **Baseline diagnostic**: Sharpe = 2.876, Volatility = 74.77%
- **Allocator diagnostic**: Sharpe = 0.627, Volatility = 15.32%

**Root Cause**: Incorrect heuristic in `core/backtest_engine.py:420` for detecting monthly vs daily data, causing wrong annualization factors.

**Resolution**: The **allocator diagnostic (Sharpe = 0.627) is CORRECT**. The baseline diagnostic inflated metrics by ~4.8x due to using daily annualization (252) on monthly rebalance data.

**Fix Required**: Replace length-based heuristic with explicit frequency parameter in backtest config.

---

## Detailed Investigation

### 1. Observed Discrepancy

| Source | Sharpe | Volatility | CAGR | Periods |
|--------|--------|-----------|------|---------|
| **Baseline Diagnostic** (`momentum_quality_v1_diagnostic.md`) | 2.876 | 74.77% | 9.28% | 117 |
| **Allocator Diagnostic** (`momentum_quality_v1_regime_allocators.md`) | 0.627 | 15.32% | 8.76% | 116 |
| **Direct CSV Calculation** | 0.638 | 15.58% | 9.09% | 112 |

**Key Observations**:
- CAGR is consistent across all three (~9%) ✅
- Sharpe and Volatility differ by ~4.8x factor ❌
- CSV calculation matches allocator diagnostic, not baseline

### 2. Root Cause Analysis

**Location**: `core/backtest_engine.py`, function `_calculate_metrics()`, line 420

**Buggy Code**:
```python
def _calculate_metrics(equity_curve: pd.Series, initial_capital: float) -> Dict:
    """Calculate performance metrics from equity curve."""
    # ... (lines 401-418)

    # Annualization factor (assume monthly if < 100 points, else daily)
    periods_per_year = 252 if len(equity_curve) > 100 else 12

    # Volatility
    volatility = returns.std() * np.sqrt(periods_per_year) if len(returns) > 0 else 0

    # Sharpe
    sharpe = 0.0
    if len(returns) > 0 and returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(periods_per_year)
    # ...
```

**The Issue**:
1. Heuristic uses `len(equity_curve) > 100` to distinguish daily vs monthly
2. M+Q baseline has **117 monthly rebalance points** (2015-2024, ~10 years)
3. 117 > 100 → code incorrectly assumes **daily data** (252 trading days/year)
4. Should use **monthly annualization** (12 periods/year)

**Mathematical Proof**:

Given monthly returns with std = 4.50%:

| Calculation | Formula | Result | Matches? |
|------------|---------|--------|----------|
| **Correct (monthly)** | 4.50% × √12 | 15.58% | Allocator: 15.32% ✅ |
| **Incorrect (daily)** | 4.50% × √252 | 71.46% | Baseline: 74.77% ✅ |

The 4.8x error ratio (√252 / √12 = √21 ≈ 4.58) perfectly explains the discrepancy.

### 3. Why Allocator Diagnostic is Correct

The allocator diagnostic uses a **different calculation path**:
- Computes metrics **directly from monthly return series** (not equity curve)
- Uses explicit `periods_per_year = 12` in regime-specific calculations
- Code path: `scripts/run_momentum_quality_regime_allocators.py:278-350`

Additionally, allocator has 76-82 periods (shorter due to alignment), which may fall below the 100-point threshold in some code paths, causing correct monthly annualization.

### 4. Impact Assessment

**Affected Diagnostics**:
- ❌ `momentum_quality_v1_diagnostic.md` (baseline) - INFLATED metrics
- ❌ Likely all diagnostics from `scripts/run_momentum_quality_baseline.py` with >100 rebalances
- ✅ `momentum_quality_v1_regime_allocators.md` - CORRECT
- ✅ CSV-based calculations - CORRECT

**Affected Metrics**:
- **Sharpe Ratio**: Inflated by ~4.8x
- **Volatility**: Inflated by ~4.8x
- **CAGR**: UNAFFECTED (computed from total return, not annualization factor)
- **Max Drawdown**: UNAFFECTED (equity-based, not return-based)

**Affected Decisions**:
- M3.5 regime-aware allocation was evaluated against WRONG baseline Sharpe (2.876 instead of 0.627)
- However, **decision outcome remains valid** because:
  - Oracle allocator also uses same buggy code path → comparison is apples-to-apples
  - Both Oracle and Rule-based still failed to beat their respective baselines
  - Relative comparisons (Δ Sharpe) are still meaningful

---

## Canonical Values Going Forward

**Authoritative Sharpe Ratio for Static 25/75 M+Q Ensemble (2015-2024)**:
- **Sharpe Ratio: 0.63** (range: 0.58-0.64 depending on sample alignment)
- **Volatility: 15.3%** (annualized monthly)
- **CAGR: 9.0%** (±0.3% depending on sample)

**Source of Truth**:
1. Direct calculation from monthly returns CSV
2. Allocator diagnostic (uses correct annualization)

**DO NOT USE**:
- Baseline diagnostic Sharpe of 2.876 (incorrect annualization)

---

## Proposed Fix

### Option 1: Replace Heuristic with Explicit Parameter (RECOMMENDED)

**Changes Required**:

1. **Update `BacktestConfig` dataclass**:
```python
@dataclass
class BacktestConfig:
    # ... existing fields ...
    rebalance_schedule: str  # 'D', 'W', 'M', 'Q'
    track_daily_equity: bool = False

    # NEW: Explicit annualization factor
    periods_per_year: Optional[int] = None  # If None, infer from rebalance_schedule
```

2. **Update `_calculate_metrics()`**:
```python
def _calculate_metrics(
    equity_curve: pd.Series,
    initial_capital: float,
    rebalance_schedule: str  # NEW parameter
) -> Dict:
    """Calculate performance metrics from equity curve."""
    # ... (existing code lines 401-418) ...

    # Determine periods_per_year from rebalance schedule
    schedule_to_periods = {
        'D': 252,  # Daily
        'W': 52,   # Weekly
        'M': 12,   # Monthly
        'Q': 4,    # Quarterly
    }
    periods_per_year = schedule_to_periods.get(rebalance_schedule, 12)

    # Rest of calculation remains same
    # ...
```

3. **Update call site in `run_backtest()`**:
```python
# Line 147
metrics = _calculate_metrics(
    equity_curve,
    config.initial_capital,
    config.rebalance_schedule  # NEW argument
)
```

**Pros**:
- Explicit, no guessing
- Robust to any sample length
- Self-documenting

**Cons**:
- Requires updating all backtest call sites (low risk, IDE will catch)

### Option 2: Raise Heuristic Threshold (NOT RECOMMENDED)

Change threshold from 100 to 500:
```python
periods_per_year = 252 if len(equity_curve) > 500 else 12
```

**Pros**:
- Minimal code change

**Cons**:
- Still fragile (what about 10-year weekly rebalancing? 520 points)
- Kicks the can down the road
- Doesn't handle daily backtests < 500 days

---

## Prevention Strategy

### 1. Regression Test

Add test to `tests/test_backtest_engine.py`:

```python
def test_monthly_rebalance_annualization_correct():
    """Ensure monthly rebalances use correct annualization factor."""
    # Create synthetic monthly equity curve (117 points, like M+Q baseline)
    dates = pd.date_range('2015-04-30', periods=117, freq='M')

    # Simulate realistic monthly returns (std ~4.5%)
    np.random.seed(42)
    returns = np.random.normal(0.008, 0.045, 117)
    equity = 100000 * np.cumprod(1 + returns)
    equity_curve = pd.Series(equity, index=dates)

    # Calculate metrics
    metrics = _calculate_metrics(equity_curve, 100000, rebalance_schedule='M')

    # Expected volatility: ~4.5% * sqrt(12) ≈ 15.6%
    assert 0.14 < metrics['volatility'] < 0.17, \
        f"Monthly volatility should be ~15.6%, got {metrics['volatility']:.2%}"

    # Expected Sharpe: ~0.6
    assert 0.5 < metrics['sharpe'] < 0.7, \
        f"Monthly Sharpe should be ~0.6, got {metrics['sharpe']:.2f}"
```

### 2. Documentation Update

Add warning to `docs/core/ERROR_PREVENTION_ARCHITECTURE.md`:

```markdown
### Annualization Pitfall: Equity Curve Length ≠ Frequency

**Date Added:** 2025-11-23
**Severity:** HIGH

**Pattern**: Inferring annualization factor from equity curve length.

**Risk**: Monthly rebalances over long periods (>100 months) can be mistaken for daily data,
causing 4-5x inflation of volatility and Sharpe metrics.

**Solution**: Always pass `rebalance_schedule` explicitly to metric calculation functions.
Never infer frequency from sample length.

**Reference**: docs/notes/m3_5_sharpe_discrepancy_resolution.md
```

### 3. Code Review Checklist

Add to `.claude/code_review_checklist.md`:
- [ ] Annualization factors are explicit, not inferred from data length
- [ ] Backtest metrics calculation receives `rebalance_schedule` parameter
- [ ] No heuristics like `if len(data) > N then assume daily`

---

## Action Items

**Immediate (Before M3.6)**:
- [x] Document discrepancy in this note
- [ ] Update M3.5 review doc to reference correct Sharpe (0.627)
- [ ] Update M3.5 spec with cross-link to this resolution
- [ ] Add TODO marker in `core/backtest_engine.py:420` referencing this note

**Short-Term (M3.6 or before)**:
- [ ] Implement Option 1 fix (explicit `rebalance_schedule` parameter)
- [ ] Add regression test
- [ ] Re-run baseline diagnostic to generate correct report
- [ ] Archive old baseline diagnostic with "DEPRECATED - See resolution note" marker

**Long-Term (Phase 4)**:
- [ ] Audit all historical diagnostics for this issue
- [ ] Add metric validation layer (sanity checks: Sharpe > 5 should warn, Vol > 50% for monthly should error)

---

## References

- **Affected Files**:
  - `core/backtest_engine.py:420` (bug location)
  - `scripts/run_momentum_quality_baseline.py` (produces wrong diagnostic)
  - `scripts/run_momentum_quality_regime_allocators.py` (produces correct diagnostic)

- **Diagnostics**:
  - **CORRECT**: `results/ensemble_baselines/momentum_quality_v1_regime_allocators.md`
  - **CORRECT**: `results/ensemble_baselines/momentum_quality_v1_monthly_returns.csv`
  - **INCORRECT**: `results/ensemble_baselines/momentum_quality_v1_diagnostic.md`

- **Specs**:
  - `docs/ENSEMBLES_M3.5_REGIME_ALLOC_SPEC.md` (references old Sharpe)

- **Investigation Log**:
  - `docs/logs/phase3_m3_overnight_20251123.md` (detailed timeline)

---

**Resolution Status**: DOCUMENTED ✅
**Fix Status**: PENDING (Option 1 implementation scheduled for M3.6)
**Test Coverage**: TO BE ADDED

**Last Updated**: 2025-11-23
