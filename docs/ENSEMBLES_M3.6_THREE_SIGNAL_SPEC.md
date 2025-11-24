# Phase 3 M3.6 - 3-Signal Ensemble (Momentum + Quality + Insider) Specification

**Created:** 2025-11-23
**Status:** RESEARCH / INFRASTRUCTURE READY
**Goal:** Add InstitutionalInsider to M+Q ensemble for 3-way signal diversification and alpha enhancement

---

## 1. Executive Summary

The static 25/75 Momentum+Quality ensemble delivers solid risk-adjusted returns (Sharpe ~0.63, CAGR ~9%) across 2015-2024. However, it relies on only **two signal families**:
- **InstitutionalMomentum:** Price-driven, trend-following
- **CrossSectionalQuality:** Fundamental-driven, defensive

Adding **InstitutionalInsider** introduces a third, orthogonal information source:
- **InstitutionalInsider:** Behavioral/information-driven, insider transaction analysis

### Diversification Hypothesis

3-signal ensemble may improve Sharpe via:
1. **Lower correlation**: Insider trades are event-driven, complementary to price trends and fundamentals
2. **Alpha potential**: Insider buying/selling contains private information (Cohen-Malloy-Pomorski 2012)
3. **Regime complementarity**: Insiders may be more active during transitions (improving signal during regime shifts)

### Infrastructure Status

- ✅ **InstitutionalInsider implementation complete** (Cohen-Malloy-Pomorski methodology)
- ✅ **Cross-sectional API added** (`generate_cross_sectional_scores()` - 2025-11-23)
- ✅ **Bulk data fetching** (50-100x performance improvement via `get_insider_trades_bulk()`)
- ✅ **PIT safety** (filing dates enforced via `as_of_date` filtering)
- ⏳ **Ensemble config**: Not yet implemented (TODO - see Section 6)
- ⏳ **Diagnostic backtest**: Not yet run (TODO - see Section 7)

### M3.6 Acceptance Gate

**Proceed to full evaluation if:**
- 3-signal Sharpe ≥ 0.63 (M+Q baseline)
- 3-signal Max DD ≤ -25% (M+Q baseline: -24.21%)
- Insider data coverage ≥ 75% of S&P 500 universe (2015-2024)

---

## 2. Problem Statement: Why Add Insider?

### Motivation

M+Q ensemble performs well but has known limitations:
- **Momentum:** Vulnerable to reversals, whipsaw in choppy markets
- **Quality:** Creates drag during steady bull markets (see M3.5 regime analysis)
- **Correlation:** M+Q may be moderately correlated (both respond to earnings quality)

**Insider signal offers orthogonal information:**
- Insiders trade based on **private knowledge** (not public price/fundamental data)
- Insider clusters signal high-conviction views (coordinated buying/selling)
- Timing: Insiders trade **before** events (leading indicator vs lagging momentum)

### Academic Basis

**Cohen, Malloy & Pomorski (2012)** "Decoding Inside Information"
- Key finding: **Routine trades** (10b5-1 plans) vs **opportunistic trades** differ in information content
- **Cluster detection**: 3+ insiders trading same direction within 7 days → strong signal
- **Role hierarchy**: CEO/CFO trades more informative than routine officer transactions

**Implementation in InstitutionalInsider:**
- Dollar-weighted transactions (larger trades = more conviction)
- Role weights (CEO=3.0, CFO=2.5, Director=1.5, Officer=1.0)
- Cluster detection (3+ insiders within 7 days → 2x weight boost)
- 90-day lookback (captures recent insider activity)
- $10K minimum transaction (filters noise)

### Expected Benefits

1. **Diversification**: Low expected correlation with M+Q
2. **Alpha**: Private information edge (if signal is well-constructed)
3. **Regime adaptability**: Insiders may trade more actively during transitions
4. **Drawdown mitigation**: Insiders may reduce exposure before crashes (early warning)

### Risks & Challenges

1. **Data coverage**: Not all S&P 500 stocks have frequent insider activity
2. **Signal sparsity**: Insider trades are intermittent (not monthly like momentum/quality)
3. **Noise vs information**: Many insider trades are routine (10b5-1 plans, option exercises)
4. **Regulatory lag**: Form 4 filing delay (typically 2 days, but can be longer)

---

## 3. Proposed 3-Signal Ensemble Design

### 3.1 Ensemble Name & Identifier

**Name:** `momentum_quality_insider_v1`
**Nickname:** "M+Q+I v1" or "3-Signal Baseline"
**Status:** RESEARCH / BACKTEST PENDING

### 3.2 Member Signals

| Signal | Class | Version | Normalization | Data Source |
|--------|-------|---------|---------------|-------------|
| InstitutionalMomentum | Momentum | v2 | Adaptive quintiles | Sharadar prices |
| CrossSectionalQuality | Quality | v1 | Adaptive quintiles | Sharadar fundamentals |
| InstitutionalInsider | Insider | v1 | Adaptive quintiles | Sharadar insiders |

**Common Configuration:**
- Universe: S&P 500 actual constituents (PIT), min_price=$5
- Rebalance: Monthly (month-end)
- Period: 2015-04-01 to 2024-12-31 (aligns with M+Q baseline)
- Long-only, equal-weight within quintiles

### 3.3 Weighting Schemes

#### Option 1: Equal 3-Way Split (Conservative Baseline)
```python
weights = {
    'momentum': 0.333,
    'quality': 0.333,
    'insider': 0.334
}
```
**Rationale:** No prior bias, treat all signals equally, test pure diversification hypothesis.

#### Option 2: Quality-Dominant (Based on M3.4 Calibration)
```python
weights = {
    'momentum': 0.25,
    'quality': 0.50,
    'insider': 0.25
}
```
**Rationale:**
- M3.4 found 25/75 M+Q optimal (quality-heavy)
- Preserve quality dominance, split "momentum budget" between M and I
- Hypothesis: Insider complements momentum (both alpha-seeking), quality anchors

#### Option 3: Insider-Light (Conservative Test)
```python
weights = {
    'momentum': 0.30,
    'quality': 0.60,
    'insider': 0.10
}
```
**Rationale:**
- Start with small insider weight to test incremental value
- Minimize risk if insider signal is noisy or low-coverage
- Easier to detect marginal Sharpe contribution

### 3.4 Recommended Starting Point

**Propose Option 2 (0.25 / 0.50 / 0.25) as v1 baseline:**

1. **Aligned with M3.4 findings**: Quality-heavy allocation already validated
2. **Symmetric M/I split**: Equal weight to alpha-seeking signals, quality as anchor
3. **Testable hypothesis**: Does splitting momentum budget with insider improve Sharpe?
4. **Clear comparison**: If 3-signal Sharpe ≥ M+Q Sharpe, insider adds value

---

## 4. Acceptance Gates (vs M+Q Baseline)

### 4.1 Baseline Performance (M+Q 25/75)

From `results/ensemble_baselines/momentum_quality_v1_regime_allocators_review.md` (CORRECT values):

| Metric | Value |
|--------|-------|
| **Sharpe** | **0.627** |
| **CAGR** | 8.76% |
| **Volatility** | 15.32% |
| **Max Drawdown** | -24.21% |
| **Total Return** | 125.24% |
| **Periods** | 116 (monthly rebalances) |

### 4.2 Minimum Acceptance Criteria

**3-signal ensemble must meet ALL of:**

1. ✅ **Sharpe ≥ M+Q baseline** (≥ 0.627)
2. ✅ **Max Drawdown ≤ M+Q baseline** (≤ -24.21%)
3. ✅ **Positive Sharpe** (> 0.0, i.e., strategy is profitable)
4. ✅ **Insider data coverage ≥ 75%** (at least 75% of rebalance dates have ≥75% of universe with insider data)

### 4.3 Strong Go Criteria (Proceed to Weight Calibration)

**If ensemble meets:**

1. ✅ **Sharpe improvement ≥ 5%** (Sharpe ≥ 0.658, i.e., 0.627 × 1.05)
2. ✅ **Max Drawdown improvement ≥ 2%** (Max DD ≥ -23.71%, i.e., -24.21% × 0.98)
3. ✅ **CAGR ≥ M+Q CAGR** (≥ 8.76%)
4. ✅ **No catastrophic periods** (no single year with return < -30%)

**Then → Proceed to:**
- Weight grid search (vary w_m, w_q, w_i)
- Optuna refinement
- Regime-specific analysis

### 4.4 No-Go Criteria (Insider Not Viable)

**Abandon M3.6 if:**

1. ❌ **Sharpe < 0.60** (worse than M+Q by >10%)
2. ❌ **Max DD < -30%** (significantly worse drawdown)
3. ❌ **Insider coverage < 50%** (too sparse for reliable signal)
4. ❌ **Turnover explosion** (>200% annualized, indicating insider signal is too noisy)

---

## 5. Implementation Status & TODOs

### 5.1 Infrastructure (COMPLETE ✅)

- [x] `InstitutionalInsider.generate_cross_sectional_scores()` method (2025-11-23)
- [x] `DataManager.get_insider_trades_bulk()` method (existing)
- [x] Bulk mode support in `InstitutionalInsider.generate_signals()` (existing)
- [x] PIT safety (filing dates enforced via `as_of_date`)

### 5.2 Ensemble Config (TODO ⏳)

**File:** `signals/ml/ensemble_configs.py`

**Add function:**
```python
def get_momentum_quality_insider_v1_ensemble(
    dm: DataManager,
    weights: Optional[Dict[str, float]] = None
) -> EnsembleSignal:
    """
    3-signal M+Q+Insider ensemble (Phase 3 M3.6).

    Default weights: {'momentum': 0.25, 'quality': 0.50, 'insider': 0.25}

    Args:
        dm: DataManager instance
        weights: Optional weight override (must sum to 1.0)

    Returns:
        EnsembleSignal with 3 member signals
    """
    if weights is None:
        weights = {'momentum': 0.25, 'quality': 0.50, 'insider': 0.25}

    # Validate weights sum to 1.0
    assert abs(sum(weights.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"

    # Create member signals
    momentum = InstitutionalMomentum({'formation_period': 252, 'skip_period': 21})
    quality = CrossSectionalQuality({
        'use_profitability': True,
        'use_growth': True,
        'use_safety': True
    }, data_manager=dm)
    insider = InstitutionalInsider({
        'lookback_days': 90,
        'cluster_window': 7,
        'cluster_min_insiders': 3
    }, data_manager=dm)

    # Create ensemble
    ensemble = EnsembleSignal(
        name='momentum_quality_insider_v1',
        members={
            'momentum': momentum,
            'quality': quality,
            'insider': insider
        },
        weights=weights,
        data_manager=dm
    )

    return ensemble
```

**Effort:** 1 hour (implementation + testing)

### 5.3 Diagnostic Script (TODO ⏳)

**File:** `scripts/run_momentum_quality_insider_baseline.py`

**Script Outline:**
```python
#!/usr/bin/env python3
"""
Phase 3 M3.6 - 3-Signal (M+Q+Insider) Baseline Diagnostic

Run full-period backtest for momentum_quality_insider_v1 ensemble
and compare vs M+Q baseline.

Outputs:
- results/ensemble_baselines/momentum_quality_insider_v1_diagnostic.csv
- results/ensemble_baselines/momentum_quality_insider_v1_diagnostic.md
"""

from data.data_manager import DataManager
from signals.ml.ensemble_configs import get_momentum_quality_insider_v1_ensemble
from core.backtest_engine import BacktestConfig, run_backtest
# ... (implementation similar to run_momentum_quality_baseline.py)

def main():
    # 1. Setup
    dm = DataManager()
    config = BacktestConfig(
        start_date='2015-04-01',
        end_date='2024-12-31',
        initial_capital=100000,
        rebalance_schedule='M'
    )

    # 2. Create ensemble
    ensemble = get_momentum_quality_insider_v1_ensemble(dm)

    # 3. Define universe and signal functions (using cross-sectional APIs)
    def universe_fn(rebal_date: str) -> List[str]:
        # S&P 500 actual PIT universe
        pass

    def signal_fn(rebal_date: str, universe: List[str]) -> pd.Series:
        # Generate 3-signal ensemble scores
        # Call each signal's generate_cross_sectional_scores()
        # Combine with weights
        pass

    # 4. Run backtest
    result = run_backtest(universe_fn, signal_fn, config)

    # 5. Write outputs
    # - CSV with monthly returns
    # - MD with performance summary + comparison vs M+Q

if __name__ == '__main__':
    main()
```

**Effort:** 2 hours (adapt from M+Q baseline script)

### 5.4 Comparison Report (TODO ⏳)

**File:** `results/ensemble_baselines/momentum_quality_insider_v1_review.md`

**Template Sections:**
1. Executive Summary (GO/NO-GO decision)
2. Performance Comparison Table (M+Q vs M+Q+I)
3. Insider Data Coverage Analysis
4. Per-Regime Breakdown (if GO)
5. Acceptance Gate Results
6. Recommendations

**Effort:** 1 hour (auto-generated from diagnostic output)

---

## 6. Data Coverage Assessment (PRE-FLIGHT CHECK)

### 6.1 Required Queries

Before running full backtest, verify insider data coverage:

```sql
-- 1. Overall insider data coverage (2015-2024)
SELECT
  MIN(filingdate) as earliest,
  MAX(filingdate) as latest,
  COUNT(DISTINCT ticker) as unique_tickers,
  COUNT(*) as total_trades
FROM sharadar_insiders
WHERE filingdate >= '2015-01-01'
  AND filingdate <= '2024-12-31';

-- 2. Trades per year (density check)
SELECT
  SUBSTR(filingdate, 1, 4) as year,
  COUNT(*) as trades_count,
  COUNT(DISTINCT ticker) as unique_tickers
FROM sharadar_insiders
WHERE filingdate >= '2015-01-01'
  AND filingdate <= '2024-12-31'
GROUP BY year
ORDER BY year;

-- 3. S&P 500 overlap (TODO: requires universe table join)
-- Estimate: 80-95% of S&P 500 should have insider activity
```

### 6.2 Coverage Acceptance Criteria

**Minimum viable:**
- ≥ 75% of S&P 500 universe has at least 1 insider trade per year (2015-2024)
- ≥ 80% of monthly rebalance dates have insider data for ≥75% of universe

**If coverage is poor (<50%):**
- Document limitation
- Create placeholder ensemble (NotImplementedError)
- Flag as future work when better data available

---

## 7. Diagnostic Script Execution Plan

### 7.1 Pre-Flight Checklist

Before running full diagnostic:

1. ✅ Verify `InstitutionalInsider.generate_cross_sectional_scores()` compiles
2. ⏳ Run data coverage queries (Section 6.1)
3. ⏳ Smoke test ensemble on 3-ticker universe (AAPL, MSFT, GOOGL)
4. ⏳ Verify bulk fetching works (single DB query for all tickers)

### 7.2 Execution Steps

```bash
# 1. Quick smoke test (3 tickers, 1 year)
python3 -c "
from signals.insider.institutional_insider import InstitutionalInsider
from data.data_manager import DataManager
import pandas as pd

dm = DataManager()
insider = InstitutionalInsider({'lookback_days': 90})
scores = insider.generate_cross_sectional_scores(
    rebal_date=pd.Timestamp('2024-01-31'),
    universe=['AAPL', 'MSFT', 'GOOGL'],
    data_manager=dm
)
print('Smoke test passed:', scores)
"

# 2. Full diagnostic (S&P 500, 2015-2024)
python3 scripts/run_momentum_quality_insider_baseline.py

# 3. Review results
less results/ensemble_baselines/momentum_quality_insider_v1_diagnostic.md

# 4. Compare vs M+Q baseline
python3 scripts/compare_ensemble_baselines.py \
  --baseline momentum_quality_v1 \
  --candidate momentum_quality_insider_v1
```

### 7.3 Expected Runtime

- Smoke test: <1 min
- Full diagnostic: ~10-15 min (bulk fetching reduces time vs per-ticker queries)
- Comparison report: <1 min

---

## 8. Next Steps & Decision Tree

### 8.1 If Pre-Flight Coverage Check Fails (<50% coverage)

**Action:**
1. Document coverage limitation in M3.6 NOTES
2. Create placeholder ensemble with NotImplementedError
3. Mark M3.6 as "DATA LIMITED - DEFERRED"
4. Move to Phase 4 (hardening/testing)

### 8.2 If Pre-Flight Coverage Check Passes (≥75% coverage)

**Action:**
1. Implement ensemble config (Section 5.2)
2. Implement diagnostic script (Section 5.3)
3. Run full backtest
4. Apply acceptance gates (Section 4)

### 8.3 If Diagnostic Passes Acceptance Gates (Sharpe ≥ 0.627)

**Action:**
1. Write comparison review (GO decision)
2. Proceed to weight calibration:
   - Grid search (w_m, w_q, w_i)
   - Optuna refinement
3. Update ENSEMBLES.md with M+Q+I
4. Add to ENSEMBLE_REGISTRY

### 8.4 If Diagnostic Fails Acceptance Gates (Sharpe < 0.627)

**Action:**
1. Write comparison review (NO-GO decision)
2. Analyze failure mode:
   - Is insider signal noisy?
   - Is correlation higher than expected?
   - Does insider hurt in specific regimes?
3. Document lessons learned
4. Archive M3.6 as "explored but not viable"

---

## 9. References

### Academic
- Cohen, Malloy & Pomorski (2012) "Decoding Inside Information"
- Seyhun (1986) "Insiders' Profits, Costs of Trading, and Market Efficiency"
- Jeng, Metrick, Zeckhauser (2003) "Estimating the Returns to Insider Trading"

### Internal
- `docs/ENSEMBLES_M3.6_NOTES.md` (infrastructure assessment)
- `signals/insider/institutional_insider.py` (implementation)
- `results/ensemble_baselines/momentum_quality_v1_regime_allocators_review.md` (M+Q baseline)
- Phase 3 M3.4 weight calibration: `results/ensemble_baselines/momentum_quality_v1_weight_optuna.md`

---

**Status:** M3.6 spec complete, infrastructure ready, awaiting data coverage check and diagnostic implementation.

**Created:** 2025-11-23
**Last Updated:** 2025-11-23
