# Phase 3 M3.5 - Regime-Aware Allocation for Momentum + Quality v1

**Created:** 2025-11-23
**Status:** RESEARCH
**Goal:** Design and implement regime-aware weight allocation to improve upon static 25/75 M+Q baseline

---

## 1. Executive Summary

Static M+Q 25/75 performs well across 2015-2024, but regime analysis shows quality's value varies significantly by market conditions:
- Quality **helps** in 4/5 regimes (COVID Crash, QE Recovery, 2022 Bear, Recent)
- Quality **hurts** in 1/5 regimes (Pre-COVID steady bull market)

This creates an opportunity for **regime-conditional weight allocation** to capture quality's defensive value during stress while reducing drag during calm bull markets.

We implement **two** allocators:
1. **Oracle Allocator** (hindsight, research-only): Uses known regime labels to find ceiling performance
2. **Rule-Based Allocator** (PIT-safe, practical): Uses observable indicators to approximate regime classification

---

## 2. Data Source & Regime Definitions

**Source:** `results/ensemble_baselines/momentum_quality_v1_regime_diagnostic.md`

**5 Macro Regimes (2015-2024):**

| Regime | Dates | Description | M-Only Sharpe | M+Q Sharpe | ΔSharpe |
|--------|-------|-------------|---------------|------------|---------|
| **Pre-COVID Expansion** | 2015-04-01 to 2019-12-31 | Steady bull, low vol | 0.768 | 0.692 | -0.075 |
| **COVID Crash** | 2020-02-01 to 2020-04-30 | Pandemic crash | -0.373 | -0.091 | +0.282 |
| **COVID/QE Recovery** | 2020-05-01 to 2021-12-31 | QE-driven rally | 1.998 | 2.315 | +0.317 |
| **2022 Bear Market** | 2022-01-01 to 2022-12-31 | Inflation/rate hikes | -0.312 | -0.200 | +0.112 |
| **Recent Period** | 2023-01-01 to 2024-12-31 | AI boom, mixed | 0.555 | 0.714 | +0.159 |

**Key Insight:** Quality adds value during stress (crashes, bears, volatile regimes) but creates drag during calm expansions.

---

## 3. Oracle Allocator Design (Hindsight-Based)

### 3.1 Allocation Strategy

**Goal:** Find performance ceiling by using perfect regime knowledge.

**Oracle Weight Determination (Grid Search per Regime):**

Instead of hand-picking weights, the oracle performs a **programmatic grid search** for each regime to find the optimal allocation:

**Grid Search Methodology:**
1. For each regime **r** ∈ {Pre-COVID, COVID Crash, QE Recovery, 2022 Bear, Recent}:
2. Search over discrete grid: `w_m ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}`
3. Set `w_q = 1.0 - w_m` (enforces sum = 1.0)
4. For each candidate (w_m, w_q):
   - Construct monthly M+Q ensemble returns using those weights
   - Compute in-regime Sharpe ratio
5. Select (w_m, w_q) that **maximizes Sharpe** in that regime
6. Tie-breaker: If Sharpe within 0.01, prefer lower max drawdown

**Constraints:**
- Minimum weight: 0.10 (enforced by grid: never abandon a signal)
- Maximum weight: 0.90 (enforced by grid: avoid single-signal risk)
- All weights sum to 1.0 (enforced by construction)

**Illustrative Expected Outcomes** (will be validated by grid search):
- **Pre-COVID Expansion:** Expect w_m ≈ 0.30-0.40 (less quality drag)
- **COVID Crash:** Expect w_m ≈ 0.10-0.15 (MAX defensive tilt)
- **COVID/QE Recovery:** Expect w_m ≈ 0.20-0.25 (high quality)
- **2022 Bear Market:** Expect w_m ≈ 0.10-0.20 (MAX defensive tilt)
- **Recent Period:** Expect w_m ≈ 0.25-0.30 (near baseline)

**Design Philosophy:**
- Let data determine optimal per-regime weights **rigorously**
- No hand-tuning bias
- Transparent, reproducible methodology
- Provides true performance ceiling for regime-aware approach

### 3.2 Implementation Method

1. **Regime Mapping Function:**
   ```python
   def get_regime_label(date: pd.Timestamp) -> str:
       """Map rebalance date to regime label using fixed boundaries."""
       if date < pd.Timestamp('2020-02-01'):
           return 'pre_covid_expansion'
       elif date < pd.Timestamp('2020-05-01'):
           return 'covid_crash'
       elif date < pd.Timestamp('2022-01-01'):
           return 'covid_recovery'
       elif date < pd.Timestamp('2023-01-01'):
           return 'bear_2022'
       else:
           return 'recent'
   ```

2. **Optimal Weight Grid Search (per regime):**
   ```python
   def find_optimal_weights_for_regime(
       regime_name: str,
       momentum_returns: pd.Series,  # Monthly returns for momentum-only
       quality_returns: pd.Series,   # Monthly returns for quality-only
       regime_dates: pd.DatetimeIndex,  # Dates in this regime
   ) -> Dict[str, float]:
       """
       Find optimal (w_m, w_q) for a regime via grid search.

       Returns:
           {'momentum': w_m, 'quality': w_q, 'sharpe': best_sharpe}
       """
       grid = np.arange(0.1, 1.0, 0.1)  # [0.1, 0.2, ..., 0.9]
       best_sharpe = -np.inf
       best_weights = None

       for w_m in grid:
           w_q = 1.0 - w_m
           # Construct ensemble returns for this regime
           ensemble_returns = w_m * momentum_returns[regime_dates] + w_q * quality_returns[regime_dates]
           # Compute annualized Sharpe (assume monthly returns, 12 periods/year)
           if len(ensemble_returns) > 1:
               sharpe = (ensemble_returns.mean() * 12) / (ensemble_returns.std() * np.sqrt(12))
           else:
               sharpe = 0.0

           if sharpe > best_sharpe:
               best_sharpe = sharpe
               best_weights = {'momentum': w_m, 'quality': w_q}

       return {**best_weights, 'sharpe': best_sharpe}
   ```

3. **Oracle Weight Determination:**
   - Run `find_optimal_weights_for_regime()` for each of the 5 regimes
   - Store optimal (w_m, w_q) per regime in lookup table
   - At each rebalance date, apply the corresponding optimal weights

4. **Per-Rebalance Weight Selection:**
   - At each monthly rebalance date, lookup regime label
   - Return corresponding optimal (w_m, w_q) tuple (computed via grid search)
   - Apply to ensemble member weights

### 3.3 Acceptance Gates

**Oracle allocator must meet:**
- ✅ Full-period Sharpe ≥ static 25/75 baseline (2.876 from diagnostic)
- ✅ Max Drawdown ≤ static baseline (-23.89%)
- ✅ Positive Sharpe improvement in ≥ 3/5 regimes
- ✅ No catastrophic regime (Sharpe < -1.0 in any single regime)
- ✅ Average weights within [0.15-0.35, 0.65-0.85] (centered on 25/75)

**If oracle fails:** Indicates regime-aware approach has fundamental limits.

---

## 4. Rule-Based Allocator Design (PIT-Safe, Practical)

### 4.1 Allocation Strategy

**Goal:** Approximate oracle performance using **only observable, PIT-safe indicators** available at rebalance time.

**Regime Indicators (All PIT-Safe):**

1. **Realized Volatility** (RealVol_6M):
   - 6-month (126 trading days) rolling volatility of S&P 500 total return
   - Threshold: High if > 20% annualized

2. **Drawdown from Peak** (CurrentDD):
   - Current drawdown from all-time high of S&P 500
   - Threshold: Crisis if < -15%

3. **Momentum Breadth** (MomBreadth):
   - % of S&P 500 constituents with positive 12-1 momentum
   - Threshold: Weak if < 40%

**Regime Classification (3 practical regimes):**

| Regime | Criteria | Interpretation |
|--------|----------|----------------|
| **CALM** | RealVol < 15% AND CurrentDD > -10% | Normal bull market, low stress |
| **STRESS** | RealVol > 25% OR CurrentDD < -15% | Crisis, crash, or severe bear |
| **CHOPPY** | All other cases | Mixed/uncertain conditions |

**Weight Assignment:**

| Regime | Momentum Weight | Quality Weight | Rationale |
|--------|----------------|----------------|-----------|
| **CALM** | 0.35 | 0.65 | Reduce quality drag in stable markets |
| **STRESS** | 0.15 | 0.85 | MAX defensive tilt during crises |
| **CHOPPY** | 0.25 | 0.75 | Use baseline (static weights) |

**Design Philosophy:**
- Default to **baseline 25/75** unless clear signal
- Tilt **toward quality** when markets show stress
- Tilt **toward momentum** only when markets are calm AND trending
- Conservative thresholds to avoid overtrading

### 4.2 Implementation Method

1. **Indicator Computation (PIT-Safe):**
   ```python
   def compute_regime_indicators(
       date: pd.Timestamp,
       data_manager: DataManager
   ) -> Dict[str, float]:
       """
       Compute PIT-safe regime indicators at rebalance date.

       Returns:
           Dict with keys: 'real_vol_6m', 'current_dd', 'mom_breadth'
       """
       # Fetch SPY/S&P 500 prices for lookback window
       end_date = date
       start_date = date - pd.Timedelta(days=252)  # ~1 year lookback

       spy_prices = data_manager.get_spy_prices(start_date, end_date)

       # 1. Realized vol (6M, annualized)
       returns = spy_prices.pct_change().dropna()
       real_vol_6m = returns.tail(126).std() * np.sqrt(252)

       # 2. Current drawdown from peak
       all_time_high = spy_prices.expanding().max().iloc[-1]
       current_price = spy_prices.iloc[-1]
       current_dd = (current_price / all_time_high) - 1.0

       # 3. Momentum breadth (optional, may skip for simplicity)
       # mom_breadth = ... (requires per-stock momentum, more complex)

       return {
           'real_vol_6m': real_vol_6m,
           'current_dd': current_dd,
       }
   ```

2. **Regime Classification:**
   ```python
   def classify_regime(indicators: Dict[str, float]) -> str:
       """
       Classify regime from PIT-safe indicators.

       Returns:
           'CALM', 'STRESS', or 'CHOPPY'
       """
       real_vol = indicators['real_vol_6m']
       current_dd = indicators['current_dd']

       # STRESS: High vol OR severe drawdown
       if real_vol > 0.25 or current_dd < -0.15:
           return 'STRESS'

       # CALM: Low vol AND small drawdown
       elif real_vol < 0.15 and current_dd > -0.10:
           return 'CALM'

       # CHOPPY: Everything else
       else:
           return 'CHOPPY'
   ```

3. **Weight Assignment:**
   ```python
   RULE_BASED_WEIGHTS = {
       'CALM':   {'momentum': 0.35, 'quality': 0.65},
       'STRESS': {'momentum': 0.15, 'quality': 0.85},
       'CHOPPY': {'momentum': 0.25, 'quality': 0.75},
   }
   ```

### 4.3 Acceptance Gates

**Rule-based allocator must meet:**
- ✅ Full-period Sharpe ≥ 90% of oracle Sharpe (allows for imperfect regime detection)
- ✅ Max Drawdown ≤ static baseline (-23.89%)
- ✅ Regime classification accuracy ≥ 60% vs oracle labels (validated on hindsight)
- ✅ No single-regime catastrophe (Sharpe < -0.5 in any regime)
- ✅ Turnover reasonable (avg ≤ 2 weight changes per year)

**If rule-based underperforms static by >10%:** Document as NO-GO for live deployment.

---

## 5. Validation Framework

### 5.1 Comparison Matrix

All three allocators (Static, Oracle, Rule-Based) must be compared on:

| Metric | Static 25/75 | Oracle | Rule-Based | Oracle vs Static | Rule vs Static |
|--------|-------------|--------|------------|------------------|----------------|
| **Full-Period Sharpe** | 2.876 | ? | ? | ? | ? |
| **Full-Period CAGR** | 9.28% | ? | ? | ? | ? |
| **Max Drawdown** | -23.89% | ? | ? | ? | ? |
| **Pre-COVID Sharpe** | 0.692 | ? | ? | ? | ? |
| **COVID Crash Sharpe** | -0.091 | ? | ? | ? | ? |
| **QE Recovery Sharpe** | 2.315 | ? | ? | ? | ? |
| **2022 Bear Sharpe** | -0.200 | ? | ? | ? | ? |
| **Recent Sharpe** | 0.714 | ? | ? | ? | ? |
| **Avg Turnover** | ~0/year | 4/year | ? | - | ? |

### 5.2 Diagnostic Outputs

**Scripts to create:**
1. `scripts/run_momentum_quality_regime_allocators.py`
   - Run all three allocators on same universe/period
   - Generate comparison MD + CSV

**Output files:**
- `results/ensemble_baselines/momentum_quality_v1_regime_allocators_diagnostic.md`
- `results/ensemble_baselines/momentum_quality_v1_regime_allocators_comparison.csv`

**MD Report Structure:**
1. Introduction & methodology
2. Per-allocator full-period metrics
3. Per-allocator per-regime metrics
4. Comparison tables (delta vs static)
5. Weight time-series visualization (text description)
6. Recommendations (GO/NO-GO for each allocator)

---

## 6. Decision Criteria

### 6.1 Oracle Allocator

**GO if:**
- Oracle Sharpe > Static Sharpe by ≥5% (2.876 * 1.05 = 3.02)
- Oracle improves Sharpe in ≥4/5 regimes

**NO-GO if:**
- Oracle Sharpe ≤ Static Sharpe (no upside from regime-awareness)
- Oracle underperforms in ≥3/5 regimes

### 6.2 Rule-Based Allocator

**GO if:**
- Rule Sharpe > Static Sharpe by ≥3% (2.876 * 1.03 = 2.96)
- Rule Sharpe ≥ 90% of Oracle Sharpe
- Max Drawdown ≤ Static baseline

**NO-GO if:**
- Rule Sharpe ≤ Static Sharpe (no practical value)
- Rule Max Drawdown > Static by >2% (defensive failure)

### 6.3 Reporting

If **either** allocator passes GO:
- Document as RESEARCH-approved for further validation
- Update signal catalog and ensemble docs
- Consider OOS validation on holdout period (2025+)

If **both** fail NO-GO:
- Static 25/75 remains canonical
- Document regime-aware as explored but not viable
- Archive spec as historical reference

---

## 7. Implementation Checklist

### Phase 1: Spec & Design ✅
- [x] Define oracle weights per regime
- [x] Define rule-based indicators & classification
- [x] Set acceptance gates
- [x] Document validation framework

### Phase 2: Code Implementation ✅
- [x] Create `signals/ml/regime_allocators.py`:
  - [x] `OracleRegimeAllocatorMQ` class
  - [x] `RuleBasedRegimeAllocatorMQ` class
- [x] Create `scripts/run_momentum_quality_regime_allocators.py`
- [N/A] Extend `signals/ml/ensemble_configs.py` (skipped due to NO-GO)

### Phase 3: Validation ✅
- [x] Run oracle allocator backtest
- [x] Run rule-based allocator backtest
- [x] Generate comparison MD + CSV
- [x] Review metrics vs acceptance gates
- [x] Make GO/NO-GO decision

### Phase 4: Testing & Docs ✅
- [x] Add tests in `tests/test_regime_allocators.py`
- [x] Archive spec as reference (NO-GO outcome)
- [N/A] Update ensemble configs (skipped due to NO-GO)

---

## 8. References

- Regime diagnostic: `results/ensemble_baselines/momentum_quality_v1_regime_diagnostic.md`
- Static 25/75 baseline: `results/ensemble_baselines/momentum_quality_v1_diagnostic.md`
- Weight calibration: `results/ensemble_baselines/momentum_quality_v1_weight_optuna.md`
- Phase 3 M3.6 notes: `docs/ENSEMBLES_M3.6_NOTES.md`

---

## 9. Outcome & Decision (2025-11-23)

**Status:** COMPLETE - NO-GO DECISION
**Decision Date:** 2025-11-23
**Reviewer:** Automated analysis with acceptance gate criteria

### Summary

Both Oracle and Rule-Based allocators were implemented, tested, and evaluated against acceptance gates over the period 2015-04-01 to 2024-12-31.

**Results:**
- **Oracle Allocator**: Sharpe declined by -8.5% vs static baseline (target: +5%)
- **Rule-Based Allocator**: Sharpe essentially flat at -0.2% vs static (target: +3%)

**Decision: NO-GO for both allocators** ❌

### Sharpe Discrepancy Note

**IMPORTANT**: Earlier diagnostics reported static 25/75 Sharpe as 2.876, which was incorrect due to annualization bug. The **correct Sharpe is 0.627** (confirmed by allocator diagnostic and direct CSV calculation). See detailed resolution: `docs/notes/m3_5_sharpe_discrepancy_resolution.md`

**Canonical Values for Static 25/75 M+Q (2015-2024)**:
- Sharpe: ~0.63
- Volatility: ~15.3% (annualized monthly)
- CAGR: ~9.0%

### Key Findings

1. **Oracle Performance Ceiling is Negative**: Even with perfect hindsight and per-regime grid-search optimization, the Oracle allocator delivered worse Sharpe than static 25/75. This proves regime-conditional weighting has fundamental limits for this signal combination.

2. **Rule-Based Adds No Value**: The PIT-safe rule-based allocator is essentially performance-neutral, failing to justify the added complexity of live regime detection and dynamic rebalancing.

3. **Static 25/75 Remains Robust**: The baseline already performs well across regimes on average. Regime-specific tilts introduce instability that offsets any within-regime benefits.

### Artifacts Generated

- Code: `signals/ml/regime_allocators.py` (Oracle + Rule-Based classes)
- Tests: `tests/test_regime_allocators.py` (24 tests, all passing)
- Script: `scripts/run_momentum_quality_regime_allocators.py`
- Diagnostic: `results/ensemble_baselines/momentum_quality_v1_regime_allocators.md`
- Comparison: `results/ensemble_baselines/momentum_quality_v1_regime_allocators.csv`
- Review: `results/ensemble_baselines/momentum_quality_v1_regime_allocators_review.md`

### Lessons Learned

- Not all reasonable hypotheses work in practice
- Perfect hindsight (Oracle) provides valuable performance ceiling insights
- "Simple and robust" often beats "complex and adaptive"
- Academic rigor requires testing ideas that may fail

### Recommendation

**Retain static 25/75 as canonical M+Q ensemble baseline**. Archive regime-aware allocators as "explored but not viable" research pathway. Proceed to Phase 3 M3.6 (3-signal ensemble with Insider).

---

**Status:** ARCHIVED - NO-GO (Explored, Tested, Not Viable)
**Next Step:** Proceed to M3.6 - Institutional Insider cross-sectional API
