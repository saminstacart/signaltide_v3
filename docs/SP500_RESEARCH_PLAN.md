# S&P 500 Advanced Quantitative Backtesting Research Plan

**Version:** 2.0 (Risk-Adjusted)
**Last Updated:** 2025-11-21
**Status:** Phase 0 (Quality Diagnostics) - In Progress
**Risk Register:** `results/risk_register.md`

---

## Executive Summary

This is a two-stage institutional quantitative research program to identify optimal signal strategies for S&P 500 trading:

**Stage 1 (Phases 0-4):** Individual signal baseline establishment, optimization, and validation
**Stage 2 (Phases 5-6):** Signal ensemble design, testing, and production candidate selection

**Key Principles:**
- Multiple regime-aware optimization windows (not single-period optimization)
- Survivorship-bias-free universe construction
- Conservative transaction cost modeling with future state-dependent enhancement
- Explicit control of research degrees of freedom and ensemble overfitting
- Comprehensive out-of-sample and walk-forward validation
- Signal health and decay monitoring

**Timeline:** 8-10 weeks (parallelizable to ~6 weeks)

---

## Risk-Aware Assumptions and Constraints

### Time Period Structure: Regime-Aware Windows

**CRITICAL:** We do NOT treat 2015-2019 as the single canonical optimization window.

**Approach:**
- Use **multiple overlapping 5-year windows** across full Sharadar history
- Identify **regime-specific periods** for stress testing:
  - Bull: 2010-2014, 2015-2019 (low vol, strong momentum)
  - COVID Stress: 2020-2021 (crash + recovery)
  - Bear: 2022 (inflation, rising rates, tech rotation)
  - Recent: 2023-2024 (recovery, AI rally)

**Proposed Window Structure:**
```
Historical (Pre-Optimization): 2010-2014
  Purpose: Robustness check, parameter stability test

Primary Optimization Windows (Rolling 5-year):
  Window 1: 2010-2014 (post-GFC normalization)
  Window 2: 2012-2016 (low vol, monetary easing)
  Window 3: 2015-2019 (momentum dominance, tech surge)
  Window 4: 2017-2021 (includes COVID shock)

Validation Windows:
  Val 1: 2020-2021 (COVID stress test)
  Val 2: 2022 (bear market, factor rotation)

Pure Out-of-Sample: 2023-2024
  Purpose: Final test, no parameter tuning allowed
```

**Regime-Aware Testing (TODO for Phase 4):**
- Define regime indicators:
  - Bull: SPY 3-month return > +5%, VIX < 20
  - Bear: SPY 3-month return < -5% OR VIX > 30  - Sideways: Neither condition
  - High Vol: VIX > 25
- Compute signal performance by regime across all windows
- Test for regime-conditioned parameter sets
- Consider regime-switching ensemble in Phase 5

**Why This Matters:** Single-period optimization risks overfitting to that regime's idiosyncratic features (e.g., momentum in 2015-2019 bull market fails in 2022 bear).

---

### Universe Definition: Survivorship-Bias-Free

**CRITICAL:** Testing on "current S&P 500 members only" introduces 1-3% annual survivorship bias.

**Required Universe:**
1. **Primary:** Point-in-time S&P 500 members via `dim_universe_membership`
   - Includes stocks that were later removed (delisted, acquired, demoted)
   - Uses `[start, end)` semantics for membership periods
   - Expected: ~500-503 stocks per rebalance date

2. **Extended (for robustness):** S&P 500 + Recently Removed
   - Include stocks removed from S&P 500 in last 5 years
   - Captures "falling angels" that may be important for signal testing
   - Verify via `membership_end_date` in universe table

3. **Optional (Phase 4+):** Russell 1000 Borderline
   - Test signal behavior on S&P 500 threshold names
   - Helps assess capacity and signal stability outside core universe

**Audit Requirements (Before Phase 1):**
- [ ] Verify `UniverseManager.get_universe('sp500_actual', as_of_date)` returns correct count
- [ ] Check `lastpricedate` handling in `_get_market_cap_data()`
- [ ] Test on known delisted stock (e.g., Enron, Lehman, recent bankruptcies)
- [ ] Compare results: survivor-only vs survivorship-free universe
- [ ] Document exact universe used in every backtest report metadata

**Why This Matters:** Excluding failed stocks artificially inflates returns. Real trading involves stocks that later blow up.

---

### Transaction Costs: Baseline → Advanced Model

**Current Baseline (Phase 1-5):**
```python
TransactionCostModel(
    commission=0.0,    # bps (Schwab free trades)
    slippage=2.0,      # bps (execution vs close)
    spread=3.0         # bps (bid-ask)
)
# Total: 5 bps per trade
```

**Justification:**
- Conservative but realistic for S&P 500 mega-caps
- Schwab $50K account has $0 commissions
- Liquid large-caps have narrow spreads

**Known Limitations:**
- Flat cost ignores market cap differences
- No volatility/VIX adjustment for stress periods
- No position size / market impact modeling
- Optimistic for smaller S&P 500 names

**Future Enhancement (Phase 6 - Production):**
```python
# State-Dependent Transaction Cost Model (TODO)
def get_transaction_cost(ticker, date, position_size, market_data):
    # Base cost by market cap bucket
    market_cap = get_market_cap(ticker, date)
    if market_cap > 200e9:      # Mega (AAPL, MSFT)
        base_cost = 3.0  # bps
    elif market_cap > 10e9:     # Large (most S&P 500)
        base_cost = 5.0  # bps
    elif market_cap > 2e9:      # Mid
        base_cost = 8.0  # bps
    else:                       # Small
        base_cost = 12.0  # bps

    # VIX multiplier for stress periods
    vix = get_vix(date)
    vol_multiplier = 1.0 + (vix / 50.0)

    # Position size impact (% of ADV)
    adv = get_average_daily_volume(ticker, date, window=20)
    impact_multiplier = 1.0 + (position_size / adv) * 0.5

    # Combined cost
    total_cost = base_cost * vol_multiplier * impact_multiplier
    return total_cost
```

**Warning:** All Phase 1-5 performance numbers are "pre-realistic-TC". Expect -1% to -3% annual return hit once advanced TC model is applied in Phase 6.

**Why This Matters:** Underestimated costs lead to unprofitable strategies in live trading. Better to be conservative.

---

### Signal Decay and Concept Drift Monitoring

**Risk:** Academic signals may decay over time due to crowding, regime changes, or data mining bias in original research.

**Signal Health Metrics (TODO for Phase 4+):**
```python
class SignalHealthMonitor:
    """Track signal performance degradation over time."""

    def compute_rolling_metrics(signal, window='1Y'):
        # Rolling IR, Sharpe, turnover by window
        # Compare to optimization window baseline
        pass

    def detect_decay(signal, optimization_sharpe, current_sharpe):
        degradation = (optimization_sharpe - current_sharpe) / optimization_sharpe
        if degradation > 0.30:
            return "SIGNAL_DECAY_WARNING"
        pass

    def regime_performance(signal, regimes):
        # Performance by bull/bear/sideways
        # Identify regime-specific failures
        pass
```

**Academic Publication Tracking:**
- Momentum: Jegadeesh-Titman (1993) - 30+ years old, well-known, may be crowded
- Quality: Asness-Frazzini-Pedersen (2018) - relatively recent
- Insider: Cohen-Malloy-Pomorski (2012) - 12 years, regulations evolving

**Decay Mitigation:**
- Use OOS windows (2022-2024) as primary validation
- Treat 30%+ performance degradation as signal death
- Build signal health dashboard for live monitoring (Phase 7)
- Consider ensemble as hedge against individual signal decay

**Why This Matters:** Historical backtests show what worked. Live trading needs signals that still work now.

---

### Ensemble Overfitting Control: Research Governance

**Risk:** Testing 100 ensemble combinations and reporting the best one is p-hacking.

**Research Governance Rules:**

1. **Pre-Declare Ensemble Forms (Before Phase 5):**
   ```
   To be tested (no more than these 5):
   1. Equal Weight (baseline)
   2. Inverse Volatility (risk parity)
   3. Performance-Weighted (trailing 6-month Sharpe)
   4. Regime-Aware Simple (VIX > 25 → defensive tilt)
   5. Adaptive Dynamic (reweight monthly based on trailing metrics)
   ```

2. **Limit Quality Variants:**
   - Build 3-4 Quality signals in Phase 0.2
   - Test all on S&P 500 baseline (Phase 1)
   - Promote only top 2 to ensemble stage (Phase 5)
   - Rationale: Reduces researcher degrees of freedom

3. **Statistical Adjustments:**
   - Use deflated Sharpe ratio (accounts for number of trials)
   - Track in-sample vs out-of-sample Sharpe gap
   - Flag if IS/OOS gap > 30% (overfitting indicator)
   - Report median ensemble performance, not just best

4. **Decision Logging:**
   - Create `results/research_decisions.md`
   - Document why we chose each ensemble form BEFORE testing
   - Record all major design choices with rationale
   - No retroactive narrative-fitting to best result

5. **Reporting Standards:**
   - Report all ensemble tests, not just winners
   - Show performance distribution across ensemble variants
   - Highlight which choices mattered (signal selection > weighting > regime)

**Why This Matters:** Cherry-picked strategies fail out-of-sample. Rigorous process produces robust strategies.

---

### Correlation Regime Shifts: Beyond Static Correlation

**Risk:** Signal correlations spike during crises, eliminating diversification when needed most.

**Static Correlation Matrix (Naive):**
```
             Momentum  Quality  Insider
Momentum        1.00     0.15     0.30
Quality         0.15     1.00     0.10
Insider         0.30     0.10     1.00

Ensemble: "Looks diversified!"
```

**Reality - Correlation by Regime:**
```
Bull Market (VIX < 20):
             Momentum  Quality  Insider
Momentum        1.00     0.10     0.25
Quality         0.10     1.00     0.05
Insider         0.25     0.05     1.00

Crisis (VIX > 30):
             Momentum  Quality  Insider
Momentum        1.00     0.60     0.75
Quality         0.60     1.00     0.50
Insider         0.75     0.50     1.00

Ensemble: "Concentrated bet during crash!"
```

**Required Analysis (Phase 5):**
- Compute correlations by regime (bull/bear/high-vol)
- Track rolling 1-year correlation windows
- Test ensemble performance assuming 0.8+ correlation in tail scenarios
- Consider adding true diversifiers (e.g., defensive quality, low-beta stocks)
- Monitor correlation clustering in live deployment (Phase 7)

**Advanced (Future - Not Phase 5):**
- DCC-GARCH dynamic correlation modeling
- Correlation clustering risk metrics
- Dynamic ensemble reweighting based on correlation forecasts

**Why This Matters:** Diversification vanishes when you need it most. Static correlation is a lie.

---

## Phase Structure and Deliverables

### Phase 0: Quality Signal Research Project (Weeks 1-2)

**CRITICAL:** Phase 0.1 identified fundamental methodology error in InstitutionalQuality v0.

**Status Update (2025-11-21):**
- ✅ Phase 0.1 Complete: Root cause identified
- ⚠️ InstitutionalQuality v0 (time-series scaled) **DEPRECATED**
- ➡️ Phase 0.2: Implement CrossSectionalQuality v1 per `docs/QUALITY_SPEC.md`

#### 0.1 Debug Current InstitutionalQuality - **COMPLETE**
**Deliverable:** `results/quality_diagnostics_report.md` ✅

**Key Finding:**
InstitutionalQuality v0 uses **time-series ranking** (each stock vs its own history), not **cross-sectional ranking** (all stocks vs each other). This is fundamentally incompatible with academic QMJ methodology.

**Root Cause Confirmed:**
- ❌ NOT data coverage (100% have fundamentals)
- ❌ NOT parameter issues (methodology is wrong)
- ✅ **Methodology mismatch**: Time-series vs cross-sectional ranking
- ✅ **Result**: 1.74% annual, Sharpe -0.157 (not capturing quality premium)
- ✅ **Solution**: Reimplement with proper cross-sectional approach

**v0 Deprecation:**
- Status: Archived, no further development
- Code location: `signals/quality/institutional_quality.py` (preserved for reference)
- Results: `results/baseline_quality_v0_DEPRECATED.json`
- Superseded by: CrossSectionalQuality v1 (Phase 0.2)

#### 0.2 Implement CrossSectionalQuality v1 - **IN PROGRESS**
**Deliverable:** `signals/quality/cross_sectional_quality.py`

**Specification:** Follow `docs/QUALITY_SPEC.md` exactly

**Implementation Approach:**
1. **First: Canonical CrossSectionalQuality ONLY**
   - Proper cross-sectional ranking at each month-end
   - Components: 40% Profitability, 30% Growth, 30% Safety (QMJ default)
   - Validate thoroughly before building variants

2. **Then: Decide on variants based on v1 performance**
   - Only build variants if v1 validates successfully
   - Candidates: QualityProfitability, QualityAccruals, QualityPiotroski
   - Maximum 2-3 variants total (avoid multiple testing explosion)

**API Compliance:**
- Inherits from `InstitutionalSignal` base class
- Monthly rebalancing (rebalance_frequency='monthly')
- Returns signals in [-1, 1] range
- Implements `generate_signals_cross_sectional()` for multi-stock universe
- Point-in-time constraints enforced (33-day filing lag)

#### 0.3 Quality Suite Baseline Comparison
**Deliverable:** `results/quality_suite_comparison.md`

**Tasks:**
- Run simple long-only top-decile backtests (2015-2024) for each variant
- Compute long-short top-minus-bottom spreads
- Compare: CAGR, Sharpe, max drawdown, turnover, capacity
- Identify which variants are complementary vs redundant (correlation analysis)
- Select top 2 Quality signals for full Phase 1 backtest

---

### Phase 1: S&P 500 Baseline Backtests (Weeks 2-3)

#### 1.1 Define Time Periods
Document window structure for all subsequent phases:
- Historical: 2010-2014
- Optimization Windows: Multiple rolling 5-year windows (2010-2014, 2012-2016, 2015-2019, 2017-2021)
- Validation: 2020-2021, 2022
- Pure OOS: 2023-2024

#### 1.2 Run Individual Signal Baselines on S&P 500
**Signals:**
- InstitutionalMomentum
- InstitutionalInsider
- Top 2 Quality variants from Phase 0

**Standard Settings:**
- Universe: `sp500_actual` (survivorship-free point-in-time)
- Rebalancing: Monthly (month-end via trading calendar)
- Position sizing: Equal weight
- Transaction costs: 5 bps (baseline model)
- Capital: $50,000
- Period: 2010-2024 (full history), segmented reporting

**Expected Runtime:** ~20-30 minutes per signal on S&P 500

**Deliverable:** `results/baseline_sp500_comprehensive.md`

#### 1.3 Generate Baseline Diagnostics
**Analysis:**
- Performance by signal and by subperiod (2010-14, 2015-19, 2020-21, 2022, 2023-24)
- vs SPY: CAGR, Sharpe, Sortino, IR, Alpha/Beta, t-stats
- Risk: Max drawdown, drawdown duration, underwater %, VaR/CVaR
- Operational: Monthly turnover, avg position count, transaction cost drag
- Signal overlap: Return correlations + position overlap matrix
- Regime sensitivity: Performance in bull/bear/high-vol periods

**Key Questions:**
1. Which signals have positive IR vs SPY on OOS period (2023-24)?
2. Do signals maintain performance across regimes or only work in certain periods?
3. What is correlation structure between signals (static + regime-conditional)?
4. Are there obvious bugs or structural issues?

---

### Phase 2: Coarse Optimization Sanity Check (Week 3)

#### 2.1 Define Optimization Objective
**Deliverable:** `optimization/objectives.py` with production objective

**Primary Objective:** Deflated Sharpe Ratio
```python
deflated_sharpe = sharpe / sqrt(1 + (trials - 1) * sharpe^2 / N)
```

**Secondary Objective:** Information Ratio vs SPY

**Constraints:**
- Max monthly turnover < 100% (avoid excessive churn)
- Min position count > 20 (diversification)
- Max drawdown < 30% (risk limit)
- Min trades per year > 50 (avoid data-starved strategies)

#### 2.2 Run Coarse Optimization (20-30 Trials)
**Purpose:** Verify infrastructure scales to S&P 500

**Execution:**
```bash
python scripts/optimize_signals_coarse.py \
    --signal InstitutionalMomentum \
    --universe sp500_actual \
    --period 2015-01-01,2019-12-31 \
    --n-trials 30 \
    --objective deflated_sharpe
```

**Deliverable:** `results/optimization/coarse_sanity_check.md`

**Success Criteria:**
- Objectives improve over trials (not random)
- Per-trial runtime: 1-2 hours (acceptable for 150 trials = 1 week wall-clock)
- No numerical errors or crashes
- Parameter bounds are reasonable (not hitting edges)

#### 2.3 Go/No-Go Decision
Review coarse results, refine if needed, proceed to full optimization

---

### Phase 3: Full Hyperparameter Optimization (Weeks 4-6)

#### 3.1 Run Full Optimization (100-200 Trials Per Signal)
**Settings:**
- Optimization windows: Multiple 5-year windows (primary: 2015-2019)
- Sampler: TPE (Tree-structured Parzen Estimator)
- Pruner: MedianPruner for early stopping
- Parallelization: 4-8 jobs
- Total compute: ~150-200 hours per signal (parallelizable to ~1 week wall-clock)

**Parameters to Optimize:**

**Momentum:**
- `formation_period`: [126, 378] days (6-18 months)
- `skip_period`: [5, 42] days (1 week to 2 months)
- `rank_window`: [60, 500] days (lookback for percentile ranking)
- `long_threshold`: [0.0, 0.3] (quintile cutoff)
- `winsorize_pct`: categorical [[1,99], [5,95], [10,90]]

**Insider:**
- `lookback_days`: [30, 180] (1-6 months aggregation)
- `min_transaction_value`: [5000, 50000]
- `cluster_window`: [3, 14] days
- `role_weights`: CEO/CFO/Director multipliers
- `rank_window`: [60, 500]

**Quality (per variant):**
- Component weights (profitability/growth/safety)
- `rank_window`: [120, 750] days (0.5-3 years)
- Winsorize bounds
- Long/short thresholds

**Deliverable:**
- `results/optimization/best_configs_{signal}.json`
- Optimization history plots
- Parameter importance analysis (Optuna built-in)

#### 3.2 Extract and Validate Best Configs
**Tasks:**
- Identify top 3 configs per signal (not just #1)
- Check parameter stability across top configs
- Test on validation window (2020-2021) before OOS
- Flag suspicious patterns (e.g., extreme parameter values)

**Deliverable:** `results/optimization/optimization_summary.md`

---

### Phase 4: Comprehensive Validation (Weeks 6-7)

#### 4.1 Purged K-Fold Cross-Validation
**Method:** López de Prado purged K-fold
- 5 folds on optimization window
- Purge: 21 days between folds
- Embargo: 21 days after each test fold

**Deliverable:** In-sample vs out-of-fold performance gaps

#### 4.2 Walk-Forward Out-of-Sample Testing
**Windows:**
1. Train: 2015-2019 → Test: 2020-2021 (COVID stress)
2. Train: 2015-2019 → Test: 2022 (bear market)
3. Train: 2015-2019 → Test: 2023-2024 (pure OOS)
4. Rolling 3-year train → 1-year test through full history

**Deliverable:** `results/validation/walk_forward_report.md`

**Key Metrics:**
- IS vs OOS Sharpe gap (flag if > 30%)
- Parameter stability over time
- Regime-specific degradation

#### 4.3 Statistical Validation Battery
**Tests:**
1. Probabilistic Sharpe Ratio (PSR > 95% confidence?)
2. Deflated Sharpe Ratio (adjusted for trials)
3. Monte Carlo permutation (10,000 shuffles, compute p-value)
4. Haircut analysis (expected live vs backtest performance)

**Deliverable:** `results/validation/statistical_tests.md`

#### 4.4 Regime Analysis
**Regime Definitions:**
- Bull: SPY 3-mo return > +5%, VIX < 20
- Bear: SPY 3-mo return < -5% OR VIX > 30
- Sideways: Neither
- High Vol: VIX > 25

**Analysis:**
- Signal performance by regime
- Drawdown behavior by regime
- Correlation shifts by regime
- Identify regime-dependent failures

**Deliverable:** `results/validation/regime_analysis.md`

---

### Phase 5: Ensemble Design & Testing (Weeks 7-9)

#### 5.1 Signal Correlation Analysis
**Static + Dynamic:**
- Overall correlation matrix (returns, positions)
- Correlations by regime (bull/bear/high-vol)
- Rolling 1-year correlation windows
- Identify regime shifts and correlation spikes

**Deliverable:** `results/ensemble/correlation_analysis.md`

#### 5.2 Pre-Declared Ensemble Forms
**To Test (exactly these 5, no more):**
1. Equal Weight (baseline)
2. Inverse Volatility (risk parity)
3. Performance-Weighted (trailing 6-month Sharpe)
4. Regime-Aware Simple (VIX > 25 → tilt to Quality/defensive)
5. Adaptive Dynamic (monthly reweighting based on trailing IR)

**Constraints:**
- All weights sum to 1.0
- No negative weights (no signal shorting)
- Max single signal weight < 0.6

#### 5.3 Optimize Ensemble Weights
**Approach:**
- Use Optuna on optimization window (2015-2019)
- Objective: Maximize IR vs SPY with drawdown penalty
- Test on validation (2020-2021) and OOS (2022-2024)
- Track IS vs OOS performance gap

**Deliverable:** `results/ensemble/ensemble_optimization.md`

#### 5.4 Final Ensemble Validation
**Full History Backtest:**
- Test best ensemble on 2010-2024 (including pre-optimization period)
- Performance attribution: which signals contributed most?
- Regime analysis: where did ensemble add value?
- Compare to individual signals and SPY

**Deliverable:** `results/ensemble/final_ensemble_report.md`

---

### Phase 6: Production Candidate Selection (Weeks 9-10)

#### 6.1 Executive Summary
**Comparison Table:**
| Strategy | CAGR | Sharpe | IR | Max DD | Win Rate | Turnover | Capacity |
|----------|------|--------|-----|--------|----------|----------|----------|
| SPY | X% | X.XX | - | -X% | - | - | Unlimited |
| Best Single | X% | X.XX | X.XX | -X% | X% | X% | $XXM |
| Best Ensemble | X% | X.XX | X.XX | -X% | X% | X% | $XXM |

#### 6.2 Risk Assessment
- Worst drawdown periods and recovery
- Tail risk (VaR/CVaR 95%, 99%)
- Stress tests: 2020 COVID, 2022 bear
- Capacity estimation

#### 6.3 Production Readiness Checklist
- [ ] Point-in-time integrity verified
- [ ] Transaction costs conservative (5 bps)
- [ ] No parameter overfitting (IS/OOS gap < 30%)
- [ ] PSR > 95% confidence
- [ ] IR vs SPY > 0.5 (target: > 1.0)
- [ ] Max drawdown < 25%
- [ ] Capacity > $10M

#### 6.4 Enhanced Transaction Cost Model (Phase 6 Priority)
Before live deployment, implement state-dependent TC model:
- Market cap buckets (mega/large/mid/small)
- VIX multiplier for volatility periods
- Position size / ADV impact modeling
- Re-run backtests with enhanced model
- Document performance hit from realistic costs

**Deliverable:** `results/production_candidate_final.md`

---

### Phase 7: Meta-Strategy and Live Monitoring (Future)

**Not Scoped in Current Plan - Placeholder for Future Work**

#### 7.1 Signal Health Dashboard
- Rolling IR, Sharpe by subperiod
- Decay detection (30%+ degradation = warning)
- Regime performance tracking
- Correlation drift monitoring

#### 7.2 Dynamic Reweighting Rules
- Adjust ensemble weights based on recent performance
- De-risk rules: IR collapse, correlation spikes, turnover explosions
- Regime-adaptive portfolio tilts

#### 7.3 Kill-Switches and Contingencies
- Max drawdown > 25% → reduce exposure 50%
- IR < 0 for 6 months → pause strategy, re-optimize
- Correlation spike (avg > 0.7) → switch to single best signal
- Turnover > 150% → freeze rebalancing, investigate

#### 7.4 Research Pipeline
- Continuous signal development
- Academic paper monitoring
- A/B testing new signals vs production
- Regular reoptimization (annual)

---

## Success Criteria

### Minimum Viable (Go/No-Go for $50K Deployment)
- At least 2 signals with IR vs SPY > 0.5 on OOS (2023-2024)
- Ensemble IR vs SPY > 0.7
- Max drawdown < 25%
- Turnover < 50% monthly (implies capacity > $10M)
- PSR > 95% confidence
- No data leakage or lookahead bias

### Institutional Quality (Aspirational)
- Ensemble IR vs SPY > 1.0
- Sharpe > 1.5
- Max drawdown < 20%
- Positive alpha in 70%+ of rolling 1-year periods
- Deflated Sharpe > 2.0
- Robust across multiple regimes (no single-regime dependency)

---

## Research Governance and Quality Control

### Decision Logging
**File:** `results/research_decisions.md`

**Required Entries:**
- Date of decision
- What was decided (e.g., "Use 2015-2019 as primary optimization window")
- Rationale (e.g., "Covers full bull cycle, sufficient data, pre-COVID")
- Alternatives considered
- Who approved (even if solo researcher)

**Example:**
```markdown
### 2025-11-21: Quality Signal Variants for Phase 0.2

**Decision:** Build exactly 3 Quality variants: QualityProfitability, QualityAccruals, QualityPiotroski

**Rationale:**
- Current InstitutionalQuality underperforming (1.74% annual)
- Need multiple quality definitions to capture different aspects of quality
- 3 variants balances exploration vs research degrees of freedom
- Academic support: Profitability (Novy-Marx), Accruals (Sloan), F-Score (Piotroski)

**Alternatives Considered:**
- Fix current signal only: Too risky if construction is fundamentally flawed
- Build 5+ variants: Increases multiple testing problem
- Use single replacement: Doesn't allow comparison and ensemble benefits

**Next Review:** After Phase 0.3 results, select top 2 for full backtest
```

### No Retroactive Narratives
- Write hypothesis BEFORE running test
- If test fails, document failure (don't cherry-pick)
- Report all ensemble tests, not just best performer

### Reporting Standards
- Always show IS and OOS performance side-by-side
- Report deflated Sharpe (accounts for multiple testing)
- Show performance by subperiod, not just aggregate
- Include regime-specific analysis (bull/bear/high-vol)
- Document all parameter choices and constraints

---

## Immediate Next Actions

1. **Complete Phase 0.1 Quality Diagnostics** (Current)
   - Run cross-sectional analysis on 2023/2024 S&P 500
   - Diagnose why current Quality signal is weak
   - Write `results/quality_diagnostics_report.md`

2. **Phase 0.2 Quality Suite Implementation**
   - Build 3 Quality variants
   - Test decile spreads and monotonicity
   - Select top 2 for Phase 1

3. **Phase 1 S&P 500 Baseline**
   - Run all signals on full survivorship-free S&P 500
   - Generate comprehensive diagnostics
   - Make go/no-go decision for optimization

---

## Appendix: TODO Tracker

### Immediate (Before Phase 1)
- [ ] Complete Phase 0.1 Quality diagnostics report
- [ ] Audit `UniverseManager` for survivorship bias handling
- [ ] Verify trading calendar coverage (2010-2024)
- [ ] Test on known delisted stock (verify survivorship-free)
- [ ] Build 3 Quality variants (Phase 0.2)
- [ ] Select top 2 Quality signals for baseline

### Phase 3-4 (Optimization/Validation)
- [ ] Implement regime detection logic (bull/bear/high-vol)
- [ ] Add regime-aware correlation analysis
- [ ] Build signal health monitoring functions
- [ ] Implement purged K-fold properly (verify purge/embargo)
- [ ] Add Monte Carlo permutation testing
- [ ] Compute deflated Sharpe for all results

### Phase 5 (Ensemble)
- [ ] Pre-declare exactly 5 ensemble forms to test
- [ ] Limit Quality variants to top 2 (avoid multiple testing)
- [ ] Track IS vs OOS performance gap for ensembles
- [ ] Test ensemble robustness to correlation spikes
- [ ] Consider adding defensive diversifiers

### Phase 6 (Production)
- [ ] Implement state-dependent transaction cost model
- [ ] Re-run backtests with enhanced TC model
- [ ] Capacity analysis and market impact modeling
- [ ] Build production monitoring dashboard scaffolding
- [ ] Document kill-switches and de-risk rules

### Future Phases (Post-Production)
- [ ] Signal health dashboard (rolling metrics)
- [ ] Dynamic reweighting rules based on regimes
- [ ] Continuous signal research pipeline
- [ ] Annual reoptimization workflow
- [ ] A/B testing framework for new signals

---

**End of Research Plan**
**Version:** 2.0 (Risk-Adjusted)
**Next Review:** After Phase 0.1 completion
**Status:** Phase 0.1 in progress
