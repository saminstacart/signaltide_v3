# Institutional Insider v2 Research Specification

**Status:** PLANNING (Not Yet Started)
**Created:** 2025-11-24
**Priority:** LOW (Phase 4+ work)
**Estimated Effort:** 3-4 weeks (if pursued)

---

## Context & Motivation

**Why this spec exists:**

Insider v1 failed validation in two independent tests:
1. **Phase 1 standalone:** Sharpe 0.034, not significant (failed 3/5 gates)
2. **Phase 3 ensemble:** -13.76% return vs M+Q baseline (10-year test)

However, the 5-year test (2020-2024) showed +7.24% incremental return, suggesting the signal MAY have value in specific market regimes or with improved construction.

**Purpose of this spec:**

Define a structured research program for Insider v2 that could:
- Test improved signal construction methodologies
- Identify regimes where insider information is predictive
- Establish clear GO/NO-GO criteria before committing to full 10-year backtests

**Decision:** This is a **FUTURE** research program. Do not start unless:
- M+Q production deployment is complete and stable
- Other higher-priority signals have been explored
- There is explicit stakeholder interest in rescuing the insider signal

---

## 1. Acceptance Criteria (GO Gates for v2)

Insider v2 must meet ALL of these criteria over a 10-year PIT backtest to be considered for production:

| Gate | Metric | Threshold | Rationale |
|------|--------|-----------|-----------|
| **Gate 1: Incremental Return** | M+Q+I vs M+Q | ≥ +5.0% | Must add meaningful alpha |
| **Gate 2: Incremental Sharpe** | M+Q+I vs M+Q | ≥ +0.05 | Must improve risk-adjusted returns |
| **Gate 3: Max DD** | M+Q+I vs M+Q | Neutral or better | Must not increase risk |
| **Gate 4: Correlation** | M+Q+I vs M+Q | ≤ 0.95 | Must provide diversification |
| **Gate 5: Robustness** | Positive in 4/5 regimes | Consistent | Must not be regime-lucky |
| **Gate 6: Runtime** | vs M+Q baseline | ≤ 3x | Must be computationally feasible |

**Philosophy:** Set the bar HIGH. If Insider v2 can't clearly beat v1's -13.76% performance, abandon the program.

---

## 2. Hypotheses for Why v1 Failed

Before designing v2, understand v1's failure modes:

### 2.1 Signal Construction Issues

1. **Simple aggregation:**
   - v1 used: `sum(transaction_value * role_weight)`
   - Problem: Treats all transactions equally (awards = sales = option exercises)
   - Improvement: Filter transaction codes, weight by information content

2. **No size adjustment:**
   - v1 ignored company size
   - $100K trade means more for $1B company than $100B company
   - Improvement: Scale by market cap or float

3. **No time decay:**
   - All transactions in 90-day window equally weighted
   - Recent trades should matter more
   - Improvement: Exponential decay (e.g., half-life 30 days)

4. **Role weighting:**
   - v1 used: CEO=3.0, CFO=2.5
   - Problem: Arbitrary, not validated
   - Improvement: Data-driven weights or role-specific models

### 2.2 Coverage & Quality Issues

1. **Transaction type mix:**
   - v1 included: 23% awards, 23% options, 20% sales
   - Awards/options often uninformative (planned compensation)
   - Improvement: Focus on S (sales) and P (purchases) codes

2. **Timing lag:**
   - Insider trades reported with ~8 day lag
   - By time of rebalance, information may be stale
   - Improvement: Shorter lookback (30-60 days) or weekly rebalancing

3. **Cluster detection:**
   - v1 had basic cluster logic (7-day window, min 3 insiders)
   - May have missed coordinated trading
   - Improvement: Network analysis, board-level coordination

### 2.3 Regime Dependency

1. **Performance by regime:**
   - v1 helped in 2020-2024 (volatile, info-rich)
   - v1 hurt in 2015-2019 (stable, low-vol)
   - Hypothesis: Insider signal only works in uncertain markets
   - Improvement: Regime detector + conditional weighting

2. **Factor crowding:**
   - Insider trading widely followed (Cohen et al. 2012 paper)
   - May be too crowded in efficient markets
   - Improvement: Focus on less-followed stocks (small/mid-cap)

---

## 3. Proposed v2 Improvements (Research Roadmap)

### Phase A: Signal Construction Enhancements (2 weeks)

**Goal:** Test alternative construction methodologies on SMALL sample (50 stocks, 3 years)

**Experiments:**

1. **Transaction Code Filtering:**
   - Variant A: S (sales) and P (purchases) only
   - Variant B: Exclude M (option exercises) and A (awards)
   - Variant C: S-only (sales as bearish signal)
   - Metric: Decile spread, Sharpe

2. **Size Adjustment:**
   - Variant A: Divide transaction $ by market cap
   - Variant B: Divide by float (shares outstanding × price)
   - Variant C: Percentile rank within size quintile
   - Metric: Correlation with returns, monotonicity

3. **Time Decay:**
   - Variant A: Equal weight (current v1)
   - Variant B: Linear decay (newest = 1.0, oldest = 0.0)
   - Variant C: Exponential decay (half-life 30 days)
   - Metric: Signal autocorrelation, turnover

4. **Role Specificity:**
   - Variant A: CEO-only model
   - Variant B: C-suite (CEO+CFO+COO) only
   - Variant C: Data-driven role weights via regression
   - Metric: Predictive R²

**Deliverable:** Ranked list of top 3 construction variants

---

### Phase B: Parameter Optimization (1 week)

**Goal:** Tune parameters on best Phase A variant using rolling windows

**Parameters to sweep:**

| Parameter | Current (v1) | Range to Test | Granularity |
|-----------|--------------|---------------|-------------|
| `lookback_days` | 90 | [30, 60, 90, 120, 180] | 5 values |
| `min_transactions` | 3 | [1, 3, 5, 10] | 4 values |
| `value_threshold` | $100K | [$0, $50K, $100K, $500K] | 4 values |

**Method:**
- Use 3-year rolling windows (2015-2017, 2016-2018, ..., 2021-2023)
- Optimize for OOS Sharpe in following year
- Select params with best average OOS performance

**Deliverable:** Optimized param set for v2

---

### Phase C: Regime-Aware Weighting (1 week)

**Goal:** Test if insider signal is predictably regime-dependent

**Regime Definitions:**

| Regime | Indicator | Insider Weight |
|--------|-----------|----------------|
| **High Volatility** | VIX > 20 | 0.25 (full) |
| **Uncertain** | Drawdown > 10% | 0.25 (full) |
| **Crisis** | VIX > 30 OR DD > 20% | 0.25 (full) |
| **Stable** | VIX < 15 AND DD < 5% | 0.0 (off) |
| **Transition** | Other | 0.10 (reduced) |

**Test:**
- Run M+Q+I with conditional insider weighting
- Compare to:
  - M+Q baseline (no insider)
  - M+Q+I v1 (fixed 25% weight)
  - M+Q+I v2 (regime-aware)

**Deliverable:** Decision on whether regime-switching adds value

---

### Phase D: Full 10-Year Validation (1 week)

**Goal:** Run best v2 variant on full 10-year PIT backtest

**Setup:**
- Period: 2015-04-01 to 2024-12-31
- Universe: S&P 500 PIT
- Rebalance: Monthly
- Ensemble: M+Q+I v2 (25/50/25 or optimized weights)

**Acceptance:**
- Must pass ALL 6 gates (see Section 1)
- Particularly critical: +5% return and +0.05 Sharpe vs M+Q

**Deliverable:** GO/NO-GO decision for Insider v2

---

## 4. Computational Budget

**Phase A (construction):** ~3 hours
- Small sample (50 stocks × 3 years)
- 4 experiments × 3 variants = 12 runs
- ~15 min per run

**Phase B (params):** ~2 hours
- Rolling windows (8 periods)
- 5×4×4 = 80 param combinations
- Subsample or early stopping

**Phase C (regime):** ~2 hours
- 3 regime strategies
- Full 10-year period each

**Phase D (validation):** ~2 hours
- 1 final 10-year run (M+Q+I v2)
- 1 baseline run (M+Q, if not cached)

**Total:** ~9 hours of compute time (spread over 4 weeks)

---

## 5. Risk Assessment

### Likelihood of Success: LOW-MEDIUM (20-40%)

**Why low:**
- v1 failed badly (-13.76%)
- Standalone Phase 1 also failed (Sharpe 0.034)
- Insider trading is crowded (Cohen et al. 2012 widely known)
- S&P 500 large-caps are efficient (less information asymmetry)

**Why not zero:**
- 5-year test showed +7.24% (regime-specific alpha exists)
- Many construction improvements untested
- Regime-aware approach not tried yet

### Opportunity Cost: HIGH

**Alternative uses of 3-4 weeks:**
- Explore earnings quality signal (Sloan 1996)
- Test short interest signal (Asquith et al. 2005)
- Optimize M+Q weights dynamically
- Add sector neutralization to M+Q

**Recommendation:** Only pursue Insider v2 if:
1. M+Q is in production and performing well
2. Higher-priority signals have been tested
3. Stakeholder specifically requests insider research

---

## 6. GO/NO-GO Decision Framework

### Before Starting (Phase 0)

**Prerequisites:**
- [ ] M+Q production deployment complete
- [ ] At least 1 month of live M+Q performance
- [ ] No higher-priority signal candidates remaining
- [ ] Explicit stakeholder approval

**If ANY prerequisite is false:** Defer Insider v2 to Phase 5+

### After Phase A (Construction)

**GO Criteria:**
- [ ] At least 1 variant shows >0.5%/mo decile spread (vs 0.02%/mo for v1)
- [ ] At least 1 variant shows Sharpe > 0.20 (vs 0.034 for v1)
- [ ] Computational time reasonable (<2x current)

**If criteria not met:** STOP. Document findings and abandon Insider v2.

### After Phase D (Full Validation)

**GO Criteria:**
- [ ] Pass ALL 6 acceptance gates (Section 1)
- [ ] Incremental return ≥ +5% vs M+Q
- [ ] Incremental Sharpe ≥ +0.05 vs M+Q
- [ ] Runtime ≤ 3x M+Q baseline

**If criteria not met:** Archive Insider v2 as RESEARCH_NO_GO and move on.

---

## 7. Academic References

### Core Insider Trading Literature

1. **Cohen, Malloy & Pomorski (2012):**
   - "Decoding Inside Information"
   - Journal of Finance
   - Key insight: Routine vs opportunistic trades

2. **Seyhun (1986):**
   - "Insiders' Profits, Costs of Trading, and Market Efficiency"
   - Journal of Financial Economics
   - Key insight: Filing lag matters

3. **Lakonishok & Lee (2001):**
   - "Are Insider Trades Informative?"
   - Review of Financial Studies
   - Key insight: Larger companies have less insider edge

### Methodology References

4. **Asness, Frazzini & Pedersen (2019):**
   - "Quality Minus Junk"
   - Review of Accounting Studies
   - Relevance: Multi-factor combination methodology

5. **Harvey, Liu & Zhu (2016):**
   - "...and the Cross-Section of Expected Returns"
   - Review of Financial Studies
   - Relevance: Multiple testing, factor decay

---

## 8. Alternative Approaches (If Insider v2 Fails)

If Insider v2 still fails after all improvements, consider:

### 8.1 Alternative Data Sources

1. **Form 4 NLP:** Text analysis of filing narratives
2. **13F filings:** Institutional investor flows
3. **Hedge fund 13F:** Smart money tracking
4. **Activist filings (13D):** Event-driven signals

### 8.2 Alternative Signal Types

1. **Earnings Quality:** Accruals, earnings manipulation
2. **Short Interest:** Days to cover, short ratio
3. **Analyst Revisions:** Upgrade/downgrade momentum
4. **Patent Activity:** Innovation as quality proxy

### 8.3 Hybrid Approaches

1. **Insider + Sentiment:** Combine with news sentiment
2. **Insider + Momentum:** Only use insider in strong momentum stocks
3. **Insider as Veto:** Screen out stocks with heavy selling

---

## 9. Documentation Requirements

If Insider v2 research proceeds, create:

1. **Phase A Report:** `docs/logs/insider_v2_construction_YYYYMMDD.md`
   - Methodology
   - Experiment results
   - Top 3 variants
   - GO/NO-GO decision

2. **Phase B Report:** `docs/logs/insider_v2_optimization_YYYYMMDD.md`
   - Parameter sweep results
   - Optimal parameter set
   - Sensitivity analysis

3. **Phase C Report:** `docs/logs/insider_v2_regime_YYYYMMDD.md`
   - Regime detection logic
   - Performance by regime
   - Switching strategy results

4. **Phase D Report:** `docs/logs/insider_v2_validation_YYYYMMDD.md`
   - Full 10-year results
   - Gate-by-gate evaluation
   - Final GO/NO-GO recommendation

5. **Code Artifacts:**
   - `signals/insider/institutional_insider_v2.py`
   - `tests/test_insider_v2.py`
   - `scripts/diagnose_insider_v2.py`

---

## 10. Final Recommendation

**Status:** DEFERRED TO PHASE 4+

**Rationale:**
- Insider v1 failed convincingly (-13.76% vs M+Q)
- Low probability of v2 success (~20-40%)
- High opportunity cost (3-4 weeks)
- M+Q already provides strong baseline (135.98% return, 0.628 Sharpe)

**Action Plan:**
1. ✅ Document v1 failure thoroughly (complete)
2. ✅ Mark M+Q+I v1 as RESEARCH_NO_GO (complete)
3. ✅ Create this v2 spec for future reference (complete)
4. ⬜ Focus Phase 4 on:
   - M+Q production deployment
   - Alternative signals (earnings quality, short interest)
   - M+Q enhancements (sector neutralization, dynamic weights)

**Only revisit Insider v2 if:**
- M+Q is stable in production
- All higher-priority signals tested
- Explicit stakeholder request

---

**Prepared by:** Claude (Anthropic Sonnet 4.5)
**Date:** 2025-11-24
**Status:** PLANNING (Not Started)
**Next Review:** After Phase 4 priorities complete

---

**END OF INSIDER V2 RESEARCH SPEC**
