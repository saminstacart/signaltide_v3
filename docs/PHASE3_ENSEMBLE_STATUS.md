# Phase 3 Ensemble Status & Recommendations

**Last Updated:** 2025-11-24
**Phase:** Phase 3 - Multi-Signal Ensemble Construction
**Status:** M+Q v1 PROD_READY (Production Evaluation Complete), M+Q+I v1 NO_GO

---

## Executive Summary

Phase 3 has completed validation of two multi-signal ensembles over a 10-year period (2015-2024):

1. **Momentum + Quality (M+Q v1):** ✅ **PROD_READY** - Production evaluation complete, pending paper trading
2. **Momentum + Quality + Insider (M+Q+I v1):** ❌ **RESEARCH_NO_GO** - Not recommended

**Flagship Recommendation:** Use `momentum_quality_v1` (M+Q) as the Phase 3 production strategy.

**Production Evaluation Completed (2025-11-24):**
- ✅ SPY benchmark comparison (IR -0.344, explained by structural Mag-7 underweight)
- ✅ Portfolio turnover metrics (18.07% mean, 11.67% median)
- ✅ Sector & size exposure analysis (~85% large-cap, 2-3% Mag-7)
- ✅ Bootstrap robustness (95% CI [0.095, 1.193] excludes zero, P>0 = 97.44%)

---

## Ensemble Status Table

| Ensemble | Status | 10Y Return | 10Y Sharpe | Max DD | Decision Date | Report |
|----------|--------|------------|------------|--------|---------------|--------|
| **Momentum + Quality v1** | ✅ **PROD_READY** | **135.98%** | **0.628** | -23.89% | 2025-11-24 | [Canonical Diag](logs/phase3_MQ_v1_canonical_diag_20251124.md) |
| Momentum + Quality + Insider v1 | ❌ **RESEARCH_NO_GO** | 122.22% | 0.591 | -23.42% | 2025-11-24 | [Full Diag](logs/phase3_m3_M3.6_full_diag_20251124.md) |

---

## 1. Momentum + Quality v1 (M+Q) - FLAGSHIP

### Configuration

```python
from signals.ml.ensemble_configs import get_momentum_quality_v1_ensemble

ensemble = get_momentum_quality_v1_ensemble(dm)
```

**Composition:**
- InstitutionalMomentum v2: 25% weight
  - 308-day formation, 0-day skip, adaptive quintiles
- CrossSectionalQuality v1: 75% weight
  - QMJ methodology, 3-factor model (profitability, growth, safety)

### Performance (2015-04-01 to 2024-12-31)

| Metric | Value |
|--------|-------|
| Total Return | **135.98%** |
| CAGR | **9.28%** |
| Volatility | 16.32% |
| Sharpe Ratio | **0.628** |
| Max Drawdown | -23.89% |
| Num Rebalances | 117 (monthly) |

### Why M+Q Works

1. **Calibrated Weights:** 25/75 allocation validated via:
   - Grid sweep: Best across all metrics
   - Optuna optimization: Converged to M≈0.20-0.25
   - Quality-heavy plateau is stable and interpretable (1:3 ratio)

2. **Factor Diversification:**
   - Low correlation between momentum and quality (~0.3-0.4)
   - Quality provides stability during momentum crashes

3. **Robust Performance:**
   - Positive in 4/5 macro regimes
   - ~30% better Sharpe than pure momentum
   - Reduced max drawdown vs momentum-only

4. **Production-Ready:**
   - Both signals have GO status
   - Computationally efficient (~10 min for 10-year backtest)
   - Clean cross-sectional implementation
   - PIT-correct with explicit annualization

### Next Steps for M+Q

**Completed (2025-11-24 Phase 3.1 Production Evaluation):**
1. ✅ Run final production validation checklist
2. ✅ Document sector/size exposures
3. ✅ Set up performance tracking vs SPY
4. ✅ Bootstrap robustness testing

**Pending for Paper Trading:**
1. ⬜ Create production monitoring dashboard
2. ⬜ Establish rebalance execution procedures
3. ⬜ Paper trading validation (1-2 months)

**Production Artifacts:**
- SPY Comparison: `results/ensemble_baselines/momentum_quality_v1_vs_spy_monthly.csv`
- Turnover Metrics: `results/ensemble_baselines/momentum_quality_v1_turnover_metrics.csv`
- Exposure Snapshot: `results/ensemble_baselines/momentum_quality_v1_exposure_snapshot.csv`
- Bootstrap Sharpe: `results/ensemble_baselines/momentum_quality_v1_bootstrap_sharpe.csv`

**Status:** ✅ PROD_READY - Approve for paper trading phase.

---

## 2. Momentum + Quality + Insider v1 (M+Q+I) - NO_GO

### ⚠️ Decision: RESEARCH_NO_GO

**The insider signal DEGRADES performance and is NOT recommended for production.**

### Configuration (For Reference Only - Do Not Use)

```python
# ⚠️ NOT RECOMMENDED - Use get_momentum_quality_v1_ensemble() instead
from signals.ml.ensemble_configs import get_momentum_quality_insider_v1_ensemble

ensemble = get_momentum_quality_insider_v1_ensemble(dm)
```

**Composition:**
- InstitutionalMomentum v2: 25% weight
- CrossSectionalQuality v1: 50% weight
- InstitutionalInsider v1: 25% weight

### Performance vs M+Q Baseline (2015-04-01 to 2024-12-31)

| Metric | M+Q Baseline | M+Q+I | Delta | Impact |
|--------|--------------|-------|-------|--------|
| Total Return | **135.98%** | 122.22% | **-13.76%** | 🔴 Negative |
| CAGR | **9.28%** | 8.61% | **-0.67%** | 🔴 Negative |
| Sharpe | **0.628** | 0.591 | **-0.037** | 🔴 Negative |
| Volatility | 16.32% | 16.25% | -0.07% | 🟡 Neutral |
| Max Drawdown | -23.89% | -23.42% | +0.47% | 🟡 Marginal |
| Correlation | - | - | **0.9894** | 🔴 No diversification |
| Runtime | ~10 min | ~107 min | **10.7x slower** | 🔴 Expensive |

### Why Insider v1 Failed

1. **Negative Alpha:**
   - Reduced total return by 13.76 percentage points
   - Lowered Sharpe by 0.037 (not worth the cost)
   - Max drawdown improvement trivial (+0.47%)

2. **Regime Instability:**
   - 5-year test (2020-2024): +7.24% return (POSITIVE)
   - 10-year test (2015-2024): -13.76% return (NEGATIVE)
   - Insider alpha is regime-dependent and unreliable

3. **High Correlation:**
   - 98.94% correlation with M+Q
   - No meaningful diversification benefit

4. **Computational Cost:**
   - 10.7x slower than M+Q (107 min vs 10 min)
   - Fetches ~80K insider transactions per rebalance
   - Not justified by performance

5. **Coverage Issues:**
   - While 98.1% coverage seems good, quality of transactions matters
   - Many transactions are uninformative (planned sales, option exercises)
   - Simple aggregation approach not capturing signal value

### Recommendation

**DO NOT USE** M+Q+I v1 in production. Options:

- ✅ **Recommended:** Use M+Q v1 as flagship
- 🔬 **Future Research:** Insider v2 with improved construction (see spec)
- 🚫 **Not Recommended:** Parameter tuning of v1 (unlikely to overcome -13.76% deficit)

### Full Diagnostic

See: [Phase 3 M3.6 Full Diagnostic](logs/phase3_m3_M3.6_full_diag_20251124.md)

---

## 3. Signal-Level Registry Alignment

The ensemble decisions align with individual signal validation:

| Signal | Version | Standalone Status | Ensemble Performance | Notes |
|--------|---------|-------------------|----------------------|-------|
| InstitutionalMomentum | v2 | ✅ GO | Strong contributor | 308d formation, adaptive quintiles |
| CrossSectionalQuality | v1 | ✅ GO | Strong contributor | QMJ methodology, 3-factor model |
| InstitutionalInsider | v1 | ⚠️ NO_GO (Phase 1) | ❌ Degrades ensemble | Failed 3/5 standalone gates, also fails in ensemble |

**Note:** Insider v1 was already marked NO_GO in Phase 1 standalone testing. The ensemble diagnostic confirms this finding - the signal does not add value either standalone or in combination.

---

## 4. Production Path Forward

### Completed (2025-11-24)

1. ✅ Complete 10-year M+Q vs M+Q+I diagnostic
2. ✅ Document NO_GO decision for M+Q+I v1
3. ✅ Run M+Q production validation checklist
4. ✅ Create M+Q canonical diagnostic report
5. ✅ Document sector/size exposures
6. ✅ Set up SPY benchmark tracking
7. ✅ Bootstrap robustness testing (5,000 resamples)

### Near-Term Actions (Weeks 1-2)

1. ⬜ Establish M+Q monitoring & execution procedures
2. ⬜ Paper trading simulation (1-2 months)
3. ⬜ Create Insider v2 research spec (for later)

### Deferred Actions (Phase 4+)

1. ⬜ Insider v2 research program (separate mini-program)
2. ⬜ Alternative third signal exploration (earnings quality, short interest, etc.)
3. ⬜ Regime-aware ensemble switching
4. ⬜ Dynamic weight optimization

---

## 5. Key Learnings

### Methodological

1. **Smoke tests insufficient:** 5-year smoke showed +7.24%, but 10-year showed -13.76%
   - Always validate over full available history
   - Short-term wins may be regime-specific

2. **Regime dependency matters:** Signals that work in one regime may fail in others
   - Insider helped in 2020-2024 (volatile, info-rich)
   - Insider hurt in 2015-2019 (stable, low-vol bull)

3. **Correlation kills diversification:** 98.94% correlation = no benefit
   - Third signal must be truly orthogonal to add value
   - High correlation + negative alpha = double penalty

### Operational

1. **Computational cost is real:** 10.7x slowdown matters for research velocity
   - Only worthwhile if performance justifies it
   - M+Q+I took 2 hours vs M+Q's 10 minutes

2. **Data quality > data coverage:** 98.1% coverage wasn't enough
   - Transaction quality matters more than quantity
   - Many insider trades are uninformative

3. **Prioritization framework worked:** Phased approach caught issues early
   - Priority 1 (annualization) prevented 4.8x Sharpe inflation bug
   - Priority 2 (API) ensured correct implementation
   - Priority 3 (coverage) validated data before full run

---

## 6. References

### Primary Diagnostics

- **M+Q Canonical Production Diagnostic:** [phase3_MQ_v1_canonical_diag_20251124.md](logs/phase3_MQ_v1_canonical_diag_20251124.md)
  - Complete production evaluation with SPY benchmark, turnover, exposure, bootstrap
  - Status: PROD_READY (pending paper trading)

- **Full 10-Year Comparison Diagnostic:** [phase3_m3_M3.6_full_diag_20251124.md](logs/phase3_m3_M3.6_full_diag_20251124.md)
  - M+Q vs M+Q+I comparison methodology and results
  - Failure mode analysis for insider signal
  - Three options: ABANDON / OPTIMIZE / CONDITIONAL

### M+Q v1 Production Artifacts (2025-11-24)

- **SPY Comparison:** [momentum_quality_v1_vs_spy_monthly.csv](../results/ensemble_baselines/momentum_quality_v1_vs_spy_monthly.csv)
- **Turnover Metrics:** [momentum_quality_v1_turnover_metrics.csv](../results/ensemble_baselines/momentum_quality_v1_turnover_metrics.csv)
- **Exposure Snapshot:** [momentum_quality_v1_exposure_snapshot.csv](../results/ensemble_baselines/momentum_quality_v1_exposure_snapshot.csv)
- **Bootstrap Sharpe:** [momentum_quality_v1_bootstrap_sharpe.csv](../results/ensemble_baselines/momentum_quality_v1_bootstrap_sharpe.csv)

### Supporting Documentation

- **M3.6 Spec:** [ENSEMBLES_M3.6_THREE_SIGNAL_SPEC.md](ENSEMBLES_M3.6_THREE_SIGNAL_SPEC.md)
- **Priority Logs:**
  - Priority 1-2: [phase3_m3_followup_M3.6_20251124.md](logs/phase3_m3_followup_M3.6_20251124.md)
  - Priority 3: [phase3_m3_M3.6_priority3_20251124.md](logs/phase3_m3_M3.6_priority3_20251124.md)
- **Weight Calibration:** [momentum_quality_v1_weight_sweep.md](../results/ensemble_baselines/momentum_quality_v1_weight_sweep.md)
- **Architecture:** [ENSEMBLES.md](core/ENSEMBLES.md)

### M+Q+I Artifacts (For Reference Only - NO_GO)

- **M+Q+I Diagnostic Report:** [mqi_three_signal_v1_diagnostic.md](../results/ensemble_baselines/mqi_three_signal_v1_diagnostic.md)
- **M+Q+I Comparison CSV:** [mqi_three_signal_v1_comparison.csv](../results/ensemble_baselines/mqi_three_signal_v1_comparison.csv)

---

## 7. Decision Authority & Sign-Off

**Prepared by:** Claude (Anthropic Opus 4.5)
**Date:** 2025-11-24
**Decision:** M+Q v1 → ✅ **PROD_READY**, M+Q+I v1 → ❌ **RESEARCH_NO_GO**

**Rationale:**
- M+Q delivers strong, robust 10-year performance (135.98% return, 0.628 Sharpe)
- Bootstrap significance confirmed (95% CI [0.095, 1.193] excludes zero, P>0 = 97.44%)
- Turnover acceptable (18.07% mean → ~11 bps/year cost drag)
- Factor exposures understood (structural Mag-7 underweight explains -0.344 IR vs SPY)
- M+Q+I degrades performance (-13.76% vs M+Q) and adds no diversification value
- Insider signal is regime-dependent and unreliable over full history
- Computational cost (10.7x) not justified by results

**Production Evaluation Complete:**
- ✅ SPY benchmark comparison
- ✅ Portfolio turnover metrics
- ✅ Sector & size exposure analysis
- ✅ Bootstrap robustness testing

**Next Review:** After paper trading validation complete

---

**END OF STATUS DOCUMENT**
