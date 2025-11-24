# Momentum Phase 2 Optimization Specification

**Date:** 2025-11-21
**Status:** ACTIVE
**Signal:** InstitutionalMomentum (Jegadeesh-Titman 12-1 variants)
**Objective:** Find optimal hyperparameters to maximize out-of-sample Sharpe while maintaining regime stability

---

## 1. Objective

Conduct controlled hyperparameter search on InstitutionalMomentum to:
1. Improve full-sample Sharpe ratio (target: >0.15)
2. Maximize out-of-sample Sharpe (target: ≥0.2)
3. Improve decile monotonicity
4. Maintain positive performance across all regimes
5. Identify 1-3 candidate configurations for paper trading

**This is NOT an automatic optimizer** - we test a small predeclared grid and dump all results for manual review.

---

## 2. Fixed Configuration

### Universe and Data
- **Universe:** S&P 500 PIT (`sp500_actual`)
- **Period:** 2015-01-01 to 2024-12-31 (effective start after momentum lookback)
- **Rebalancing:** Monthly (end of month)
- **Capital:** $50,000
- **Transaction costs:** 5 bps flat (commission 0 + slippage 2 bps + spread 3 bps)

### Sample Splits
- **In-sample (IS):** 2015-01-01 to 2022-12-31 (train window)
- **Out-of-sample (OOS):** 2023-01-01 to 2024-12-31 (validation holdout)
- **Full sample:** 2015-01-01 to 2024-12-31 (final evaluation)

**Rationale for OOS window:**
- Phase 1.5 showed recent regime (2023-2024) is STRONGEST (Sharpe 0.31)
- Using 2023-2024 as OOS tests if momentum remains strong in most recent data
- Simulates realistic "deploy in 2023, test through 2024" scenario

---

## 3. Hyperparameter Search Space

**Small discrete grid (27 total configurations):**

| Parameter | Values | Interpretation |
|-----------|--------|----------------|
| `formation_period` | [126, 189, 252] | 6, 9, 12 months |
| `skip_period` | [5, 10, 21] | 1 week, 2 weeks, 1 month |
| `winsorize_pct` | [1, 5, 10] | Two-sided outlier clipping |

**Parameter interpretations:**
- **Formation period:** Lookback window for momentum calculation
  - 126 days (6 months): Faster momentum, higher turnover
  - 189 days (9 months): Medium-term momentum
  - 252 days (12 months): Standard Jegadeesh-Titman

- **Skip period:** Buffer to avoid short-term reversals
  - 5 days (1 week): Minimal skip, captures recent momentum
  - 10 days (2 weeks): Moderate buffer
  - 21 days (1 month): Standard academic skip period

- **Winsorization:** Outlier handling at [lower%, 100-upper%] percentiles
  - 1%: Minimal clipping (keeps extreme winners/losers)
  - 5%: Standard institutional practice
  - 10%: Aggressive outlier suppression

**Total configurations:** 3 × 3 × 3 = 27

---

## 4. Metrics Per Configuration

### Primary Metrics
1. **Full-sample Sharpe** (2015-2024)
2. **In-sample Sharpe** (2015-2022)
3. **Out-of-sample Sharpe** (2023-2024)
4. **Deflated Sharpe Ratio (DSR)** - simple proxy for overfitting adjustment
5. **Maximum drawdown** (full sample)
6. **Annual return** (full sample)
7. **Volatility** (annualized, full sample)

### Regime-Specific Metrics
Split OOS window into sub-regimes and compute monthly return & Sharpe:
- **COVID (2020):** Jan 2020 - Dec 2020
- **Bear 2022:** Jan 2021 - Dec 2022
- **Recent (2023-2024):** Jan 2023 - Dec 2024

**Why these regimes?**
- COVID: Extreme volatility stress test
- Bear 2022: Rising rates, tech drawdown
- Recent: Current market regime (most predictive of future)

### Overfitting Controls (Phase 2.0 - Simple Version)
**Deflated Sharpe Ratio (DSR)** - simplified version:
- **Formula:** `DSR = Sharpe / sqrt(2 * ln(n_trials) / (n_months - 1))`
- **Purpose:** Penalize Sharpe based on number of trials tested
- **Threshold:** DSR > 1.0 suggests robust signal (not data-mined)

**Future enhancements (Phase 2.1+):**
- Purged K-Fold CV (López de Prado, 2018)
- Combinatorial Purged CV (CPCV)
- Probability of Backtest Overfitting (PBO)

---

## 5. Acceptance Gates

Each configuration is evaluated against these **predeclared thresholds**:

| Gate | Threshold | Rationale |
|------|-----------|-----------|
| **Full-sample Sharpe** | > 0.15 | Minimum economic viability |
| **OOS Sharpe** | ≥ 0.2 | Must work in holdout period |
| **Regime stability** | All regimes > -0.3 | No catastrophic failures |
| **DSR** | > 0.5 | Basic overfitting guard |

**Pass criteria:**
- **PASS:** All 4 gates passed
- **CONDITIONAL:** 3/4 gates passed (subject to manual review)
- **FAIL:** ≤2 gates passed

---

## 6. Output Deliverables

### 6.1 Trial Results CSV
**File:** `results/momentum_phase2_trials.csv`

**Columns:**
```
formation_period, skip_period, winsorize_pct,
full_sharpe, is_sharpe, oos_sharpe, dsr,
annual_return, volatility, max_drawdown,
regime_covid_return, regime_covid_sharpe,
regime_2022_return, regime_2022_sharpe,
regime_recent_return, regime_recent_sharpe,
passes_full_gate, passes_oos_gate, passes_regime_gate, passes_dsr_gate,
passes_all_gates
```

### 6.2 Summary Report
**File:** `results/MOMENTUM_PHASE2_SUMMARY.md`

**Contents:**
1. Executive summary
   - Total configs tested
   - Configs passing all gates
   - Configs passing 3/4 gates (conditional)
2. Table of passing configurations
   - Sorted by OOS Sharpe descending
   - Hyperparameters + key metrics
3. Regime stability analysis
4. Overfitting assessment (DSR distribution)
5. **NO automatic winner selection** - manual review required

### 6.3 Detailed Trials Report
**File:** `results/momentum_phase2_trials.md`

**Contents:**
- Full table of all 27 configs
- Sorted by OOS Sharpe
- Visual indicators for gate pass/fail
- Regime-by-regime breakdown

---

## 7. Implementation Notes

### 7.1 Data Requirements
- Price data must start **400 days before backtest start** to compute momentum on first rebalance
- Effective backtest start will be later than 2015-01-01 due to momentum lookback
- For 252-day formation + 21-day skip, need 273 trading days minimum

### 7.2 Signal Construction
Reuse `InstitutionalMomentum` class from `signals/momentum/institutional_momentum.py`:
```python
params = {
    'formation_period': formation,
    'skip_period': skip,
    'winsorize_pct': [winsor, 100-winsor],
    'quintiles': True,
    'rebalance_frequency': 'monthly'
}
signal = InstitutionalMomentum(params)
```

### 7.3 Backtest Mechanics
- Build universe using `UniverseManager.get_universe('sp500_actual', as_of_date)`
- Monthly rebalancing via `get_rebalance_dates(schedule='M', ...)`
- Track equity curve daily, compute returns monthly
- Apply transaction costs via `TransactionCostModel`

### 7.4 Determinism
- Fix random seed: `np.random.seed(42)`
- Universe construction is deterministic (PIT S&P 500)
- Winsorization uses fixed percentiles (no randomness)
- Rebalance dates from trading calendar (deterministic)

---

## 8. Success Criteria

**Phase 2 is successful if:**
1. ✅ Script runs without errors for all 27 configs
2. ✅ Results CSV and MD files are generated
3. ✅ At least 1 config passes all 4 acceptance gates
4. ✅ OOS Sharpe of passing configs ≥ 0.2
5. ✅ Recent regime (2023-2024) Sharpe ≥ 0.2 for passing configs

**Phase 2 is a failure if:**
- ❌ Zero configs pass all gates
- ❌ Passing configs have negative recent regime returns
- ❌ DSR < 0 for all passing configs (suggests pure overfitting)

**Next steps if successful:**
- Manual review of 1-3 passing configs
- Select final config(s) for Phase 3 (ensemble design)
- Run Phase 1.5 diagnostics on selected config to generate formal report

**Next steps if failure:**
- Expand search space (e.g., add 63-day formation, 0-day skip)
- Reconsider acceptance thresholds (may be too strict)
- Investigate if momentum premium has decayed in S&P 500

---

## 9. Timeline

- **Phase 2.0 (Grid search):** 2025-11-21 to 2025-11-22
- **Manual review:** 2025-11-22
- **Phase 2.1 (Advanced validation):** TBD (only if Phase 2.0 passes)

---

## 10. Risk Factors

### Overfitting Risks
- **Small grid mitigates:** Only 27 trials reduces multiple testing burden
- **OOS holdout guards:** True holdout validation (no peeking at 2023-2024 during IS training)
- **DSR penalty:** Adjusts Sharpe for number of trials

### Data Risks
- **Regime shift:** OOS period (2023-2024) may not represent future
- **Survivorship bias:** S&P 500 PIT mitigates but not eliminates
- **Cost model:** Flat 5 bps may underestimate real costs for some stocks

### Implementation Risks
- **Lookback bias:** Ensure price data fetched early enough
- **Duplicate dates:** Handle corporate actions (learned from Phase 1.5)
- **Sign errors:** Verify long-short direction (decile 1 = winners)

---

**Document Owner:** SignalTide Research Team
**Last Updated:** 2025-11-21
**Status:** Phase 2 Specification ACTIVE
