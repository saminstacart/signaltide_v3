# Signal Registry - Stage 1 Status

**Purpose:** This registry tracks all signals that have completed Phase 1 baseline validation. It provides a single source of truth for which signals passed acceptance gates and are ready for ensemble construction vs which failed and should be archived.

**Last Updated:** 2025-11-21
**Stage:** Stage 1 Complete (Momentum v2, Quality v1, Insider v1 evaluated)

---

## Signal Status Table

| Signal Name | Version | Universe | Status | Full Sharpe | OOS Sharpe | Recent Sharpe | Verdict Notes | Report Path |
|-------------|---------|----------|--------|-------------|------------|---------------|---------------|-------------|
| **InstitutionalMomentum** | v2 (Trial 11) | sp500_actual | **GO** | 0.245 | 0.742 | 0.309 | Ready for ensemble. Canonical config: 308d formation, 0d skip, 9.2% winsor. Strong OOS performance. | `results/MOMENTUM_PHASE2_TRIAL11_DIAGNOSTIC.md` |
| **CrossSectionalQuality** | v1 | sp500_actual | **NO_GO** | N/A | N/A | N/A | Failed Phase 1 acceptance gates. Poor decile monotonicity and weak long-short spread. Archived. | `results/quality_v1_phase1_report.md` |
| **InstitutionalInsider** | v1 | sp500_actual | **NO_GO** | 0.034 | 0.374 | 0.132 | Failed 3/5 gates. No decile monotonicity (D1-D10 spread only 0.28%/yr). Statistically insignificant (t-stat 0.180). Archived. | `results/INSIDER_PHASE1_REPORT.md` |

---

## Detailed Signal Descriptions

### InstitutionalMomentum v2 - **GO**

**Methodology:** Jegadeesh-Titman cross-sectional momentum with optimized parameters
**Canonical Config:**
- Formation period: 308 days (~14.7 months)
- Skip period: 0 days
- Winsorization: 9.2% two-sided
- Rebalance: Monthly (end of month)
- Position sizing: Equal-weight top quintile

**Performance Metrics:**
- **Full Sample (2015-2024):** Sharpe 0.245, Return 3.33%/yr, Max DD -44.76%
- **In-Sample (2015-2022):** Sharpe 0.155, Return 2.22%/yr
- **Out-of-Sample (2023-2024):** Sharpe 0.742, Return 7.72%/yr, Max DD -12.08%

**Regime Performance:**
- COVID (2020): Sharpe 0.139, Return 1.01%/mo
- Bear 2022 (2021-2022): Sharpe 0.188, Return 1.31%/mo
- Recent (2023-2024): Sharpe 0.309, Return 1.36%/mo

**Gates Passed:** All Phase 2 gates (Full Sharpe > 0.15, OOS Sharpe ≥ 0.20, OOS Max DD < 30%, Recent Sharpe > 0.20)

**Next Steps:**
- Ready for ensemble design (Phase 3)
- Consider combining with complementary signals
- Monitor OOS performance going forward

**Config File:** `results/MOMENTUM_PHASE2_CANONICAL_CONFIG.json`

---

### CrossSectionalQuality v1 - **NO_GO**

**Methodology:** Asness QMJ-style quality signal (profitability + growth + safety)
**Phase 1 Config:**
- Profitability weight: 40%
- Growth weight: 30%
- Safety weight: 30%
- Winsorization: 5-95%
- Rebalance: Monthly

**Failure Reasons:**
- Poor decile monotonicity (expected relationship between quality and returns not observed)
- Weak long-short spread (insufficient separation between high and low quality stocks)
- Failed minimum IR threshold
- Inconsistent regime behavior

**Diagnosis:** The quality signal as currently specified does not exhibit sufficient predictive power on S&P 500 PIT universe over 2015-2024 period.

**Status:** Archived - Do not optimize
**Report:** `results/quality_v1_phase1_report.md`

---

### InstitutionalInsider v1 - **NO_GO**

**Methodology:** Cohen-Malloy-Pomorski insider trading signal with role weighting and cluster detection
**Phase 1 Config:**
- Lookback: 90 days
- Min transaction value: $10,000
- Cluster window: 7 days, min 3 insiders
- CEO weight: 3.0x, CFO weight: 2.5x
- Rebalance: Monthly

**Performance Metrics:**
- **Full Sample:** Sharpe 0.034, Return 0.15%/yr, Max DD -9.22%
- **OOS:** Sharpe 0.374, Return 1.22%/yr (OOS better than IS, but likely noise)
- **t-statistic:** 0.180 (p > 0.05, not statistically significant)

**Critical Failure: No Decile Monotonicity**
- D1 (high insider buying): 11.03%/yr
- D4 (middle): 13.29%/yr (best performance!)
- D10 (high insider selling): 10.75%/yr
- **Spread (D1 - D10):** 0.28%/yr (threshold: ≥6%/yr for GO)

**Gates Failed:**
1. ❌ Decile monotonicity: 0.02%/mo (need ≥0.5%/mo)
2. ❌ Full Sharpe: 0.034 (need ≥0.30)
3. ❌ t-statistic: 0.180 (need ≥2.0)

**Gates Passed:**
4. ✅ Recent mean return: 0.10%/mo (need >0%)
5. ✅ OOS Sharpe: 0.374 (need ≥0.20)

**Diagnosis:**
- Insider activity has no predictive relationship with future returns on S&P 500 large caps
- Market efficiently prices publicly available insider transaction data
- Signal may work better on small/mid-caps where information asymmetry is higher
- Post-SOX regulatory environment may have reduced insider edge

**Status:** Archived - Do not optimize
**Report:** `results/INSIDER_PHASE1_REPORT.md`

---

## Stage 1 Summary

**Signals Evaluated:** 3
**Signals Passing Gates:** 1 (Momentum v2)
**Signals Failed:** 2 (Quality v1, Insider v1)

**Portfolio Composition for Stage 2 (Ensemble Design):**
- **Single signal:** InstitutionalMomentum v2 (Trial 11 canonical config)
- **Ensemble options:**
  - Add leverage/volatility overlay
  - Combine with macro regime filters
  - Consider alternative quality/insider specifications if research warrants

**Decision Date:** 2025-11-21
**Next Phase:** Ensemble design (Phase 3) with Momentum v2 as anchor signal

---

## Usage

**For Research:**
```python
from core.signal_registry import get_signal_status, list_signals

# Get specific signal
momentum = get_signal_status('InstitutionalMomentum', 'v2')
print(f"Status: {momentum.status}, OOS Sharpe: {momentum.oos_sharpe}")

# List all signals
all_signals = list_signals()
go_signals = [s for s in all_signals if s.status == 'GO']
```

**For Documentation:**
- Always check this registry before proposing new signal research
- Update this file when completing Phase 1 for any new signal
- Archive NO_GO signals but keep registry entry for institutional memory

---

## Maintenance Protocol

**When adding a new signal:**
1. Complete Phase 1 baseline validation
2. Run diagnostic script and generate report
3. Evaluate against acceptance gates
4. Add row to Signal Status Table above
5. Add detailed description section
6. Update `core/signal_registry.py` with new entry
7. Commit changes with message: `Add [SignalName] to registry: [GO/NO_GO]`

**Quarterly review:**
- Re-run OOS performance for GO signals on latest data
- Check if any NO_GO signals warrant re-evaluation with new data/methodology
- Update "Last Updated" date at top of file
