# Phase 3 Milestone 3.6 - 3-Signal M+Q+Insider Ensemble Notes

**Created:** 2025-11-23
**Status:** M3.6 Planning / Infrastructure Assessment
**Goal:** Add InstitutionalInsider to momentum+quality ensemble for 3-signal diversification

---

## Section 3.1 - Insider Infrastructure Inventory

### Insider Signals Found

1. **InstitutionalInsider** (`signals/insider/institutional_insider.py`)
   - **Status:** ✅ Implemented (legacy API only)
   - **Academic Basis:** Cohen, Malloy & Pomorski (2012) "Decoding Inside Information"
   - **Methodology:**
     - Dollar-weighted insider transactions
     - Role hierarchy (CEO=3.0, CFO=2.5, Director=1.5, Officer=1.0, Other=0.5)
     - Cluster detection (3+ insiders within 7 days)
     - 90-day lookback window
     - Min transaction value: $10,000

   - **Current API:**
     ```python
     def generate_signals(self,
                         data: pd.DataFrame,
                         bulk_insider_data: Optional[pd.DataFrame] = None
                        ) -> pd.Series
     ```
     - Returns signals in [-1, 1] range
     - Supports bulk mode (pre-fetched insider data) for performance
     - Supports legacy mode (per-ticker DB queries) for compatibility

   - **Data Dependencies:**
     - Sharadar insiders table (`sharadar_insiders`)
     - Accessed via `data_manager.get_insider_trades()`
     - Point-in-time safe (uses filing dates)

   - **Missing for Ensemble Integration:**
     - ❌ No `generate_cross_sectional_scores(rebal_date, universe, data_manager)` method
     - ❌ Not compatible with `make_multisignal_ensemble_fn()` adapter
     - ❌ Cannot be used in Phase 3 cross-sectional ensemble pathway

2. **SimpleInsider** (`signals/insider/simple_insider.py`)
   - **Status:** ⚠️ Likely deprecated/legacy
   - **Not assessed** - InstitutionalInsider is the production candidate

### Signal Catalog Documentation

From `docs/signal_catalog.md` (lines 200-252):
- Status: "Implemented (needs backtest integration)"
- Backtest-Ready: "⏳ Planned"
- Ensemble-Ready: "⏳ Planned"
- Recommended Use: "Future integration"

**Interpretation:** InstitutionalInsider is a well-implemented signal that needs the Phase 3 cross-sectional interface added for ensemble use.

---

## Section 3.2 - Path Forward for M3.6

### Option A: Add Cross-Sectional Interface (RECOMMENDED)

**Scope:** Extend InstitutionalInsider with new method while preserving legacy API.

**Implementation:**

```python
# Add to InstitutionalInsider class
def generate_cross_sectional_scores(
    self,
    rebal_date: pd.Timestamp,
    universe: Sequence[str],
    data_manager: "DataManager",
) -> pd.Series:
    """
    Generate cross-sectional insider scores for universe at rebalance date.

    Workflow:
    1. Bulk-fetch insider trades for all tickers in universe
       - Date range: [rebal_date - lookback_days, rebal_date]
       - Point-in-time filter: filing_date <= rebal_date
    2. For each ticker, compute dollar-weighted insider score
    3. Cross-sectional rank/normalize scores → [-1, 1] or quintiles
    4. Return pd.Series indexed by ticker

    Returns:
        pd.Series: Insider scores (subset of universe with valid data)
    """
    # 1. Bulk fetch insider data for entire universe
    lookback_start = rebal_date - pd.Timedelta(days=self.lookback_days)
    bulk_insider_data = data_manager.get_bulk_insider_trades(
        tickers=universe,
        start_date=lookback_start.strftime('%Y-%m-%d'),
        end_date=rebal_date.strftime('%Y-%m-%d'),
        as_of_date=rebal_date.strftime('%Y-%m-%d')  # PIT filter
    )

    # 2. Compute per-ticker insider scores
    scores = {}
    for ticker in universe:
        ticker_insiders = bulk_insider_data.xs(ticker, level='ticker', drop_level=False)
        if len(ticker_insiders) > 0:
            score = self._compute_insider_score(ticker_insiders)  # refactor from generate_signals
            scores[ticker] = score

    # 3. Cross-sectional ranking
    scores_series = pd.Series(scores)
    ranked_scores = self.to_cross_sectional_rank(
        scores_series,
        winsorize_pct=self.winsorize_pct
    )  # from InstitutionalSignal base class

    # 4. Optional: Convert to quintiles if self.quintiles == True
    if self.quintiles:
        ranked_scores = self.to_quintiles(
            ranked_scores,
            mode=self.quintile_mode
        )

    return ranked_scores
```

**Refactoring Required:**
1. Extract score computation logic from `generate_signals()` into helper:
   ```python
   def _compute_insider_score(self, insider_trades: pd.DataFrame) -> float:
       """Compute dollar-weighted insider score from trades."""
       # Extract logic from generate_signals that computes:
       # - Dollar weighting
       # - Role weighting
       # - Cluster bonuses
       # Return single score in continuous range
   ```

2. Add bulk data fetcher to DataManager if not exists:
   ```python
   # data/data_manager.py
   def get_bulk_insider_trades(
       self,
       tickers: List[str],
       start_date: str,
       end_date: str,
       as_of_date: str
   ) -> pd.DataFrame:
       """
       Fetch insider trades for multiple tickers in single query.

       Returns:
           MultiIndex DataFrame [(ticker, filing_date)] with insider trade data
       """
   ```

**Effort Estimate:** 2-3 hours (medium complexity)
- Refactor score computation: 1 hour
- Implement cross-sectional method: 1 hour
- Test on small universe: 30 min
- Smoke test in ensemble: 30 min

**Testing:**
- Unit test: `test_institutional_insider_cross_sectional()` in `tests/test_institutional_signals.py`
- Smoke test: Add to `test_multisignal_ensemble.py` with small 3-ticker universe

---

### Option B: Placeholder Ensemble (INTERIM)

If time-constrained, create ensemble config with placeholder/stub:

```python
# signals/ml/ensemble_configs.py
def get_momentum_quality_insider_v1_ensemble(dm: DataManager) -> EnsembleSignal:
    """
    3-signal M+Q+Insider ensemble (Phase 3 M3.6).

    STATUS: STUB - Insider signal not yet ensemble-ready

    InstitutionalInsider lacks generate_cross_sectional_scores() as of 2025-11-23.
    See: docs/ENSEMBLES_M3.6_NOTES.md for implementation path.

    DO NOT USE until insider integration completed.
    """
    raise NotImplementedError(
        "InstitutionalInsider needs cross-sectional interface. "
        "See docs/ENSEMBLES_M3.6_NOTES.md Section 3.2 Option A."
    )
```

**Effort Estimate:** 15 min (documentation only)

**Use Case:** Marks M3.6 as "in progress" without blocking other Phase 3 work.

---

## Section 3.3 - Proposed 3-Signal Ensemble Spec

**Name:** `momentum_quality_insider_v1`
**Status:** RESEARCH
**Universe:** S&P 500 actual constituents (PIT), min_price=$5

### Member Signals

| Signal | Version | Weight (v1) | Normalization | Rationale |
|--------|---------|-------------|---------------|-----------|
| InstitutionalMomentum | v2 | 0.25 | none (adaptive quintiles) | Price-based, trend capture |
| CrossSectionalQuality | v1 | 0.50 | none (adaptive quintiles) | Fundamental, defensive |
| InstitutionalInsider | v1 | 0.25 | none (adaptive quintiles) | Behavioral, alpha source |

**Total:** 1.00

### Weight Rationale (v1 Starting Point)

- **Quality-heavy (0.50):** Proven defensive value from M3.4 calibration (25/75 M+Q)
- **Momentum + Insider balanced (0.25 each):** Both are alpha sources, start symmetric
- **Diversification hypothesis:**
  - Momentum: Price-driven, medium-term trends
  - Quality: Fundamental-driven, valuation anchor
  - Insider: Information-driven, short-term edge
  - Low expected correlation → potential Sharpe lift

### Evaluation Plan

1. **Baseline diagnostic** (compare vs M+Q 25/75):
   - Full period: 2015-2024
   - Metrics: Sharpe, CAGR, Max DD, turnover
   - Output: `results/ensemble_baselines/momentum_quality_insider_v1_diagnostic.md`

2. **Per-regime analysis:**
   - Use same 5 regimes as M+Q regime diagnostic
   - Identify where insider adds value vs drag

3. **Weight calibration** (if promising):
   - Small grid sweep: (w_m, w_q, w_i) with sum=1.0
   - Focus on quality weight range [0.4, 0.6], vary M/I split
   - Optuna refinement if warranted

---

## Section 3.4 - Data Coverage Assessment

### Sharadar Insiders Table

**Need to verify:**
- [ ] Date range coverage (does it support 2015-2024 backtest?)
- [ ] S&P 500 coverage (% of universe with insider data)
- [ ] Transaction density (avg trades per ticker per quarter)
- [ ] Data quality (filing date consistency, PIT correctness)

**Action:** Query database for coverage stats before full implementation.

```sql
-- Coverage check
SELECT
  MIN(filing_date) as earliest_filing,
  MAX(filing_date) as latest_filing,
  COUNT(DISTINCT ticker) as unique_tickers,
  COUNT(*) as total_trades
FROM sharadar_insiders
WHERE filing_date >= '2015-01-01'
  AND filing_date <= '2024-12-31';

-- S&P 500 overlap (requires universe table join)
SELECT
  COUNT(DISTINCT si.ticker) as sp500_with_insider,
  COUNT(DISTINCT um.ticker) as sp500_total
FROM sharadar_insiders si
RIGHT JOIN dim_universe_membership um
  ON si.ticker = um.ticker
 AND um.universe = 'sp500_actual'
WHERE si.filing_date >= '2015-01-01';
```

**Expected Coverage:** 80-95% of S&P 500 should have insider data (large caps have more filing activity).

---

## Section 3.5 - Decision Matrix

| Approach | Pros | Cons | Recommendation |
|----------|------|------|----------------|
| **Option A: Full Implementation** | - Proper ensemble integration<br>- Enables weight calibration<br>- Reusable for future ensembles | - 2-3 hours work<br>- Requires refactoring<br>- Needs testing | ✅ **DO THIS** if data coverage is good |
| **Option B: Placeholder** | - Fast (15 min)<br>- Documents intent<br>- Unblocks other M3 work | - No actual ensemble<br>- M3.6 incomplete | ⚠️ Use only if time-critical |

---

## Section 3.6 - Next Actions (Recommended)

### Immediate (M3.6 Phase 1 - Infrastructure)

1. ✅ Inventory insider signals (DONE - this document)
2. **Verify data coverage** (30 min):
   ```bash
   python3 -c "
   from data.data_manager import DataManager
   dm = DataManager()
   # Run coverage queries above
   # Print results
   "
   ```
3. **If coverage is good (>75% of S&P 500):**
   - Proceed with Option A (full implementation)
4. **If coverage is poor (<50%):**
   - Document limitation
   - Create placeholder (Option B)
   - Flag as future work when better insider data available

### Follow-on (M3.6 Phase 2 - Integration)

1. Add `generate_cross_sectional_scores()` to InstitutionalInsider
2. Add unit tests
3. Create ensemble config with v1 weights (0.25/0.50/0.25)
4. Run baseline diagnostic
5. Update docs (signal_catalog.md, ENSEMBLES.md)
6. Add to ENSEMBLE_REGISTRY

### Stretch (M3.6 Phase 3 - Calibration)

1. Weight grid sweep
2. Per-regime analysis
3. Optuna refinement

---

## References

- Cohen, Malloy & Pomorski (2012) "Decoding Inside Information"
- Phase 3 M3.4 weight calibration: `results/ensemble_baselines/momentum_quality_v1_weight_optuna.md`
- Signal catalog: `docs/signal_catalog.md` lines 200-252
- Insider signal implementation: `signals/insider/institutional_insider.py`

---

**Status:** M3.6 assessment complete, ready for decision on implementation approach.
