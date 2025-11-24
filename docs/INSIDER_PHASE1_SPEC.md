# Insider Trading Signal - Phase 1 Specification

**Status:** Phase 1 - Baseline Economic Validation
**Signal:** InstitutionalInsider (Cohen-Malloy-Pomorski methodology)
**Date Created:** 2025-11-21
**Owner:** SignalTide Research Team

---

## 1. Objective

Establish baseline economic performance for the **InstitutionalInsider** signal on the S&P 500 point-in-time (PIT) universe over the period 2015-2024.

**Primary Goal:** Decide GO vs NO-GO for Insider Phase 2 optimization based on long-short factor behavior and decile monotonicity.

**Phase 1 Focus:**
- Does the Insider signal exhibit a monotonic relationship between insider activity and future returns?
- Is the long-short spread (high insider buying vs low/selling) statistically significant?
- Does the signal show consistent behavior across market regimes?
- Are the economic magnitudes large enough to justify optimization effort?

---

## 2. Signal Implementation

**Signal Class:** `InstitutionalInsider`
**Location:** `signals/insider/institutional_insider.py`

### Methodology (Cohen-Malloy-Pomorski 2012)

The Insider signal uses a professional implementation of academic insider trading research:

1. **Dollar-weighted transactions**
   - Weight each insider trade by transaction value (log scale)
   - Focus on economically meaningful trades (≥$10,000 default)

2. **Role hierarchy weighting**
   - CEO: 3.0x weight (highest information content)
   - CFO: 2.5x weight
   - President: 2.5x weight
   - COO: 2.0x weight
   - Director: 1.5x weight
   - Officer: 1.0x weight
   - Other: 0.5x weight

3. **Cluster detection**
   - Identify coordinated insider activity (≥3 insiders within 7 days)
   - Amplify cluster signals by 2x (Cohen-Malloy-Pomorski finding)

4. **Cross-sectional ranking**
   - Aggregate insider activity over 90-day lookback window
   - Rank stocks by net insider score (buys minus sells)
   - Normalize to [-1, 1] range using rolling percentile rank

5. **Monthly rebalancing**
   - Hold positions for entire month (reduce transaction costs)
   - Signal updates at month-end only

### Default Parameters (Phase 1)

```python
{
    'lookback_days': 90,           # 3-month insider activity window
    'min_transaction_value': 10000, # Minimum $10K transaction
    'cluster_window': 7,            # 7-day cluster detection window
    'cluster_min_insiders': 3,      # Minimum 3 insiders for cluster
    'ceo_weight': 3.0,              # CEO weighting
    'cfo_weight': 2.5,              # CFO weighting
    'winsorize_pct': [5, 95],      # Two-sided winsorization
    'rebalance_frequency': 'monthly'
}
```

### Academic References

**Primary:**
- Cohen, Malloy, Pomorski (2012). "Decoding Inside Information". *Journal of Finance*, 67(3), 1009-1043.
  - Finding: Routine insider trades vs opportunistic trades
  - Cluster purchases predict 6-12% annual alpha
  - CEO/CFO trades have highest information content

**Supporting:**
- Seyhun (1986). "Insiders' Profits, Costs of Trading, and Market Efficiency". *Journal of Financial Economics*, 16(2), 189-212.
  - Early evidence on insider trading profitability
  - Documentation of role hierarchy effects

- Jeng, Metrick, Zeckhauser (2003). "Estimating the Returns to Insider Trading: A Performance-Evaluation Perspective". *Review of Economics and Statistics*, 85(2), 453-471.
  - Insider purchases predict returns, sales do not
  - Dollar-weighting improves signal quality

---

## 3. Universe and Backtest Settings

**Universe:** S&P 500 PIT (`sp500_actual`)
- Point-in-time membership (no survivorship bias)
- Minimum price: $5.00 per share
- Expected ~500 stocks per month

**Date Range:**
- **Start:** 2015-04-01 (align with Momentum/Quality Phase 1)
- **End:** 2024-12-31
- **In-sample cutoff:** 2022-12-31
- **Out-of-sample period:** 2023-01-01 to 2024-12-31

**Rebalancing:**
- **Frequency:** Monthly (end of month)
- **Schedule:** `get_rebalance_dates(schedule='M', ...)`
- **Holding period:** Entire month (no intra-month rebalancing)

**Capital:**
- **Initial:** $50,000
- **Position sizing:** Equal-weight within portfolios

**Transaction Costs:**
- **Model:** 20 basis points per trade (10 commission + 5 slippage + 5 spread)
- **Application:** Applied on all rebalances

---

## 4. Portfolio Construction

Phase 1 tests TWO portfolio types to understand factor behavior:

### Portfolio 1: Long-Only Top Quintile
- **Long positions:** Top 20% of stocks by Insider score (highest insider buying)
- **Equal-weight:** 1/N position sizing
- **Purpose:** Test if high insider activity stocks outperform

### Portfolio 2: Long-Short Factor (Top vs Bottom Deciles)
- **Long positions:** Top decile (highest insider buying)
- **Short positions:** Bottom decile (highest insider selling OR no activity)
- **Equal-weight:** 1/N within each leg
- **Purpose:** Isolate pure insider factor exposure, remove market beta

**Primary Analysis:** Long-short portfolio (Portfolio 2) is the main acceptance test.

**Decile Monotonicity:** Build all 10 decile portfolios to test:
- Do returns increase monotonically from bottom to top decile?
- Is the spread between top and bottom economically significant?

---

## 5. Phase 1 Acceptance Gates

Phase 1 uses **long-short factor portfolio** (top decile vs bottom decile) as the primary acceptance metric.

### Gate 1: Decile Monotonicity
**Requirement:** Long-short spread shows increasing returns from bottom to top deciles

**Test:**
- Build 10 equal-weighted decile portfolios
- Compute average monthly return for each decile (full sample)
- Check that top decile average > bottom decile average

**Threshold:**
- Top decile mean monthly return ≥ bottom decile mean monthly return
- Spread ≥ 0.5% per month (annualized: 6%+)

**Rationale:** If the Insider signal is economically meaningful, we expect stocks with high insider buying to outperform those with high insider selling.

### Gate 2: Long-Short Sharpe Ratio (Full Sample)
**Requirement:** Long-short Sharpe ≥ 0.30 (full sample 2015-2024)

**Test:**
- Build long-short portfolio (top decile long, bottom decile short)
- Compute annualized Sharpe ratio over full period
- Compare to noise threshold (0.30 minimum for factor signals)

**Threshold:**
- **Minimum:** Sharpe ≥ 0.30 (basic signal quality)
- **Target:** Sharpe ≥ 0.50 (strong signal)

**Rationale:** Sharpe < 0.30 suggests the signal is mostly noise. At 0.30+, the signal has potential economic value.

### Gate 3: Statistical Significance (t-statistic)
**Requirement:** Long-short mean return has t-stat ≥ 2.0 (p < 0.05)

**Test:**
- Compute monthly returns for long-short portfolio
- Calculate t-statistic: `t = mean(returns) / (std(returns) / sqrt(n_months))`
- Compare to threshold 2.0

**Threshold:**
- t-statistic ≥ 2.0 (two-sided test, 5% significance level)

**Rationale:** Statistical significance ensures the observed returns are unlikely to be random chance.

### Gate 4: Recent Regime Performance (2023-2024)
**Requirement:** Long-short mean return > 0 in recent period (2023-2024)

**Test:**
- Compute average monthly return for long-short portfolio in 2023-2024
- Check that mean > 0

**Threshold:**
- Mean monthly return > 0.0% (positive on average)
- Sharpe ≥ 0.10 preferred (not required)

**Rationale:** If the signal is "dead" in the most recent regime, optimization may be futile. We require at least positive average returns.

### Gate 5: In-Sample vs Out-of-Sample Consistency
**Requirement:** OOS Sharpe ≥ 0.20 (no catastrophic degradation)

**Test:**
- Compute Sharpe for in-sample period (2015-04-01 to 2022-12-31)
- Compute Sharpe for out-of-sample period (2023-01-01 to 2024-12-31)
- Check that OOS Sharpe ≥ 0.20

**Threshold:**
- OOS Sharpe ≥ 0.20 (minimum)
- OOS Sharpe within 50% of IS Sharpe (consistency check)

**Rationale:** Catastrophic OOS degradation suggests overfitting or regime change.

---

## 6. Acceptance Criteria Summary

| Gate | Metric | Threshold | Purpose |
|------|--------|-----------|---------|
| 1 | Decile Spread | Top - Bottom ≥ 0.5%/mo | Monotonicity |
| 2 | Full Sharpe | ≥ 0.30 (target 0.50) | Economic significance |
| 3 | t-statistic | ≥ 2.0 | Statistical significance |
| 4 | Recent Mean Return | > 0.0%/mo | Signal not dead |
| 5 | OOS Sharpe | ≥ 0.20 | No catastrophic degradation |

**GO Decision:** Pass **ALL 5 gates** → Proceed to Phase 2 optimization

**CONDITIONAL GO:** Pass 4/5 gates, with Gate 2 or Gate 5 marginal → Proceed with caution

**NO-GO:** Fail ≥2 gates, OR fail Gate 1 (monotonicity) → Do not optimize

---

## 7. Regime Definitions

Analyze performance across market regimes to understand signal behavior:

| Regime | Period | Description |
|--------|--------|-------------|
| COVID | 2020-01-01 to 2020-12-31 | Pandemic crash and recovery |
| Bear 2022 | 2021-01-01 to 2022-12-31 | Fed hiking cycle, growth selloff |
| Recent | 2023-01-01 to 2024-12-31 | Post-bear recovery, AI boom |

**Metrics per Regime:**
- Mean monthly return
- Annualized Sharpe ratio
- Number of months

**Expected Behavior:**
- Insider signal may perform better in volatile regimes (information advantage)
- May underperform in strong momentum regimes (2023-2024)
- Should remain positive on average across regimes

---

## 8. Diagnostic Outputs

### `results/INSIDER_PHASE1_BASELINE.md`

**Content:**
- Configuration summary (parameters, universe, dates)
- Long-only top quintile performance metrics
- Long-short factor portfolio performance metrics
- Decile return table (all 10 deciles)
- Regime split table (COVID, Bear 2022, Recent)
- Headline metrics for quick review

### `results/INSIDER_PHASE1_REPORT.md`

**Content:**
- Full diagnostic report with GO/NO-GO verdict
- Full-sample, in-sample, out-of-sample metrics
- Gate-by-gate evaluation with PASS/FAIL flags
- Regime performance analysis
- Decile monotonicity chart (text table)
- Recommendations for Phase 2 (if GO)

### `results/INSIDER_PHASE1_DECILES.csv`

**Content:**
- CSV table with decile-level returns
- Columns: `decile`, `mean_return`, `volatility`, `sharpe`, `num_months`
- Used for plotting and further analysis

---

## 9. Success Criteria (Phase 1 → Phase 2)

**Minimum Viable Signal (GO):**
- ✅ Decile monotonicity (top > bottom by ≥0.5%/mo)
- ✅ Full Sharpe ≥ 0.30
- ✅ t-statistic ≥ 2.0
- ✅ Recent regime mean return > 0
- ✅ OOS Sharpe ≥ 0.20

**Strong Signal (Ideal):**
- 🎯 Full Sharpe ≥ 0.50
- 🎯 OOS Sharpe ≥ 0.40
- 🎯 Decile spread ≥ 1.0%/mo
- 🎯 t-statistic ≥ 3.0
- 🎯 Positive returns in all 3 regimes

**NO-GO Signal:**
- ❌ Full Sharpe < 0.30
- ❌ Decile spread < 0.5%/mo OR non-monotonic
- ❌ t-statistic < 2.0
- ❌ Recent regime mean return < 0
- ❌ OOS Sharpe < 0.20

---

## 10. Known Risks and Mitigations

### Risk 1: Sparse Insider Data
**Issue:** Not all S&P 500 stocks have regular insider activity. Many stocks may have zero insider trades in a 90-day window.

**Mitigation:**
- Allow zero scores (neutral) for stocks with no insider activity
- Focus on cross-sectional ranking (stocks WITH activity rank relative to each other)
- Monitor percentage of universe with non-zero signals

### Risk 2: Filing Lag Bias
**Issue:** Insider trades must be filed within 2 business days (SEC Rule 10b5-1), but delays can occur.

**Mitigation:**
- Use `as_of_date` parameter in `get_insider_trades()` to filter by filing date, not trade date
- Assume 5-day average lag (conservative)
- Document any data issues found

### Risk 3: Regulatory Changes
**Issue:** Insider trading regulations changed in 2002 (Sarbanes-Oxley) and 2023 (10b5-1 amendments).

**Mitigation:**
- Test starts in 2015 (post-SOX, pre-2023 changes)
- Monitor regime performance for breaks in 2023+
- Document if recent regime shows degradation

### Risk 4: Market Microstructure
**Issue:** Insider buying may cluster in small-cap value stocks, creating unintended style tilts.

**Mitigation:**
- Analyze sector tilts in top decile
- Check market cap distribution
- Compare to S&P 500 index weights
- Document style exposures for ensemble design

---

## 11. Next Steps After Phase 1

### If GO (Pass All Gates):
1. **Phase 2.0:** Grid search over key parameters:
   - `lookback_days`: [30, 60, 90, 120, 180]
   - `min_transaction_value`: [5000, 10000, 25000, 50000]
   - `cluster_window`: [3, 5, 7, 10, 14]
   - `cluster_min_insiders`: [2, 3, 4, 5]

2. **Phase 2.1:** Optuna Bayesian optimization:
   - Continuous parameter tuning
   - Multi-objective optimization (Sharpe + max drawdown)
   - Walk-forward validation

3. **Phase 3:** Ensemble design:
   - Combine with Momentum v2
   - Test correlation between signals
   - Optimize position sizing and risk management

### If CONDITIONAL GO (4/5 Gates):
1. Investigate which gate failed and why
2. Consider alternative specifications:
   - Longer lookback periods
   - Different role weights
   - Alternative cluster definitions
3. Re-test with modifications before optimization

### If NO-GO (Fail ≥2 Gates):
1. Document failure mode in ERROR_PREVENTION_ARCHITECTURE.md
2. Archive results for reference
3. Consider alternative insider methodologies:
   - Focus on purchases only (sales have weaker signal)
   - Use only CEO/CFO trades
   - Combine with other fundamental factors

---

## 12. References

### Academic Literature

1. **Cohen, L., Malloy, C., & Pomorski, L. (2012).** "Decoding Inside Information". *Journal of Finance*, 67(3), 1009-1043.
   - Main methodology reference
   - Cluster detection approach
   - Role hierarchy weighting

2. **Seyhun, H. N. (1986).** "Insiders' Profits, Costs of Trading, and Market Efficiency". *Journal of Financial Economics*, 16(2), 189-212.
   - Early insider trading evidence
   - Transaction cost considerations

3. **Jeng, L. A., Metrick, A., & Zeckhauser, R. (2003).** "Estimating the Returns to Insider Trading: A Performance-Evaluation Perspective". *Review of Economics and Statistics*, 85(2), 453-471.
   - Dollar-weighting methodology
   - Purchases vs sales asymmetry

### Implementation References

- `signals/insider/institutional_insider.py` - Signal implementation
- `data/data_manager.py` - Insider data retrieval (`get_insider_trades()`)
- `core/universe_manager.py` - S&P 500 PIT universe construction
- `validation/simple_validation.py` - Performance metric calculations

---

**Document Status:** Phase 1 Specification Complete
**Last Updated:** 2025-11-21
**Next Review:** After Phase 1 diagnostic completion

---

## Appendix A: Data Layer Notes

### Current Implementation (Per-Ticker Queries)

The initial Insider Phase 1 baseline used per-ticker, per-rebalance database calls:
- **Method**: `DataManager.get_insider_trades(ticker, start, end, as_of)`
- **Query Pattern**: N tickers × M rebalances = ~58,000 individual database queries
- **Runtime**: ~2.5 hours for full Phase 1 diagnostic (116 rebalances, 500 stocks)
- **Bottleneck**: Database I/O (each query fetches insider data independently)

### New Implementation (Bulk Data Path)

Added `DataManager.get_insider_trades_bulk()` for scalable research runs:
- **Method**: `DataManager.get_insider_trades_bulk(tickers, start, end, as_of)`
- **Query Pattern**: 1 bulk query fetches all tickers at once
- **Expected Speedup**: 50-100x (from 2.5 hours → 1-3 minutes)
- **Returns**: Multi-indexed DataFrame `[(ticker, filingdate)]` for fast lookups

**Usage Pattern:**
```python
# At start of backtest, fetch all data once
bulk_data = dm.get_insider_trades_bulk(
    tickers=universe,
    start_date='2015-01-01',
    end_date='2024-12-31',
    as_of_date='2024-12-31'
)

# At each rebalance, lookup specific ticker's trades
for rebal_date in rebalance_dates:
    for ticker in universe:
        # Fast in-memory lookup (no DB query)
        ticker_trades = bulk_data.xs(ticker, level='ticker')
        # ... process trades ...
```

### Architecture Alignment (3-Tier Data Warehouse)

The bulk data path fits into the existing 3-tier architecture:

**Tier 1 (Raw Tables - Current):**
- `sharadar_insiders` table (345K+ transactions)
- `get_insider_trades_bulk()` = Tier 1 helper (direct table access)
- **Use case**: Research, backtesting, one-off analysis

**Tier 2 (Fact Tables - Future):**
- Planned: `fact_signals_panel` or `insider_monthly_signals`
- Pre-aggregated insider scores at monthly frequency
- **Use case**: Production signal serving, fast portfolio rebalancing

**Tier 3 (Application Layer - Future):**
- Portfolio management system
- Real-time signal updates
- **Use case**: Live trading, monitoring dashboards

**Current Status:** Tier 1 complete with bulk optimization. Tier 2/3 to be implemented if Insider signal becomes viable (currently NO-GO).

### Recommendation for Future Research

Any future Insider signal research (alternative specifications, different universes, etc.) should:
1. Use `get_insider_trades_bulk()` instead of per-ticker queries
2. Fetch data once at start of backtest, store in memory
3. Consider caching to Parquet files for repeated runs on same date range
4. If Insider becomes GO signal, build Tier 2 aggregated table for production

---

## Appendix B: Expected Data Schema

### Insider Trades Table (`sharadar_insiders`)

**Key Columns:**
- `ticker`: Stock ticker symbol
- `date`: Trade date (transaction date)
- `filingdate`: Date filed with SEC (use for point-in-time filtering)
- `transactioncode`: 'P' = purchase, 'S' = sale
- `transactionshares`: Number of shares traded
- `transactionpricepershare`: Price per share
- `transactionvalue`: Total dollar value (shares * price)
- `officertitle`: Insider's role/title (for role classification)

**Point-in-Time Filtering:**
Use `filingdate <= as_of_date` to avoid lookahead bias.

**Expected Volume:**
- ~345,000+ insider transactions in database
- ~50-100 transactions per S&P 500 stock per year
- Concentrated in earnings windows and corporate events
