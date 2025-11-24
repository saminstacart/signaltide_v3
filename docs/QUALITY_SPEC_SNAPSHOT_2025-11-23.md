# Quality Signal Mathematical Specification

**Version:** 1.0
**Date:** 2025-11-21
**Status:** Phase 0.2 - Implementation Specification
**Supersedes:** InstitutionalQuality v0 (time-series scaled, deprecated)

---

## 1. Motivation and Academic Basis

### Quality Minus Junk (QMJ) Factor

The quality premium is documented in academic literature as a robust, positive expected return anomaly:

**Primary Reference:**
- Asness, Frazzini, Pedersen (2018). "Quality Minus Junk." Review of Accounting Studies.
  - Quality = profitable, growing, safely financed, shareholder-friendly
  - Cross-sectional ranking at each rebalance date (CRITICAL)
  - Long high-quality, short low-quality (or long-only top decile)

**Supporting References:**
- Novy-Marx (2013). "The Other Side of Value." Journal of Financial Economics.
  - Gross profitability metric
- Piotroski (2000). "Value Investing: The Use of Historical Financial Statement Information."
  - F-Score methodology (binary signals)
- Sloan (1996). "Do Stock Prices Fully Reflect Information in Accruals?"
  - Accruals anomaly (earnings quality)

### Why v0 (InstitutionalQuality) Failed

**v0 Implementation Error:**
- Used time-series ranking: each stock vs its own 2-year history
- Captured quality momentum (is AAPL more quality than AAPL 1Y ago?)
- Did NOT capture cross-sectional quality premium (is AAPL more quality than MSFT today?)

**Result:**
- 1.74% annual return, Sharpe -0.157
- Methodology-misaligned, not parameter-broken
- Not testing academic QMJ at all

**v1 Correction:**
- Cross-sectional ranking: all stocks vs each other at each month-end
- Proper academic methodology
- Expected significant performance improvement

---

## 2. Mathematical Definition

### 2.1 Universe and Time Notation

**Universe:**
```
U_SP500_PIT(t) = {i : ticker i is S&P 500 member on date t according to dim_universe_membership}

U_EXT(t) = U_SP500_PIT(t) ∪ {recently removed constituents} ∪ {borderline R1000 names} [optional]
```

**Time Windows:**
```
t_rebal = {t_1, t_2, ..., t_N} : Set of monthly rebalance dates (month-ends via trading calendar)
```

**Point-in-Time Constraint:**
```
For all data used at time t:
  - Prices: As of t or earlier
  - Fundamentals: Filed ≤ t (with 33-day lag minimum for quarterly filings)
  - All data: Known and available on date t (no lookahead)
```

### 2.2 Raw Metrics Computation

For each date t ∈ t_rebal and ticker i ∈ U_SP500_PIT(t):

#### Profitability Metrics

**ROE (Return on Equity):**
```
ROE_i(t) = NetIncome_i(t_q) / Equity_i(t_q)

where:
  t_q = most recent quarterly report filed ≤ (t - 33 days)
  Sharadar column: 'roe' from sharadar_sf1 (dimension='ARQ')
  If missing: computed from 'netinc' / 'equity'
```

**ROA (Return on Assets):**
```
ROA_i(t) = NetIncome_i(t_q) / TotalAssets_i(t_q)

Sharadar column: 'roa' from sharadar_sf1 (dimension='ARQ')
If missing: computed from 'netinc' / 'assets'
```

**Gross Profitability / Assets:**
```
GPA_i(t) = GrossProfit_i(t_q) / TotalAssets_i(t_q)

Sharadar columns: 'gp' / 'assets' from sharadar_sf1
```

**Profitability Score:**
```
P_i(t) = mean(ROE_i(t), ROA_i(t), GPA_i(t)) if all available
       = mean(available metrics)               if some missing
       = 0                                     if all missing
```

#### Growth Metrics

**Revenue Growth (YoY):**
```
RevGrowth_i(t) = (Revenue_i(t_q) - Revenue_i(t_q-4)) / Revenue_i(t_q-4)

where:
  t_q = most recent quarter
  t_q-4 = same quarter 1 year ago (4 quarters back)
  Sharadar column: 'revenue' from sharadar_sf1
```

**Earnings Growth (YoY):**
```
NIGrowth_i(t) = (NetIncome_i(t_q) - NetIncome_i(t_q-4)) / |NetIncome_i(t_q-4)|

Sharadar column: 'netinc' from sharadar_sf1

Special handling:
  - If denominator changes sign: cap growth at [-2.0, 2.0]
  - If denominator = 0: use revenue growth instead
```

**Growth Score:**
```
G_i(t) = mean(RevGrowth_i(t), NIGrowth_i(t)) if both available
       = RevGrowth_i(t)                      if only revenue
       = 0                                   if neither
```

#### Safety Metrics

**Leverage (Debt/Equity):**
```
DE_i(t) = TotalDebt_i(t_q) / Equity_i(t_q)

Sharadar column: 'de' from sharadar_sf1

Safety contribution: -DE_i(t)  [inverted: low leverage = high safety]
```

**Earnings Stability (ROE Volatility):**
```
ROEVol_i(t) = std(ROE_i(t_q-7), ..., ROE_i(t_q))  [8 quarters = 2 years]

Safety contribution: -ROEVol_i(t)  [inverted: low vol = high safety]

Require: At least 8 quarters of ROE history
```

**Profitability Consistency:**
```
PosROE_i(t) = 1  if ROE_i(t) > 0
             = 0  otherwise

Binary indicator for positive profitability
```

**Safety Score:**
```
S_i(t) = mean(-DE_i(t), -ROEVol_i(t), PosROE_i(t)) if all available
       = mean(available metrics)                    if some missing
       = 0                                          if all missing
```

### 2.3 Cross-Sectional Standardization

**At each rebalance date t:**

For each component (Profitability, Growth, Safety):

```
# Compute across all stocks in universe
P_vec(t) = [P_1(t), P_2(t), ..., P_N(t)] for all i ∈ U_SP500_PIT(t)

# Winsorize to handle outliers
P_vec_wins(t) = winsorize(P_vec(t), lower=5th percentile, upper=95th percentile)

# Cross-sectional z-score
Z_P_i(t) = (P_i(t) - mean(P_vec_wins(t))) / std(P_vec_wins(t))

# Repeat for G and S
Z_G_i(t) = zscore_cross_section(G_i(t))
Z_S_i(t) = zscore_cross_section(S_i(t))
```

**Handling Edge Cases:**
- If std = 0 (all stocks identical): Z = 0 for all
- If < 20 stocks have valid data: skip this rebalance date
- Missing Z_P, Z_G, or Z_S for a stock: use 0 for that component

### 2.4 Composite Quality Score

**Weighted Combination:**
```
Q_i(t) = w_P * Z_P_i(t) + w_G * Z_G_i(t) + w_S * Z_S_i(t)

Default weights (QMJ style):
  w_P = 0.4  (profitability emphasis)
  w_G = 0.3  (growth)
  w_S = 0.3  (safety)

Constraint: w_P + w_G + w_S = 1.0
```

**Parameterization for Variants:**
- **QualityProfitability**: (0.8, 0.1, 0.1) - heavy profitability
- **QualityBalanced**: (0.4, 0.3, 0.3) - default QMJ
- **QualitySafety**: (0.2, 0.2, 0.6) - defensive quality

### 2.5 Ranking and Signal Generation

**Cross-Sectional Percentile Rank:**
```
R_i(t) = percentile_rank(Q_i(t) over all i ∈ U_SP500_PIT(t))

Range: R_i(t) ∈ [0, 1]
  0 = lowest quality stock
  1 = highest quality stock
```

**Convert to Signal Range [-1, 1]:**
```
Signal_i(t) = 2 * R_i(t) - 1

Range: Signal_i(t) ∈ [-1, 1]
  -1 = lowest quality (short candidate)
  +1 = highest quality (long candidate)
   0 = median quality (neutral)
```

**Quintile Signals (Optional, for cleaner interpretation):**
```
if R_i(t) ∈ [0.0, 0.2):   Signal_i(t) = -1.0  (quintile 1, lowest quality)
if R_i(t) ∈ [0.2, 0.4):   Signal_i(t) = -0.5  (quintile 2)
if R_i(t) ∈ [0.4, 0.6):   Signal_i(t) =  0.0  (quintile 3, neutral)
if R_i(t) ∈ [0.6, 0.8):   Signal_i(t) = +0.5  (quintile 4)
if R_i(t) ∈ [0.8, 1.0]:   Signal_i(t) = +1.0  (quintile 5, highest quality)
```

### 2.6 Temporal Extension

**Signal Persistence (Forward-Fill):**
```
For each date d between rebalance dates (t_k, t_k+1):
  Signal_i(d) = Signal_i(t_k)  for all d ∈ [t_k, t_k+1)

Monthly rebalancing:
  - Compute signals at month-ends only
  - Hold signals constant for entire following month
  - Zero transaction cost within month (no rebalancing)
```

**Missing Data Handling:**
```
If ticker i ∉ U_SP500_PIT(t) (not in universe):
  Signal_i(t) = 0

If ticker i has insufficient fundamental data:
  Signal_i(t) = 0 (neutral, do not trade)

Do NOT forward-fill from previous rebalance if no current data
```

---

## 3. Implementation Requirements

### 3.1 Data Dependencies

**Required Tables:**
- `sharadar_sf1`: Fundamental data (dimension='ARQ' for quarterly, as-reported)
- `dim_universe_membership`: Point-in-time S&P 500 membership
- `dim_trading_calendar`: NYSE trading calendar for month-end dates

**Required Columns (sharadar_sf1):**
- `roe`, `roa`, `gp`, `assets`: Profitability
- `revenue`, `netinc`: Growth
- `de`, `equity`: Safety (leverage)
- `datekey`, `ticker`, `dimension`, `lastupdated`: Metadata

**Point-in-Time Enforcement:**
```python
# When querying fundamentals on date t:
WHERE lastupdated <= t AND datekey <= (t - 33 days)

# 33-day filing lag assumption:
#   Quarterly reports filed within ~45 days of quarter end
#   Use 33 days as conservative floor
#   This is CRITICAL for no lookahead bias
```

### 3.2 Code Structure

**Class Definition:**
```python
class CrossSectionalQuality(InstitutionalSignal):
    """
    Quality Minus Junk signal with proper cross-sectional ranking.

    Academic basis: Asness-Frazzini-Pedersen (2018)

    Methodology:
    1. Compute P/G/S scores for all stocks at each month-end
    2. Cross-sectional z-score standardization
    3. Weighted composite: Q = 0.4*P + 0.3*G + 0.3*S
    4. Percentile rank to [-1, 1] signal range
    5. Forward-fill to daily

    Parameters:
        w_profitability: Weight on profitability (default: 0.4)
        w_growth: Weight on growth (default: 0.3)
        w_safety: Weight on safety (default: 0.3)
        quintiles: Use quintile discretization (default: True)
        sector_neutral: Rank within sectors (default: False)
    """

    def generate_signals_cross_sectional(
        self,
        universe: List[str],
        rebalance_dates: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Generate quality signals for entire universe at each rebalance date.

        Args:
            universe: List of tickers (e.g., S&P 500 on as_of_date)
            rebalance_dates: Monthly rebalance dates

        Returns:
            DataFrame with columns = tickers, index = all trading days,
            values = signals in [-1, 1]
        """
        # Implementation follows spec above
        pass
```

**API Compliance:**
- Inherits from `InstitutionalSignal` base class
- Uses monthly rebalancing (rebalance_frequency='monthly')
- Returns signals in [-1, 1] range
- Logs all major operations for debugging
- Implements `get_parameter_space()` for optimization

### 3.3 Validation Requirements

**Before Deployment:**

1. **Data Coverage Check:**
   - Must have ≥ 90% of S&P 500 with non-null quality scores
   - Flag tickers with missing fundamentals
   - Document coverage gaps by sector

2. **Monotonicity Test:**
   - Top decile vs bottom decile spread must be positive
   - Test on multiple 3-year rolling windows
   - Require: ≥ 70% of windows show positive spread

3. **Sector Neutrality Check:**
   - Compute sector weights in top quintile vs S&P 500 index weights
   - Flag if any single sector: |weight_top_quintile - weight_index| > 20%
   - Document intentional sector tilts (e.g., Quality naturally underweights Utilities)

4. **Temporal Stability:**
   - Check signal autocorrelation (should be high due to monthly rebalancing)
   - Verify: corr(Signal_i(t), Signal_i(t+1 month)) > 0.7 for most stocks
   - Low correlation suggests noisy or broken signal

5. **Out-of-Sample Sanity:**
   - Even with no optimization, long-only top decile should:
     - Not catastrophically underperform SPY (max tracking error < 20% annual)
     - Have positive Sharpe ratio over 10-year period
     - Avoid extended drawdowns > 40%

---

## 4. Parameter Space for Optimization

### Tunable Parameters

```python
{
    # Component weights (must sum to 1.0)
    'w_profitability': ('float', 0.2, 0.6),
    'w_growth': ('float', 0.1, 0.5),
    'w_safety': ('float', 0.1, 0.5),

    # Winsorization bounds
    'winsorize_lower': ('categorical', [1, 5, 10]),  # percentile
    'winsorize_upper': ('categorical', [90, 95, 99]),

    # Signal discretization
    'quintiles': ('categorical', [True, False]),

    # Long/short thresholds (if not using quintiles)
    'long_threshold': ('float', 0.6, 0.9),   # percentile for long positions
    'short_threshold': ('float', 0.1, 0.4),  # percentile for short positions

    # Sector neutrality
    'sector_neutral': ('categorical', [True, False]),

    # Lookback for earnings stability (quarters)
    'roe_vol_window': ('int', 4, 12),  # 1-3 years
}
```

### Fixed Parameters (Not Optimized)

- Rebalancing frequency: Monthly (end-of-month)
- Ranking method: Cross-sectional (by definition)
- Point-in-time lag: 33 days (conservative filing lag)
- Universe: S&P 500 point-in-time membership

### Constraints

```python
# Weight constraint
w_profitability + w_growth + w_safety == 1.0

# Threshold constraint (if not using quintiles)
short_threshold < 0.5 < long_threshold

# History requirement
roe_vol_window <= min_quarters_available - 1
```

---

## 5. Expected Performance Characteristics

### Academic Benchmarks (Asness et al. 2018)

From original QMJ paper (US Large Caps, 1957-2012):
- **Sharpe Ratio:** ~0.4-0.6 (depending on sample period)
- **Information Ratio vs Market:** ~0.3-0.4
- **Annualized Return (long-short):** ~5-8%
- **Correlation with Momentum:** ~0.2-0.3 (low, diversifying)
- **Correlation with Value:** ~0.1-0.2 (low)

### Our Targets (S&P 500, 2015-2024)

**Minimum Viable (v1 Acceptance Criteria):**
- Information Ratio vs SPY: > 0.3
- Sharpe Ratio: > 0.5
- Positive alpha: p < 0.05
- Max drawdown: < 30%
- Turnover: < 40% monthly (manageable costs)

**Institutional Quality (Phase 1 Goal):**
- Information Ratio vs SPY: > 0.5
- Sharpe Ratio: > 1.0
- Positive alpha: p < 0.01
- Max drawdown: < 25%
- Win rate (1Y rolling): > 60%

### Known Regime Dependencies

Quality tends to:
- **Outperform in recessions:** Stable, profitable companies hold up better
- **Underperform in speculative booms:** 2020-2021 growth mania (tech > quality)
- **Outperform in rising rate environments:** 2022 bear (quality > growth)
- **Correlate with "flight to quality":** Pairs well with defensive strategies

**Implication:** Phase 4 regime analysis is critical. Don't reject signal if it underperforms in 2020-2021 tech boom - that's expected behavior.

---

## 6. Deprecation Notice for v0

### InstitutionalQuality (Time-Series Scaled) - DEPRECATED

**Status:** Replaced by CrossSectionalQuality v1
**Last Used:** Phase 0.1 diagnostics (2025-11-21)
**Reason for Deprecation:** Methodology mismatch with academic QMJ

**v0 Issues:**
1. Time-series ranking (stock vs own history)
2. Does not capture cross-sectional quality premium
3. Empirically failed: 1.74% annual return, Sharpe -0.157
4. Not suitable for production or further optimization

**Migration Path:**
- All future Quality backtests use CrossSectionalQuality v1
- v0 results archived for reference only
- No further development or parameter tuning on v0

**v0 Code Location:** `signals/quality/institutional_quality.py` (archived, not deleted)
**v0 Results:** `results/baseline_quality_v0_DEPRECATED.json`

---

## 7. Variants and Extensions

### 7.1 QualityProfitability (Novy-Marx Style)

**Specification:**
```
w_profitability = 0.8
w_growth = 0.1
w_safety = 0.1

Profitability emphasis:
- ROE, ROA, GPA (as above)
- Add: ROIC = NetIncome / (Debt + Equity)
- Add: CF/Assets = OperatingCashFlow / TotalAssets

Academic basis: Novy-Marx (2013) "gross profitability premium"
```

**Use Case:** Pure profitability exposure, minimal growth/leverage considerations

### 7.2 QualityAccruals (Sloan Style)

**Specification:**
```
Accruals-based quality metrics:

AccrualsRatio_i(t) = (NetIncome_i - OperatingCashFlow_i) / TotalAssets_i

Lower accruals = higher quality (less earnings manipulation)

Signal: -zscore(AccrualsRatio)  [inverted]

Academic basis: Sloan (1996)
```

**Use Case:** Earnings quality / manipulation detection

### 7.3 QualityPiotroski (F-Score)

**Specification:**
```
9-point binary scorecard:

Profitability (4 points):
  1. ROA > 0
  2. CFO > 0
  3. ROA increased vs last year
  4. CFO > Net Income (quality of earnings)

Leverage/Liquidity (3 points):
  5. Decreased leverage vs last year
  6. Increased current ratio
  7. No new equity issuance

Operating Efficiency (2 points):
  8. Increased gross margin
  9. Increased asset turnover

F-Score = sum(above) ∈ [0, 9]

Signal: F-Score / 9  (normalized to [0, 1])

Academic basis: Piotroski (2000)
```

**Use Case:** Value-quality hybrid, binary signal simplicity

### 7.4 QualitySectorNeutral

**Specification:**
```
Same as CrossSectionalQuality, but:

Within each GICS sector s:
  1. Compute Q_i(t) for all i in sector s
  2. Rank within sector: R_i_s(t) = percentile_rank_within_sector(Q_i(t))
  3. Convert to signal: Signal_i(t) = 2 * R_i_s(t) - 1

Eliminates sector tilts, pure stock selection within sectors
```

**Use Case:** Sector-neutral long-short, avoid unintended sector bets

---

## 8. Testing and Validation Protocol

### Phase 0.2 Acceptance Tests

Before promoting CrossSectionalQuality v1 to Phase 1 baseline:

1. **Code Review:**
   - [ ] Matches mathematical spec exactly
   - [ ] Point-in-time constraints enforced (33-day lag)
   - [ ] Cross-sectional ranking at each rebalance (not time-series)
   - [ ] Type hints and docstrings complete
   - [ ] Unit tests for edge cases (missing data, single stock, etc.)

2. **Data Validation:**
   - [ ] Coverage ≥ 90% of S&P 500
   - [ ] No lookahead bias (verify lastupdated filtering)
   - [ ] Fundamentals align with known values for spot checks

3. **Signal Validation:**
   - [ ] Decile monotonicity: top > bottom in ≥ 70% of 3Y windows
   - [ ] Sector tilts documented and explainable
   - [ ] Signal autocorrelation > 0.7 (monthly persistence)
   - [ ] No extreme outliers (>99th percentile signals investigated)

4. **Performance Validation:**
   - [ ] Long-only top decile: positive Sharpe over 2015-2024
   - [ ] Long-short spread: positive in most regimes
   - [ ] IR vs SPY > 0.3 (minimum threshold)
   - [ ] Max drawdown < 35% (sanity check)

5. **Diagnostic Report:**
   - [ ] `results/quality_diagnostics_v1.md` generated
   - [ ] Side-by-side comparison vs v0 (deprecated)
   - [ ] Clear recommendation: proceed to Phase 1 or iterate

### Ongoing Monitoring (Post-Deployment)

- Monthly signal health checks (rolling IC, Sharpe)
- Quarterly regime analysis (bull/bear/sideways performance)
- Annual reoptimization review (parameter stability)
- Continuous data quality monitoring (missing fundamentals alerts)

---

## 9. Open Questions and Future Research

### Near-Term (Phase 0.2-1)

1. **ROE/ROA Availability:**
   - Sharadar may not have pre-computed ROE/ROA columns
   - Confirm: compute from raw fields (netinc/equity, netinc/assets)
   - Document which fields are available vs need computation

2. **Sector Neutral vs Sector Tilts:**
   - Academic QMJ allows sector tilts (e.g., overweight Tech in 2020s)
   - Decide: sector-neutral by default, or allow tilts with monitoring?

3. **Earnings Quality Metrics:**
   - Sloan accruals require cash flow statement data
   - Verify: do we have `ncfo` (net cash from operations) in SF1?

### Medium-Term (Phase 3-5)

4. **Quality Factor Timing:**
   - Can we detect when Quality will outperform (regime switching)?
   - VIX, credit spreads, macro indicators as Quality timing signals?

5. **Quality + Momentum Interaction:**
   - Academic: Quality and Momentum are ~0.2-0.3 correlated
   - Ensemble: How to optimally combine (equal weight, risk parity, adaptive)?

6. **Global Quality:**
   - Extend beyond US S&P 500 to international markets?
   - Regional quality definitions (Europe, Asia, EM)

### Long-Term (Phase 6+)

7. **Machine Learning Quality:**
   - Can ML improve on hand-crafted Quality metrics?
   - Gradient boosting on fundamentals → quality proxy?
   - Risk: overfitting, interpretability loss

8. **ESG-Adjusted Quality:**
   - Incorporate ESG scores into safety/governance component?
   - Academic: mixed evidence on ESG alpha

---

## 10. References and Further Reading

### Academic Papers (Must Read)

1. **Asness, Frazzini, Pedersen (2018)**
   - "Quality Minus Junk"
   - Review of Accounting Studies
   - URL: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2312432

2. **Novy-Marx (2013)**
   - "The Other Side of Value: The Gross Profitability Premium"
   - Journal of Financial Economics
   - URL: https://rnm.simon.rochester.edu/research/OSV.pdf

3. **Piotroski (2000)**
   - "Value Investing: The Use of Historical Financial Statement Information"
   - Journal of Accounting Research
   - URL: https://www.jstor.org/stable/2672906

4. **Sloan (1996)**
   - "Do Stock Prices Fully Reflect Information in Accruals?"
   - The Accounting Review
   - Classic accruals anomaly paper

### Practitioner Resources

5. **AQR Capital Management**
   - "Quality: The Other Dimension of Value"
   - White paper explaining QMJ in practice

6. **Alpha Architect**
   - "The Quantitative Value Investing Philosophy"
   - Blog series on quality metrics implementation

### Data Documentation

7. **Sharadar Fundamentals (SF1) Data Dictionary**
   - Column definitions, dimensions (ARQ/MRQ/ART/MRT)
   - Filing lag assumptions
   - URL: https://data.nasdaq.com/databases/SF1/documentation

---

**End of Specification**
**Version:** 1.0
**Status:** Ready for Implementation
**Next Step:** Implement `CrossSectionalQuality` class following this spec
