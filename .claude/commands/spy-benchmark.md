Execute comprehensive SPY benchmark analysis (CURRENT_STATE.md Phase 1):

## ⚠️ Before Starting (CLAUDE.md Protocol)

**Common Pitfalls to Avoid:**
- **Pitfall #2: Lookahead Bias**
  Wrong: `df[df.date <= end_date]`
  Right: `df[(df.date <= as_of_date) & (df.date_known <= as_of_date)]`

- **Pitfall #4: Ignoring Transaction Costs**
  Always include realistic costs (~5 bps default; stress test at 10-20 bps) in all return calculations

- **Pitfall #6: File Reading**
  NEVER use `limit` parameter when reading CURRENT_STATE.md (744 lines)

**See** `.claude/CLAUDE.md` → "Common Claude Pitfalls"

## Prerequisites
1. Read complete `CURRENT_STATE.md` (NO limit parameter!) focusing on lines 177-214 for Phase 1 requirements
2. Verify backtest results exist in `results/institutional_backtest_report.md`
3. Load SPY price data for comparison period (2020-01-01 to 2024-12-31)

## Required Calculations

### 1. Information Ratio
- Formula: (Strategy Return - SPY Return) / Tracking Error
- Target: > 0.5 (good), > 1.0 (excellent), > 2.0 (world-class)
- Calculate tracking error (std of excess returns * sqrt(252))
- Report annualized

### 2. Alpha/Beta Decomposition
- Run regression: Strategy Returns = α + β*SPY Returns + ε
- Calculate: alpha (annualized), beta, R², p-value
- Test statistical significance (p < 0.05)
- Interpret:
  - Alpha > 0 and significant → genuine value add
  - Beta ≈ 1.0 → market-neutral (not levered SPY)
  - Beta < 1.0 → defensive
  - Beta > 1.0 → levered exposure

### 3. Risk-Adjusted Performance
Compare strategy vs SPY:
- **Sharpe Ratio**: (Return - RiskFree) / Volatility
- **Sortino Ratio**: (Return - RiskFree) / Downside Deviation
- **Calmar Ratio**: Annual Return / Max Drawdown
- Use risk-free rate = 2% (from config)

### 4. Drawdown Comparison
- Strategy max drawdown (should be NEGATIVE, e.g., -28.72%)
- SPY max drawdown (e.g., -34.10%)
- **Remember**: Less negative = BETTER (see ERROR_PREVENTION_ARCHITECTURE.md)
- Calculate drawdown duration (peak to trough)
- Calculate recovery time (trough to new peak)

### 5. Regime-Specific Performance
Define regimes:
- **Bull Market**: SPY > 200-day MA
- **Bear Market**: SPY < 200-day MA
- **High Volatility**: 21-day rolling vol > 20% annualized

For each regime:
- Strategy return (annualized)
- SPY return (annualized)
- Outperformance
- Strategy Sharpe
- Days in regime

### 6. Consistency Analysis
Rolling 1-year windows:
- Calculate strategy vs SPY returns for each 252-day window
- Count wins (strategy > SPY)
- Calculate win rate
- Target: > 60% (excellent), > 50% (good)

### 7. Calendar Year Performance
For each year (2020, 2021, 2022, 2023, 2024):
- Strategy total return
- SPY total return
- Outperformance
- Strategy Sharpe
- SPY Sharpe

## Statistical Significance Tests

1. **T-test for alpha**: Is alpha significantly different from zero?
2. **T-test for excess returns**: Is mean(strategy - SPY) > 0?
3. **Permutation test**: Randomly permute strategy returns, compare

## Output Format

Generate comprehensive markdown report: `results/spy_benchmark_report.md`

```markdown
# SPY Benchmark Analysis Report
**Generated:** YYYY-MM-DD HH:MM:SS
**Period:** 2020-01-01 to 2024-12-31
**Strategy:** [Signal Name or Composite]

## Executive Summary
[2-3 sentences: Do we beat SPY? How confident are we?]

## Key Metrics

| Metric | Strategy | SPY | Difference |
|--------|----------|-----|------------|
| Total Return | X% | Y% | Z% |
| Annual Return | X% | Y% | Z% |
| Sharpe Ratio | X.XX | Y.YY | Z.ZZ |
| Sortino Ratio | X.XX | Y.YY | Z.ZZ |
| Calmar Ratio | X.XX | Y.YY | Z.ZZ |
| Max Drawdown | -X% | -Y% | Better by Z% |
| Information Ratio | X.XX | - | - |

## Alpha/Beta Analysis

**Regression: Strategy = α + β*SPY + ε**

- **Alpha**: X.XX% annualized (p = 0.XXX) [Significant/Not Significant]
- **Beta**: X.XX (95% CI: [X.XX, Y.YY])
- **R²**: X.XX (XX% of variance explained by SPY)

**Interpretation:**
[Is alpha positive and significant? Is beta near 1.0? What does this mean?]

## Risk Analysis

### Drawdown Comparison
- **Strategy Max DD**: -X.XX% (Date: YYYY-MM-DD)
- **SPY Max DD**: -Y.YY% (Date: YYYY-MM-DD)
- **Advantage**: Strategy has Z.ZZ% better drawdown control
- **Duration**: X days from peak to trough
- **Recovery**: Y days from trough to new peak

### Regime-Specific Performance

#### Bull Market (SPY > 200 MA)
- Days: XXX
- Strategy Return: X.X%
- SPY Return: Y.Y%
- Outperformance: Z.Z%
- Strategy Sharpe: X.XX

#### Bear Market (SPY < 200 MA)
- Days: XXX
- Strategy Return: X.X%
- SPY Return: Y.Y%
- Outperformance: Z.Z%
- Strategy Sharpe: X.XX

#### High Volatility (Vol > 20%)
- Days: XXX
- Strategy Return: X.X%
- SPY Return: Y.Y%
- Outperformance: Z.Z%
- Strategy Sharpe: X.XX

## Consistency Analysis

### Rolling 1-Year Windows
- Total windows: XXX
- Strategy wins: XXX (XX%)
- SPY wins: XXX (XX%)
- **Win Rate**: XX% [Excellent > 60%, Good > 50%]

### Calendar Year Performance

| Year | Strategy | SPY | Outperformance |
|------|----------|-----|----------------|
| 2020 | XX% | YY% | ZZ% |
| 2021 | XX% | YY% | ZZ% |
| 2022 | XX% | YY% | ZZ% |
| 2023 | XX% | YY% | ZZ% |
| 2024 | XX% | YY% | ZZ% |

## Statistical Significance

- **Alpha t-statistic**: X.XX (p = 0.XXX)
- **Excess return t-test**: p = 0.XXX
- **Monte Carlo permutation test**: p = 0.XXX

## Go/No-Go Assessment

### Minimum Viable Product Criteria (from CURRENT_STATE.md)
- [ ] Information Ratio > 0.5: **[PASS/FAIL]** (Actual: X.XX)
- [ ] Positive alpha (p < 0.05): **[PASS/FAIL]** (p = 0.XXX)
- [ ] Max drawdown < 25%: **[PASS/FAIL]** (Actual: -XX%)
- [ ] Win 50%+ of 1-year periods: **[PASS/FAIL]** (Actual: XX%)
- [ ] No data leakage: **[PASS/FAIL]** (Verified)

**Score: X/5 checks passed**

### Stretch Goals (Institutional Quality)
- [ ] Information Ratio > 1.0: **[PASS/FAIL]** (Actual: X.XX)
- [ ] Alpha > 3% annualized: **[PASS/FAIL]** (Actual: X.X%)
- [ ] Max drawdown < 20%: **[PASS/FAIL]** (Actual: -XX%)
- [ ] Win 70%+ of 1-year periods: **[PASS/FAIL]** (Actual: XX%)

## Recommendation

**[DEPLOY / DON'T DEPLOY / NEEDS IMPROVEMENT]**

**Rationale:**
[Based on the metrics above, explain the recommendation]

## Next Steps
1. [Action item 1 based on results]
2. [Action item 2]
3. [Action item 3]

---
**Report generated by SPY Benchmark Analysis**
**See CURRENT_STATE.md Phase 1 for complete methodology**
```

## Post-Generation Actions

1. Save report to `results/spy_benchmark_report.md`
2. Update `CURRENT_STATE.md` with results summary
3. If all MVP criteria passed:
   - Mark Phase 1 as ✅ COMPLETE in CURRENT_STATE.md
   - Proceed to Phase 2 (Data Integrity Verification)
4. If any MVP criteria failed:
   - Document issues in ERROR_PREVENTION_ARCHITECTURE.md
   - Suggest fixes based on which criteria failed

## Important Notes

- **Always read complete CURRENT_STATE.md first** (no limit parameter!)
- **Check drawdown logic carefully** - less negative is better!
- **Cite statistical significance** - p-values for all tests
- **Be honest about results** - don't massage numbers
- **Reference ERROR_PREVENTION_ARCHITECTURE.md** if issues found
