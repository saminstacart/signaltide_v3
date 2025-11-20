Run comprehensive data integrity checks (CURRENT_STATE.md Phase 2):

## ⚠️ Before Starting (CLAUDE.md Protocol)

**Integrity Quiz (Must Answer):**
Before claiming data integrity verified, answer:
1. Can you trace every data point back to its source?
2. Have you tested with delisted stocks (survivorship bias)?
3. Are all fundamentals lagged by 45-60 days?
4. Did you verify with historical date simulation?

**Common Pitfalls:**
- **Pitfall #2: Lookahead Bias** - Check ALL `as_of` parameters
- **Pitfall #6: File Reading** - Read COMPLETE CURRENT_STATE.md (744 lines)

**See** `.claude/CLAUDE.md` → "Final Integrity Check"

## Prerequisites
1. Read complete `CURRENT_STATE.md` (NO limit parameter!) focusing on lines 218-244 for Phase 2 requirements
2. Verify database connection: `echo $SIGNALTIDE_DB_PATH`
3. Access to Sharadar database tables (prices, fundamentals, insiders, tickers)

## Phase 2 Requirements (from CURRENT_STATE.md)

### 1. Point-in-Time Universe Construction

**Objective:** Verify we only use stocks that existed at signal date

**Steps:**
```python
# For each backtest date:
# 1. Query tickers that were active on that date
# 2. Exclude tickers with delisted_date < backtest_date
# 3. Verify our universe matches point-in-time universe
```

**Checks:**
- [ ] No stocks used before their first trading date
- [ ] No stocks used after their delisting date
- [ ] Universe changes appropriately over time
- [ ] Delisted stocks properly included until delisting

**Query Example:**
```sql
SELECT ticker, name, firstpricedate, lastpricedate, delisted
FROM sharadar_tickers
WHERE delisted = 1
  AND lastpricedate >= '2020-01-01'
  AND lastpricedate <= '2024-12-31'
ORDER BY lastpricedate;
```

### 2. Fundamental Data Timing

**Objective:** Verify 45-60 day filing lag respected

**Rule:** Use fundamentals from 2 months ago (safe lag)

**Checks:**
- [ ] All `get_fundamentals()` calls have `as_of` parameter
- [ ] Fundamentals not used until filing_date + 45 days minimum
- [ ] Quality signal specifically checked (was missing `as_of` - now fixed)
- [ ] Test with historical dates to verify point-in-time access

**Files to Check:**
- `signals/quality/institutional_quality.py` line 101 (verified fixed)
- Any other signals using fundamentals
- Portfolio construction code

**Test:**
```python
# Simulate backtest on date 2020-03-15
# Verify only fundamentals with filingdate < 2020-02-01 are used
# (45-60 day lag)
```

### 3. Delisted Stocks Analysis

**Objective:** Verify survivorship bias eliminated

**Critical Check:** Are delisted stocks included in backtest?

**Steps:**
1. Query all stocks delisted 2020-2024
2. Check if they appear in backtest universe
3. Verify final losses captured (e.g., WISH $20 → $1)
4. Confirm delisting date handling is correct

**High-Profile Delistings to Check:**
- WISH (ContextLogic) - Delisted 2023
- UBER (check if still active)
- Any stocks that went to $0

**Query:**
```sql
SELECT ticker, name, lastpricedate, lastupdated
FROM sharadar_tickers
WHERE delisted = 1
  AND lastpricedate >= '2020-01-01'
  AND lastpricedate <= '2024-12-31'
ORDER BY lastpricedate DESC
LIMIT 50;
```

**Verification:**
```python
# For each delisted stock:
# 1. Check if it's in our universe during its active period
# 2. Verify we have price data up to delisting date
# 3. Confirm we don't use it after delisting
# 4. Calculate P&L if we held through delisting
```

### 4. Lookahead Bias Validation

**Objective:** For each signal at date T, verify all data < T

**Systematic Checks:**
- [ ] Price data: Only use data up to current timestamp
- [ ] Fundamental data: Respect filing lag
- [ ] Insider data: Only use transactions before signal date
- [ ] Corporate actions: Splits/dividends properly backdated

**Test Cases:**
```python
# Test 1: Check momentum signal on 2020-01-15
# Verify only uses prices up to 2020-01-15

# Test 2: Check quality signal on 2020-03-31
# Verify only uses fundamentals filed before 2020-02-15

# Test 3: Check insider signal on 2020-06-30
# Verify only uses transactions before 2020-06-30
```

**Validation Function:**
```python
def validate_no_lookahead(signal, data, date):
    """
    For given signal and date, verify no future data used.

    Returns: True if valid, False if lookahead detected
    """
    # Generate signal for date T
    # Check that signal[T] only uses data < T
    # Use institutional_base.validate_no_lookahead() method
```

### 5. Corporate Actions Verification

**Objective:** Verify splits/dividends handled correctly

**Checks:**
- [ ] Stock splits properly adjusted in price history
- [ ] Dividends not creating artificial signals
- [ ] Reverse splits handled correctly
- [ ] Spin-offs tracked appropriately

**Test:**
- AAPL: 4-for-1 split on 2020-08-31
- TSLA: 5-for-1 split on 2020-08-31
- Verify prices adjusted properly

## Output Format

Generate data integrity report: `results/data_integrity_report.md`

```markdown
# Data Integrity Verification Report
**Generated:** YYYY-MM-DD HH:MM:SS
**Backtest Period:** 2020-01-01 to 2024-12-31
**Phase:** 2 (Data Integrity Verification)

## Executive Summary
[Pass/Fail overall + key findings]

## 1. Point-in-Time Universe

### Universe Statistics
- Total unique tickers in backtest: XXX
- Delisted tickers included: XXX (XX%)
- Active tickers: XXX (XX%)
- Average universe size per date: XXX stocks

### Verification Results
- [x] No stocks before first trading date
- [x] No stocks after delisting date
- [x] Universe evolves correctly
- [x] Delisted stocks properly included

### Issues Found
[List any issues or "None - All checks passed"]

## 2. Fundamental Data Timing

### Filing Lag Analysis
- Average filing lag: XX days
- Minimum lag used: 45 days ✓
- Maximum safe lag: 60 days

### Code Verification
- [x] Quality signal has `as_of` parameter (line 101)
- [x] All fundamentals calls have proper lag
- [x] No signals use same-day fundamentals

### Test Results
**Historical Date Test (2020-03-15):**
- Latest fundamental used: 2020-01-31 ✓ (45-day lag)
- Filing date: 2020-02-01
- Safe lag confirmed: YES

### Issues Found
[List any issues or "None - All checks passed"]

## 3. Delisted Stocks

### Delisting Statistics
- Total delistings 2020-2024: XX stocks
- Delistings in our universe: XX stocks
- Inclusion rate: XX%

### Verification Results
| Ticker | Delisted Date | In Universe? | Final Price | Loss Captured? |
|--------|---------------|--------------|-------------|----------------|
| WISH | 2023-01-XX | YES | $0.XX | YES |
| ... | ... | ... | ... | ... |

### High-Profile Cases
**WISH (ContextLogic):**
- IPO Price: $XX
- Peak Price: $XX (Date: YYYY-MM-DD)
- Delisting Price: $XX
- Total Loss: -XX%
- **Captured in backtest:** YES/NO

### Issues Found
[List any issues or "None - All checks passed"]

## 4. Lookahead Bias Testing

### Systematic Checks

**Price Data:**
- [x] Momentum signal: No lookahead (validated)
- [x] All signals use .shift() or rolling() correctly
- [x] No future price data in signal generation

**Fundamental Data:**
- [x] Quality signal: 45-day lag enforced
- [x] `as_of` parameter used correctly
- [x] Historical test passed

**Insider Data:**
- [x] Insider signal: Only past transactions
- [x] No future-dated transactions used
- [x] Proper temporal ordering

### Test Case Results

**Test 1: Momentum Signal (2020-01-15)**
- Data used: 2019-01-15 to 2020-01-15 ✓
- No future data detected ✓
- **PASS**

**Test 2: Quality Signal (2020-03-31)**
- Fundamentals used: Filed before 2020-02-15 ✓
- Safe 45-day lag verified ✓
- **PASS**

**Test 3: Insider Signal (2020-06-30)**
- Transactions used: Before 2020-06-30 ✓
- No future transactions ✓
- **PASS**

### Issues Found
[List any issues or "None - All checks passed"]

## 5. Corporate Actions

### Splits Verified
| Ticker | Split Date | Ratio | Verified? |
|--------|-----------|-------|-----------|
| AAPL | 2020-08-31 | 4:1 | ✓ |
| TSLA | 2020-08-31 | 5:1 | ✓ |

### Verification Method
- Compare pre-split and post-split prices
- Verify continuity in returns
- Check for artificial momentum signals

### Issues Found
[List any issues or "None - All checks passed"]

## Overall Assessment

### All Checks Status
- [ ] Point-in-Time Universe: **PASS/FAIL**
- [ ] Fundamental Timing: **PASS/FAIL**
- [ ] Delisted Stocks: **PASS/FAIL**
- [ ] Lookahead Bias: **PASS/FAIL**
- [ ] Corporate Actions: **PASS/FAIL**

**Final Score: X/5 checks passed**

### Certification

**Data Integrity Grade: A+++/A/B/C/F**

**CERTIFICATION:**
- [ ] Zero lookahead bias confirmed
- [ ] Zero survivorship bias confirmed
- [ ] Point-in-time data access verified
- [ ] Safe for production deployment

**Confidence Level:** HIGH/MEDIUM/LOW

## Issues Summary

### Critical Issues (Must Fix)
[List or "None"]

### Minor Issues (Should Fix)
[List or "None"]

### Recommendations
1. [Recommendation 1]
2. [Recommendation 2]

## Next Steps

**If ALL checks passed:**
- ✅ Mark Phase 2 as COMPLETE in CURRENT_STATE.md
- ➡️ Proceed to Phase 3 (Transaction Cost Right-Sizing)
- ➡️ Begin production deployment preparation

**If ANY checks failed:**
- 🔴 Document in ERROR_PREVENTION_ARCHITECTURE.md
- 🔴 Fix issues before proceeding
- 🔴 Re-run verification after fixes

---
**Report generated by Data Integrity Verification**
**See CURRENT_STATE.md Phase 2 for complete methodology**
```

## Post-Verification Actions

1. Save report to `results/data_integrity_report.md`
2. Update `CURRENT_STATE.md` with certification status
3. If all checks passed:
   - Mark Phase 2 as ✅ COMPLETE
   - Add "Data Integrity: A+++" badge to README.md
4. If any checks failed:
   - Log issues in ERROR_PREVENTION_ARCHITECTURE.md
   - Create fix plan with priority
   - DO NOT proceed to production

## Important Notes

- **This is CRITICAL for production deployment**
- **Be thorough - survivorship bias can destroy real performance**
- **Document everything - institutional investors will audit this**
- **If unsure, err on side of caution - mark as FAIL**
- **Reference ERROR_PREVENTION_ARCHITECTURE.md for known patterns**
