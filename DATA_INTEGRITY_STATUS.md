# Data Integrity Status Report

**Date:** 2025-11-20
**Phase:** Pre-Production Validation
**Overall Status:** ⚠️ **NOT CERTIFIED** - Critical gaps identified

## Executive Summary

SignalTide v3 has **NOT passed comprehensive data integrity certification** required for production deployment with real capital. While the architecture and signal design are A+++, **we cannot certify zero lookahead/survivorship bias** without completing the validation steps below.

**DO NOT DEPLOY TO PRODUCTION** until all checks below show ✅ VERIFIED.

## Certification Checklist

### Critical for Production

| Category | Status | Priority | Blocker? |
|----------|--------|----------|----------|
| Point-in-Time Universe | ❌ UNVERIFIED | CRITICAL | YES |
| Fundamental Data Lag | ⚠️ PARTIAL | CRITICAL | YES |
| Survivorship Bias | ❌ UNVERIFIED | CRITICAL | YES |
| Lookahead Bias (Prices) | ⚠️ PARTIAL | CRITICAL | YES |
| Lookahead Bias (Fundamentals) | ⚠️ PARTIAL | CRITICAL | YES |
| Lookahead Bias (Insider) | ❌ UNVERIFIED | HIGH | YES |
| Corporate Actions | ❌ UNVERIFIED | MEDIUM | NO |
| Data Quality | ✅ BASIC | MEDIUM | NO |

**Blockers:** 5 CRITICAL, 1 HIGH
**Pass Rate:** 0/8 fully verified

## Detailed Status

### 1. Point-in-Time Universe Construction

**Status:** ❌ **UNVERIFIED**
**Priority:** CRITICAL
**Blocks Production:** YES

#### What This Means

At each backtest date (e.g., 2020-03-15), we must use ONLY the stocks that:
1. Were actively trading on that date
2. Had NOT been delisted yet
3. Were included in our universe criteria at that time

**Example Risk:**
- Stock XYZ delisted in 2023 after dropping 95%
- If we exclude it from our 2020-2023 backtest, we miss the loss
- This inflates returns (survivorship bias)

#### Current Implementation

**Code Location:** `scripts/run_institutional_backtest.py`

```python
# Lines 200-250 (approximate)
def run_signal_backtest(self, signal_class, signal_params, signal_name, needs_dm=False):
    # Universe is fixed for entire backtest period
    universe = self.universe  # STATIC LIST
```

**Problem:** Universe is **static** for entire backtest period. No logic to:
- Check if ticker was trading on signal date
- Exclude tickers after delisting date
- Include tickers only after IPO date

#### What Needs Verification

1. **Query Database for Active Tickers:**
   ```sql
   SELECT ticker, firstpricedate, lastpricedate, delisted
   FROM sharadar_tickers
   WHERE ticker IN (universe)
     AND firstpricedate <= ?  -- Signal date
     AND (lastpricedate >= ? OR lastpricedate IS NULL)  -- Not delisted yet
   ```

2. **Test Cases:**
   - WISH (ContextLogic): Delisted 2023-01-XX after $20 → $0.50 collapse
   - Verify WISH is included in backtest 2020-2022
   - Verify WISH is excluded after delisting date
   - Verify final loss is captured in performance

3. **Validation Script:**
   ```python
   # Check if delisted stocks are in universe
   delisted_count = len([t for t in universe if is_delisted(t, end_date)])
   print(f"Delisted stocks in universe: {delisted_count}")
   # Should be > 0 for 2020-2024 period
   ```

#### Required Fix

Create `validate_universe_timeline()` method:
```python
def validate_universe_timeline(self, as_of_date: str) -> List[str]:
    """
    Get valid universe for specific date.

    Returns only tickers that:
    - Were trading on as_of_date
    - Not delisted before as_of_date
    - Pass our liquidity/data quality filters
    """
    query = """
    SELECT ticker
    FROM sharadar_tickers
    WHERE ticker IN ({})
      AND firstpricedate <= ?
      AND (delisted = 0 OR lastpricedate >= ?)
    """.format(','.join('?' * len(self.universe)))

    return self.dm.query(query, self.universe + [as_of_date, as_of_date])
```

#### Success Criteria

- [ ] Universe changes appropriately over backtest period
- [ ] Delisted stocks included until delisting date
- [ ] No stocks before their IPO date
- [ ] Count of delisted stocks > 0 for 2020-2024 (sanity check)
- [ ] Visual inspection: Plot universe size over time, should vary

---

### 2. Fundamental Data Timing

**Status:** ⚠️ **PARTIAL** (Quality signal fixed, others unverified)
**Priority:** CRITICAL
**Blocks Production:** YES

#### What This Means

When using quarterly earnings (e.g., 2020 Q1 earnings), we must account for:
1. **Reporting lag**: Company reports ~30-45 days after quarter end
2. **Filing lag**: SEC filing happens after earnings call
3. **Safe lag**: Assume 45-60 days from quarter end to public availability

**Example Risk:**
- Using 2020-03-31 earnings data on 2020-03-31 signal date
- Earnings not filed until 2020-05-15 (45 days later)
- This is **lookahead bias** - using future data in past

#### Current Implementation

**Verified Fixed:**
- ✅ Quality signal (`signals/quality/institutional_quality.py` line 101):
  ```python
  fundamentals = self.dm.get_fundamentals(
      ticker=ticker,
      dimensions=['MRQ'],
      as_of=as_of_date  # ✅ CORRECT
  )
  ```

**Unverified:**
- ❌ Momentum signal: Uses prices only (no fundamentals) - **LOW RISK**
- ❌ Insider signal: May use fundamentals for quality filter - **NEEDS AUDIT**
- ❌ Any composite signals - **NEEDS AUDIT**

#### What Needs Verification

1. **Grep all `get_fundamentals()` calls:**
   ```bash
   grep -r "get_fundamentals" --include="*.py" signals/
   ```

2. **For each call, verify:**
   - Has `as_of` parameter
   - `as_of` is set to signal date (not None)
   - Comment explains lag assumption

3. **Test with Historical Date:**
   ```python
   # Simulate backtest on 2020-03-15
   # Get fundamentals that would have been available
   fundamentals = dm.get_fundamentals(
       ticker='AAPL',
       dimensions=['MRQ'],
       as_of='2020-03-15'
   )
   # Verify latest data is from 2019-12-31 (Q4 2019)
   # NOT from 2020-03-31 (Q1 2020 - not filed yet)
   assert fundamentals['calendardate'].max() <= '2020-02-01'
   ```

#### Required Fix

1. **Audit all signals:**
   - Read each signal implementation completely
   - Identify all data sources (prices, fundamentals, insider, events)
   - Verify temporal discipline for each

2. **Add validation to DataManager:**
   ```python
   def get_fundamentals(self, ticker, dimensions, as_of=None):
       if as_of is None:
           raise ValueError("as_of parameter is REQUIRED (prevent lookahead)")

       # Get data with filing lag
       filing_lag_days = 45
       safe_date = pd.to_datetime(as_of) - pd.Timedelta(days=filing_lag_days)

       # Query only data filed before safe_date
       return self._query_fundamentals(ticker, dimensions, max_date=safe_date)
   ```

3. **Document lag assumptions:**
   - Create `docs/DATA_TEMPORAL_RULES.md`
   - Document each data source's lag
   - Reference in CLAUDE.md

#### Success Criteria

- [ ] All `get_fundamentals()` calls have `as_of` parameter
- [ ] Historical date test passes (2020-03-15 → latest Q4 2019)
- [ ] DataManager enforces `as_of` requirement
- [ ] Documentation explains 45-day lag assumption
- [ ] No fundamentals used within 45 days of filing

---

### 3. Survivorship Bias Validation

**Status:** ❌ **UNVERIFIED**
**Priority:** CRITICAL
**Blocks Production:** YES

#### What This Means

**Survivorship bias** = Only backtesting stocks that are still trading today.

This is **catastrophic** for backtest validity because:
- Delisted stocks often went to $0 (100% loss)
- Excluding them removes the worst performers
- Can inflate Sharpe ratio by 0.3-0.5 (huge!)

**Real Example:**
- WISH (ContextLogic): IPO $24 (Dec 2020) → Delisted $0.50 (Jan 2023)
- If excluded: Miss 98% loss
- If included: Captures real risk

#### Current Implementation

**Code Location:** `scripts/run_institutional_backtest.py`

```python
# Universe is manually specified or from screening
universe = ['AAPL', 'MSFT', 'GOOGL', ...]  # All still trading today?
```

**Problem:** No evidence that delisted stocks are included.

#### What Needs Verification

1. **Query Delisted Stocks 2020-2024:**
   ```sql
   SELECT ticker, name, lastpricedate, delisted
   FROM sharadar_tickers
   WHERE delisted = 1
     AND lastpricedate >= '2020-01-01'
     AND lastpricedate <= '2024-12-31'
   ORDER BY lastpricedate DESC
   LIMIT 100;
   ```

2. **Check High-Profile Delistings:**
   - WISH (ContextLogic) - Delisted 2023
   - CVNA (Carvana) - Check if survived
   - Any stock that went to $0

3. **Calculate Inclusion Rate:**
   ```python
   delisted_in_period = get_delisted_stocks('2020-01-01', '2024-12-31')
   delisted_in_universe = [t for t in delisted_in_period if t in universe]

   inclusion_rate = len(delisted_in_universe) / len(delisted_in_period)
   print(f"Delisted stock inclusion rate: {inclusion_rate:.1%}")
   # Should be > 0% (ideally 10-30% depending on universe criteria)
   ```

4. **Verify Loss Capture:**
   ```python
   # For WISH specifically
   wish_trades = [t for t in backtest.trades if t.symbol == 'WISH']
   if len(wish_trades) > 0:
       # Verify final exit price is near delisting price (~$0.50)
       final_exit = wish_trades[-1].price
       assert final_exit < 1.0, "Should capture near-zero delisting price"
   else:
       raise ValueError("WISH not in universe - survivorship bias!")
   ```

#### Required Fix

1. **Expand Universe to Include Delisted:**
   - Don't manually specify universe
   - Use database query with survivorship-bias-free criteria
   - Example: "All stocks with >$500M market cap on 2020-01-01"

2. **Create Survivorship Report:**
   ```python
   # scripts/analyze_survivorship_bias.py
   def generate_survivorship_report(universe, start, end):
       """
       Analyze survivorship bias in universe.

       Reports:
       - Total delisted stocks in period
       - Delisted stocks in our universe
       - Inclusion rate
       - Impact on returns (with/without delisted)
       """
   ```

3. **Automated Check:**
   ```python
   def validate_no_survivorship_bias(backtest_results):
       """
       Raise error if universe shows survivorship bias.

       Heuristic: If ZERO stocks delisted during 5-year backtest,
       highly suspicious (expect 5-15% delisting rate).
       """
       delisted_count = count_delisted_in_universe(backtest_results)
       if delisted_count == 0:
           raise ValueError("SURVIVORSHIP BIAS: Zero delistings detected!")
   ```

#### Success Criteria

- [ ] Query identifies 50+ delisted stocks 2020-2024
- [ ] At least 5-10 delisted stocks in our universe
- [ ] WISH (or similar) appears in trades
- [ ] Final prices capture delisting losses
- [ ] Report documents inclusion rate > 5%
- [ ] Automated check prevents zero-delisting universes

---

### 4. Lookahead Bias - Price Data

**Status:** ⚠️ **PARTIAL** (Likely correct, needs verification)
**Priority:** CRITICAL
**Blocks Production:** YES

#### What This Means

When generating signal on date T, use ONLY price data up to date T.

**Common Mistakes:**
```python
# WRONG: Uses future data
signal = prices.rolling(252).mean()  # Includes today's close in calculation

# CORRECT: Shift to use only past data
signal = prices.shift(1).rolling(252).mean()  # Uses yesterday's close
```

#### Current Implementation

**Momentum Signal (`signals/momentum/institutional_momentum.py`):**
- Uses `rolling()` and `pct_change()`
- **NEEDS VERIFICATION:** Are we shifting correctly?

**Expected Pattern:**
```python
# Generate signal for date T
# Should use prices[T-252:T-1], NOT prices[T-251:T]
returns = prices.pct_change()
momentum = returns.rolling(window=252, min_periods=200).mean().shift(1)
```

#### What Needs Verification

1. **Manual Inspection:**
   - Read momentum signal implementation line by line
   - Check every `.rolling()` call
   - Check every `.pct_change()` call
   - Verify `.shift(1)` before using signals

2. **Historical Test:**
   ```python
   # Generate signal for 2020-03-15
   signal_date = '2020-03-15'
   signal_value = momentum_signal.generate_signals(data, as_of=signal_date)

   # Verify signal uses data up to 2020-03-14 only
   latest_price_used = get_latest_price_in_calculation(signal_date)
   assert latest_price_used <= '2020-03-14'
   ```

3. **Correlation Test:**
   ```python
   # Signal should NOT be correlated with same-day returns
   same_day_returns = prices.pct_change()
   signal_values = momentum_signal.generate_signals(data)

   correlation = signal_values.corr(same_day_returns)
   assert abs(correlation) < 0.1, "Signal correlated with same-day returns!"
   ```

#### Required Fix

1. **Add Shift Validation:**
   ```python
   class BaseSignal:
       def validate_no_lookahead(self, signals, prices):
           """
           Validate signals don't use same-day prices.

           Method: Check correlation with same-day returns.
           If high correlation, likely lookahead bias.
           """
           same_day_returns = prices.pct_change()
           for ticker in signals.columns:
               corr = signals[ticker].corr(same_day_returns[ticker])
               if abs(corr) > 0.3:
                   raise ValueError(
                       f"LOOKAHEAD BIAS: {ticker} signal correlated "
                       f"with same-day returns (corr={corr:.2f})"
                   )
   ```

2. **Documentation:**
   - Add comment to every rolling calculation
   - Explain why shift is necessary
   - Reference temporal discipline in CLAUDE.md

#### Success Criteria

- [ ] Manual code inspection confirms `.shift(1)` usage
- [ ] Historical test passes (uses T-1 data for signal T)
- [ ] Correlation test passes (|corr| < 0.1)
- [ ] All signals have validation in unit tests
- [ ] Documentation explains shift necessity

---

### 5. Lookahead Bias - Insider Trading Data

**Status:** ❌ **UNVERIFIED**
**Priority:** HIGH
**Blocks Production:** YES

#### What This Means

Insider transactions are **reported with lag**:
1. Transaction date (when insider bought/sold)
2. Filing date (when Form 4 filed with SEC)
3. **Lag: 2 business days** (legally required)

**Example:**
- Insider buys on 2020-03-13 (Friday)
- Form 4 filed by 2020-03-17 (Tuesday, 2 days later)
- **We can't use this trade until 2020-03-18** (after market close on filing day)

#### Current Implementation

**Code Location:** `signals/insider/institutional_insider.py`

**NEEDS AUDIT:** Does the signal use:
- Transaction date (filingdate) ✅ CORRECT
- Or actual trade date (transactiondate) ❌ LOOKAHEAD

#### What Needs Verification

1. **Read Insider Signal Code:**
   ```python
   # Check which date field is used
   insider_data = self.dm.get_insider_trades(ticker=ticker, as_of=as_of_date)

   # Verify query filters by filing date, not transaction date:
   # WHERE filingdate <= as_of_date  ✅ CORRECT
   # NOT: WHERE transactiondate <= as_of_date  ❌ WRONG
   ```

2. **Schema Verification:**
   ```sql
   -- Check sharadar_insiders schema
   SELECT filingdate, transactiondate, ticker, shares
   FROM sharadar_insiders
   WHERE ticker = 'AAPL'
     AND filingdate >= '2020-01-01'
   ORDER BY filingdate DESC
   LIMIT 10;

   -- Verify lag between transactiondate and filingdate
   SELECT AVG(julianday(filingdate) - julianday(transactiondate)) as avg_lag_days
   FROM sharadar_insiders
   WHERE filingdate IS NOT NULL
     AND transactiondate IS NOT NULL;
   -- Should be ~2-3 days
   ```

3. **Test Case:**
   ```python
   # Insider bought on 2020-03-13
   # Filed on 2020-03-17
   # Generate signal for 2020-03-16 (before filing)
   signal = insider_signal.generate_signals(data, as_of='2020-03-16')

   # This trade should NOT be included in signal
   insider_trades = dm.get_insider_trades('AAPL', as_of='2020-03-16')
   assert '2020-03-13' not in insider_trades['transactiondate'].values

   # Generate signal for 2020-03-18 (after filing)
   signal = insider_signal.generate_signals(data, as_of='2020-03-18')

   # NOW this trade should be included
   insider_trades = dm.get_insider_trades('AAPL', as_of='2020-03-18')
   assert '2020-03-13' in insider_trades['transactiondate'].values
   ```

#### Required Fix

1. **Verify DataManager Query:**
   ```python
   def get_insider_trades(self, ticker, as_of=None):
       """
       Get insider trades known as of date.

       CRITICAL: Filter by filingdate, NOT transactiondate!
       This accounts for 2-day reporting lag.
       """
       if as_of is None:
           raise ValueError("as_of required to prevent lookahead")

       query = """
       SELECT *
       FROM sharadar_insiders
       WHERE ticker = ?
         AND filingdate <= ?  -- Use filing date, not transaction date!
       ORDER BY filingdate DESC
       """
       return self.query(query, [ticker, as_of])
   ```

2. **Add Lag Verification:**
   ```python
   # In DataManager.__init__
   def validate_insider_lag(self):
       """Verify insider data has expected 2-day filing lag."""
       query = """
       SELECT AVG(julianday(filingdate) - julianday(transactiondate)) as avg_lag
       FROM sharadar_insiders
       WHERE filingdate IS NOT NULL
         AND transactiondate IS NOT NULL
       """
       avg_lag = self.query(query)[0]['avg_lag']

       if avg_lag < 1.0:
           raise ValueError("Insider filing lag suspiciously low!")

       logger.info(f"Insider average filing lag: {avg_lag:.1f} days")
   ```

#### Success Criteria

- [ ] Code inspection confirms `filingdate` usage
- [ ] Schema analysis shows ~2-day lag between transaction/filing
- [ ] Test case passes (transaction excluded before filing)
- [ ] DataManager enforces `as_of` parameter
- [ ] Validation runs on startup

---

### 6. Corporate Actions Verification

**Status:** ❌ **UNVERIFIED**
**Priority:** MEDIUM
**Blocks Production:** NO (but important)

#### What This Means

Stock splits, dividends, and spin-offs affect historical prices:
- **2-for-1 split**: $100 stock → $50 stock (2x shares)
- **Historical prices must be adjusted** to maintain continuity
- **Example**: Apple 4-for-1 split on 2020-08-31

#### Current Implementation

**Data Source:** Sharadar provides split-adjusted prices

**ASSUMPTION:** `sharadar_prices` table has split-adjusted data

**NEEDS VERIFICATION:** Is this assumption correct?

#### What Needs Verification

1. **Check Known Splits:**
   ```python
   # Apple 4-for-1 split on 2020-08-31
   aapl_before = dm.get_prices('AAPL', start='2020-08-28', end='2020-08-28')
   aapl_after = dm.get_prices('AAPL', start='2020-09-01', end='2020-09-01')

   # Pre-split close should be ~$500 (unadjusted) or ~$125 (adjusted)
   # Post-split close should be ~$125 (both adjusted and unadjusted)
   # If continuity broken, data not adjusted correctly
   ```

2. **Check Return Continuity:**
   ```python
   # Calculate returns across split date
   aapl_returns = dm.get_prices('AAPL', '2020-08-01', '2020-09-30').pct_change()

   # Split day should NOT show huge return jump
   split_day_return = aapl_returns.loc['2020-08-31']
   assert abs(split_day_return) < 0.05, "Split not adjusted correctly"
   ```

3. **Query Events Table:**
   ```sql
   SELECT ticker, date, eventcode, splitratio
   FROM sharadar_events
   WHERE ticker = 'AAPL'
     AND date >= '2020-01-01'
     AND eventcode LIKE '%split%'
   ```

#### Required Fix

1. **Automated Split Verification:**
   ```python
   def verify_split_adjustments(ticker, start, end):
       """
       Verify price continuity across corporate actions.

       Checks that returns don't show artificial jumps on split dates.
       """
       prices = dm.get_prices(ticker, start, end)
       returns = prices['close'].pct_change()

       # Get split events
       events = dm.get_events(ticker, start, end, eventcode='split')

       for event in events:
           split_date = event['date']
           split_return = returns.loc[split_date]

           # Split should not create >5% artificial return
           if abs(split_return) > 0.05:
               raise ValueError(
                   f"{ticker} split on {split_date} not properly adjusted "
                   f"(return = {split_return:.2%})"
               )
   ```

2. **Documentation:**
   - Confirm Sharadar provides split-adjusted data
   - Document any adjustments we make
   - Add to ARCHITECTURE.md data section

#### Success Criteria

- [ ] Apple split verified (2020-08-31)
- [ ] Tesla split verified (2020-08-31)
- [ ] No artificial return jumps on split dates
- [ ] Events table query confirms splits exist
- [ ] Documentation confirms adjustment methodology

---

### 7. Data Quality Baseline

**Status:** ✅ **BASIC** (Has basic checks, not comprehensive)
**Priority:** MEDIUM
**Blocks Production:** NO

#### Current Implementation

**DataManager has basic checks:**
- Missing data detection
- Duplicate removal
- Date range validation

**What's Missing:**
- Outlier detection (fat-finger trades)
- Volume validation (0 volume days)
- Spread validation (unrealistic spreads)

#### Success Criteria

- [x] Basic missing data detection (EXISTS)
- [ ] Outlier detection for prices
- [ ] Volume > 0 validation
- [ ] Price continuity checks

---

## Summary & Next Steps

### Current State

**Verified:** 0/8 checks
**Partial:** 3/8 checks
**Unverified:** 5/8 checks

**CRITICAL BLOCKERS:**
1. Point-in-Time Universe (❌)
2. Survivorship Bias (❌)
3. Fundamental Lag (⚠️ Partial)
4. Price Lookahead (⚠️ Partial)
5. Insider Lookahead (❌)

### Immediate Next Steps

**Phase 1: Audit (1-2 days)**
1. Read all signal implementations completely
2. Verify temporal discipline for each data source
3. Document findings in this file

**Phase 2: Database Analysis (1 day)**
1. Query delisted stocks 2020-2024
2. Check fundamental filing lags
3. Verify insider filing lags
4. Test corporate action adjustments

**Phase 3: Validation Scripts (2-3 days)**
1. Create `scripts/validate_data_integrity.py`
2. Implement all verification checks above
3. Generate comprehensive report
4. Fix any issues found

**Phase 4: Certification (1 day)**
1. Run all validation checks
2. Review results with user
3. Update this document with ✅ or ❌
4. Make go/no-go decision for production

### Estimated Timeline

**Best Case:** 5-7 days if all checks pass
**Realistic:** 10-14 days including fixes
**Worst Case:** 30+ days if major issues found

### Production Deployment Criteria

**DO NOT DEPLOY** until:
- [ ] All CRITICAL items show ✅ VERIFIED
- [ ] All HIGH items show ✅ VERIFIED or ⚠️ ACCEPTABLE
- [ ] Comprehensive report generated
- [ ] User reviews and approves
- [ ] This document updated with certification

---

## References

**CURRENT_STATE.md:**
- Phase 2: Data Integrity Verification (lines 218-244)

**.claude/commands/check-data-integrity.md:**
- Detailed validation procedures

**Academic Standards:**
- Bailey et al. (2014) "Backtest Overfitting"
- Harvey & Liu (2015) "Backtesting"
- De Prado (2018) "Advances in Financial Machine Learning" Chapter 7

---

**Last Updated:** 2025-11-20
**Next Review:** After Phase 1 audit complete
**Certification Status:** ❌ NOT CERTIFIED FOR PRODUCTION
