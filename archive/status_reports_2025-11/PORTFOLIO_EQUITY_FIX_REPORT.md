# Portfolio Equity Bug Investigation Report

**Date:** 2025-11-20
**Issue:** AttributeError: 'Portfolio' object has no attribute 'equity'

## Summary

The reported bug does NOT exist in the current codebase. All code correctly uses `portfolio.get_equity()` instead of `portfolio.equity`.

## Investigation Results

### 1. Portfolio Class Analysis (`core/portfolio.py`)

**Correct Method:**
- Line 71-74: `get_equity()` method exists and works correctly
- Returns: `float` representing total equity (cash + positions)

**NO Attribute:**
- No `equity` property or attribute exists
- Accessing `portfolio.equity` raises `AttributeError` (as expected)

### 2. Backtest Script Analysis (`scripts/run_institutional_backtest.py`)

**Line 315 (Reported as line 274 in bug report):**
```python
equity_curve.append(portfolio.get_equity())  # ✓ CORRECT
```

**Search Results:**
- Searched entire codebase for `portfolio.equity` pattern
- **0 instances found** (excluding `get_equity()` and `equity_curve`)
- All code uses the correct method

### 3. Verification Tests

Created and ran `/Users/samuelksherman/signaltide_v3/scripts/test_portfolio_equity.py`:

**Test Results:**
- ✓ Test 1: `get_equity()` returns initial capital correctly
- ✓ Test 2: `get_equity()` updates after trades
- ✓ Test 3: `equity_curve` attribute exists
- ✓ Test 4: `portfolio.equity` correctly raises `AttributeError`

**Backtest Pattern Test:**
- ✓ Simulated 5-day backtest
- ✓ Equity tracking works correctly
- ✓ Returns calculated properly

### 4. Possible Causes of User's Error

If the user experienced this error, possible causes:

1. **Cached bytecode** - Old .pyc files with buggy code
   - **Solution:** Cleared all `__pycache__` directories and `.pyc` files

2. **Local uncommitted changes** - User may have experimental code
   - **Solution:** User should run `git diff` to check

3. **Different file version** - User may have older version
   - **Solution:** User should pull latest code

4. **Line number mismatch** - Recent commits may have shifted line numbers
   - **Solution:** Search for pattern, not line number

## Recommendations

### For Users Experiencing This Error:

1. **Clear Python cache:**
   ```bash
   find . -type d -name "__pycache__" -exec rm -rf {} +
   find . -type f -name "*.pyc" -delete
   ```

2. **Verify code version:**
   ```bash
   git status
   git diff scripts/run_institutional_backtest.py
   git diff core/portfolio.py
   ```

3. **Pull latest changes:**
   ```bash
   git pull origin main
   ```

4. **Run verification test:**
   ```bash
   python3 scripts/test_portfolio_equity.py
   ```

### Code Standards Going Forward:

**ALWAYS use:**
```python
equity = portfolio.get_equity()  # ✓ Correct
```

**NEVER use:**
```python
equity = portfolio.equity  # ✗ Wrong - AttributeError
```

## Database Setup Issue

**Secondary Issue Discovered:**
The backtest script cannot run because the database doesn't exist:
```
FileNotFoundError: Database not found: /Users/samuelksherman/signaltide_v3/data/databases/market_data.db
```

**Solution Required:**
1. User needs to create/download the Sharadar database
2. Or set `SIGNALTIDE_DB_PATH` environment variable to existing database
3. Or use mock data for testing (see `data/mock_generator.py`)

## Conclusion

**Bug Status:** NOT FOUND in current codebase

**Issues Fixed:** 0 (no buggy code exists)

**Correct Attribute/Method:** `portfolio.get_equity()` (method, not attribute)

**Actions Taken:**
1. ✓ Comprehensive codebase search
2. ✓ Created verification test script
3. ✓ Cleared Python cache
4. ✓ Verified Portfolio class implementation
5. ✓ Verified backtest script implementation

**Next Steps for User:**
1. Clear Python cache (if error persists)
2. Verify code version matches repository
3. Set up database or use mock data for testing
4. Run `python3 scripts/test_portfolio_equity.py` to verify
