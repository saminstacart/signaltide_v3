# Transaction Cost Discrepancy Analysis

**Date:** 2025-11-20
**Severity:** CRITICAL
**Status:** ✅ RESOLVED (2025-11-21)

**RESOLUTION NOTE (2025-11-21):**
This document analyzed a historical discrepancy between documented (20 bps) and implemented (5 bps) transaction costs. After review, we determined that **5 bps is the correct default** for a $50K Schwab account with zero commissions and tight spreads on liquid stocks. All documentation has been updated to reflect:
- **Default production assumption:** ~5 bps per round-trip
- **Stress testing:** 10-20 bps to ensure robustness under worse liquidity

The analysis below is preserved for historical context.

---

## Executive Summary

SignalTide v3 has a **critical discrepancy** between documented transaction costs (20 bps) and implemented transaction costs (5 bps) in backtests. This means all backtest performance metrics are potentially **4x too optimistic** regarding transaction cost impact.

**UPDATE:** After analysis, 5 bps is deemed appropriate for the production environment. Documentation has been aligned accordingly.

## The Discrepancy

### What Documentation Says: 20 bps

**config.py (lines 103-107):**
```python
DEFAULT_TRANSACTION_COSTS = {
    'commission_pct': 0.001,  # 0.1% commission (10 bps)
    'slippage_pct': 0.0005,  # 0.05% slippage (5 bps)
    'spread_pct': 0.0005,  # 0.05% spread (5 bps)
}
# Total: 20 basis points
```

**.claude/CLAUDE.md (line X):**
> Transaction costs: Always 20bps (10 commission + 5 slippage + 5 spread)

**docs/PRODUCTION_READY.md (lines 60-65):**
```markdown
**Cost Model Verification:**
- Commission: 10 bps (verify with broker)
- Slippage: 5 bps (monitor vs actual)
- Spread: 5 bps (half-spread cost)
- **Total: 20 bps per trade**
```

### What Backtest Actually Uses: 5 bps

**scripts/run_institutional_backtest.py (lines 146-152):**
```python
# Transaction costs (default to realistic Schwab model)
if transaction_costs is None:
    self.transaction_costs = {
        'commission_pct': 0.0,      # $0 commission
        'slippage_pct': 0.0002,     # 2 bps
        'spread_pct': 0.0003,       # 3 bps
        # Total: 5 bps per trade
    }
```

**results/institutional_backtest_report.md (line 6):**
```markdown
**Transaction Costs:** 5.0 bps
```

## Root Cause Analysis

### Why This Happened

1. **Schwab Zero-Commission Model**: Comment in backtest script says "realistic Schwab model"
   - Schwab offers $0 commissions for stocks/ETFs
   - But we still have market impact (slippage + spread)

2. **Dual Configuration Sources**:
   - `config.py` has DEFAULT_TRANSACTION_COSTS (20 bps)
   - `run_institutional_backtest.py` has its own hardcoded defaults (5 bps)
   - Backtest script **never uses** config.py defaults unless explicitly passed

3. **No Cross-Validation**:
   - No validation that backtest costs match documented costs
   - No check that config.py values are being used

### Code Flow

```
run_institutional_backtest.py
  ↓
  __init__(transaction_costs=None)  # User didn't pass costs
  ↓
  if transaction_costs is None:  # True!
  ↓
  self.transaction_costs = {5 bps hardcoded}  # USES HARDCODED, NOT CONFIG
```

The `TransactionCostModel` in `core/execution.py` is correctly implemented and uses config.py defaults, but the **backtest script never creates it** with those defaults.

## Impact Assessment

### Overly Optimistic Results?

With 5 bps instead of 20 bps, we're underestimating costs by **15 bps per trade**.

**Monthly Rebalancing Analysis:**
- Rebalance frequency: Monthly (12x per year)
- Assume 50% turnover per rebalance (conservative)
- Annual trades per position: 12 rebalances × 50% turnover = 6 full round-trips
- Cost difference per round-trip: (20 bps - 5 bps) × 2 sides = 30 bps
- **Annual cost underestimate: 6 × 30 bps = 180 bps = 1.8% per year**

### Performance Impact

**Current Backtest Results (with 5 bps):**
- Momentum: Annual 28.82%, Sharpe 0.757
- Quality: Annual 1.74%, Sharpe -0.157
- Insider: Annual 27.59%, Sharpe 0.975

**Estimated with 20 bps (reducing by ~1.8%):**
- Momentum: Annual ~27.0%, Sharpe ~0.72 (still good)
- Quality: Annual ~0.0%, Sharpe ~-0.30 (worse, already negative)
- Insider: Annual ~25.8%, Sharpe ~0.93 (still excellent)

**VERDICT:** Results still likely positive but **degraded by ~6-7% relative**.

## Which Cost Model is Correct?

### The 5 bps "Schwab Model" (Current)

**Pros:**
- Reflects modern zero-commission brokers (Schwab, Fidelity, TD Ameritrade)
- More realistic for retail/small institutional
- 2 bps slippage + 3 bps spread is reasonable for liquid stocks

**Cons:**
- Ignores opportunity cost of free commissions (payment for order flow)
- May be optimistic for large positions
- Doesn't match our documented standard

### The 20 bps "Conservative Model" (Documented)

**Pros:**
- More conservative, safer for production
- Includes 10 bps commission buffer for adverse selection
- Accounts for market impact on larger trades
- Matches institutional best practices

**Cons:**
- May be pessimistic for very liquid stocks (AAPL, MSFT, etc.)
- 10 bps commission unrealistic with modern brokers
- Could underestimate true alpha

### Recommendation: Use 10 bps (Split the Difference)

**Proposed Model:**
```python
{
    'commission_pct': 0.0002,  # 2 bps (PFOF + execution quality)
    'slippage_pct': 0.0004,    # 4 bps (market impact)
    'spread_pct': 0.0004,      # 4 bps (bid-ask)
}
# Total: 10 basis points
```

**Rationale:**
- More realistic than 20 bps for liquid stocks
- More conservative than 5 bps for safety margin
- 2 bps accounts for payment-for-order-flow costs
- 4 bps slippage reasonable for $5K-$10K trades
- 4 bps spread conservative for S&P 500 stocks

## Action Plan

### Phase 1: Immediate Documentation Sync

1. **Update config.py:**
   - Change DEFAULT_TRANSACTION_COSTS to 10 bps model
   - Add comment explaining rationale
   - Document assumptions (liquid stocks, <5% ADV)

2. **Fix run_institutional_backtest.py:**
   - Remove hardcoded transaction costs
   - Import and use DEFAULT_TRANSACTION_COSTS from config
   - Add validation that costs match config

3. **Update all documentation:**
   - CLAUDE.md: Change "20bps" to "10bps"
   - PRODUCTION_READY.md: Update cost model
   - Add footnote about Schwab zero-commission era

### Phase 2: Re-Run Critical Backtests

Re-run with corrected 10 bps costs:
1. Individual signal backtests (momentum, quality, insider)
2. SPY benchmark comparison
3. Rebalancing frequency analysis
4. Update all reports in results/

### Phase 3: Add Validation

Create test to ensure backtest costs match config:
```python
def test_backtest_costs_match_config():
    """Ensure backtest uses config transaction costs."""
    from config import DEFAULT_TRANSACTION_COSTS
    from scripts.run_institutional_backtest import InstitutionalBacktest

    bt = InstitutionalBacktest(
        universe=['AAPL'],
        start_date='2020-01-01',
        end_date='2020-12-31'
    )

    # Should use config defaults when not specified
    assert bt.transaction_costs == DEFAULT_TRANSACTION_COSTS
```

### Phase 4: Sensitivity Analysis

Document performance under different cost models:
- 5 bps (optimistic - modern broker)
- 10 bps (baseline - recommended)
- 15 bps (moderate - market impact included)
- 20 bps (conservative - institutional standard)

Create `results/transaction_cost_sensitivity.md` showing how Sharpe changes.

## Governance Changes

### Prevention Measures

1. **Single Source of Truth**:
   - All transaction costs MUST come from config.py
   - No hardcoded cost values allowed in scripts
   - Add linter rule to catch this

2. **Cross-Validation**:
   - Add assertion in backtest that costs match config
   - Report must show which config was used
   - Test suite validates cost propagation

3. **Documentation Standards**:
   - All docs reference config.py for canonical values
   - Update quarterly to reflect broker changes
   - Track cost model changes in git

### Code Review Checklist

Add to ERROR_PREVENTION_ARCHITECTURE.md:
```markdown
### Transaction Cost Verification
- [ ] No hardcoded transaction costs in scripts?
- [ ] Uses config.py DEFAULT_TRANSACTION_COSTS?
- [ ] Report shows correct cost basis points?
- [ ] Costs validated against broker reality?
```

## References

**Schwab Zero Commission:**
- Announced October 2019
- $0 online stock/ETF trades
- Still have SEC fees (~0.2 bps), PFOF spread impact

**Academic Literature:**
- Almgren & Chriss (2000): Market impact models
- Grinold & Kahn (2000): 5-30 bps typical for institutional
- Modern HFT era: Spreads compressed to 1-5 bps for liquid stocks

**Broker Comparisons (2024):**
- Interactive Brokers: $0.005/share (IBKR Lite: $0)
- Schwab: $0 stocks/ETFs
- Fidelity: $0 stocks/ETFs
- Institutional: Often volume-based, ~0.2-2 bps

## Conclusion

This is a **critical but fixable** issue. The 4x discrepancy between documented (20 bps) and implemented (5 bps) costs is significant, but:

1. **Impact is manageable**: ~1.8% annual drag difference
2. **Results still positive**: Even with 20 bps, Sharpe ratios remain good
3. **Easy to fix**: Single source of truth in config.py
4. **Opportunity for improvement**: 10 bps model is better than either extreme

**Next Steps:**
1. ✅ Document issue (this file)
2. ⏳ Update config.py to 10 bps model
3. ⏳ Fix backtest script to use config
4. ⏳ Re-run all backtests with corrected costs
5. ⏳ Update ERROR_PREVENTION_ARCHITECTURE.md
6. ⏳ Add validation tests

**Priority:** HIGH - Complete before any production deployment

---

**Last Updated:** 2025-11-20
**Next Review:** After fix implementation
