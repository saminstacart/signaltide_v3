# Error Prevention Architecture

## Problem Statement

We've experienced recurring errors due to:
1. **API inconsistencies**: `generate()` vs `calculate()` vs `generate_signals()`
2. **Parameter naming**: `lookback_days` vs `formation_period`
3. **Initialization patterns**: `data_manager` sometimes required, sometimes not
4. **Documentation drift**: Code changes but examples/docs don't update

These errors are caught at **runtime** (too late) instead of **design time** or **import time**.

## Solution: Contractual Architecture

### Phase 1: Define Contracts (IMMEDIATE)

**1.1 Create `core/api_contracts.py`**
- Define `BaseSignal` abstract base class
- All signals MUST inherit from it
- Enforces method signatures: `generate_signals(data) -> Series`
- Standardizes initialization: `__init__(params, data_manager=None)`

**1.2 Create `core/signal_registry.py`**
- Decorator-based registration: `@SignalRegistry.register('momentum')`
- Auto-discovery of available signals
- Validation at import time (fail fast)

**1.3 Create parameter schemas**
```python
@dataclass
class MomentumParams:
    formation_period: int  # Required
    skip_period: int       # Required
    quintiles: bool = True # Optional with default
```

### Phase 2: Retrofit Existing Code (NEXT)

**2.1 Update all signals to inherit from BaseSignal**
- InstitutionalMomentum extends BaseSignal
- InstitutionalQuality extends BaseSignal
- InstitutionalInsider extends BaseSignal

**2.2 Standardize method names**
- ALL use `generate_signals(data)` - no exceptions
- Remove `calculate()`, `generate()` aliases

**2.3 Standardize parameters**
- Use dataclasses for type safety
- Validation at construction time

### Phase 3: Auto-Documentation (AUTOMATED)

**3.1 Generate API docs from code**
```bash
python scripts/generate_api_docs.py
```
Creates `docs/API_CONTRACTS.md` automatically

**3.2 Generate usage examples**
```bash
python scripts/generate_examples.py
```
Creates `docs/SIGNAL_USAGE_EXAMPLES.md`

### Phase 4: Continuous Validation (CI/CD)

**4.1 Pre-commit hooks**
- Validate all signals implement BaseSignal
- Check docs are up-to-date
- Run type checking (mypy)

**4.2 Error trend tracking**
- Maintain `docs/ERROR_TRENDS.md`
- Log every error pattern we encounter
- Track solutions and prevention measures
- Review quarterly to identify systemic issues

## Benefits

1. **Fail Fast**: Errors caught at import time, not runtime
2. **Self-Documenting**: API docs generated from code
3. **Type Safety**: mypy catches issues before execution
4. **Consistency**: One true way to implement signals
5. **Discoverability**: Registry shows all available signals
6. **Prevention**: Pre-commit hooks catch issues before commit

## Example: Before vs After

### Before (Error-Prone)
```python
# Script A uses this:
signal = InstitutionalMomentum({'lookback_days': 252})
signals = signal.generate(ticker, start, end)

# Script B uses this:
signal = InstitutionalMomentum({'formation_period': 252}, data_manager=dm)
signals = signal.calculate(prices)

# Script C uses this:
signal = InstitutionalMomentum({'formation_period': 252})
signals = signal.generate_signals(prices)
```
**Result:** Runtime errors, confusion, wasted time

### After (Enforced Contract)
```python
# ONLY ONE WAY - enforced by BaseSignal:
signal = SignalRegistry.get_signal('momentum')(
    params={'formation_period': 252, 'skip_period': 21},
    data_manager=dm  # Optional, None if not needed
)
signals = signal.generate_signals(prices)

# If you try the wrong way:
signal.calculate(prices)  # AttributeError at runtime
# But mypy catches it: "BaseSignal has no attribute 'calculate'"
```

## Implementation Timeline

- **Week 1**: Create contracts, registry, base classes
- **Week 2**: Retrofit existing signals
- **Week 3**: Add validation scripts and pre-commit hooks
- **Week 4**: Documentation generation and error tracking

## Error Trends Log

### 2025-11-20: API Naming Mismatches
**Count:** 5 occurrences
**Pattern:** `generate()` vs `generate_signals()`
**Solution:** BaseSignal ABC with `@abstractmethod`
**Status:** Architecture designed, pending implementation

### 2025-11-20: Parameter Inconsistencies
**Count:** 3 occurrences
**Pattern:** `lookback_days` vs `formation_period`
**Solution:** Dataclass parameter schemas
**Status:** Architecture designed, pending implementation

### 2025-11-20: Data Manager Confusion
**Count:** 4 occurrences
**Pattern:** Unclear when `data_manager` is needed
**Solution:** Optional kwarg with clear docstring
**Status:** Architecture designed, pending implementation

### 2025-11-20: Inverted Comparison Logic in Benchmark Analysis
**Count:** 1 occurrence (CRITICAL)
**Pattern:** Drawdown comparison logic inverted - reported better performance as worse
**Location:** `scripts/spy_benchmark_analysis.py:567-574`
**Issue:** Max drawdown is negative; -28.72% > -34.10% means BETTER, but code reported as WORSE
**Impact:** False negative in benchmark scoring (4/5 instead of 5/5)
**Solution:** Fixed comparison logic with explanatory comment
**Status:** ✅ FIXED
**Prevention:** Add unit tests for comparison logic; code review checklist for negative metrics

### 2025-11-20: Reading Files with Limit Parameter (META-ERROR)
**Count:** 1 occurrence (CRITICAL)
**Pattern:** Claude reading files with `limit=100` parameter, truncating critical context
**Location:** Analysis of CURRENT_STATE.md, ARCHITECTURE.md (only read first 100 lines)
**Issue:** Used Read tool with limit parameter when full context was needed
**Impact:** Missing 80%+ of documentation context, incomplete understanding of project
**Solution:** Always read complete files unless explicitly confirmed file is too large
**Status:** ✅ FIXED (re-read full files)
**Prevention:**
  - Default to reading full files without limit
  - Only use limit for confirmed large files (>1000 lines)
  - Verify file size before deciding to limit
  - Add reminder to CLAUDE.md about reading complete documentation

## Integration with Claude Code

**CRITICAL**: Before ANY work session, Claude must:
1. Read `.claude/CLAUDE.md` COMPLETELY (version 2.0.0+)
2. Pass the Final Integrity Quiz (8 questions)
3. Review this error log for relevant patterns
4. Apply the behavioral contract (ALWAYS/NEVER rules)

**See `.claude/CLAUDE.md` for:**
- **Common Claude Pitfalls** (Section: 🚫 Common Claude Pitfalls & Solutions)
  - Wrong: `calculate()` Right: `generate_signals()`
  - Wrong: Lookahead bias patterns Right: Point-in-time queries
  - Wrong: File reading with limits Right: Complete file reads
- **Self-Verification Checklist** (Section: ✅ Claude Self-Verification Checklist)
  - Pre-commit quality gates for data integrity, code quality, testing, documentation
- **Code Review Simulation** (Section: 🔍 Code Review Simulation)
  - Critical questions to ask before any commit
  - Red flags to catch (SQL injection, pandas warnings, etc.)

**Governance Process:**
- CLAUDE.md reviewed quarterly (next: 2026-02-20)
- Common pitfalls capped at 10 items (full list here)
- Error trends inform CLAUDE.md updates

## Next Steps

1. ✅ Document architecture (this file)
2. ⏳ Implement `core/api_contracts.py`
3. ⏳ Implement `core/signal_registry.py`
4. ⏳ Retrofit existing signals
5. ⏳ Add validation scripts
6. ⏳ Set up pre-commit hooks
7. ⏳ Generate API docs

## References

- Python ABCs: https://docs.python.org/3/library/abc.html
- Type Hints: https://docs.python.org/3/library/typing.html
- Dataclasses: https://docs.python.org/3/library/dataclasses.html
- Pre-commit: https://pre-commit.com/
