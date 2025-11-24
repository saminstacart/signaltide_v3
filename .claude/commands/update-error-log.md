Update ERROR_PREVENTION_ARCHITECTURE.md with newly discovered error:

## ⚠️ Before Starting (CLAUDE.md Protocol)

**Self-Verification Requirements:**
Before logging any error, answer:
1. Is this a new pattern or recurring issue?
2. Have I checked ERROR_PREVENTION_ARCHITECTURE.md for similar entries?
3. Do I have complete context (file:line, impact, solution)?
4. Have I classified severity correctly (CRITICAL/HIGH/MEDIUM/LOW)?

**Common Pitfalls to Avoid:**
- **Pitfall #6: File Reading** - Read COMPLETE ERROR_PREVENTION_ARCHITECTURE.md (NO limit!)
- **Pitfall #1: Wrong Method Names** - If logging API errors, verify correct signatures
- **Pitfall #2: Lookahead Bias** - If logging data errors, check temporal discipline

**See** `.claude/CLAUDE.md` → "Self-Verification Checklist" and "Error Prevention Protocol"

**Last Verified:** 2025-11-20

## When to Use This Command

Use whenever you encounter:
- New error pattern not in existing log
- Recurring issue that needs documentation
- Quality/architectural issue discovered
- Bug fix that should be tracked

## Steps

### 1. Read Current Error Log
Read complete `docs/core/ERROR_PREVENTION_ARCHITECTURE.md` (NO limit parameter!)
- Check if error pattern already documented
- Find the "Error Trends Log" section
- Identify pattern count for similar issues

### 2. Gather Error Information

Collect all relevant details:
- **Date**: When error occurred (YYYY-MM-DD format)
- **Pattern Name**: Short, descriptive name
- **Count**: How many times observed (or estimate)
- **Location**: File:line or general area
- **Technical Issue**: What went wrong technically
- **Impact**: What broke or could have broken
- **Solution**: How it was fixed (if fixed)
- **Prevention**: How to prevent in future

### 3. Add New Entry

Add entry to Error Trends Log section using this template:

```markdown
### YYYY-MM-DD: [Error Pattern Name]
**Count:** X occurrences
**Pattern:** [Brief description of pattern]
**Location:** [File:line or general area]
**Issue:** [Technical explanation]
**Impact:** [What broke or could break]
**Solution:** [How it was fixed]
**Status:** [✅ FIXED / ⏳ IN PROGRESS / 📋 PLANNED]
**Prevention:** [How to prevent in future]
```

### 4. Suggest Architectural Improvements

Based on the error pattern, suggest:
- Code review checklist items
- Automated tests to add
- Documentation updates needed
- Architecture changes to consider

## Example Entries (from existing log)

### API Naming Mismatches
```markdown
### 2025-11-20: API Naming Mismatches
**Count:** 5 occurrences
**Pattern:** `generate()` vs `generate_signals()`
**Solution:** BaseSignal ABC with `@abstractmethod`
**Status:** Architecture designed, pending implementation
```

### Inverted Comparison Logic
```markdown
### 2025-11-20: Inverted Comparison Logic in Benchmark Analysis
**Count:** 1 occurrence (CRITICAL)
**Pattern:** Drawdown comparison logic inverted
**Location:** `scripts/spy_benchmark_analysis.py:567-574`
**Issue:** Max drawdown is negative; -28.72% > -34.10% means BETTER
**Impact:** False negative in benchmark scoring (4/5 instead of 5/5)
**Solution:** Fixed comparison logic with explanatory comment
**Status:** ✅ FIXED
**Prevention:** Add unit tests for comparison logic
```

### File Reading with Limit Parameter
```markdown
### 2025-11-20: Reading Files with Limit Parameter (META-ERROR)
**Count:** 1 occurrence (CRITICAL)
**Pattern:** Claude reading files with `limit=100` parameter
**Location:** Analysis of CURRENT_STATE.md, ARCHITECTURE.md
**Issue:** Used Read tool with limit when full context needed
**Impact:** Missing 80%+ of documentation context
**Solution:** Always read complete files unless explicitly confirmed large
**Status:** ✅ FIXED (re-read full files)
**Prevention:**
  - Default to reading full files without limit
  - Only use limit for confirmed large files (>1000 lines)
  - Add reminder to CLAUDE.md about reading complete docs
```

## Template for New Entry

Use this as starting point:

```markdown
### 2025-11-20: [Your Error Pattern Name]
**Count:** [Number] occurrence(s) [CRITICAL/HIGH/MEDIUM/LOW priority]
**Pattern:** [One sentence description]
**Location:** [File path:line or module/area]
**Issue:** [Technical explanation of what went wrong]
**Impact:** [What broke, could break, or was degraded]
**Solution:** [How it was fixed or how to fix it]
**Status:** [✅ FIXED / ⏳ IN PROGRESS / 📋 PLANNED]
**Prevention:** [Specific measures to prevent recurrence]
  - [Prevention measure 1]
  - [Prevention measure 2]
  - [Prevention measure 3]
```

## After Adding Entry

1. **Update Next Steps** section if new architecture needed
2. **Reference in CLAUDE.md** if it's a critical pattern
3. **Create GitHub issue** if fix requires significant work (optional)
4. **Add to code review checklist** if pattern should be caught in review

## Error Classification

### CRITICAL
- Production blockers
- Data integrity issues
- Security vulnerabilities
- Silent failures (wrong results, no error)

### HIGH
- Performance degradation
- API inconsistencies
- Missing validation
- Poor error messages

### MEDIUM
- Code quality issues
- Documentation gaps
- Non-critical bugs
- Inefficient algorithms

### LOW
- Style inconsistencies
- Minor usability issues
- Non-impactful edge cases

## Prevention Strategies by Category

### For API/Interface Errors
- Abstract base classes with `@abstractmethod`
- Type hints and mypy checking
- Unit tests for interface compliance
- Code review checklist

### For Logic Errors
- Unit tests for edge cases
- Property-based testing
- Code review with examples
- Inline comments explaining tricky logic

### For Data Errors
- Validation at boundaries
- Type checking (runtime and static)
- Logging of data transformations
- Integration tests with realistic data

### For Process Errors
- Documentation of procedures
- Checklists for common tasks
- Automated checks where possible
- Post-mortems after incidents

## Example Workflow

```python
# 1. Discovered error while testing
Error: AttributeError: 'InstitutionalMomentum' object has no attribute 'calculate'

# 2. Analyze
- Pattern: Called wrong method name (calculate instead of generate_signals)
- Root cause: No enforced API contract
- Impact: Runtime error, wasted time debugging
- Already seen 5 times before

# 3. Document in ERROR_PREVENTION_ARCHITECTURE.md
### 2025-11-20: Wrong Method Name Called (Recurring)
**Count:** 6 occurrences
**Pattern:** Called `calculate()` instead of `generate_signals()`
**Location:** Multiple scripts calling signal classes
**Issue:** No enforced API contract for signal interface
**Impact:** Runtime errors, debugging time wasted
**Solution:** Need to implement BaseSignal ABC from architecture doc
**Status:** 📋 PLANNED (architecture designed in this file)
**Prevention:**
  - Implement BaseSignal ABC with @abstractmethod
  - Add mypy type checking to CI/CD
  - Update CLAUDE.md to emphasize correct method name

# 4. Take action
- Priority: HIGH (recurring 6 times)
- Next step: Implement BaseSignal ABC (Phase: Error Prevention Architecture)
```

## Important Notes

- **Be honest about impact** - don't downplay serious issues
- **Be specific about prevention** - actionable items, not vague suggestions
- **Reference locations** - file:line for precision
- **Track trends** - note if this is recurring pattern
- **Update CLAUDE.md** if pattern is critical for future sessions
- **This is A+++ quality control** - essential for maintaining standards

## Output

After updating ERROR_PREVENTION_ARCHITECTURE.md, provide summary:

```markdown
## Error Log Updated

**Error Added:** [Pattern Name]
**Priority:** [CRITICAL/HIGH/MEDIUM/LOW]
**Status:** [✅ FIXED / ⏳ IN PROGRESS / 📋 PLANNED]

**Summary:** [One paragraph explaining the error, impact, and prevention]

**Next Actions:**
1. [Action item 1 if any]
2. [Action item 2 if any]

**Error Log:** `docs/core/ERROR_PREVENTION_ARCHITECTURE.md` (updated)
```
